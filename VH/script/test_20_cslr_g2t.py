import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import PhoenixDataset, build_transforms, collate_fn
from data.gloss_text_dataset import (
    CONFIDENCE_TOKEN_BY_BUCKET,
    confidence_bucket_from_score,
    extract_weather_slots,
    slot_dim,
)
from models.gloss_to_text import DomainSlotGlossToText
from script.export_swin_gloss import (
    apply_path_overrides,
    estimate_ctc_confidences,
    load_state,
    load_vocab_from_checkpoint_or_annotations,
    patch_dataset_paths,
    resolve_path,
)
from training.train_final import build_cslr_model
from utils.ctc_decoder import batch_ctc_decode
from utils.metrics import compute_bleu, compute_wer


def as_path_or_none(path_value: Optional[str]) -> Optional[Path]:
    if path_value in (None, ""):
        return None
    return Path(resolve_path(path_value))


def prepare_cslr_config(cfg: dict, encoder: str) -> dict:
    if encoder == "swin" and "video_swin" in cfg:
        cfg["video_swin"]["pretrained"] = False
    if encoder == "resnet34" and "cnn" in cfg:
        cfg["cnn"]["pretrained"] = False
    return cfg


def load_g2t_model(ckpt_path: Path, device: torch.device):
    saved = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = saved["cfg"]
    gloss_vocab = saved["gloss_vocab"]
    text_vocab = saved["text_vocab"]
    tc = cfg["g2t_slt"]

    gloss_pad_idx = gloss_vocab.token2idx.get(gloss_vocab.PAD, 1)
    text_pad_idx = text_vocab.token2idx.get(text_vocab.PAD, 1)
    model = DomainSlotGlossToText(
        gloss_vocab_size=len(gloss_vocab),
        text_vocab_size=len(text_vocab),
        gloss_pad_idx=gloss_pad_idx,
        text_pad_idx=text_pad_idx,
        slot_dim=slot_dim(),
        d_model=tc["d_model"],
        nhead=tc["nhead"],
        num_encoder_layers=tc["num_encoder_layers"],
        num_decoder_layers=tc["num_decoder_layers"],
        dim_feedforward=tc["dim_feedforward"],
        dropout=tc["dropout"],
        max_src_len=tc["max_src_len"],
        max_tgt_len=tc["max_tgt_len"],
        domain_prefix_len=tc.get("domain_prefix_len", 1),
        use_weather_slots=tc.get("use_weather_slots", True),
        use_conv_gate=tc.get("use_conv_gate", True),
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, gloss_vocab, text_vocab, tc


@torch.no_grad()
def translate_predicted_gloss(model, gloss_vocab, text_vocab, tc, pred_tokens, device, confidence_bucket=None):
    model_tokens = list(pred_tokens)
    if tc.get("use_confidence_token", False):
        model_tokens = [confidence_bucket or CONFIDENCE_TOKEN_BY_BUCKET["mid"]] + model_tokens

    if pred_tokens:
        gloss_ids = gloss_vocab.encode(model_tokens)
    else:
        fallback = [confidence_bucket or CONFIDENCE_TOKEN_BY_BUCKET["mid"]] if tc.get("use_confidence_token", False) else []
        gloss_ids = gloss_vocab.encode(fallback) or [gloss_vocab.token2idx.get(gloss_vocab.UNK, 2)]

    gloss = torch.tensor([gloss_ids], dtype=torch.long, device=device)
    gloss_lens = torch.tensor([len(gloss_ids)], dtype=torch.long, device=device)
    slots = torch.tensor([extract_weather_slots(pred_tokens)], dtype=torch.float32, device=device)
    bos_idx = text_vocab.token2idx[text_vocab.BOS]
    eos_idx = text_vocab.token2idx[text_vocab.EOS]
    pred = model.greedy_decode(
        gloss,
        gloss_lens,
        bos_idx,
        eos_idx,
        slots,
        max_len=tc.get("max_tgt_len", 128),
    )
    return " ".join(text_vocab.decode(pred[0].tolist()))


@torch.no_grad()
def run_test20(args):
    random.seed(args.seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_path_overrides(cfg, args)
    cfg = prepare_cslr_config(cfg, args.encoder)
    patch_dataset_paths(cfg)

    cslr_ckpt_path = Path(resolve_path(args.cslr_ckpt))
    if not cslr_ckpt_path.exists():
        raise FileNotFoundError(f"CSLR checkpoint not found: {cslr_ckpt_path}")

    print(f"[Test20] device={device}", flush=True)
    print(f"[Test20] encoder={args.encoder}", flush=True)
    print(f"[Test20] cslr_ckpt={cslr_ckpt_path}", flush=True)

    print("[Test20] loading CSLR checkpoint...", flush=True)
    cslr_ckpt = torch.load(cslr_ckpt_path, map_location="cpu", weights_only=False)
    gloss_vocab, text_vocab = load_vocab_from_checkpoint_or_annotations(cslr_ckpt, cfg)

    print("[Test20] building dataset...", flush=True)
    ds = PhoenixDataset(
        split=args.split,
        gloss_vocab=gloss_vocab,
        text_vocab=text_vocab,
        max_frames=cfg["data"]["max_frames"],
        temporal_stride=cfg["data"]["temporal_stride"],
        transform=build_transforms(args.split, cfg["data"]["img_height"], cfg["data"]["img_width"]),
        return_translation=True,
    )
    n_samples = min(args.n_samples, len(ds))
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(ds)), n_samples)
    loader = DataLoader(
        Subset(ds, indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    print("[Test20] building CSLR model...", flush=True)
    cslr = build_cslr_model(cfg, len(gloss_vocab), args.encoder).to(device)
    load_state(cslr, cslr_ckpt)
    cslr.eval()

    g2t_model = None
    g2t_gloss_vocab = None
    g2t_text_vocab = None
    g2t_tc = None
    g2t_ckpt_path = as_path_or_none(args.g2t_ckpt)
    if g2t_ckpt_path and g2t_ckpt_path.exists():
        print(f"[Test20] loading G2T checkpoint={g2t_ckpt_path}", flush=True)
        g2t_model, g2t_gloss_vocab, g2t_text_vocab, g2t_tc = load_g2t_model(g2t_ckpt_path, device)
    elif g2t_ckpt_path:
        print(f"[Test20] G2T checkpoint not found, skip G2T text: {g2t_ckpt_path}", flush=True)

    samples = []
    hyp_glosses = []
    ref_glosses = []
    hyp_texts = []
    ref_texts = []

    for batch in tqdm(loader, desc=f"[Test20 {args.encoder}]"):
        frames = batch["frames"].to(device)
        frame_lens = batch["frame_lens"].to(device)
        ref_gloss = batch["gloss"]
        ref_lens = batch["gloss_lens"]
        ref_text_tensor = batch["translation"]

        log_probs, _, adj_lens = cslr(frames, frame_lens)
        cslr_confidence = estimate_ctc_confidences(
            log_probs,
            adj_lens.clamp(max=log_probs.size(0)),
            blank_idx=cfg["cslr"]["ctc_blank_idx"],
        )[0]
        confidence_bucket = confidence_bucket_from_score(cslr_confidence)
        pred_ids = batch_ctc_decode(
            log_probs,
            adj_lens.clamp(max=log_probs.size(0)),
            blank_idx=cfg["cslr"]["ctc_blank_idx"],
            mode="greedy",
        )[0]

        pred_tokens = gloss_vocab.decode(pred_ids)
        ref_tokens = gloss_vocab.decode(
            ref_gloss[0, :ref_lens[0].item()].tolist(),
            skip_special=True,
        )
        ref_text = " ".join(text_vocab.decode(ref_text_tensor[0].tolist()))
        hyp_text = None

        if g2t_model is not None:
            hyp_text = translate_predicted_gloss(
                g2t_model,
                g2t_gloss_vocab,
                g2t_text_vocab,
                g2t_tc,
                pred_tokens,
                device,
                confidence_bucket,
            )
            hyp_texts.append(hyp_text)
            ref_texts.append(ref_text)

        hyp_glosses.append(pred_tokens)
        ref_glosses.append(ref_tokens)

        item = {
            "video_id": batch["video_ids"][0],
            "ref_gloss": " ".join(ref_tokens),
            "pred_gloss": " ".join(pred_tokens),
            "confidence": round(cslr_confidence, 6),
            "confidence_bucket": confidence_bucket,
            "ref_text": ref_text,
            "pred_text": hyp_text,
        }
        samples.append(item)
        print(
            f"\n[{len(samples):02d}/{n_samples}] {item['video_id']}\n"
            f"  GLOSS REF : {item['ref_gloss']}\n"
            f"  GLOSS PRED: {item['pred_gloss']}\n"
            f"  TEXT REF  : {item['ref_text']}\n"
            f"  TEXT PRED : {item['pred_text'] if item['pred_text'] is not None else '<skip>'}",
            flush=True,
        )

    wer = compute_wer(hyp_glosses, ref_glosses)
    summary = {
        "encoder": args.encoder,
        "split": args.split,
        "n_samples": n_samples,
        "seed": args.seed,
        "cslr_ckpt": str(cslr_ckpt_path),
        "g2t_ckpt": str(g2t_ckpt_path) if g2t_ckpt_path else None,
        "wer": wer,
        "wer_percent": wer * 100.0,
    }
    if hyp_texts:
        bleu = compute_bleu(hyp_texts, ref_texts)
        summary["bleu"] = bleu
        summary["bleu4"] = bleu["bleu"]

    output_path = Path(resolve_path(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": samples}, f, ensure_ascii=False, indent=2)

    jsonl_path = output_path.with_suffix(".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("\n[Test20] Summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[Test20] wrote {output_path}")
    print(f"[Test20] wrote {jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test 20 PHOENIX samples with a CSLR checkpoint and optional G2T checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--encoder", required=True, choices=["swin", "resnet34"])
    parser.add_argument("--cslr_ckpt", required=True)
    parser.add_argument("--g2t_ckpt", default="checkpoints_g2t_slt/g2t_proposed_adat_gloss2text/best_g2t_slt.pth")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phoenix_root", default=None)
    parser.add_argument("--frames_root", default=None)
    parser.add_argument("--annot_root", default=None)
    run_test20(parser.parse_args())
