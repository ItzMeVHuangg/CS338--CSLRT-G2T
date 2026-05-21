import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

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

import data.dataset as phoenix_data
from data.dataset import PhoenixDataset, Vocabulary, build_transforms, build_vocabularies, collate_fn
from data.gloss_text_dataset import confidence_bucket_from_score
from training.train_final import build_cslr_model as build_final_cslr_model
from utils.ctc_decoder import batch_ctc_decode
from utils.metrics import compute_wer


def resolve_path(path_value: str | None) -> str | None:
    if path_value in (None, ""):
        return None
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def patch_dataset_paths(cfg: dict):
    paths = cfg["paths"]
    phoenix_data.PHOENIX_ROOT = Path(resolve_path(paths["phoenix_root"]))
    phoenix_data.FRAMES_ROOT = Path(resolve_path(paths["frames_root"]))
    phoenix_data.ANNOT_ROOT = Path(resolve_path(paths["annot_root"]))


def apply_path_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    paths = cfg.setdefault("paths", {})
    if args.phoenix_root:
        paths["phoenix_root"] = args.phoenix_root
    if args.frames_root:
        paths["frames_root"] = args.frames_root
    if args.annot_root:
        paths["annot_root"] = args.annot_root
    return cfg


def as_vocabulary(obj) -> Vocabulary:
    if isinstance(obj, Vocabulary):
        return obj
    if hasattr(obj, "token2idx") and hasattr(obj, "idx2token"):
        return obj
    if isinstance(obj, dict):
        if "token2idx" in obj:
            obj = obj["token2idx"]
        vocab = Vocabulary()
        vocab.token2idx = {str(k): int(v) for k, v in obj.items()}
        vocab.idx2token = {int(v): str(k) for k, v in obj.items()}
        for token in [vocab.BLANK, vocab.PAD, vocab.UNK, vocab.BOS, vocab.EOS]:
            if token not in vocab.token2idx:
                vocab._add(token)
        return vocab
    raise TypeError(f"Unsupported vocabulary object: {type(obj)}")


def load_vocab_from_checkpoint_or_annotations(ckpt, cfg):
    fallback = None
    if isinstance(ckpt, dict):
        gloss_vocab = ckpt.get("gloss_vocab") or ckpt.get("gloss_vocabulary")
        text_vocab = ckpt.get("text_vocab") or ckpt.get("translation_vocab")
        if gloss_vocab is not None and text_vocab is not None:
            return as_vocabulary(gloss_vocab), as_vocabulary(text_vocab)
        if gloss_vocab is not None:
            fallback = build_vocabularies()
            return as_vocabulary(gloss_vocab), fallback[1]
    return fallback or build_vocabularies()


def infer_seq_model(ckpt, requested: str) -> str:
    if requested and requested != "auto":
        return requested
    if isinstance(ckpt, dict):
        seq_model = ckpt.get("seq_model") or ckpt.get("sequence_model")
        if isinstance(seq_model, str):
            return seq_model
    return "bilstm"


def build_export_cslr_model(
    cfg,
    num_classes: int,
    encoder: str,
    seq_model: str,
    builder: str = "auto",
):
    if builder == "auto":
        # Variant G Swin checkpoints were trained by train_ablation.py. That
        # wrapper uses a different temporal-length adjustment from train_final.py,
        # so export must rebuild the same wrapper or CTC decoding degrades badly.
        builder = "ablation" if encoder == "swin" or seq_model != "bilstm" else "final"

    if builder == "final":
        if seq_model != "bilstm":
            raise ValueError("The final builder only supports seq_model=bilstm")
        return build_final_cslr_model(cfg, num_classes, encoder)

    from training.train_ablation import build_cslr_model as build_ablation_cslr_model

    encoder_map = {
        "swin": "video_swin",
        "resnet34": "cnn_2d",
    }
    if encoder not in encoder_map:
        raise ValueError(f"Unsupported encoder for ablation export: {encoder}")
    return build_ablation_cslr_model(
        cfg,
        num_classes,
        encoder_type=encoder_map[encoder],
        seq_model_type=seq_model,
    )


def load_state(model, ckpt):
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict")
    state = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if state is None:
        raise KeyError("Checkpoint has no model/model_state_dict/state_dict key")

    def remap_legacy_swin_state(raw_state):
        mapped = {}
        for key, value in raw_state.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module."):]
            if new_key.startswith("model."):
                new_key = new_key[len("model."):]

            # Older Variant G checkpoints saved VideoSwinEncoder under
            # encoder.backbone.*, while the current encoder exposes the same
            # modules directly as encoder.patch_embed/stages/swin_norm.
            if new_key.startswith("encoder.backbone.patch_embed."):
                new_key = "encoder.patch_embed." + new_key[len("encoder.backbone.patch_embed."):]
            elif new_key.startswith("encoder.backbone.features."):
                new_key = "encoder.stages." + new_key[len("encoder.backbone.features."):]
            elif new_key.startswith("encoder.backbone.norm."):
                new_key = "encoder.swin_norm." + new_key[len("encoder.backbone.norm."):]

            # Older ablation checkpoints saved the BiLSTM-CTC sequence model as
            # flat seq_* modules instead of seq_model.<module>.
            if new_key.startswith("seq_bilstm."):
                new_key = "seq_model.bilstm." + new_key[len("seq_bilstm."):]
            elif new_key.startswith("seq_projection."):
                new_key = "seq_model.projection." + new_key[len("seq_projection."):]
            elif new_key.startswith("seq_ctc_head."):
                new_key = "seq_model.ctc_head." + new_key[len("seq_ctc_head."):]
            elif new_key.startswith("seq_layer_norm."):
                new_key = "seq_model.layer_norm." + new_key[len("seq_layer_norm."):]

            mapped[new_key] = value
        return mapped

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        candidates = [
            {k.replace("module.", "", 1): v for k, v in state.items()},
            {k.replace("cslr.", "", 1): v for k, v in state.items()},
            {k.replace("model.", "", 1): v for k, v in state.items()},
            remap_legacy_swin_state(state),
        ]
        last_error = None
        for candidate in candidates:
            try:
                model.load_state_dict(candidate, strict=True)
                print("[Export] loaded checkpoint with key remapping.", flush=True)
                return
            except RuntimeError as exc:
                last_error = exc
        raise last_error


def estimate_ctc_confidences(
    log_probs: torch.Tensor,
    adjusted_lengths: torch.Tensor,
    blank_idx: int,
) -> list[float]:
    """Mean non-blank greedy-frame probability per sample."""
    probs = log_probs.exp()
    top_prob, top_idx = probs.max(dim=-1)
    max_steps = log_probs.size(0)
    scores = []
    for batch_idx in range(log_probs.size(1)):
        valid_len = int(adjusted_lengths[batch_idx].clamp(min=1, max=max_steps).item())
        frame_prob = top_prob[:valid_len, batch_idx]
        frame_idx = top_idx[:valid_len, batch_idx]
        non_blank = frame_idx.ne(blank_idx)
        if non_blank.any():
            score = frame_prob[non_blank].mean()
        else:
            score = frame_prob.mean()
        scores.append(float(score.detach().cpu().item()))
    return scores


@torch.no_grad()
def export_gloss(
    cfg,
    cslr_ckpt_path: str,
    split: str,
    output: str,
    encoder: str,
    seq_model: str,
    builder: str,
    batch_size: int,
    max_samples: int | None = None,
    confidence_low: float = 0.55,
    confidence_high: float = 0.75,
):
    patch_dataset_paths(cfg)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cslr_ckpt_path = resolve_path(cslr_ckpt_path)
    print(f"[Export] device={device}", flush=True)
    print(f"[Export] checkpoint={cslr_ckpt_path}", flush=True)
    print(f"[Export] split={split} | encoder={encoder}", flush=True)
    if not Path(cslr_ckpt_path).exists():
        raise FileNotFoundError(f"CSLR checkpoint not found: {cslr_ckpt_path}")

    if encoder == "swin" and "video_swin" in cfg:
        # The checkpoint contains trained weights, so avoid any torchvision
        # pretrained-weight download during inference/export.
        cfg["video_swin"]["pretrained"] = False
    if encoder == "resnet34" and "cnn" in cfg:
        cfg["cnn"]["pretrained"] = False

    print("[Export] loading checkpoint...", flush=True)
    ckpt = torch.load(cslr_ckpt_path, map_location="cpu", weights_only=False)
    seq_model = infer_seq_model(ckpt, seq_model)
    print(f"[Export] seq_model={seq_model}", flush=True)
    print(f"[Export] builder={builder}", flush=True)
    print("[Export] loading vocabulary...", flush=True)
    gloss_vocab, text_vocab = load_vocab_from_checkpoint_or_annotations(ckpt, cfg)

    print("[Export] building dataset...", flush=True)
    ds = PhoenixDataset(
        split=split,
        gloss_vocab=gloss_vocab,
        text_vocab=text_vocab,
        max_frames=cfg["data"]["max_frames"],
        temporal_stride=cfg["data"]["temporal_stride"],
        transform=build_transforms(split, cfg["data"]["img_height"], cfg["data"]["img_width"]),
        return_translation=True,
    )
    if max_samples is not None and max_samples > 0:
        ds = Subset(ds, list(range(min(max_samples, len(ds)))))
        print(f"[Export] max_samples={len(ds)}", flush=True)

    export_num_workers = int(os.environ.get("EXPORT_NUM_WORKERS", "0"))
    export_pin_memory = os.environ.get("EXPORT_PIN_MEMORY", "0") == "1" and torch.cuda.is_available()
    print(
        f"[Export] building dataloader... num_workers={export_num_workers} "
        f"pin_memory={export_pin_memory}",
        flush=True,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=export_num_workers,
        collate_fn=collate_fn,
        pin_memory=export_pin_memory,
    )

    print("[Export] building CSLR model...", flush=True)
    model = build_export_cslr_model(cfg, len(gloss_vocab), encoder, seq_model, builder).to(device)
    print("[Export] loading CSLR weights...", flush=True)
    load_state(model, ckpt)
    model.eval()

    output_path = Path(resolve_path(output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hyp_glosses, ref_glosses = [], []
    bucket_counts = Counter()

    with open(output_path, "w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc=f"[Export {split}]"):
            frames = batch["frames"].to(device)
            frame_lens = batch["frame_lens"].to(device)
            ref_gloss = batch["gloss"]
            ref_lens = batch["gloss_lens"]
            ref_text = batch.get("translation")

            log_probs, _, adj_lens = model(frames, frame_lens)
            ctc_confidences = estimate_ctc_confidences(
                log_probs,
                adj_lens.clamp(max=log_probs.size(0)),
                blank_idx=cfg["cslr"]["ctc_blank_idx"],
            )
            preds = batch_ctc_decode(
                log_probs,
                adj_lens.clamp(max=log_probs.size(0)),
                blank_idx=cfg["cslr"]["ctc_blank_idx"],
                mode="greedy",
            )

            for i, pred in enumerate(preds):
                pred_tokens = gloss_vocab.decode(pred)
                ref_tokens = gloss_vocab.decode(
                    ref_gloss[i, :ref_lens[i].item()].tolist(),
                    skip_special=True,
                )
                hyp_glosses.append(pred_tokens)
                ref_glosses.append(ref_tokens)
                confidence = ctc_confidences[i]
                confidence_bucket = confidence_bucket_from_score(
                    confidence,
                    low_threshold=confidence_low,
                    high_threshold=confidence_high,
                )
                bucket_counts[confidence_bucket] += 1

                item = {
                    "video_id": batch["video_ids"][i],
                    "gloss": " ".join(pred_tokens),
                    "gloss_tokens": pred_tokens,
                    "confidence": round(confidence, 6),
                    "confidence_bucket": confidence_bucket,
                    "reference_gloss": " ".join(ref_tokens),
                }
                if ref_text is not None:
                    item["reference_text"] = " ".join(text_vocab.decode(ref_text[i].tolist()))
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    wer = compute_wer(hyp_glosses, ref_glosses)
    metrics_path = output_path.with_suffix(output_path.suffix + ".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "split": split,
            "encoder": encoder,
            "seq_model": seq_model,
            "builder": builder,
            "checkpoint": cslr_ckpt_path,
            "output": str(output_path),
            "num_samples": len(hyp_glosses),
            "wer": wer,
            "wer_percent": wer * 100.0,
            "confidence_low_threshold": confidence_low,
            "confidence_high_threshold": confidence_high,
            "confidence_bucket_counts": dict(bucket_counts),
        }, f, ensure_ascii=False, indent=2)
    print(f"[Export] wrote {output_path}")
    print(f"[Export] {split} WER = {wer * 100:.2f}%")
    print(f"[Export] metrics = {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export predicted gloss from a frozen CSLR checkpoint")
    parser.add_argument("--config", default="configs/config_swin.yaml")
    parser.add_argument("--cslr_ckpt", required=True)
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder", default="swin", choices=["swin", "resnet34"])
    parser.add_argument("--seq_model", default="auto", choices=["auto", "bilstm", "transformer", "conformer"])
    parser.add_argument("--builder", default="auto", choices=["auto", "final", "ablation"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument("--confidence_low", type=float, default=0.55, help="CTC confidence threshold for <conf_low>")
    parser.add_argument("--confidence_high", type=float, default=0.75, help="CTC confidence threshold for <conf_high>")
    parser.add_argument("--phoenix_root", default=None, help="Override PHOENIX root")
    parser.add_argument("--frames_root", default=None, help="Override frame directory root")
    parser.add_argument("--annot_root", default=None, help="Override annotation directory")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_path_overrides(cfg, args)
    export_gloss(
        cfg,
        args.cslr_ckpt,
        args.split,
        args.output,
        args.encoder,
        args.seq_model,
        args.builder,
        args.batch_size,
        args.max_samples,
        args.confidence_low,
        args.confidence_high,
    )
