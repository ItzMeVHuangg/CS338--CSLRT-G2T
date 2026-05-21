import argparse
import json
import math
import os
import random
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import Vocabulary
from data.gloss_text_dataset import (
    GlossTextDataset,
    add_confidence_tokens,
    build_vocabularies_from_annotations,
    collate_gloss_text,
    load_predicted_gloss,
    slot_dim,
)
from models.gloss_to_text import DomainSlotGlossToText


def resolve_path(path_value: Optional[str], base_dir: Path = PROJECT_ROOT) -> Optional[str]:
    if path_value in (None, ""):
        return None
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_vocabulary(obj) -> Vocabulary:
    if isinstance(obj, Vocabulary):
        return obj
    if hasattr(obj, "token2idx") and hasattr(obj, "idx2token"):
        obj = obj.token2idx
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


def load_vocabularies(annot_root: str, cslr_ckpt: Optional[str]) -> Tuple[Vocabulary, Vocabulary]:
    fallback = None
    if cslr_ckpt:
        ckpt_path = Path(cslr_ckpt)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                gloss_obj = ckpt.get("gloss_vocab") or ckpt.get("gloss_vocabulary")
                text_obj = ckpt.get("text_vocab") or ckpt.get("translation_vocab")
                if gloss_obj is not None and text_obj is not None:
                    print(f"[Vocab] Loaded gloss/text vocab from checkpoint: {ckpt_path}")
                    return as_vocabulary(gloss_obj), as_vocabulary(text_obj)
                if gloss_obj is not None:
                    print(f"[Vocab] Loaded gloss vocab from checkpoint; text vocab from annotations: {ckpt_path}")
                    fallback = build_vocabularies_from_annotations(annot_root)
                    return as_vocabulary(gloss_obj), fallback[1]
            print(f"[Vocab] Checkpoint has no usable vocab fields, falling back to annotations: {ckpt_path}")
        else:
            print(f"[Vocab] CSLR checkpoint not found, falling back to annotations: {ckpt_path}")
    print(f"[Vocab] Building from annotations: {annot_root}")
    return fallback or build_vocabularies_from_annotations(annot_root)


class LabelSmoothedCE(nn.Module):
    def __init__(self, smoothing: float, pad_idx: int):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=smoothing, ignore_index=pad_idx)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bsz, steps, vocab = logits.shape
        return self.loss(logits.reshape(bsz * steps, vocab), targets.reshape(bsz * steps))


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int, eta_min_ratio: float):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(eta_min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def simple_bleu(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    def ngrams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))]

    precisions = []
    hyp_len = 0
    ref_len = 0
    for n in range(1, 5):
        match = 0
        total = 0
        for hyp, ref in zip(hypotheses, references):
            hyp_t = hyp.split()
            ref_t = ref.split()
            hyp_len += len(hyp_t) if n == 1 else 0
            ref_len += len(ref_t) if n == 1 else 0
            ref_counts = {}
            for gram in ngrams(ref_t, n):
                ref_counts[gram] = ref_counts.get(gram, 0) + 1
            for gram in ngrams(hyp_t, n):
                total += 1
                if ref_counts.get(gram, 0) > 0:
                    match += 1
                    ref_counts[gram] -= 1
        precisions.append(100.0 * match / max(total, 1))

    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / max(hyp_len, 1))
    bleu4 = bp * math.exp(sum(math.log(max(p / 100.0, 1e-9)) for p in precisions) / 4.0) * 100.0
    return {"bleu": bleu4, "bleu1": precisions[0], "bleu2": precisions[1], "bleu3": precisions[2], "bleu4": precisions[3]}


def compute_metrics(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    try:
        from utils.metrics import compute_bleu
        return compute_bleu(hypotheses, references)
    except Exception as exc:
        print(f"[Metric] Falling back to built-in BLEU approximation: {exc}")
        return simple_bleu(hypotheses, references)


def decode_batch(text_vocab: Vocabulary, token_batch: torch.Tensor) -> List[str]:
    return [" ".join(text_vocab.decode(row.tolist())) for row in token_batch]


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def evaluate(model, loader, criterion, text_vocab, device, bos_idx, eos_idx, max_len: int):
    model.eval()
    total_loss = 0.0
    hypotheses, references = [], []
    examples = []

    for batch in tqdm(loader, desc="[G2T Eval]", leave=False):
        if is_pretrained_seq2seq(model):
            loss = model.forward_loss(batch, device)
            total_loss += loss.item()
            hyp_sents = model.generate_text(batch, device, max_len=max_len)
            ref_sents = list(batch["translation_text"])
        else:
            gloss = batch["gloss"].to(device)
            gloss_lens = batch["gloss_lens"].to(device)
            slots = batch["weather_slots"].to(device)
            tgt = batch["translation"].to(device)
            tgt_lens = batch["translation_lens"].to(device)

            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            tgt_lens_in = (tgt_lens - 1).clamp(min=1)
            logits = model(gloss, gloss_lens, tgt_in, tgt_lens_in, slots)
            total_loss += criterion(logits, tgt_out).item()

            pred = model.greedy_decode(gloss, gloss_lens, bos_idx, eos_idx, slots, max_len=max_len)
            hyp_sents = decode_batch(text_vocab, pred.cpu())
            ref_sents = decode_batch(text_vocab, tgt.cpu())
        hypotheses.extend(hyp_sents)
        references.extend(ref_sents)

        if len(examples) < 8:
            batch_size = len(batch["video_ids"])
            for i in range(min(batch_size, 8 - len(examples))):
                examples.append({
                    "video_id": batch["video_ids"][i],
                    "gloss": batch["gloss_text"][i],
                    "confidence_bucket": batch["confidence_bucket"][i],
                    "reference": ref_sents[i],
                    "hypothesis": hyp_sents[i],
                })

    metrics = compute_metrics(hypotheses, references)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics, examples


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, gloss_pad_idx: int, text_pad_idx: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(
            collate_gloss_text,
            gloss_pad_idx=gloss_pad_idx,
            text_pad_idx=text_pad_idx,
        ),
    )


def is_pretrained_seq2seq(model: nn.Module) -> bool:
    return hasattr(model, "forward_loss") and hasattr(model, "generate_text")


def build_g2t_model(
    tc: dict,
    gloss_vocab: Vocabulary,
    text_vocab: Vocabulary,
    gloss_pad_idx: int,
    text_pad_idx: int,
) -> nn.Module:
    model_type = str(tc.get("model_type", "domain_slot_transformer")).lower()

    if model_type in {"mbart", "mt5", "pretrained_seq2seq"}:
        from models.gloss_to_text_mbart import PretrainedSeq2SeqGlossToText

        return PretrainedSeq2SeqGlossToText(
            model_name=tc.get("pretrained_model_name", "facebook/mbart-large-50"),
            source_prefix=tc.get("source_prefix", "translate gloss to German: "),
            src_lang=tc.get("src_lang", "de_DE"),
            tgt_lang=tc.get("tgt_lang", "de_DE"),
            max_src_len=tc.get("max_src_len", 128),
            max_tgt_len=tc.get("max_tgt_len", 128),
            num_beams=tc.get("num_beams", 4),
            length_penalty=tc.get("length_penalty", 1.0),
            freeze_encoder=tc.get("freeze_encoder", False),
            gradient_checkpointing=tc.get("gradient_checkpointing", False),
            use_fast_tokenizer=tc.get("use_fast_tokenizer", False),
        )

    common_kwargs = dict(
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
    )

    if model_type in {"cvae", "latent_cvae", "latent_transformer"}:
        from models.gloss_to_text_cvae import LatentDomainSlotGlossToText

        return LatentDomainSlotGlossToText(
            **common_kwargs,
            latent_dim=tc.get("latent_dim", 64),
        )

    if model_type not in {"domain_slot_transformer", "transformer", "vanilla_transformer"}:
        raise ValueError(f"Unknown g2t_slt.model_type: {model_type}")

    return DomainSlotGlossToText(**common_kwargs)


def auxiliary_loss(model: nn.Module, tc: dict, update_step: int) -> Optional[torch.Tensor]:
    if not hasattr(model, "auxiliary_loss"):
        return None
    aux = model.auxiliary_loss()
    if aux is None:
        return None
    weight = float(tc.get("kl_weight", 0.0))
    if weight <= 0:
        return None
    warmup_steps = int(tc.get("kl_warmup_steps", 0) or 0)
    if warmup_steps > 0:
        weight *= min(1.0, float(update_step + 1) / float(warmup_steps))
    return weight * aux


def train_g2t_slt(cfg: dict, cslr_ckpt: Optional[str] = None):
    seed = cfg.get("seed", 42)
    set_seed(seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[G2T] Device: {device}")

    paths = cfg["paths"]
    annot_root = resolve_path(paths["annot_root"])
    cslr_ckpt = resolve_path(cslr_ckpt) if cslr_ckpt else None
    gloss_vocab, text_vocab = load_vocabularies(annot_root, cslr_ckpt)

    tc = cfg["g2t_slt"]
    use_confidence_token = bool(tc.get("use_confidence_token", False))
    if use_confidence_token:
        add_confidence_tokens(gloss_vocab)
    print(f"[Vocab] Gloss={len(gloss_vocab)} | Text={len(text_vocab)}")

    gloss_pad_idx = gloss_vocab.token2idx.get(gloss_vocab.PAD, 1)
    text_pad_idx = text_vocab.token2idx.get(text_vocab.PAD, 1)
    bos_idx = text_vocab.token2idx[text_vocab.BOS]
    eos_idx = text_vocab.token2idx[text_vocab.EOS]

    experiment_name = cfg.get("experiment_name") or tc.get("experiment_name") or "g2t_slt"
    pred_cfg = cfg.get("predicted_gloss", {})
    pred_train = load_predicted_gloss(resolve_path(pred_cfg.get("train")), include_metadata=use_confidence_token)
    pred_dev = load_predicted_gloss(resolve_path(pred_cfg.get("dev")), include_metadata=use_confidence_token)
    pred_test = load_predicted_gloss(resolve_path(pred_cfg.get("test")), include_metadata=use_confidence_token)
    train_source = tc.get("train_source", "gt").lower()
    use_pred_train = train_source == "predicted" and pred_train is not None

    train_ds = GlossTextDataset(
        "train", annot_root, gloss_vocab, text_vocab,
        predicted_gloss=pred_train if use_pred_train else None,
        use_weather_slots=tc.get("use_weather_slots", True),
        use_confidence_token=use_confidence_token,
        confidence_augmentation=tc.get("confidence_augmentation"),
    )
    dev_ds = GlossTextDataset(
        "dev", annot_root, gloss_vocab, text_vocab,
        use_weather_slots=tc.get("use_weather_slots", True),
        use_confidence_token=use_confidence_token,
    )
    dev_pred_ds = GlossTextDataset(
        "dev", annot_root, gloss_vocab, text_vocab, predicted_gloss=pred_dev,
        use_weather_slots=tc.get("use_weather_slots", True),
        use_confidence_token=use_confidence_token,
    ) if pred_dev else None
    test_pred_ds = GlossTextDataset(
        "test", annot_root, gloss_vocab, text_vocab, predicted_gloss=pred_test,
        use_weather_slots=tc.get("use_weather_slots", True),
        use_confidence_token=use_confidence_token,
    ) if pred_test else None

    num_workers = cfg.get("data", {}).get("num_workers", 0)
    train_loader = make_loader(train_ds, tc["batch_size"], True, num_workers, gloss_pad_idx, text_pad_idx)
    dev_loader = make_loader(dev_ds, tc["eval_batch_size"], False, num_workers, gloss_pad_idx, text_pad_idx)
    dev_pred_loader = make_loader(dev_pred_ds, tc["eval_batch_size"], False, num_workers, gloss_pad_idx, text_pad_idx) if dev_pred_ds else None
    test_pred_loader = make_loader(test_pred_ds, tc["eval_batch_size"], False, num_workers, gloss_pad_idx, text_pad_idx) if test_pred_ds else None

    model = build_g2t_model(tc, gloss_vocab, text_vocab, gloss_pad_idx, text_pad_idx).to(device)
    trainable_params, total_params = count_parameters(model)

    criterion = None if is_pretrained_seq2seq(model) else LabelSmoothedCE(tc.get("label_smoothing", 0.1), text_pad_idx)
    optimizer = AdamW(model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    grad_acc = tc.get("grad_accumulation_steps", 1)
    update_steps = math.ceil(len(train_loader) / grad_acc) * tc["num_epochs"]
    eta_min_ratio = tc.get("eta_min", 1e-7) / tc["learning_rate"]
    scheduler = cosine_with_warmup(optimizer, tc.get("warmup_steps", 200), update_steps, eta_min_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and tc.get("use_amp", True)))

    ckpt_root = Path(paths.get("checkpoint_dir", "checkpoints_g2t_slt"))
    if not ckpt_root.is_absolute():
        ckpt_root = PROJECT_ROOT / ckpt_root
    ckpt_dir = ckpt_root / experiment_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_g2t_slt.pth"
    examples_path = ckpt_dir / "dev_examples.json"
    history_path = ckpt_dir / "metrics_history.jsonl"
    summary_path = ckpt_dir / "summary.json"
    if history_path.exists():
        history_path.unlink()

    print(
        f"[G2T] Experiment={experiment_name} | Train samples={len(train_ds)} | Dev={len(dev_ds)} | "
        f"batch={tc['batch_size']} | epochs={tc['num_epochs']} | "
        f"model_type={tc.get('model_type', 'domain_slot_transformer')} | "
        f"weather_slots={tc.get('use_weather_slots', True)} | conv_gate={tc.get('use_conv_gate', True)} | "
        f"confidence_token={use_confidence_token}"
    )
    print(f"[G2T] Params trainable={trainable_params:,} | total={total_params:,}")
    if use_pred_train:
        print(f"[G2T] Train source: predicted gloss ({len(pred_train)} samples)")
    else:
        print("[G2T] Train source: ground-truth gloss")
    if pred_dev:
        print(f"[G2T] Predicted dev gloss loaded: {len(pred_dev)} samples")
    if pred_test:
        print(f"[G2T] Predicted test gloss loaded: {len(pred_test)} samples")

    best_bleu = -1.0
    best_record = {}
    no_improve = 0
    patience = tc.get("early_stopping", {}).get("patience", 8)
    do_es = tc.get("early_stopping", {}).get("enabled", True)
    update_step = 0

    for epoch in range(tc["num_epochs"]):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        start = time.time()
        pbar = tqdm(train_loader, desc=f"[G2T Train] Epoch {epoch + 1}", leave=False)

        for step, batch in enumerate(pbar):
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and tc.get("use_amp", True))):
                if is_pretrained_seq2seq(model):
                    loss = model.forward_loss(batch, device)
                else:
                    gloss = batch["gloss"].to(device)
                    gloss_lens = batch["gloss_lens"].to(device)
                    slots = batch["weather_slots"].to(device)
                    tgt = batch["translation"].to(device)
                    tgt_lens = batch["translation_lens"].to(device)

                    tgt_in = tgt[:, :-1]
                    tgt_out = tgt[:, 1:]
                    tgt_lens_in = (tgt_lens - 1).clamp(min=1)

                    logits = model(gloss, gloss_lens, tgt_in, tgt_lens_in, slots)
                    loss = criterion(logits, tgt_out)
                    aux = auxiliary_loss(model, tc, update_step)
                    if aux is not None:
                        loss = loss + aux
                loss = loss / grad_acc

            scaler.scale(loss).backward()
            should_step = ((step + 1) % grad_acc == 0) or (step + 1 == len(train_loader))
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["gradient_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                update_step += 1

            total_loss += loss.item() * grad_acc
            pbar.set_postfix(loss=f"{loss.item() * grad_acc:.4f}")

        train_loss = total_loss / max(len(train_loader), 1)
        dev_metrics, dev_examples = evaluate(
            model, dev_loader, criterion, text_vocab, device, bos_idx, eos_idx, tc["max_tgt_len"]
        )
        elapsed = time.time() - start
        print(
            f"Epoch [{epoch + 1:03d}/{tc['num_epochs']}] "
            f"loss={train_loss:.4f} | dev_BLEU4={dev_metrics['bleu']:.2f} | "
            f"dev_loss={dev_metrics['loss']:.4f} | lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.0f}s"
        )

        pred_metrics = None
        if dev_pred_loader is not None:
            pred_metrics, pred_examples = evaluate(
                model, dev_pred_loader, criterion, text_vocab, device, bos_idx, eos_idx, tc["max_tgt_len"]
            )
            print(f"  Pred-gloss dev BLEU4={pred_metrics['bleu']:.2f} | loss={pred_metrics['loss']:.4f}")

        epoch_record = {
            "experiment": experiment_name,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "oracle_dev": dev_metrics,
            "pred_dev": pred_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_sec": elapsed,
        }
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")

        if dev_metrics["bleu"] > best_bleu:
            best_bleu = dev_metrics["bleu"]
            best_record = epoch_record
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "bleu": best_bleu,
                "dev_metrics": dev_metrics,
                "gloss_vocab": gloss_vocab,
                "text_vocab": text_vocab,
                "cfg": cfg,
            }, best_path)
            with open(examples_path, "w", encoding="utf-8") as f:
                json.dump(dev_examples, f, ensure_ascii=False, indent=2)
            if pred_metrics is not None:
                write_json(ckpt_dir / "dev_pred_metrics_at_best_oracle.json", pred_metrics)
                write_json(ckpt_dir / "dev_pred_examples_at_best_oracle.json", {"examples": pred_examples})
            print(f"  Saved best checkpoint: {best_path} (BLEU4={best_bleu:.2f})")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{patience})")

        if do_es and no_improve >= patience:
            print(f"[G2T] Early stopping at epoch {epoch + 1}")
            break

    final_eval = {}
    if best_path.exists():
        saved = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        oracle_metrics, oracle_examples = evaluate(
            model, dev_loader, criterion, text_vocab, device, bos_idx, eos_idx, tc["max_tgt_len"]
        )
        final_eval["oracle_dev"] = oracle_metrics
        write_json(ckpt_dir / "oracle_dev_examples_best.json", {"examples": oracle_examples})

    if best_path.exists() and dev_pred_loader is not None:
        pred_metrics, pred_examples = evaluate(
            model, dev_pred_loader, criterion, text_vocab, device, bos_idx, eos_idx, tc["max_tgt_len"]
        )
        final_eval["pred_dev"] = pred_metrics
        write_json(ckpt_dir / "pred_dev_examples_best.json", {"examples": pred_examples})
        print(f"[G2T] Best checkpoint pred-gloss dev BLEU4={pred_metrics['bleu']:.2f} | loss={pred_metrics['loss']:.4f}")

    if best_path.exists() and test_pred_loader is not None:
        test_metrics, test_examples = evaluate(
            model, test_pred_loader, criterion, text_vocab, device, bos_idx, eos_idx, tc["max_tgt_len"]
        )
        test_path = ckpt_dir / "test_pred_examples.json"
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_examples, f, ensure_ascii=False, indent=2)
        final_eval["pred_test"] = test_metrics
        print(f"[G2T] Pred-gloss test BLEU4={test_metrics['bleu']:.2f} | examples={test_path}")

    summary = {
        "experiment": experiment_name,
        "checkpoint": str(best_path),
        "history": str(history_path),
        "best_oracle_dev_bleu4": best_bleu,
        "best_record": best_record,
        "final_eval": final_eval,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "model_switches": {
            "model_type": tc.get("model_type", "domain_slot_transformer"),
            "pretrained_model_name": tc.get("pretrained_model_name"),
            "domain_prefix_len": tc.get("domain_prefix_len", 1),
            "use_weather_slots": tc.get("use_weather_slots", True),
            "use_conv_gate": tc.get("use_conv_gate", True),
            "latent_dim": tc.get("latent_dim"),
            "kl_weight": tc.get("kl_weight"),
            "use_confidence_token": use_confidence_token,
            "confidence_augmentation": tc.get("confidence_augmentation", {}),
            "label_smoothing": tc.get("label_smoothing", 0.1),
            "train_source": "predicted" if use_pred_train else "gt",
        },
    }
    write_json(summary_path, summary)
    print(f"[G2T] Done. Best oracle-dev BLEU4={best_bleu:.2f}")
    print(f"[G2T] Summary: {summary_path}")
    return best_bleu


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.experiment_name:
        cfg["experiment_name"] = args.experiment_name
    if args.annot_root:
        cfg.setdefault("paths", {})["annot_root"] = args.annot_root
    if args.checkpoint_dir:
        cfg.setdefault("paths", {})["checkpoint_dir"] = args.checkpoint_dir
    pred_cfg = cfg.setdefault("predicted_gloss", {})
    if args.pred_train:
        pred_cfg["train"] = args.pred_train
    if args.pred_dev:
        pred_cfg["dev"] = args.pred_dev
    if args.pred_test:
        pred_cfg["test"] = args.pred_test
    if args.num_epochs is not None:
        cfg.setdefault("g2t_slt", {})["num_epochs"] = args.num_epochs
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fast Gloss-to-Text SLT without retraining CSLR")
    parser.add_argument("--config", default="configs/config_g2t_slt.yaml")
    parser.add_argument("--cslr_ckpt", default=None, help="Optional checkpoint to reuse vocabularies")
    parser.add_argument("--annot_root", default=None, help="Override PHOENIX annotation directory")
    parser.add_argument("--checkpoint_dir", default=None, help="Override output checkpoint directory")
    parser.add_argument("--experiment_name", default=None, help="Override experiment/checkpoint subdirectory name")
    parser.add_argument("--pred_train", default=None, help="Optional predicted-gloss JSONL for train")
    parser.add_argument("--pred_dev", default=None, help="Optional predicted-gloss JSONL for dev evaluation")
    parser.add_argument("--pred_test", default=None, help="Optional predicted-gloss JSONL for test evaluation")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override number of G2T training epochs")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_cli_overrides(cfg, args)
    train_g2t_slt(cfg, args.cslr_ckpt)
