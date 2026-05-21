import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from data.dataset import Vocabulary


CONFIDENCE_TOKEN_BY_BUCKET = {
    "low": "<conf_low>",
    "mid": "<conf_mid>",
    "high": "<conf_high>",
}
CONFIDENCE_TOKENS = tuple(CONFIDENCE_TOKEN_BY_BUCKET.values())

WEATHER_SLOT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "time": (
        "HEUTE", "MORGEN", "UEBERMORGEN", "NACHT", "TAG", "FRUEH",
        "ABEND", "MONTAG", "DIENSTAG", "MITTWOCH", "DONNERSTAG",
        "FREITAG", "SAMSTAG", "SONNTAG",
    ),
    "precipitation": (
        "REGEN", "SCHAUER", "SCHNEE", "GEWITTER", "HAGEL",
        "NIESEL", "NIEDERSCHLAG", "UNWETTER",
    ),
    "wind": ("WIND", "STURM", "BOE", "STURMBOE", "WINDIG"),
    "temperature": (
        "TEMPERATUR", "GRAD", "WARM", "KALT", "KUEHL", "HEISS",
        "MINUS", "PLUS",
    ),
    "cloud": ("WOLKE", "BEWOELKT", "NEBEL", "SONNE", "SONNIG", "HEITER"),
    "region": (
        "NORD", "SUED", "WEST", "OST", "MITTE", "ALPEN", "KUESTE",
        "RHEIN", "BAYERN", "DEUTSCHLAND",
    ),
}


def extract_weather_slots(gloss_tokens: Iterable[str]) -> List[float]:
    token_set = set(gloss_tokens)
    slots = []
    for keywords in WEATHER_SLOT_KEYWORDS.values():
        slots.append(1.0 if any(k in token_set for k in keywords) else 0.0)
    return slots


def slot_dim() -> int:
    return len(WEATHER_SLOT_KEYWORDS)


def add_confidence_tokens(vocab: Vocabulary) -> Vocabulary:
    for token in CONFIDENCE_TOKENS:
        vocab._add(token)
    return vocab


def normalize_confidence_bucket(value: Optional[Union[str, float, int]]) -> str:
    if value is None:
        return CONFIDENCE_TOKEN_BY_BUCKET["high"]
    if isinstance(value, (float, int)):
        return confidence_bucket_from_score(float(value))

    bucket = str(value).strip().lower()
    if bucket in CONFIDENCE_TOKEN_BY_BUCKET:
        return CONFIDENCE_TOKEN_BY_BUCKET[bucket]
    if bucket in CONFIDENCE_TOKENS:
        return bucket
    if bucket.replace(".", "", 1).isdigit():
        return confidence_bucket_from_score(float(bucket))
    return CONFIDENCE_TOKEN_BY_BUCKET["high"]


def confidence_bucket_from_score(score: float, low_threshold: float = 0.55, high_threshold: float = 0.75) -> str:
    if score < low_threshold:
        return CONFIDENCE_TOKEN_BY_BUCKET["low"]
    if score < high_threshold:
        return CONFIDENCE_TOKEN_BY_BUCKET["mid"]
    return CONFIDENCE_TOKEN_BY_BUCKET["high"]


def build_vocabularies_from_annotations(annot_root: str | Path) -> Tuple[Vocabulary, Vocabulary]:
    annot_root = Path(annot_root)
    csv_path = annot_root / "PHOENIX-2014-T.train.corpus.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Train CSV not found: {csv_path}")

    gloss_vocab = Vocabulary()
    text_vocab = Vocabulary()
    all_gloss, all_text = [], []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            all_gloss.append(row["orth"].strip().split())
            all_text.append(row["translation"].strip().split())

    gloss_vocab.build_from_sequences(all_gloss)
    text_vocab.build_from_sequences(all_text)
    return gloss_vocab, text_vocab


def _coerce_gloss_tokens(gloss) -> List[str]:
    if gloss is None:
        return []
    return gloss.split() if isinstance(gloss, str) else list(gloss)


def _predicted_record(item: Dict, include_metadata: bool):
    gloss = item.get("gloss") or item.get("pred_gloss") or item.get("prediction")
    tokens = _coerce_gloss_tokens(gloss)
    if not include_metadata:
        return tokens

    confidence = item.get("confidence")
    bucket = (
        item.get("confidence_bucket")
        or item.get("confidence_token")
        or item.get("bucket")
        or confidence
    )
    return {
        "tokens": tokens,
        "confidence": confidence,
        "confidence_bucket": normalize_confidence_bucket(bucket),
    }


def split_predicted_value(value) -> Tuple[List[str], Optional[float], str]:
    if isinstance(value, dict):
        tokens = (
            value.get("tokens")
            or value.get("gloss_tokens")
            or value.get("gloss")
            or value.get("pred_gloss")
            or value.get("prediction")
            or []
        )
        confidence = value.get("confidence")
        bucket = (
            value.get("confidence_bucket")
            or value.get("confidence_token")
            or value.get("bucket")
            or confidence
        )
        return _coerce_gloss_tokens(tokens), confidence, normalize_confidence_bucket(bucket)
    return _coerce_gloss_tokens(value), None, CONFIDENCE_TOKEN_BY_BUCKET["high"]


def load_predicted_gloss(path: Optional[str | Path], include_metadata: bool = False) -> Optional[Dict[str, object]]:
    if not path:
        return None

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Predicted gloss file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        pred = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                video_id = item.get("video_id") or item.get("name") or item.get("id")
                gloss = item.get("gloss") or item.get("pred_gloss") or item.get("prediction")
                if video_id is None or gloss is None:
                    continue
                pred[str(video_id)] = _predicted_record(item, include_metadata)
        return pred

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        items = raw["samples"] if "samples" in raw and isinstance(raw["samples"], list) else raw.items()
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Unsupported predicted gloss JSON format: {path}")

    pred = {}
    if isinstance(items, list):
        iterable = items
    else:
        iterable = []
        for key, value in items:
            if isinstance(value, dict):
                value = {**value, "video_id": value.get("video_id", key)}
            else:
                value = {"video_id": key, "gloss": value}
            iterable.append(value)

    for item in iterable:
        video_id = item.get("video_id") or item.get("name") or item.get("id")
        gloss = item.get("gloss") or item.get("pred_gloss") or item.get("prediction")
        if video_id is None or gloss is None:
            continue
        pred[str(video_id)] = _predicted_record(item, include_metadata)
    return pred


class GlossTextDataset(Dataset):
    """
    Annotation-only PHOENIX gloss-to-text dataset.

    This loader never touches video frames, so SLT training is fast and does
    not retrain or forward the CSLR backbone.
    """

    def __init__(
        self,
        split: str,
        annot_root: str | Path,
        gloss_vocab: Vocabulary,
        text_vocab: Vocabulary,
        predicted_gloss: Optional[Dict[str, object]] = None,
        use_weather_slots: bool = True,
        use_confidence_token: bool = False,
        confidence_augmentation: Optional[Dict] = None,
    ):
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"split must be train/dev/test, got {split}")

        self.split = split
        self.annot_root = Path(annot_root)
        self.gloss_vocab = gloss_vocab
        self.text_vocab = text_vocab
        self.predicted_gloss = predicted_gloss
        self.use_weather_slots = use_weather_slots
        self.use_confidence_token = use_confidence_token
        self.confidence_augmentation = confidence_augmentation or {}

        csv_path = self.annot_root / f"PHOENIX-2014-T.{split}.corpus.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Annotation CSV not found: {csv_path}")
        self.samples = self._load(csv_path)

    def _load(self, csv_path: Path) -> List[Dict]:
        samples = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                video_id = row["name"].strip()
                ref_gloss = row["orth"].strip().split()
                confidence = None
                confidence_bucket = CONFIDENCE_TOKEN_BY_BUCKET["high"]
                if self.predicted_gloss is not None and video_id in self.predicted_gloss:
                    src_gloss, confidence, confidence_bucket = split_predicted_value(
                        self.predicted_gloss[video_id]
                    )
                else:
                    src_gloss = ref_gloss
                text = row["translation"].strip().split()
                samples.append({
                    "video_id": video_id,
                    "gloss": src_gloss,
                    "ref_gloss": ref_gloss,
                    "confidence": confidence,
                    "confidence_bucket": confidence_bucket,
                    "translation": text,
                })
        return samples

    def _sample_train_bucket(self) -> str:
        probs = self.confidence_augmentation.get("train_bucket_probs", {})
        low = float(probs.get("low", 0.15))
        mid = float(probs.get("mid", 0.35))
        high = float(probs.get("high", max(0.0, 1.0 - low - mid)))
        total = max(low + mid + high, 1e-8)
        r = random.random() * total
        if r < low:
            return CONFIDENCE_TOKEN_BY_BUCKET["low"]
        if r < low + mid:
            return CONFIDENCE_TOKEN_BY_BUCKET["mid"]
        return CONFIDENCE_TOKEN_BY_BUCKET["high"]

    def _augment_gloss_for_confidence(self, gloss_tokens: List[str], bucket_token: str) -> List[str]:
        if self.split != "train" or not self.confidence_augmentation.get("enabled", False):
            return gloss_tokens

        drop_probs = self.confidence_augmentation.get("drop_token_prob", {})
        unk_probs = self.confidence_augmentation.get("unk_token_prob", {})
        bucket_name = bucket_token.replace("<conf_", "").replace(">", "")
        drop_p = float(drop_probs.get(bucket_name, 0.0))
        unk_p = float(unk_probs.get(bucket_name, 0.0))

        augmented = []
        for token in gloss_tokens:
            if len(gloss_tokens) > 1 and random.random() < drop_p:
                continue
            augmented.append(self.gloss_vocab.UNK if random.random() < unk_p else token)
        return augmented or gloss_tokens[:1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        gloss_tokens = list(sample["gloss"])
        confidence_bucket = sample.get("confidence_bucket") or CONFIDENCE_TOKEN_BY_BUCKET["high"]
        if self.use_confidence_token and self.split == "train":
            confidence_bucket = self._sample_train_bucket()
            gloss_tokens = self._augment_gloss_for_confidence(gloss_tokens, confidence_bucket)
        slot_tokens = gloss_tokens
        if self.use_confidence_token:
            gloss_tokens = [confidence_bucket] + gloss_tokens
        text_tokens = [self.text_vocab.BOS] + sample["translation"] + [self.text_vocab.EOS]

        slots = (
            extract_weather_slots(slot_tokens)
            if self.use_weather_slots
            else [0.0] * slot_dim()
        )

        return {
            "video_id": sample["video_id"],
            "gloss": torch.tensor(self.gloss_vocab.encode(gloss_tokens), dtype=torch.long),
            "translation": torch.tensor(self.text_vocab.encode(text_tokens), dtype=torch.long),
            "weather_slots": torch.tensor(slots, dtype=torch.float32),
            "gloss_text": " ".join(gloss_tokens),
            "ref_gloss_text": " ".join(sample["ref_gloss"]),
            "confidence": sample.get("confidence"),
            "confidence_bucket": confidence_bucket,
            "translation_text": " ".join(sample["translation"]),
        }


def collate_gloss_text(batch: List[Dict], gloss_pad_idx: int, text_pad_idx: int) -> Dict:
    max_gloss = max(x["gloss"].numel() for x in batch)
    max_text = max(x["translation"].numel() for x in batch)

    gloss = torch.full((len(batch), max_gloss), gloss_pad_idx, dtype=torch.long)
    text = torch.full((len(batch), max_text), text_pad_idx, dtype=torch.long)
    gloss_lens, text_lens = [], []

    for i, item in enumerate(batch):
        g = item["gloss"]
        t = item["translation"]
        gloss[i, :g.numel()] = g
        text[i, :t.numel()] = t
        gloss_lens.append(g.numel())
        text_lens.append(t.numel())

    return {
        "video_ids": [x["video_id"] for x in batch],
        "gloss": gloss,
        "gloss_lens": torch.tensor(gloss_lens, dtype=torch.long),
        "translation": text,
        "translation_lens": torch.tensor(text_lens, dtype=torch.long),
        "weather_slots": torch.stack([x["weather_slots"] for x in batch]),
        "gloss_text": [x["gloss_text"] for x in batch],
        "ref_gloss_text": [x["ref_gloss_text"] for x in batch],
        "confidence": [x["confidence"] for x in batch],
        "confidence_bucket": [x["confidence_bucket"] for x in batch],
        "translation_text": [x["translation_text"] for x in batch],
    }
