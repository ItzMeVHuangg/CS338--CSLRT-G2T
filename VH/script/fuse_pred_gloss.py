import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def confidence_bucket_from_score(score: float, low_threshold: float = 0.55, high_threshold: float = 0.75) -> str:
    if score < low_threshold:
        return "<conf_low>"
    if score < high_threshold:
        return "<conf_mid>"
    return "<conf_high>"


def load_jsonl(path: str | Path) -> Dict[str, dict]:
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            video_id = item.get("video_id") or item.get("name") or item.get("id")
            if not video_id:
                continue
            gloss = item.get("gloss_tokens") or item.get("gloss") or item.get("pred_gloss") or item.get("prediction")
            if isinstance(gloss, str):
                gloss = gloss.split()
            item["gloss_tokens"] = list(gloss or [])
            item["confidence"] = float(item.get("confidence", 1.0) or 1.0)
            records[str(video_id)] = item
    return records


def levenshtein_alignment(seq_a: List[str], seq_b: List[str]) -> List[Tuple[Optional[int], Optional[int]]]:
    m, n = len(seq_a), len(seq_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    back = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
        back[i][0] = ("del", i - 1, 0)
    for j in range(1, n + 1):
        dp[0][j] = j
        back[0][j] = ("ins", 0, j - 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            choices = [
                (dp[i - 1][j - 1] + cost, "sub", i - 1, j - 1),
                (dp[i - 1][j] + 1, "del", i - 1, j),
                (dp[i][j - 1] + 1, "ins", i, j - 1),
            ]
            score, op, prev_i, prev_j = min(choices, key=lambda x: x[0])
            dp[i][j] = score
            back[i][j] = (op, prev_i, prev_j)

    path = []
    i, j = m, n
    while i > 0 or j > 0:
        op, prev_i, prev_j = back[i][j]
        if op == "sub":
            path.append((prev_i, prev_j))
            i, j = prev_i, prev_j
        elif op == "del":
            path.append((prev_i, None))
            i = prev_i
        else:
            path.append((None, prev_j))
            j = prev_j
    path.reverse()
    return path


def slot_winner(slot: dict) -> str:
    scores = defaultdict(float)
    for token, score, _source in slot["candidates"]:
        scores[token] += score
    return max(scores.items(), key=lambda item: item[1])[0]


def build_wtn(source_items: List[dict]) -> List[dict]:
    first = source_items[0]
    slots = [
        {"candidates": [(token, first["token_weight"], first["name"])]}
        for token in first["tokens"]
    ]

    for item in source_items[1:]:
        current = [slot_winner(slot) for slot in slots]
        alignment = levenshtein_alignment(current, item["tokens"])
        new_slots = []
        for slot_idx, token_idx in alignment:
            if slot_idx is not None and token_idx is not None:
                slot = {"candidates": list(slots[slot_idx]["candidates"])}
                slot["candidates"].append((item["tokens"][token_idx], item["token_weight"], item["name"]))
                new_slots.append(slot)
            elif slot_idx is not None:
                new_slots.append(slots[slot_idx])
            else:
                new_slots.append({
                    "candidates": [(item["tokens"][token_idx], item["token_weight"], item["name"])]
                })
        slots = new_slots
    return slots


def vote_slots(slots: List[dict], min_token_score: float = 0.0) -> tuple[List[str], List[dict]]:
    fused = []
    diagnostics = []
    for slot in slots:
        scores = defaultdict(float)
        sources = defaultdict(list)
        for token, score, source_name in slot["candidates"]:
            scores[token] += score
            sources[token].append(source_name)
        total = sum(scores.values())
        token, score = max(scores.items(), key=lambda item: item[1])
        agreement = score / max(total, 1e-8)
        diagnostics.append({
            "winner": token,
            "score": round(score, 6),
            "agreement": round(agreement, 6),
            "sources": sources[token],
            "candidates": {key: round(value, 6) for key, value in scores.items()},
        })
        if score >= min_token_score:
            fused.append(token)
    return fused, diagnostics


def fuse_one(video_id: str, source_records: List[tuple[str, float, dict]], min_token_score: float) -> dict:
    source_items = []
    source_glosses = {}
    source_confidences = {}
    for name, weight, record in source_records:
        tokens = record["gloss_tokens"]
        if not tokens:
            continue
        confidence = float(record.get("confidence", 1.0) or 1.0)
        token_weight = max(1e-6, weight * confidence)
        source_items.append({
            "name": name,
            "tokens": tokens,
            "confidence": confidence,
            "token_weight": token_weight,
        })
        source_glosses[name] = " ".join(tokens)
        source_confidences[name] = round(confidence, 6)

    if not source_items:
        return {
            "video_id": video_id,
            "gloss": "",
            "gloss_tokens": [],
            "confidence": 0.0,
            "confidence_bucket": "<conf_low>",
            "source_glosses": source_glosses,
            "source_confidences": source_confidences,
        }

    source_items.sort(key=lambda item: item["token_weight"], reverse=True)
    slots = build_wtn(source_items)
    fused_tokens, diagnostics = vote_slots(slots, min_token_score=min_token_score)
    if not fused_tokens:
        fused_tokens = list(source_items[0]["tokens"])

    avg_conf = sum(item["confidence"] for item in source_items) / len(source_items)
    avg_agreement = sum(slot["agreement"] for slot in diagnostics) / max(len(diagnostics), 1)
    fused_confidence = max(0.0, min(1.0, 0.5 * avg_conf + 0.5 * avg_agreement))

    base_record = source_records[0][2]
    output = {
        "video_id": video_id,
        "gloss": " ".join(fused_tokens),
        "gloss_tokens": fused_tokens,
        "confidence": round(fused_confidence, 6),
        "confidence_bucket": confidence_bucket_from_score(fused_confidence),
        "source_glosses": source_glosses,
        "source_confidences": source_confidences,
        "fusion_diagnostics": diagnostics,
    }
    for key in ("reference_gloss", "reference_text"):
        if key in base_record:
            output[key] = base_record[key]
    return output


def wer(hypotheses: List[List[str]], references: List[List[str]]) -> float:
    edits = 0
    total = 0
    for hyp, ref in zip(hypotheses, references):
        m, n = len(hyp), len(ref)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if hyp[i - 1] == ref[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )
        edits += dp[m][n]
        total += n
    return edits / max(total, 1)


def parse_source(values: List[str]) -> tuple[str, Path, float]:
    if len(values) not in {2, 3}:
        raise argparse.ArgumentTypeError("--source expects: NAME PATH [WEIGHT]")
    name = values[0]
    path = Path(values[1])
    weight = float(values[2]) if len(values) == 3 else 1.0
    return name, path, weight


def main():
    parser = argparse.ArgumentParser(description="Fuse multiple predicted-gloss JSONL files into one pseudo-gloss JSONL")
    parser.add_argument("--source", nargs="+", action="append", required=True, metavar="SOURCE_ARG")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_token_score", type=float, default=0.0)
    parser.add_argument("--require_all", action="store_true", help="Only fuse samples present in every source")
    args = parser.parse_args()

    sources = [parse_source(value) for value in args.source]
    loaded = [(name, weight, load_jsonl(path)) for name, path, weight in sources]

    if args.require_all:
        video_ids = set.intersection(*(set(records) for _name, _weight, records in loaded))
    else:
        video_ids = set.union(*(set(records) for _name, _weight, records in loaded))
    video_ids = sorted(video_ids)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bucket_counts = Counter()
    hyps, refs = [], []

    with open(output_path, "w", encoding="utf-8") as f:
        for video_id in video_ids:
            available = [
                (name, weight, records[video_id])
                for name, weight, records in loaded
                if video_id in records
            ]
            fused = fuse_one(video_id, available, args.min_token_score)
            bucket_counts[fused["confidence_bucket"]] += 1
            if fused.get("reference_gloss"):
                hyps.append(fused["gloss_tokens"])
                refs.append(fused["reference_gloss"].split())
            f.write(json.dumps(fused, ensure_ascii=False) + "\n")

    metrics = {
        "output": str(output_path),
        "num_samples": len(video_ids),
        "sources": [
            {"name": name, "path": str(path), "weight": weight, "num_samples": len(records)}
            for (name, path, weight), (_n, _w, records) in zip(sources, loaded)
        ],
        "confidence_bucket_counts": dict(bucket_counts),
    }
    if refs:
        metrics["wer"] = wer(hyps, refs)
        metrics["wer_percent"] = metrics["wer"] * 100.0

    metrics_path = output_path.with_suffix(output_path.suffix + ".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[Fuse] wrote {output_path}")
    print(f"[Fuse] samples={len(video_ids)}")
    if "wer_percent" in metrics:
        print(f"[Fuse] WER={metrics['wer_percent']:.2f}%")
    print(f"[Fuse] metrics={metrics_path}")


if __name__ == "__main__":
    main()
