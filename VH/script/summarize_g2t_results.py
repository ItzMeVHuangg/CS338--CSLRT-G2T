import argparse
import json
from pathlib import Path


def get_nested(payload, *keys, default=None):
    cur = payload
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Summarize G2T SLT experiments as a markdown table")
    parser.add_argument("--root", default="checkpoints_g2t_slt", help="Directory containing experiment summaries")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent.parent / root

    summaries = sorted(root.glob("*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No summary.json files found under {root}")

    rows = []
    for path in summaries:
        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)
        switches = item.get("model_switches", {})
        rows.append({
            "experiment": item.get("experiment", path.parent.name),
            "model_type": switches.get("model_type", "domain_slot_transformer"),
            "oracle_dev": get_nested(item, "final_eval", "oracle_dev", "bleu"),
            "pred_dev": get_nested(item, "final_eval", "pred_dev", "bleu"),
            "pred_test": get_nested(item, "final_eval", "pred_test", "bleu"),
            "conv_gate": switches.get("use_conv_gate"),
            "weather_slots": switches.get("use_weather_slots"),
            "confidence": switches.get("use_confidence_token"),
            "params_m": item.get("trainable_params", 0) / 1_000_000,
            "summary": str(path),
        })

    print("| Experiment | Model Type | Oracle Dev BLEU-4 | Pred Dev BLEU-4 | Pred Test BLEU-4 | Conv Gate | Weather Slots | Confidence | Trainable Params |")
    print("|---|---|---:|---:|---:|---|---|---|---:|")
    for row in rows:
        print(
            f"| {row['experiment']} | {row['model_type']} | {fmt(row['oracle_dev'])} | {fmt(row['pred_dev'])} | "
            f"{fmt(row['pred_test'])} | {row['conv_gate']} | {row['weather_slots']} | {row['confidence']} | "
            f"{row['params_m']:.2f}M |"
        )


if __name__ == "__main__":
    main()
