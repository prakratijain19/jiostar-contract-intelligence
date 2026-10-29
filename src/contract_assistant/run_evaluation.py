"""Evaluate a named config on one split and persist raw summary metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cli import build_index
from .data import load_prepared_negatives, load_prepared_split
from .evaluation import abstention_metrics, query_scores, select_abstention_threshold


def run(data: Path, config_path: Path, split: str, output: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = build_index(data, config)
    train_pos = load_prepared_split(data, "train")
    train_negative_path = data / "derived" / "splits" / "unanswerable_train.jsonl"
    train_neg = [q["question"] for q in load_prepared_negatives(train_negative_path, "train")]
    threshold = select_abstention_threshold(train_pos, train_neg, index)
    selected = train_pos if split == "train" else load_prepared_split(data, "test")
    negative_path = data / "derived" / "splits" / f"unanswerable_{split}.jsonl"
    selected_neg = [q["question"] for q in load_prepared_negatives(negative_path, split)]
    per_query = []
    for example in selected:
        hits = index.search(example.query, 64, threshold)
        scores = query_scores(example, hits)
        metrics = {**scores}
        if split == "test":
            metrics["query"] = example.query
            metrics["gold"] = [{"document": s["file_path"], "start": s["span"][0], "end": s["span"][1], "answer": s["answer"]} for s in example.snippets]
            metrics["retrieved"] = [{"document": h.document_id, "start": h.start, "end": h.end, "score": h.score, "text": h.text} for h in hits[:5]]
            per_query.append(metrics)
        else:
            per_query.append(scores)
    keys = [f"char_{metric}@{k}" for k in (1, 2, 4, 8, 16, 32, 64) for metric in ("precision", "recall")]
    metrics = {key: sum(item[key] for item in per_query) / len(per_query) for key in keys}
    metrics["queries"] = len(selected)
    answerable = [e.query for e in selected]
    metrics["abstention"] = abstention_metrics(answerable, selected_neg, index, threshold)
    result = {"split": split, "test_set_evaluated": split == "test", "config": config,
              "threshold_selected_on": "train", "abstention_threshold": threshold, "metrics": metrics,
              **({"per_query": per_query} if split == "test" else {})}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/legalbench-rag"))
    p.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    p.add_argument("--split", choices=("train", "test"), required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run(args.data, args.config, args.split, args.output)
    print(json.dumps({"output": str(args.output), "split": result["split"],
                      "threshold_selected_on": result["threshold_selected_on"],
                      "abstention_threshold": result["abstention_threshold"],
                      "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
