"""Run reproducible training-only configuration experiments."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .data import load_prepared_negatives, load_prepared_split, validate_spans
from .evaluation import evaluate, select_abstention_threshold, query_scores
from .cli import build_index


PROPOSALS = [
    ({"max_chars": 1200, "overlap": 120}, "Smaller passages may improve local evidence focus."),
    ({"max_chars": 2400, "overlap": 240}, "Larger passages may cover long annotated clauses."),
    ({"max_chars": 1800, "overlap": 0}, "No overlap tests whether duplicate spans dilute precision."),
    ({"max_chars": 1800, "overlap": 360}, "More overlap may preserve evidence near boundaries."),
    ({"max_chars": 900, "overlap": 180}, "Shorter passages test fine-grained retrieval."),
    ({"max_chars": 3000, "overlap": 300}, "Long passages test contextual clause retrieval."),
    ({"retriever": "tfidf", "max_chars": 3000, "overlap": 300}, "Compare sparse cosine TF-IDF with BM25 on the same train split."),
]


def accept_proposal(candidate: dict, incumbent: dict) -> bool:
    """Primary objective is recall@8 with a 0.02 precision@8 guardrail."""
    return candidate["char_recall@8"] > incumbent["char_recall@8"] and candidate["char_precision@8"] >= incumbent["char_precision@8"] - .02


def run(root: Path, log_path: Path, iterations: int = 5, final_config_path: Path = Path("configs/final.json")) -> dict:
    train = load_prepared_split(root, "train")
    validate_spans(train, root)
    negatives_path = root / "derived" / "splits" / "unanswerable_train.jsonl"
    train_negatives = [q["question"] for q in load_prepared_negatives(negatives_path, "train")]
    rows = []

    def score(config: dict) -> dict:
        index = build_index(root, config)
        threshold = select_abstention_threshold(train, train_negatives, index)
        return {**evaluate(train, index, top_k=64, threshold=threshold),
                "abstention": {"threshold": threshold, "negatives": len(train_negatives)}}

    def diagnose(config: dict) -> dict[str, int]:
        index = build_index(root, config)
        counts = {"wrong_document_at_8": 0, "right_document_missed_at_8": 0, "gold_partially_covered_at_8": 0, "gold_covered_at_8": 0}
        for example in train:
            hits = index.search(example.query, 8)
            target = example.snippets[0]["file_path"]
            if not any(h.document_id == target for h in hits):
                counts["wrong_document_at_8"] += 1
            elif query_scores(example, hits, (8,))["char_recall@8"] == 0:
                counts["right_document_missed_at_8"] += 1
            elif query_scores(example, hits, (8,))["char_recall@8"] < 1:
                counts["gold_partially_covered_at_8"] += 1
            else:
                counts["gold_covered_at_8"] += 1
        return counts

    best_config = {"retriever": "bm25", "max_chars": 1800, "overlap": 180}
    best_metrics = score(best_config)
    rows.append({"iteration": 0, "proposal_source": "fixed baseline", "reason": "BM25 baseline", "config": best_config, "diagnosis": diagnose(best_config), "metrics": best_metrics, "accepted": True})
    for i in range(iterations):
        candidate, reason = PROPOSALS[i % len(PROPOSALS)]
        metrics = score(candidate)
        # Primary metric: training character recall@5. Guardrail: precision@5 no drop > .02.
        accepted = accept_proposal(metrics, best_metrics)
        rows.append({"iteration": i + 1, "proposal_source": "predeclared parameter experiment", "reason": reason, "config": candidate, "diagnosis": diagnose(best_config), "metrics": metrics, "accepted": accepted,
                     "decision": "accepted: higher recall with precision guardrail" if accepted else "rejected: objective or precision guardrail not met"})
        if accepted:
            best_config, best_metrics = candidate, metrics
    payload = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "data_scope": "training split only", "test_accessed": False,
        "objective": "maximize mean character recall@8", "guardrail": "character precision@8 may not fall more than 0.02", "best_config": best_config,
               "best_train_metrics": best_metrics, "experiments": rows}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    final_config_path.parent.mkdir(parents=True, exist_ok=True)
    final_config_path.write_text(json.dumps({"retriever": best_config.get("retriever", "bm25"), "k1": best_config.get("k1", 1.5),
                                             "b": best_config.get("b", 0.75), **best_config, "top_k": 5,
                                             "abstention_threshold": best_metrics["abstention"]["threshold"]}, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/legalbench-rag"))
    parser.add_argument("--log", type=Path, default=Path("experiments/history.json"))
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--final-config", type=Path, default=Path("configs/final.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.log, args.iterations, args.final_config), indent=2))


if __name__ == "__main__":
    main()
