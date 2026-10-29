from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .data import load_document_manifest
from .retrieval import BM25Index, TfidfIndex


def build_index(root: Path, config: dict):
    names = load_document_manifest(root)
    retriever = config.get("retriever", "bm25")
    options = {"max_chars": config.get("max_chars", 1800), "overlap": config.get("overlap", 180)}
    if retriever == "bm25":
        return BM25Index.from_root(root, names, **options, k1=config.get("k1", 1.5), b=config.get("b", 0.75))
    if retriever == "tfidf":
        return TfidfIndex.from_root(root, names, **options)
    raise ValueError(f"Unknown retriever: {retriever}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search CUAD contract passages")
    parser.add_argument("question", help="Natural-language contract question")
    parser.add_argument("--data", type=Path, default=Path("data/legalbench-rag"))
    parser.add_argument("--config", type=Path, default=Path("configs/final.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=None, help="overrides the saved train-selected threshold")
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("question must not be empty")
    try:
        config = json.loads(args.config.read_text(encoding="utf-8")) if args.config.exists() else {"max_chars": 1800, "overlap": 180}
        threshold = config.get("abstention_threshold", 0.0) if args.threshold is None else args.threshold
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be a finite non-negative number")
        index = build_index(args.data, config)
        results = index.search(args.question, args.top_k, threshold)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Data error: {exc}\n")
    if not results:
        print("No relevant passage found. Try rephrasing or check another contract.")
        return
    for result in results:
        print(f"[{result.rank}] {result.document_id} chars {result.start}:{result.end} score={result.score:.4f}\n{result.text}\n")


if __name__ == "__main__":
    main()
