"""Materialize split-isolated benchmark views and the selected corpus manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_benchmark, load_unanswerable, referenced_documents, validate_spans


def prepare(root: Path, negatives_path: Path) -> dict[str, int]:
    examples = load_benchmark(root / "benchmarks/cuad.json")
    spans = validate_spans(examples, root)
    negatives = load_unanswerable(negatives_path)
    derived = root / "derived"
    split_dir = derived / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        records = [{"query": item.query, "snippets": item.snippets, "split": item.split}
                   for item in examples if item.split == split]
        (split_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        negative_rows = [row for row in negatives if row["split"] == split]
        (split_dir / f"unanswerable_{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in negative_rows), encoding="utf-8")
    docs = referenced_documents(examples)
    (derived / "selected_documents.json").write_text(json.dumps(docs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"queries": len(examples), "documents": len(docs), "validated_spans": spans,
            "train": sum(item.split == "train" for item in examples),
            "test": sum(item.split == "test" for item in examples),
            "unanswerable_train": sum(item["split"] == "train" for item in negatives),
            "unanswerable_test": sum(item["split"] == "test" for item in negatives)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/legalbench-rag"))
    parser.add_argument("--negatives", type=Path, default=Path("data/unanswerable.jsonl"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, args.negatives), indent=2))


if __name__ == "__main__":
    main()
