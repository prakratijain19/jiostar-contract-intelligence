"""Audit the frozen negative questions against only the selected CUAD corpus."""
from __future__ import annotations

import json
import re
from pathlib import Path

from contract_assistant.data import load_benchmark, load_unanswerable, referenced_documents, corpus_path

ROOT = Path(__file__).resolve().parents[1]


def validate(data_root: Path, negatives_path: Path) -> dict:
    examples = load_benchmark(data_root / "benchmarks/cuad.json")
    negatives = load_unanswerable(negatives_path)
    corpus = "\n".join(corpus_path(data_root, name).read_text(encoding="utf-8") for name in referenced_documents(examples))
    normalized = re.sub(r"\s+", " ", corpus).casefold()
    expected_terms = {
        "financial": "audited total revenue",
        "market": "market capitalization",
    }
    for item in negatives:
        question = item["question"].casefold()
        category = next((name for name, term in expected_terms.items() if term in question), None)
        if category is None:
            raise ValueError(f"No recognized external-fact category in {item['id']}")
        term = expected_terms[category]
        if term in normalized:
            raise ValueError(f"The requested fact category appears in indexed text: {item['id']} ({term})")
        if not item.get("basis"):
            raise ValueError(f"Missing answerability rationale: {item['id']}")
    return {"negative_queries": len(negatives),
            "train": sum(row["split"] == "train" for row in negatives),
            "test": sum(row["split"] == "test" for row in negatives),
            "selected_documents_checked": len(referenced_documents(examples)),
            "validation": "requested fact categories and rationale labels absent from all selected agreement text"}


if __name__ == "__main__":
    print(json.dumps(validate(ROOT / "data/legalbench-rag", ROOT / "data/unanswerable.jsonl"), indent=2))
