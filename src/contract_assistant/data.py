"""Benchmark loading, exact mini selection, and annotation validation."""
from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def mini(tests: list[dict[str, Any]], n: int = 194) -> list[dict[str, Any]]:
    """LegalBench-RAG authors' exact deterministic mini selector."""
    if len(tests) <= n:
        return tests

    def key(test: dict[str, Any]) -> float:
        random.seed(test["snippets"][0]["file_path"])
        return random.random()

    return sorted(tests, key=key)[:n]


def assign_split(query: str) -> str:
    """Assignment-prescribed SHA-256 split; approximately 30% test."""
    digest = int(hashlib.sha256(query.strip().encode("utf-8")).hexdigest(), 16)
    return "test" if digest % 10 < 3 else "train"


@dataclass(frozen=True)
class Example:
    query: str
    snippets: list[dict[str, Any]]
    split: str


def load_benchmark(path: Path) -> list[Example]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        tests = raw["tests"]
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Benchmark not found: {path}. Run `python -m contract_assistant.download_data`.") from exc
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Malformed benchmark JSON at {path}: {exc}") from exc
    if not isinstance(tests, list):
        raise ValueError("Benchmark field 'tests' must be a list")
    chosen = mini(tests)
    if len(chosen) != 194:
        raise ValueError(f"Expected exactly 194 CUAD mini queries, found {len(chosen)}")
    results = []
    for i, record in enumerate(chosen):
        if not isinstance(record, dict) or not isinstance(record.get("query"), str) or not record["query"].strip():
            raise ValueError(f"Malformed query record at selected index {i}")
        spans = record.get("snippets")
        if not isinstance(spans, list) or not spans:
            raise ValueError(f"Query {i} has no snippets")
        results.append(Example(record["query"], spans, assign_split(record["query"])))
    return results


def corpus_path(root: Path, file_path: str) -> Path:
    """Resolve dataset path safely, allowing Unicode normalization differences."""
    base = (root / "corpus").resolve()
    direct = (base / file_path).resolve()
    if direct == base or base not in direct.parents:
        raise ValueError(f"Unsafe corpus path: {file_path!r}")
    if direct.is_file():
        return direct
    target = unicodedata.normalize("NFC", file_path)
    for candidate in base.rglob("*.txt"):
        if unicodedata.normalize("NFC", candidate.relative_to(base).as_posix()) == target:
            return candidate
    raise FileNotFoundError(f"Corpus document referenced by benchmark is missing: {file_path}")


def validate_spans(examples: list[Example], root: Path) -> int:
    """Validate each unique referenced document and every annotated offset."""
    cache: dict[str, str] = {}
    count = 0
    for example in examples:
        for snippet in example.snippets:
            try:
                file_path = snippet["file_path"]
                start, end = snippet["span"]
                answer = snippet["answer"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Malformed snippet in query {example.query!r}: {snippet!r}") from exc
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
                raise ValueError(f"Invalid offsets [{start}, {end}) in {file_path}")
            if file_path not in cache:
                cache[file_path] = corpus_path(root, file_path).read_text(encoding="utf-8")
            document = cache[file_path]
            if end > len(document) or document[start:end] != answer:
                raise ValueError(f"Gold span mismatch in {file_path} [{start}:{end}]")
            count += 1
    return count


def referenced_documents(examples: list[Example]) -> list[str]:
    return sorted({s["file_path"] for e in examples for s in e.snippets})


def load_unanswerable(path: Path) -> list[dict[str, str]]:
    records = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSONL at {path}:{line_number}: {exc}") from exc
                if not all(isinstance(record.get(key), str) and record[key].strip() for key in ("id", "question", "label")):
                    raise ValueError(f"Missing id/question/label at {path}:{line_number}")
                if record["label"] != "unanswerable":
                    raise ValueError(f"Unexpected label at {path}:{line_number}")
                records.append(record)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Unanswerable set not found: {path}") from exc
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Unanswerable query IDs must be unique")
    if len(records) < 30:
        raise ValueError(f"Expected at least 30 unanswerable questions, found {len(records)}")
    return [{**record, "split": assign_split(record["question"])} for record in records]


def load_prepared_split(root: Path, split: str) -> list[Example]:
    """Load only a previously materialized split; the other split is never opened."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    path = root / "derived" / "splits" / f"{split}.jsonl"
    examples = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                    if record.get("split") != split:
                        raise ValueError(f"Wrong split label at {path}:{line_number}")
                    examples.append(Example(record["query"], record["snippets"], split))
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(f"Malformed prepared split at {path}:{line_number}: {exc}") from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Prepared {split} split not found: {path}. Run `python -m contract_assistant.prepare_data`.") from exc
    return examples


def load_prepared_negatives(path: Path, split: str) -> list[dict[str, str]]:
    """Load one pre-split negative file without opening the other split."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    records = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                    if record.get("split") != split or record.get("label") != "unanswerable":
                        raise ValueError(f"Unexpected label or split at {path}:{line_number}")
                    records.append(record)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed prepared negatives at {path}:{line_number}: {exc}") from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Prepared negative {split} file missing: {path}. Run `python -m contract_assistant.prepare_data`.") from exc
    return records


def load_document_manifest(root: Path) -> list[str]:
    path = root / "derived" / "selected_documents.json"
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Selected-document manifest missing: {path}. Run `python -m contract_assistant.prepare_data`.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed document manifest at {path}: {exc}") from exc
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError("Selected-document manifest must be a JSON string array")
    return values
