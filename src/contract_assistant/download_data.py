"""Download the official HF mirror's CUAD benchmark and selected contracts."""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .data import load_benchmark, referenced_documents, validate_spans

BASE = "https://huggingface.co/datasets/awinml/legalbench-rag/resolve/main/"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "contract-intelligence-assistant/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc
    if not payload:
        raise RuntimeError(f"Downloaded empty file: {url}")
    temp = destination.with_suffix(destination.suffix + ".part")
    temp.write_bytes(payload)
    temp.replace(destination)


def prepare(root: Path) -> dict[str, int]:
    benchmark = root / "benchmarks" / "cuad.json"
    if not benchmark.exists():
        download(BASE + "benchmarks/cuad.json", benchmark)
    examples = load_benchmark(benchmark)
    for relative in referenced_documents(examples):
        target = root / "corpus" / relative
        if not target.exists():
            encoded = urllib.parse.quote("corpus/" + relative, safe="/()'!._-")
            download(BASE + encoded, target)
    snippets = validate_spans(examples, root)
    splits = {name: sum(e.split == name for e in examples) for name in ("train", "test")}
    return {"queries": len(examples), "documents": len(referenced_documents(examples)), "validated_spans": snippets, **splits}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/legalbench-rag"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.data), indent=2))


if __name__ == "__main__":
    main()
