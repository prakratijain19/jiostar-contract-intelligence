"""Character-span evaluation and train-only threshold selection."""
from __future__ import annotations

from collections.abc import Iterable

from .data import Example
from .retrieval import SearchResult

K_VALUES = (1, 2, 4, 8, 16, 32, 64)

def _merge(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(a, b) for a, b in merged]


def span_metrics(gold: list[tuple[int, int]], retrieved: list[tuple[int, int]]) -> tuple[float, float]:
    g, r = _merge(gold), _merge(retrieved)
    intersection = sum(max(0, min(ge, re) - max(gs, rs)) for gs, ge in g for rs, re in r)
    rlen = sum(b - a for a, b in r)
    glen = sum(b - a for a, b in g)
    return (intersection / rlen if rlen else 0.0, intersection / glen if glen else 0.0)


def query_scores(example: Example, results: list[SearchResult], ks: tuple[int, ...] = K_VALUES) -> dict[str, float]:
    doc = example.snippets[0]["file_path"]
    gold = [(s["span"][0], s["span"][1]) for s in example.snippets]
    values: dict[str, float] = {}
    for k in ks:
        retrieved = results[:k]
        # LegalBench-RAG's official denominator counts every returned character,
        # while intersections count only spans from the gold document.
        retrieved_chars = sum(max(0, r.end - r.start) for r in retrieved)
        gold_chars = sum(max(0, end - start) for start, end in gold)
        overlap = sum(max(0, min(r.end, end) - max(r.start, start))
                      for r in retrieved if r.document_id == doc
                      for start, end in gold)
        precision = overlap / retrieved_chars if retrieved_chars else 0.0
        recall = overlap / gold_chars if gold_chars else 0.0
        values[f"char_precision@{k}"] = precision
        values[f"char_recall@{k}"] = recall
    return values


def evaluate(examples: list[Example], index, top_k: int = 10, threshold: float = 0.0) -> dict[str, float | int]:
    if not examples:
        raise ValueError("Cannot evaluate an empty split")
    values = [query_scores(e, index.search(e.query, max(top_k, max(K_VALUES)), threshold)) for e in examples]
    keys = values[0]
    return {**{key: sum(v[key] for v in values) / len(values) for key in keys}, "queries": len(examples)}


def select_abstention_threshold(answerable_train: list[Example], unanswerable_train: list[str], index) -> float:
    """Maximize balanced accuracy on train positives and frozen train negatives."""
    positives = [index.search(e.query, 1) for e in answerable_train]
    negatives = [index.search(q, 1) for q in unanswerable_train]
    scores = [hits[0].score if hits else 0.0 for hits in positives + negatives]
    candidates = sorted(set([0.0, *scores]))
    best_score, best_threshold = -1.0, 0.0
    n_pos, n_neg = len(positives), len(negatives)
    for threshold in candidates:
        sensitivity = sum(bool(hits) and hits[0].score >= threshold for hits in positives) / max(n_pos, 1)
        specificity = sum(not hits or hits[0].score < threshold for hits in negatives) / max(n_neg, 1)
        balanced = (sensitivity + specificity) / 2
        if balanced > best_score or (balanced == best_score and threshold > best_threshold):
            best_score, best_threshold = balanced, threshold
    return best_threshold


def abstention_metrics(answerable: list[str], unanswerable: list[str], index, threshold: float) -> dict[str, float | int]:
    pos_hits = [index.search(q, 1) for q in answerable]
    neg_hits = [index.search(q, 1) for q in unanswerable]
    tp = sum(bool(h) and h[0].score >= threshold for h in pos_hits)
    fn = len(pos_hits) - tp
    tn = sum(not h or h[0].score < threshold for h in neg_hits)
    fp = len(neg_hits) - tn
    return {"answerable_queries": len(pos_hits), "unanswerable_queries": len(neg_hits),
            "answerable_coverage": tp / len(pos_hits) if pos_hits else 0.0,
            "incorrect_abstention_rate": fn / len(pos_hits) if pos_hits else 0.0,
            "correct_abstention_rate": tn / len(neg_hits) if neg_hits else 0.0,
            "false_acceptance_rate": fp / len(neg_hits) if neg_hits else 0.0,
            "correct_abstentions": tn, "false_acceptances": fp}
