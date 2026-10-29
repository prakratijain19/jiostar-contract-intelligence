"""A transparent BM25 retriever over offset-preserving passages."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .chunking import Passage, chunk_document
from .data import corpus_path


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:['’-][a-z0-9]+)?", text.lower())


@dataclass(frozen=True)
class SearchResult:
    document_id: str
    text: str
    start: int
    end: int
    score: float
    rank: int


class BM25Index:
    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and b in [0, 1]")
        self.passages, self.k1, self.b = passages, k1, b
        self.terms = [tokenize(p.text) for p in passages]
        self.lengths = [len(t) for t in self.terms]
        self.avg_len = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        df: dict[str, int] = {}
        for tokens in self.terms:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        n = len(passages)
        self.idf = {t: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for t, freq in df.items()}

    @classmethod
    def from_documents(cls, documents: dict[str, str], max_chars: int = 1800, overlap: int = 180, k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        chunks = [p for name, text in documents.items() for p in chunk_document(name, text, max_chars, overlap)]
        return cls(chunks, k1, b)

    @classmethod
    def from_root(cls, root: Path, file_paths: list[str], **options: float) -> "BM25Index":
        documents = {}
        for file_path in file_paths:
            path = corpus_path(root, file_path)
            try:
                documents[file_path] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise ValueError(f"Cannot load contract {path}: {exc}") from exc
        return cls.from_documents(documents, **options)

    def search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        tokens = tokenize(query)
        if not tokens or not self.passages:
            return []
        scored = []
        for i, terms in enumerate(self.terms):
            freq: dict[str, int] = {}
            for term in terms:
                freq[term] = freq.get(term, 0) + 1
            score = 0.0
            for term in tokens:
                tf = freq.get(term, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / max(self.avg_len, 1))
                    score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / norm
            if score >= threshold and score > 0:
                scored.append((score, i))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [SearchResult(self.passages[i].document_id, self.passages[i].text, self.passages[i].start, self.passages[i].end, score, rank)
                for rank, (score, i) in enumerate(scored[:top_k], 1)]


class TfidfIndex:
    """Sparse cosine TF-IDF baseline using the same offset-aware result shape."""

    def __init__(self, passages: list[Passage]):
        self.passages = passages
        self.terms = [Counter(tokenize(p.text)) for p in passages]
        document_frequency: Counter[str] = Counter()
        for counts in self.terms:
            document_frequency.update(counts.keys())
        n = len(passages)
        self.idf = {term: math.log((1 + n) / (1 + freq)) + 1 for term, freq in document_frequency.items()}
        self.vectors = []
        self.norms = []
        for counts in self.terms:
            vector = {term: count * self.idf[term] for term, count in counts.items()}
            self.vectors.append(vector)
            self.norms.append(math.sqrt(sum(value * value for value in vector.values())))

    @classmethod
    def from_documents(cls, documents: dict[str, str], max_chars: int = 1800, overlap: int = 180) -> "TfidfIndex":
        chunks = [p for name, text in documents.items() for p in chunk_document(name, text, max_chars, overlap)]
        return cls(chunks)

    @classmethod
    def from_root(cls, root: Path, file_paths: list[str], max_chars: int = 1800, overlap: int = 180) -> "TfidfIndex":
        documents = {}
        for file_path in file_paths:
            path = corpus_path(root, file_path)
            try:
                documents[file_path] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise ValueError(f"Cannot load contract {path}: {exc}") from exc
        return cls.from_documents(documents, max_chars, overlap)

    def search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_counts = Counter(tokenize(query))
        if not query_counts:
            return []
        qvector = {term: count * self.idf[term] for term, count in query_counts.items() if term in self.idf}
        qnorm = math.sqrt(sum(value * value for value in qvector.values()))
        if not qnorm:
            return []
        scored = []
        for i, vector in enumerate(self.vectors):
            if not self.norms[i]:
                continue
            dot = sum(value * vector.get(term, 0.0) for term, value in qvector.items())
            score = dot / (qnorm * self.norms[i])
            if score >= threshold and score > 0:
                scored.append((score, i))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [SearchResult(self.passages[i].document_id, self.passages[i].text, self.passages[i].start, self.passages[i].end, score, rank)
                for rank, (score, i) in enumerate(scored[:top_k], 1)]
