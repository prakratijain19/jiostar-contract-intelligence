"""Structure-aware chunks retaining exact offsets into source text."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Passage:
    document_id: str
    text: str
    start: int
    end: int


def chunk_document(document_id: str, text: str, max_chars: int = 1800, overlap: int = 180) -> list[Passage]:
    if max_chars < 1 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Require max_chars > overlap >= 0")
    if not text:
        return []
    # Prefer section/paragraph boundaries; use a hard window for oversized blocks.
    boundaries = [m.end() for m in re.finditer(r"(?:\n\s*\n+|\n(?=(?:\d+(?:\.\d+)*[.)]?\s|[A-Z][A-Z\s]{4,}$)))", text)]
    boundaries.append(len(text))
    blocks: list[tuple[int, int]] = []
    start = 0
    for end in boundaries:
        if end - start <= max_chars:
            blocks.append((start, end))
            start = end
            continue
        cursor = start
        while cursor < end:
            stop = min(cursor + max_chars, end)
            if stop < end:
                pivot = text.rfind(" ", cursor + max_chars // 2, stop)
                if pivot > cursor:
                    stop = pivot
            blocks.append((cursor, stop))
            if stop == end:
                break
            cursor = max(cursor + 1, stop - overlap)
        start = end
    # Merge tiny adjacent blocks while staying inside the configured limit.
    merged: list[tuple[int, int]] = []
    for a, b in blocks:
        if merged and b - merged[-1][0] <= max_chars and a == merged[-1][1]:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [Passage(document_id, text[a:b], a, b) for a, b in merged if text[a:b].strip()]
