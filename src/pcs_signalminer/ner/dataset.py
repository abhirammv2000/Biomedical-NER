"""Convert BC5CDR documents into token-classification examples with BIO labels.

Uses a fast tokenizer's offset mapping to align character-level entity spans to
sub-word tokens, so no external sentence/word tokenizer is required.
"""
from __future__ import annotations

from ..data.pubtator import Document

LABELS = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def char_labels(doc: Document) -> list[str | None]:
    """Per-character BIO-ish tag over doc.text: 'Chemical', 'Disease', or None."""
    text = doc.text
    tags: list[str | None] = [None] * len(text)
    # sort so that, on overlap, longer spans are written first then short ones
    for e in sorted(doc.entities, key=lambda x: (x.start, -(x.end - x.start))):
        for i in range(e.start, min(e.end, len(text))):
            if tags[i] is None:
                tags[i] = e.type
    return tags


def align_labels(doc: Document, offset_mapping: list[tuple[int, int]]) -> list[int]:
    """Map tokenizer offsets to BIO label ids using per-character tags."""
    ctags = char_labels(doc)
    labels: list[int] = []
    prev_type: str | None = None
    for (start, end) in offset_mapping:
        if start == end:                      # special token
            labels.append(-100)
            prev_type = None
            continue
        span_type = ctags[start] if start < len(ctags) else None
        if span_type is None:
            labels.append(LABEL2ID["O"])
            prev_type = None
        else:
            prefix = "B" if span_type != prev_type else "I"
            labels.append(LABEL2ID[f"{prefix}-{span_type}"])
            prev_type = span_type
    return labels
