"""Featurize document-level relation candidates for a transformer classifier.

Uses the *entity-marker* technique: every mention of the candidate chemical is
wrapped in [CHEM]...[/CHEM] and every mention of the candidate disease in
[DIS]...[/DIS], so the model sees which concept pair it must judge while still
reading the full document context (the CID relation is document-level and often
cross-sentence).
"""
from __future__ import annotations

CHEM_START, CHEM_END = "[CHEM]", "[/CHEM]"
DIS_START, DIS_END = "[DIS]", "[/DIS]"
MARKER_TOKENS = [CHEM_START, CHEM_END, DIS_START, DIS_END]


def mark_text(text: str,
              chem_offsets: list[tuple[int, int]],
              dis_offsets: list[tuple[int, int]]) -> str:
    """Insert entity markers around the target chemical and disease mentions.

    Offsets are inserted right-to-left so earlier offsets stay valid. Overlapping
    spans (rare) are handled by skipping a marker that would land inside an
    already-inserted one via simple sorted, non-overlapping application.
    """
    events = []  # (position, text_to_insert, priority) priority: ends before starts at same pos
    for s, e in chem_offsets:
        events.append((s, CHEM_START, 1))
        events.append((e, CHEM_END, 0))
    for s, e in dis_offsets:
        events.append((s, DIS_START, 1))
        events.append((e, DIS_END, 0))
    # insert from the largest position to the smallest
    events.sort(key=lambda x: (x[0], x[2]), reverse=True)
    out = text
    for pos, token, _ in events:
        pos = max(0, min(pos, len(out)))
        out = out[:pos] + token + out[pos:]
    return out
