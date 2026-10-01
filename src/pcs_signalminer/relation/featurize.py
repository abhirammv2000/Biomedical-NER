"""Prepare relation candidates for the classifier with entity markers: every mention of the chemical is wrapped in
[CHEM]...[/CHEM] and every mention of the disease in [DIS]...[/DIS]. The model sees which pair to judge and still reads
the whole abstract, which matters since the relation is often across sentences.
"""
from __future__ import annotations

CHEM_START, CHEM_END = "[CHEM]", "[/CHEM]"
DIS_START, DIS_END = "[DIS]", "[/DIS]"
MARKER_TOKENS = [CHEM_START, CHEM_END, DIS_START, DIS_END]


def mark_text(text: str,
              chem_offsets: list[tuple[int, int]],
              dis_offsets: list[tuple[int, int]]) -> str:
    """Put markers around the target chemical and disease mentions. They go in from right to left so the earlier
    offsets stay valid, and a marker that would land inside another one is skipped."""
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
