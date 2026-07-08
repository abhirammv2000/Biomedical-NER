"""Unit tests for the core PCS-SignalMiner pipeline components."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcs_signalminer.data.pubtator import load_jsonl, Document, Entity, Relation  # noqa: E402
from pcs_signalminer.relation import candidates as C  # noqa: E402
from pcs_signalminer.relation.featurize import mark_text, CHEM_START, DIS_START  # noqa: E402
from pcs_signalminer.linking.normalizer import DictionaryNormalizer  # noqa: E402

PROC = ROOT / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROC / "bc5cdr_train.jsonl").exists(),
    reason="processed data not built (run scripts/run_download_data.py)")


def test_offsets_align_to_text():
    docs = load_jsonl(PROC / "bc5cdr_train.jsonl")
    assert len(docs) == 500
    for d in docs[:50]:
        for e in d.entities:
            assert d.text[e.start:e.end] == e.text


def test_candidate_labels_match_gold():
    doc = Document(pmid="1", title="A causes B.", abstract="A induced B in rats.",
                   entities=[Entity(0, 1, "A", "Chemical", ["D1"]),
                             Entity(9, 10, "B", "Disease", ["D2"])],
                   relations=[Relation("D1", "D2")])
    cands = C.build_candidates(doc)
    assert len(cands) == 1
    assert cands[0].label == 1
    assert cands[0].chemical_mesh == "D1" and cands[0].disease_mesh == "D2"


def test_recall_ceiling_full_on_train():
    docs = load_jsonl(PROC / "bc5cdr_train.jsonl")
    total = recoverable = 0
    for d in docs:
        chem = {m for e in d.entities if e.type == "Chemical" for m in e.mesh_ids}
        dis = {m for e in d.entities if e.type == "Disease" for m in e.mesh_ids}
        for r in d.relations:
            total += 1
            recoverable += int(r.chemical_mesh in chem and r.disease_mesh in dis)
    assert recoverable == total  # 100% recall ceiling


def test_mark_text_inserts_markers():
    out = mark_text("aspirin caused ulcers", [(0, 7)], [(15, 21)])
    assert CHEM_START in out and DIS_START in out
    assert "aspirin" in out and "ulcers" in out


def test_normalizer_exact_and_unknown():
    doc = Document(pmid="1", title="", abstract="aspirin",
                   entities=[Entity(0, 7, "aspirin", "Chemical", ["D001241"])],
                   relations=[])
    norm = DictionaryNormalizer().fit([doc])
    assert norm.predict("Aspirin", "Chemical") == "D001241"   # case-insensitive
    assert norm.predict("zzzzz", "Chemical") is None          # unknown -> None
