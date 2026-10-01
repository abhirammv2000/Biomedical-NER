"""Build the document-level relation-extraction candidate dataset.

For each split, enumerate (chemical concept, disease concept) candidate pairs,
label them against gold CID relations, and report class balance plus the
"recall ceiling" (share of gold relations recoverable from co-occurring linked
concept pairs).

Run:
    python scripts/run_build_re_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcs_signalminer.data.pubtator import load_jsonl  # noqa: E402
from pcs_signalminer.relation import candidates  # noqa: E402

PROC_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"


def recall_ceiling(docs) -> dict:
    """A gold relation is recoverable only if both its chemical and disease
    concepts are linked entities of the correct type in that document."""
    total = 0
    recoverable = 0
    for d in docs:
        chem_ids = {m for e in d.entities if e.type == "Chemical" for m in e.mesh_ids}
        dis_ids = {m for e in d.entities if e.type == "Disease" for m in e.mesh_ids}
        for r in d.relations:
            total += 1
            if r.chemical_mesh in chem_ids and r.disease_mesh in dis_ids:
                recoverable += 1
    return {"gold_relations": total, "recoverable": recoverable,
            "recall_ceiling": round(recoverable / total, 4) if total else 0.0}


def main() -> None:
    report = {}
    for name in ("train", "dev", "test"):
        docs = load_jsonl(PROC_DIR / f"bc5cdr_{name}.jsonl")
        cands = candidates.build_dataset(docs)
        out = PROC_DIR / f"re_candidates_{name}.jsonl"
        candidates.candidates_to_jsonl(cands, out)
        stats = candidates.dataset_stats(cands)
        ceil = recall_ceiling(docs)
        report[name] = {**stats, **ceil}
        print(f"[{name}] pairs={stats['candidate_pairs']} "
              f"pos={stats['positives']} ({stats['positive_rate']:.1%}) "
              f"recall_ceiling={ceil['recall_ceiling']:.1%} -> {out.relative_to(ROOT)}")

    (REPORT_DIR / "re_dataset_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport -> {(REPORT_DIR / 're_dataset_report.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
