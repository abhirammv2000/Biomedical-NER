"""Stage 1: parse the BC5CDR PubTator files, run data-quality checks, and
write processed JSONL + a QC report.

Assumes the raw corpus has been extracted to:
    data/raw/CDR_Data/CDR.Corpus.v010516/CDR_{Training,Development,Test}Set.PubTator.txt

Run:
    python scripts/run_download_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcs_signalminer.data.pubtator import parse_pubtator, documents_to_jsonl  # noqa: E402
from pcs_signalminer.data import quality  # noqa: E402

RAW_DIR = ROOT / "data" / "raw" / "CDR_Data" / "CDR.Corpus.v010516"
PROC_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

SPLIT_FILES = {
    "train": "CDR_TrainingSet.PubTator.txt",
    "dev": "CDR_DevelopmentSet.PubTator.txt",
    "test": "CDR_TestSet.PubTator.txt",
}


def main() -> None:
    splits = {}
    for name, fname in SPLIT_FILES.items():
        path = RAW_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Extract CDR_Data.zip into data/raw/ first."
            )
        docs = parse_pubtator(path)
        splits[name] = docs
        out = PROC_DIR / f"bc5cdr_{name}.jsonl"
        documents_to_jsonl(docs, out)
        print(f"[{name}] {len(docs)} docs -> {out.relative_to(ROOT)}")

    report = quality.run_all(splits)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "data_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nQC report -> {report_path.relative_to(ROOT)}")

    # console summary
    print("\n=== Corpus summary ===")
    for name, info in report["per_split"].items():
        s = info["summary"]
        off = info["offsets"]["offset_mismatches"]
        dang = info["relation_consistency"]["dangling_relations"]
        print(f"{name:>5}: {s['documents']} docs | "
              f"{s['chemical_mentions']} chem | {s['disease_mentions']} dis | "
              f"{s['cid_relations']} CID | unlinked={s['unlinked_mentions']} | "
              f"offset_err={off} | dangling_rel={dang}")
    print(f"\nSplit disjoint: {report['global']['split_disjoint']['disjoint']}")


if __name__ == "__main__":
    main()
