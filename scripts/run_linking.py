"""Stage: fit the MeSH dictionary normalizer on train and evaluate on dev/test.

Run:
    python scripts/run_linking.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcs_signalminer.data.pubtator import load_jsonl  # noqa: E402
from pcs_signalminer.linking.normalizer import DictionaryNormalizer, evaluate  # noqa: E402

PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORT = ROOT / "reports"


def main():
    train = load_jsonl(PROC / "bc5cdr_train.jsonl")
    norm = DictionaryNormalizer().fit(train)
    MODELS.mkdir(parents=True, exist_ok=True)
    norm.save(MODELS / "mesh_normalizer.json")

    report = {}
    for name in ("dev", "test"):
        docs = load_jsonl(PROC / f"bc5cdr_{name}.jsonl")
        exact = evaluate(norm, docs)
        fuzzy = evaluate(norm, docs, fuzzy_cutoff=0.9)
        report[name] = {"exact": exact, "fuzzy_0.9": fuzzy}
        me, mf = exact["micro"], fuzzy["micro"]
        print(f"[{name}] exact: acc={me['accuracy']:.3f} cov={me['coverage']:.3f} | "
              f"fuzzy0.9: acc={mf['accuracy']:.3f} cov={mf['coverage']:.3f}")

    (REPORT / "linking_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nNormalizer -> models/mesh_normalizer.json | metrics -> reports/linking_metrics.json")


if __name__ == "__main__":
    main()
