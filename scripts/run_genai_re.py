"""GenAI study — Claude vs. the supervised model on document-level CID relation
extraction.

For each test abstract, give Claude the gold chemical/disease concepts (same
inputs the supervised model sees) and ask which pairs are chemical-induced
disease. Score against gold at the concept-pair level, and report cost + latency.

Run (mock, no API key needed — validates the pipeline):
    python scripts/run_genai_re.py --mock --limit 50

Run for real (needs ANTHROPIC_API_KEY in env or .env):
    python scripts/run_genai_re.py --limit 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcs_signalminer.data.pubtator import load_jsonl  # noqa: E402
from pcs_signalminer.relation.candidates import _concept_index  # noqa: E402
from pcs_signalminer.genai import claude_re  # noqa: E402

PROC = ROOT / "data" / "processed"
REPORT = ROOT / "reports"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="number of test docs (0=all)")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    # load .env if present (for ANTHROPIC_API_KEY)
    env = ROOT / ".env"
    if env.exists() and not args.mock:
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                import os
                os.environ.setdefault(k.strip(), v.strip())

    docs = load_jsonl(PROC / "bc5cdr_test.jsonl")
    if args.limit:
        docs = docs[:args.limit]

    client = None
    if not args.mock:
        try:
            import anthropic
            client = anthropic.Anthropic()
        except Exception as e:  # noqa: BLE001
            print(f"No Anthropic client ({e}); falling back to mock.")
            args.mock = True

    tp = fp = fn = 0
    total_cost = 0.0
    latencies = []
    for d in docs:
        chem = {cid: info["names"] for cid, info in _concept_index_dc(d, "Chemical").items()}
        dis = {did: info["names"] for did, info in _concept_index_dc(d, "Disease").items()}
        gold = {(r.chemical_mesh, r.disease_mesh) for r in d.relations}

        res = claude_re.extract_cid(d.text, chem, dis, client=client, mock=args.mock)
        pred = res.pairs
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
        total_cost += res.cost_usd
        if res.latency_s:
            latencies.append(res.latency_s)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    out = {
        "mode": "mock" if args.mock else "claude:" + claude_re.MODEL,
        "documents": len(docs),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "total_cost_usd": round(total_cost, 4),
        "mean_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "cost_per_doc_usd": round(total_cost / len(docs), 5) if docs else None,
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "genai_re_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


def _concept_index_dc(doc, etype):
    # thin wrapper so we can pass Document dataclasses from load_jsonl
    return _concept_index(doc, etype)


if __name__ == "__main__":
    main()
