"""Build the safety knowledge graph, load DuckDB, run graph algorithms.

Source = gold CID relations across all splits (dev path). Swap
`signals_from_gold` for predicted relations once the relation model is trained.

Run:
    python scripts/run_build_graph.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcs_signalminer.data.pubtator import load_jsonl  # noqa: E402
from pcs_signalminer.graph import build, store  # noqa: E402

PROC = ROOT / "data" / "processed"
REPORT = ROOT / "reports"
DUCKDB = ROOT / "data" / "processed" / "signals.duckdb"


def main():
    docs = []
    for name in ("train", "dev", "test"):
        docs.extend(load_jsonl(PROC / f"bc5cdr_{name}.jsonl"))

    signals = build.signals_from_gold(docs)
    g = build.build_graph(signals)

    # SQL layer
    df = store.signals_to_df(signals)
    store.write_duckdb(df, DUCKDB)
    sql = store.example_queries(DUCKDB)

    report = {
        "summary": build.graph_summary(g),
        "centrality": build.centrality(g),
        "communities": build.communities(g),
        "novel_signal_hypotheses": build.link_prediction(g),
        "sql_examples": sql,
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "graph_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    s = report["summary"]
    print(f"Graph: {s['nodes']} nodes ({s['chemicals']} chem, {s['diseases']} dis), "
          f"{s['edges']} edges -> {DUCKDB.relative_to(ROOT)}")
    print("\nTop inducer chemicals (by # distinct diseases):")
    for r in report["centrality"]["top_inducer_chemicals"][:5]:
        print(f"  {r['label']:<22} induces {r['degree']} diseases")
    print("\nTop novel signal hypotheses (link prediction):")
    for r in report["novel_signal_hypotheses"][:5]:
        print(f"  {r['chemical']:<20} -> {r['disease']:<25} score={r['score']}")
    print(f"\nFull report -> {(REPORT / 'graph_report.json').relative_to(ROOT)}")
    print(f"{len(report['communities'])} toxicity communities detected.")


if __name__ == "__main__":
    main()
