"""Stage 2 — Exploratory data analysis + biostatistics on BC5CDR.

Produces:
  - reports/figures/*.png      (corpus visualizations)
  - reports/eda_summary.md     (narrative + key stats)
  - reports/chemical_enrichment.csv  (Fisher-exact CID enrichment per chemical)

Run:
    python scripts/run_eda.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcs_signalminer.data.pubtator import load_jsonl  # noqa: E402
from pcs_signalminer.relation import candidates as C  # noqa: E402

PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
REPORT = ROOT / "reports"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})


def load_all():
    return {n: load_jsonl(PROC / f"bc5cdr_{n}.jsonl") for n in ("train", "dev", "test")}


def fig_mentions_by_type(splits):
    rows = []
    for name, docs in splits.items():
        c = Counter(e.type for d in docs for e in d.entities)
        rows.append({"split": name, "Chemical": c["Chemical"], "Disease": c["Disease"]})
    df = pd.DataFrame(rows).set_index("split")
    ax = df.plot(kind="bar", color=["#2b8cbe", "#e34a33"])
    ax.set_ylabel("mentions"); ax.set_title("Entity mentions by type and split")
    plt.savefig(FIG / "mentions_by_type.png"); plt.close()
    return df


def fig_mention_length(splits):
    lens = {"Chemical": [], "Disease": []}
    for docs in splits.values():
        for d in docs:
            for e in d.entities:
                lens[e.type].append(len(e.text))
    plt.figure()
    plt.hist(lens["Chemical"], bins=40, alpha=0.6, label="Chemical", color="#2b8cbe")
    plt.hist(lens["Disease"], bins=40, alpha=0.6, label="Disease", color="#e34a33")
    plt.xlabel("mention length (chars)"); plt.ylabel("count")
    plt.title("Mention length distribution"); plt.legend()
    plt.xlim(0, 60)
    plt.savefig(FIG / "mention_length.png"); plt.close()


def fig_top_concepts(splits):
    chem, dis = Counter(), Counter()
    for docs in splits.values():
        for d in docs:
            for e in d.entities:
                for m in e.mesh_ids:
                    (chem if e.type == "Chemical" else dis)[(m, e.text.lower())] += 1
    # aggregate by concept id, keep a representative name
    def top(counter, k=15):
        agg, name = Counter(), {}
        for (mid, nm), v in counter.items():
            agg[mid] += v
            name.setdefault(mid, nm)
        return [(name[mid], v) for mid, v in agg.most_common(k)]

    for title, data, color, fname in [
        ("Top chemicals", top(chem), "#2b8cbe", "top_chemicals.png"),
        ("Top diseases", top(dis), "#e34a33", "top_diseases.png"),
    ]:
        labels = [n for n, _ in data][::-1]
        vals = [v for _, v in data][::-1]
        plt.figure(figsize=(6, 5))
        plt.barh(labels, vals, color=color)
        plt.title(title); plt.xlabel("mentions")
        plt.savefig(FIG / fname); plt.close()


def fig_cid_degree(splits):
    """Distribution of #diseases induced per chemical (gold CID)."""
    deg = Counter()
    for docs in splits.values():
        for d in docs:
            per_chem = Counter(r.chemical_mesh for r in d.relations)
            for ch, k in per_chem.items():
                deg[ch] += k
    vals = list(deg.values())
    plt.figure()
    plt.hist(vals, bins=range(1, max(vals) + 2), color="#756bb1", align="left")
    plt.xlabel("# CID relations per chemical (corpus-wide)")
    plt.ylabel("# chemicals"); plt.title("Chemical -> disease CID degree distribution")
    plt.savefig(FIG / "cid_degree.png"); plt.close()


def chemical_enrichment(splits) -> pd.DataFrame:
    """Fisher-exact test: is a chemical's candidate pairs CID-positive more often
    than the rest of the corpus? Identifies chemicals statistically enriched as
    disease inducers (a literature-derived safety signal)."""
    all_docs = [d for docs in splits.values() for d in docs]
    cands = C.build_dataset(all_docs)
    df = pd.DataFrame([{"chem": c.chemical_mesh,
                        "chem_name": c.chemical_names[0] if c.chemical_names else c.chemical_mesh,
                        "label": c.label} for c in cands])
    tot_pos = int(df.label.sum()); tot = len(df)
    rows = []
    for chem, g in df.groupby("chem"):
        n = len(g)
        if n < 5:                       # require minimum support
            continue
        pos = int(g.label.sum())
        a, b = pos, n - pos             # this chemical: pos, neg
        c_, d_ = tot_pos - pos, (tot - n) - (tot_pos - pos)   # rest: pos, neg
        odds, p = fisher_exact([[a, b], [c_, d_]], alternative="greater")
        rows.append({"chemical": g.chem_name.iloc[0], "mesh": chem,
                     "candidate_pairs": n, "cid_positive": pos,
                     "cid_rate": round(pos / n, 3), "odds_ratio": round(odds, 2), "p": p})
    res = pd.DataFrame(rows)
    res["q_bh"] = multipletests(res.p, method="fdr_bh")[1]
    res = res.sort_values("q_bh").reset_index(drop=True)
    return res


def main():
    splits = load_all()
    df_types = fig_mentions_by_type(splits)
    fig_mention_length(splits)
    fig_top_concepts(splits)
    fig_cid_degree(splits)
    enr = chemical_enrichment(splits)
    enr.to_csv(REPORT / "chemical_enrichment.csv", index=False)

    n_docs = sum(len(v) for v in splits.values())
    uniq_chem = len({m for v in splits.values() for d in v for e in d.entities
                     if e.type == "Chemical" for m in e.mesh_ids})
    uniq_dis = len({m for v in splits.values() for d in v for e in d.entities
                    if e.type == "Disease" for m in e.mesh_ids})
    n_rel = sum(len(d.relations) for v in splits.values() for d in v)
    sig = enr[enr.q_bh < 0.05]

    lines = [
        "# EDA Summary — BC5CDR\n",
        f"- Documents: **{n_docs}** (500 train / 500 dev / 500 test)",
        f"- Chemical mentions: **{int(df_types['Chemical'].sum())}**, "
        f"Disease mentions: **{int(df_types['Disease'].sum())}**",
        f"- Unique chemical concepts (MeSH): **{uniq_chem}**, "
        f"unique disease concepts: **{uniq_dis}**",
        f"- Gold CID relations: **{n_rel}**\n",
        "## Biostatistics — chemicals enriched as disease inducers",
        f"Fisher's exact test (one-sided) with Benjamini-Hochberg FDR correction. "
        f"**{len(sig)}** chemicals are significantly enriched (q < 0.05) for "
        f"chemical-induced-disease relations vs. the rest of the corpus.\n",
        "Top 10 by significance:\n",
        sig.head(10).to_markdown(index=False) if len(sig) else "_none_",
        "\n## Figures",
        "- `figures/mentions_by_type.png`",
        "- `figures/mention_length.png`",
        "- `figures/top_chemicals.png`, `figures/top_diseases.png`",
        "- `figures/cid_degree.png`",
    ]
    (REPORT / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"EDA complete. {len(sig)} significantly enriched chemicals (q<0.05).")
    print(f"Figures -> {FIG.relative_to(ROOT)} | summary -> reports/eda_summary.md")


if __name__ == "__main__":
    main()
