"""PCS-SignalMiner drug-safety signal explorer.

Run:
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DUCKDB = ROOT / "data" / "processed" / "signals.duckdb"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"

st.set_page_config(page_title="PCS-SignalMiner", page_icon="💊", layout="wide")


@st.cache_data
def load_table(sql: str) -> pd.DataFrame:
    con = duckdb.connect(str(DUCKDB), read_only=True)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def load_json(name: str) -> dict | None:
    p = REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


if not DUCKDB.exists():
    st.error("Knowledge graph not built yet. Run `python scripts/run_build_graph.py`.")
    st.stop()

signals = load_table("SELECT * FROM signals")
chemicals = load_table("SELECT * FROM chemicals ORDER BY n_diseases DESC")
diseases = load_table("SELECT * FROM diseases ORDER BY n_chemicals DESC")

st.title("💊 PCS-SignalMiner")
st.caption("Literature-based drug-safety signal mining (BC5CDR): chemical-induced "
           "disease signals with provenance, for preclinical safety review.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Chemicals", f"{len(chemicals):,}")
c2.metric("Diseases", f"{len(diseases):,}")
c3.metric("Signals (edges)", f"{len(signals):,}")
c4.metric("Evidence mentions", f"{int(signals.n_pmids.sum()):,}")

tab_drug, tab_graph, tab_trends, tab_models = st.tabs(
    ["🔎 Drug Safety Explorer", "🕸️ Graph Insights", "📈 Trends & Topics", "🤖 Models & GenAI"])

# --------------------------------------------------------------------------- #
with tab_drug:
    st.subheader("Search a chemical → reported adverse conditions")
    drug = st.selectbox("Chemical", chemicals.name.tolist(), index=0)
    sub = signals[signals.chemical_name == drug].sort_values("n_pmids", ascending=False)
    st.markdown(f"**{drug}** is reported to induce **{len(sub)}** distinct conditions.")
    left, right = st.columns([1, 1])
    with left:
        st.dataframe(
            sub[["disease_name", "n_pmids"]].rename(
                columns={"disease_name": "Induced condition", "n_pmids": "Evidence (PMIDs)"}),
            hide_index=True, use_container_width=True)
    with right:
        if len(sub):
            fig = px.bar(sub.head(15), x="n_pmids", y="disease_name", orientation="h",
                         labels={"n_pmids": "evidence (PMIDs)", "disease_name": ""},
                         title=f"Top reported toxicities for {drug}")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
            st.plotly_chart(fig, use_container_width=True)
    with st.expander("Evidence (source PMIDs)"):
        for _, r in sub.head(20).iterrows():
            pmids = r.pmids.split(";")
            links = ", ".join(
                f"[{p}](https://pubmed.ncbi.nlm.nih.gov/{p}/)" for p in pmids[:8])
            st.markdown(f"- **{drug} → {r.disease_name}**: {links}")

# --------------------------------------------------------------------------- #
with tab_graph:
    gr = load_json("graph_report.json")
    if not gr:
        st.info("Run `python scripts/run_build_graph.py` to generate graph insights.")
    else:
        st.subheader("Centrality: who drives the safety graph")
        a, b = st.columns(2)
        a.markdown("**Top inducer chemicals** (by # distinct diseases)")
        a.dataframe(pd.DataFrame(gr["centrality"]["top_inducer_chemicals"])[["label", "degree"]]
                    .rename(columns={"label": "chemical", "degree": "# diseases"}),
                    hide_index=True, use_container_width=True)
        b.markdown("**Most-implicated diseases** (by # distinct chemicals)")
        b.dataframe(pd.DataFrame(gr["centrality"]["most_induced_diseases"])[["label", "degree"]]
                    .rename(columns={"label": "disease", "degree": "# chemicals"}),
                    hide_index=True, use_container_width=True)

        st.subheader("🔬 Novel safety-signal hypotheses (link prediction)")
        st.caption("Chemical→disease pairs NOT in the literature graph, ranked by "
                   "similarity to known inducers. Hypotheses for human review, not confirmed findings.")
        st.dataframe(pd.DataFrame(gr["novel_signal_hypotheses"])[["chemical", "disease", "score"]],
                     hide_index=True, use_container_width=True)

        st.subheader("🧩 Toxicity communities")
        for c in gr["communities"][:6]:
            st.markdown(f"**Community {c['community']}** ({c['size']} nodes): "
                        f"chemicals: {', '.join(c['chemicals'][:5])} … "
                        f"diseases: {', '.join(c['diseases'][:5])}")

# --------------------------------------------------------------------------- #
with tab_trends:
    tt = load_json("topics_trends_report.json")
    if (FIGS / "signals_per_year.png").exists():
        st.subheader("Safety signals reported over time")
        st.image(str(FIGS / "signals_per_year.png"))
    if tt:
        st.subheader("Abstract topics (BERTopic)")
        rows = [{"topic": t["topic"], "n_docs": t["count"], "terms": ", ".join(t["terms"])}
                for t in tt["topics"]["top_topics"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    enr = REPORTS / "chemical_enrichment.csv"
    if enr.exists():
        st.subheader("Chemicals statistically enriched as inducers (Fisher exact, BH-FDR)")
        df = pd.read_csv(enr)
        st.dataframe(df[df.q_bh < 0.05][["chemical", "candidate_pairs", "cid_rate",
                                          "odds_ratio", "q_bh"]].head(20),
                     hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------- #
with tab_models:
    st.subheader("Model performance")
    ner = load_json("ner_test_metrics.json")
    rel = load_json("relation_test_metrics.json")
    link = load_json("linking_metrics.json")
    genai = load_json("genai_re_metrics.json")

    cols = st.columns(3)
    if ner:
        cols[0].metric("NER F1 (entity-level)", f"{ner.get('f1', 0):.3f}")
    else:
        cols[0].info("NER metrics pending (GPU training)")
    if rel:
        cols[1].metric("Relation F1 (CID)", f"{rel.get('f1', 0):.3f}")
    else:
        cols[1].info("Relation metrics pending")
    if link:
        m = link.get("test", {}).get("fuzzy_0.9", link.get("test", {})).get("micro", {})
        if m:
            cols[2].metric("MeSH linking accuracy", f"{m.get('accuracy', 0):.3f}")

    if genai:
        st.subheader("Build-vs-buy: GenAI relation extraction")
        st.caption(f"Mode: {genai.get('mode')}")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Precision", f"{genai.get('precision', 0):.3f}")
        g2.metric("Recall", f"{genai.get('recall', 0):.3f}")
        g3.metric("F1", f"{genai.get('f1', 0):.3f}")
        g4.metric("Cost / doc", f"${genai.get('cost_per_doc_usd') or 0:.4f}")
    st.caption("Supervised models train on GCP T4; the GenAI arm runs Claude Opus 4.8 "
               "with structured outputs. See reports/ for full metrics.")
