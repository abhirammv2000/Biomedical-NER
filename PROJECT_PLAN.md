# PCS-SignalMiner: Project Plan

**Literature-based Drug-Safety Signal Mining for Preclinical Safety (PCS)**

End-to-end NLP system that reads biomedical literature (PubMed abstracts) and
automatically **extracts, normalizes, and ranks "chemical induces adverse
condition" signals**, assembling them into a queryable **toxicology knowledge
graph** with a stakeholder-facing dashboard.

> Target role context: Novartis *Expert Data Science, Preclinical Safety (PCS)*,
> a drug-safety / toxicology org that works with scientists & pathologists. This
> project mirrors a real PCS function: surfacing chemical/toxicity signals from
> the literature to support safety assessment.

---

## 1. Why this dataset & framing

**BC5CDR** (BioCreative V Chemical-Disease Relation corpus):
- 1,500 PubMed abstracts, split 500/500/500 (train/dev/test).
- 4,409 chemical mentions, 5,818 disease mentions, 3,116 chemical-induced-disease
  (CID) relations.
- Every entity normalized to a **MeSH** concept ID.
- Relations curated by **CTD (Comparative Toxicogenomics Database)** curators,
  the same toxicogenomics resource used in real drug-safety work.

The NER half of BC5CDR is **saturated** (chem ~93-95 F1, disease ~86-90 F1). We
treat NER as a solved component and put the emphasis on the harder,
still-open parts: document-level relation extraction, entity
normalization, a knowledge graph, and a GenAI build-vs-buy study.

## 2. Design principles
- **Real-world over leaderboard.** Optimize for a deployable, auditable pipeline,
  not a single F1 number. Every model decision ties back to the safety use case.
- **Reproducible.** Config-driven stages, fixed seeds, deterministic data splits,
  pinned deps, staged `scripts/` entrypoints.
- **Auditable / governed.** Provenance back to PMIDs on every extracted signal;
  data card + governance doc; licensing tracked.

## 3. Pipeline stages

| # | Stage | Output | Key JD skill hit |
|---|-------|--------|------------------|
| 0 | Scaffold & env | repo, env, configs | Git, reproducible practices |
| 1 | Data acquisition & QC | BC5CDR + CTD + MeSH loaded, quality report | multi-source structured+unstructured integration, data quality |
| 2 | EDA & biostatistics | corpus stats, IAA, co-occurrence enrichment | statistical analysis, EDA, visualization |
| 3 | NER (PubMedBERT) | chemical+disease tagger in SOTA band | entity recognition, model validation |
| 4 | Entity normalization | spans → MeSH IDs | text structuring, data integration |
| 5 | Relation extraction (supervised) | document-level CID classifier | relationships, classification, predictive modeling |
| 6 | **GenAI study** | Claude vs fine-tuned vs hybrid + cost/latency/error analysis | GenAI, model evaluation |
| 7 | Knowledge graph + SQL | DuckDB tables + NetworkX graph + graph algos | SQL/databases, graph algorithms |
| 8 | Topics, clustering, trends | BERTopic clusters + time-series of signals | topic modelling, clustering, time series |
| 9 | Dashboard | Streamlit/Plotly safety explorer | dashboards, viz, communication |
| 10 | Report & governance | exec summary, data card, limitations | communication, data governance |

## 4. Evaluation
- **NER:** entity-level micro/macro F1 (CoNLL eval), per-type breakdown.
- **Normalization:** accuracy of MeSH ID assignment on gold spans.
- **Relation extraction:** document-level CID precision/recall/F1 on the official
  test split; calibration; error taxonomy (cross-sentence misses, hallucinated
  relations, normalization-induced errors).
- **GenAI study:** F1 + $/1k-relations + latency + qualitative failure modes;
  build-vs-buy recommendation.
- **Graph / signals:** sanity-check top-ranked signals against CTD ground truth,
  plus a link-prediction held-out evaluation for "novel" signal discovery.

## 5. Stack
Python 3.11 · pandas/numpy/scikit-learn/matplotlib/seaborn · HuggingFace
Transformers · DuckDB (SQL) · NetworkX (graph algorithms) · BERTopic · Streamlit
+ Plotly · Anthropic SDK (Claude Opus 4.8) · cloud GPU for fine-tuning + batched
LLM inference.

## 6. Repo layout
```
data/            raw | interim | processed | external   (gitignored; see DATA_CARD)
src/pcs_signalminer/
  data/      loaders, parsers, QC
  ner/       training, inference, eval
  linking/   MeSH normalization
  relation/  supervised RE
  genai/     Claude-based RE + RAG/NL-query
  graph/     KG build, graph algorithms, SQL
  topics/    topic modeling, clustering, trends
  viz/       plotting helpers
scripts/         staged CLI entrypoints (run_*.py)
configs/         YAML configs per stage
notebooks/       EDA & analysis narratives
app/             Streamlit dashboard
reports/         figures + final report
docs/            DATA_CARD.md, GOVERNANCE.md, model cards
tests/           unit tests for parsers/metrics
```

## 7. Deliverables for the portfolio
1. Clean, reproducible repo with staged pipeline.
2. Technical report (modeling + results + error analysis + GenAI build-vs-buy).
3. Executive one-pager (non-technical, safety-stakeholder framing).
4. Live Streamlit dashboard (search a drug, see reported toxicities + evidence + graph).
5. Data card + governance/compliance note.
