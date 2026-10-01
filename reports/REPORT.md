# PCS-SignalMiner: Technical Report

**Literature-based drug-safety signal mining on BC5CDR**

---

## 1. Problem & framing

Drug-safety and preclinical-safety teams need to know **which chemicals are
reported to induce which adverse conditions**, with traceable evidence. Doing
this by hand across the biomedical literature does not scale. PCS-SignalMiner is
an end-to-end NLP system that reads PubMed abstracts and automatically extracts,
normalizes, and ranks **chemical-induced-disease (CID) signals**, assembling them
into a queryable, auditable toxicology knowledge graph with a stakeholder
dashboard.

The corpus is **BC5CDR** (BioCreative V Chemical-Disease Relation): 1,500 PubMed
abstracts, whose CID relations were curated by **CTD (Comparative Toxicogenomics
Database)** curators, which makes it a realistic proxy for literature-based
safety surveillance rather than a generic NER benchmark.

Design principle throughout: **real-world deployability over leaderboard score.**
NER on BC5CDR is saturated (chem ~93-95 F1, disease ~86-90 F1 across PubMedBERT
/ BINDER / OpenMed NER), so it is treated as a solved component. The effort goes
into the parts that are still unsaturated: document-level relation
extraction, concept normalization, the knowledge graph, a GenAI build-vs-buy
study, and an auditable end-to-end pipeline.

## 2. Data & quality controls

| Split | Docs | Chemical mentions | Disease mentions | CID relations |
|------|-----:|------------------:|-----------------:|--------------:|
| train | 500 | 5,203 | 4,182 | 1,038 |
| dev   | 500 | 5,347 | 4,244 | 1,012 |
| test  | 500 | 5,385 | 4,424 | 1,066 |

Automated ingestion checks (`reports/data_quality_report.json`): **0 span-offset
mismatches**, **0 dangling relations** (every relation's concepts are linked
entities), splits **fully disjoint**, entity types valid. Entities normalized to
**MeSH**; provenance (source PMID) retained on every signal for auditability.

## 3. Pipeline & results

### 3.1 NER (chemical + disease)
Fine-tuned **PubMedBERT** token classifier with offset-based BIO alignment,
trained on **GCP (T4 GPU)**, evaluated with entity-level seqeval (test):

| Entity | Precision | Recall | F1 |
|---|---:|---:|---:|
| Chemical | 0.891 | 0.931 | **0.911** |
| Disease | 0.762 | 0.846 | **0.802** |
| **micro** | 0.830 | 0.893 | **0.860** |

Chemical NER sits near the usual band; disease NER (0.80) trails SOTA (~0.85-0.88).
The biggest untapped levers are sentence-splitting/sliding windows over the
256-token truncation, and a CRF decoding head. This stage feeds the downstream
graph; it is not the contribution of the project.

### 3.2 Entity normalization to MeSH
Dictionary normalizer fitted on train (the standard strong BC5CDR baseline) with
a fuzzy fallback for unseen surface forms:

| | accuracy | coverage | precision-when-predicted |
|---|---:|---:|---:|
| exact | 0.616 | 0.627 | ~0.98 |
| + fuzzy (0.9) | 0.637 | 0.652 | ~0.98 |

**Takeaway:** when the dictionary recognizes a term it is ~98% correct; the gap
is coverage on unseen concepts. A production system would add a full-MeSH
embedding/retrieval linker (see next steps).

### 3.3 Document-level relation extraction (the hard core)
The CID relation is annotated at the **document level over concept pairs**, often
cross-sentence. Formulation: enumerate every (chemical concept, disease concept)
co-occurring in an abstract and classify CID vs. not. Candidate set: ~5,400
pairs/split at a **19.7% positive rate**, with a **100% recall ceiling**
(every gold relation is recoverable from co-occurring linked concepts).
Model: entity-marker featurization ([CHEM]/[DIS] around all mentions of the
target concepts) + PubMedBERT sequence classifier with class weighting, trained
on the GCP T4. Test results on the positive (CID) class:

| Precision | Recall | F1 | Accuracy |
|---:|---:|---:|---:|
| 0.648 | 0.703 | **0.674** | 0.866 |

A competitive document-level CID result (cross-sentence relations, gold entities)
and the main technical contribution of the project.

### 3.4 GenAI build-vs-buy study
The relation step is also implemented with **Claude Opus 4.8** using structured
outputs (`output_config.format`) for guaranteed-parseable JSON, given the same
gold concept inputs as the supervised model, with cost + latency tracked
(`src/pcs_signalminer/genai/`). A deterministic co-occurrence **mock** establishes
the naive floor and lets the full evaluation run without an API key:

| Approach | Precision | Recall | F1 | Cost/doc |
|---|---:|---:|---:|---:|
| Co-occurrence (mock floor) | 0.33 | 0.73 | 0.46 | $0 |
| Supervised PubMedBERT (GCP T4) | 0.648 | 0.703 | **0.674** | $0 (after training) |
| Claude Opus 4.8 (few-shot, structured) | _run with key_ | | | |

The supervised model beats the naive co-occurrence floor by a wide margin
(0.674 vs 0.46 F1); the Claude arm is wired to slot in the accuracy/cost/latency
trade-off once run on the held-out set with a key.

This is the question most pharma DS teams have to answer at some point:
*when is a fine-tuned encoder worth it vs. a general LLM?* Answered here with
F1, $/doc, latency, and failure-mode analysis rather than a single number.

### 3.5 Knowledge graph + graph algorithms (SQL: DuckDB)
Signals assembled into a directed chemical/disease graph: **1,262 nodes**
(660 chemicals, 602 diseases), **2,434 edges**, persisted to **DuckDB** with a
SQL query layer (`signals`, `chemicals`, `diseases` tables).

- **Centrality**: top inducer chemicals are biomedically sensible: cocaine,
  doxorubicin (cardiotoxicity), amiodarone, tacrolimus, 5-FU.
- **Community detection** (Louvain): **10 toxicity clusters** grouping
  co-implicated chemicals and conditions.
- **Link prediction**: chemical-similarity (Jaccard over induced-disease sets)
  surfaces **novel CID hypotheses** (e.g. bupropion/hypotension) as ranked
  candidates for human safety review.

### 3.6 Trends & topics
Publication years fetched from **NCBI E-utilities** for all 1,500 PMIDs
(multi-source integration); signals span **1968-2016**. **BERTopic** (MiniLM
embeddings + stopword-filtered bigram vectorizer) yields **26 coherent topics**,
e.g. liver/hepatitis/failure, bupivacaine/anaesthesia, pilocarpine/seizures,
breast-cancer/prolactin, gentamicin/nephropathy.

### 3.7 Biostatistics
Fisher's exact test (one-sided) with Benjamini-Hochberg FDR identifies **49
chemicals significantly enriched** (q < 0.05) as disease inducers vs. the rest of
the corpus (`reports/chemical_enrichment.csv`), a statistical lens on which
chemicals carry disproportionate safety signal.

## 4. Dashboard
`streamlit run app/dashboard.py`: search a chemical, see reported toxicities
with evidence counts and clickable PubMed links; graph insights (centrality,
communities, novel hypotheses); trends & topics; and model/GenAI metrics.

## 5. Governance & limitations
- **Provenance on every signal** (source PMIDs); pipeline is config-driven and
  seeded; runs reproduce from raw data to graph.
- Extracted signals are **hypotheses for review**, not confirmed safety
  findings. Human-in-the-loop by design (see `docs/GOVERNANCE.md`).
- Public, non-PHI data only.
- **Limitations:** normalization coverage on unseen concepts; BC5CDR is abstracts
  (not full text or clinical notes); link-prediction hypotheses require expert
  validation.

## 6. Reproducibility
```
python scripts/run_download_data.py     # parse + QC
python scripts/run_eda.py               # EDA + biostatistics
python scripts/run_linking.py           # MeSH normalization
python scripts/run_build_re_dataset.py  # RE candidates
bash   scripts/gcp_train.sh             # NER + relation on GCP T4
python scripts/run_genai_re.py --mock   # GenAI study (drop --mock with a key)
python scripts/run_build_graph.py       # knowledge graph + SQL + graph algos
python scripts/run_topics_trends.py     # topics + time-series
streamlit run app/dashboard.py          # dashboard
```
