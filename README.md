# PCS-SignalMiner

Literature-based drug-safety signal mining for Preclinical Safety.

PCS-SignalMiner reads PubMed abstracts, extracts and normalizes "chemical
induces adverse condition" signals, and assembles them into a queryable
toxicology knowledge graph with a stakeholder dashboard.

Built on the **BC5CDR** corpus (BioCreative V Chemical-Disease Relation), with
relations curated by **CTD** (Comparative Toxicogenomics Database), which makes
it a realistic proxy for literature-based safety surveillance.

Drug-safety teams need to know which chemicals are reported to induce which
adverse conditions, with evidence and provenance. This pipeline automates that
mining end to end.

## What it does
1. **NER**: finds chemical & disease mentions (fine-tuned PubMedBERT).
2. **Normalization**: links mentions to MeSH concept IDs.
3. **Relation extraction**: decides, document-level, which chemical/disease
   pairs are *chemical-induced disease* (a supervised model and a Claude-based
   GenAI approach, compared head-to-head).
4. **Knowledge graph**: assembles signals into a DuckDB + NetworkX graph and
   runs graph algorithms (centrality, communities, link prediction).
5. **Analytics**: topic modeling, clustering, and time-series trends of signals.
6. **Dashboard**: search a drug, see reported toxicities, evidence, and graph
   neighborhood.

## Quickstart
```bash
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate biomed-ner
python scripts/run_download_data.py       # stage 1
python scripts/run_eda.py                 # stage 2
# ... see PROJECT_PLAN.md for the full staged pipeline
streamlit run app/dashboard.py            # dashboard
```

## Documentation
- [PROJECT_PLAN.md](PROJECT_PLAN.md): full plan, stages and evaluation.
- [docs/DATA_CARD.md](docs/DATA_CARD.md): datasets, licensing, provenance.
- [docs/GOVERNANCE.md](docs/GOVERNANCE.md): data governance & compliance notes.

## Results at a glance
- **Data:** 1,500 abstracts parsed; QC clean (0 offset errors, splits disjoint, 3,116 CID relations).
- **Relation task:** document-level CID candidate set, ~5.4k pairs/split, **100% recall ceiling**.
- **Normalization:** MeSH dictionary linker, **~98% precision-when-predicted** (+fuzzy fallback).
- **Knowledge graph:** 1,262 nodes / 2,434 edges in DuckDB; centrality, **10 toxicity communities**, link-prediction hypotheses.
- **Biostatistics:** **49 chemicals** significantly enriched as inducers (Fisher exact, BH-FDR q<0.05).
- **Topics/trends:** 26 coherent BERTopic topics; signal time-series 1968-2016 (PubMed years via E-utilities).
- **GenAI study:** a Claude Opus 4.8 structured-output relation extractor is built, but only the no-API baseline (co-occurrence, F1 0.46) has been run. The Claude arm has not been evaluated.
- **Supervised models:** PubMedBERT NER + relation trained on **GCP T4** (`scripts/gcp_train.sh`).

See [reports/REPORT.md](reports/REPORT.md) for the full technical writeup and [PROJECT_PLAN.md](PROJECT_PLAN.md) for the roadmap.
