# Data Card

## Primary corpus — BC5CDR (BioCreative V Chemical-Disease Relation)
- **What:** 1,500 PubMed abstracts manually annotated for chemical and disease
  mentions and chemical-induced-disease (CID) relations.
- **Splits:** 500 train / 500 dev / 500 test (official).
- **Counts:** 4,409 chemicals, 5,818 diseases, 3,116 CID relations.
- **Normalization:** entities mapped to **MeSH** concept identifiers.
- **Curation:** MeSH indexers (entities) + CTD curators (relations).
- **Source:** BioCreative V CDR task; corpus paper *Li et al., Database (2016),
  doi:10.1093/database/baw068*.
- **License/usage:** publicly released for research; cite the corpus paper.
  PubMed abstracts are public; no patient-level / PHI data is involved.

## Structured reference — CTD (Comparative Toxicogenomics Database)
- **Use:** ground-truth / enrichment for chemical–disease associations; sanity
  check of extracted signals and link-prediction evaluation.
- **Note:** respect CTD's data usage terms; used here for research evaluation.

## Ontology — MeSH (Medical Subject Headings)
- **Use:** canonical vocabulary for normalizing chemical & disease mentions.
- **Source:** U.S. National Library of Medicine.

## Provenance & quality
- Every extracted signal retains its source **PMID** for auditability.
- Data-quality checks: span offset validity, label consistency, split disjointness,
  duplicate-abstract detection, MeSH-ID resolvability.

## Privacy / governance
- No PHI; literature-derived only. See [GOVERNANCE.md](GOVERNANCE.md).
