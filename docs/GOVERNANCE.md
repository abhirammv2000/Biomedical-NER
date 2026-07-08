# Data Governance & Compliance Notes

This project is built to mirror the governance expectations of a regulated
drug-safety environment, even though it uses only public literature data.

## Data classification
- **Public, non-PHI.** All inputs are published PubMed abstracts and public
  reference databases (CTD, MeSH). No patient-level or proprietary data.

## Provenance & auditability
- Every extracted chemical→disease signal carries its **source PMID** and the
  text span(s) supporting it, so any claim is traceable to evidence.
- Pipeline stages are config-driven and seeded; runs are reproducible from raw
  data to final graph.

## Quality controls
- Automated data-quality checks at ingestion (offset validity, label
  consistency, split disjointness, MeSH resolvability).
- Model evaluation on the held-out official test split only; no test leakage.
- Extracted signals are framed as **hypotheses for review**, never as
  confirmed safety findings — a human safety scientist remains in the loop.

## Responsible-use notes
- GenAI (LLM) outputs are validated against structured references and never
  surfaced without provenance, to mitigate hallucinated relations.
- Limitations and assumptions are documented in the final report.

## Licensing
- BC5CDR: research use; cite corpus paper. CTD/MeSH: per their respective terms.
- See [DATA_CARD.md](DATA_CARD.md) for details.
