"""Build the candidate chemical and disease pairs for relation extraction.

BC5CDR annotates the relation per document over MeSH concept ids, not between mention spans. So for every chemical and disease
that appear together in an abstract, the task is to decide whether the chemical is reported to induce the disease. This lists the
pairs and labels them against the gold relations, with the source of every pair traceable.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

from ..data.pubtator import Document


@dataclass
class Candidate:
    pmid: str
    chemical_mesh: str
    disease_mesh: str
    label: int                       # 1 = chemical-induced disease, else 0
    chemical_names: list[str]        # surface mentions of the chemical concept
    disease_names: list[str]         # surface mentions of the disease concept
    chemical_offsets: list[tuple[int, int]]
    disease_offsets: list[tuple[int, int]]
    text: str                        # title + " " + abstract (provenance)


def _concept_index(doc: Document, etype: str) -> dict[str, dict]:
    """Map each normalized concept id -> its mentions/offsets in the document."""
    idx: dict[str, dict] = {}
    for e in doc.entities:
        if e.type != etype:
            continue
        for mid in e.mesh_ids:               # skip unlinked (empty list)
            slot = idx.setdefault(mid, {"names": [], "offsets": []})
            if e.text not in slot["names"]:
                slot["names"].append(e.text)
            slot["offsets"].append((e.start, e.end))
    return idx


def build_candidates(doc: Document) -> list[Candidate]:
    chem_idx = _concept_index(doc, "Chemical")
    dis_idx = _concept_index(doc, "Disease")
    gold = {(r.chemical_mesh, r.disease_mesh) for r in doc.relations}

    cands: list[Candidate] = []
    for cid, cinfo in chem_idx.items():
        for did, dinfo in dis_idx.items():
            cands.append(
                Candidate(
                    pmid=doc.pmid,
                    chemical_mesh=cid,
                    disease_mesh=did,
                    label=int((cid, did) in gold),
                    chemical_names=cinfo["names"],
                    disease_names=dinfo["names"],
                    chemical_offsets=cinfo["offsets"],
                    disease_offsets=dinfo["offsets"],
                    text=doc.text,
                )
            )
    return cands


def build_dataset(docs: list[Document]) -> list[Candidate]:
    out: list[Candidate] = []
    for d in docs:
        out.extend(build_candidates(d))
    return out


def candidates_to_jsonl(cands: list[Candidate], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in cands:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def dataset_stats(cands: list[Candidate]) -> dict:
    pos = sum(c.label for c in cands)
    n = len(cands)
    # recall ceiling: fraction of gold relations recoverable from co-occurring
    # concept pairs (a gold relation is unrecoverable if either concept was not
    # linked as an entity of the right type in the document).
    return {
        "candidate_pairs": n,
        "positives": pos,
        "negatives": n - pos,
        "positive_rate": round(pos / n, 4) if n else 0.0,
    }
