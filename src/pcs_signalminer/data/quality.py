"""Data-quality checks for the parsed BC5CDR corpus.

These mirror the ingestion checks expected in a governed data pipeline:
offset integrity, label validity, split disjointness, duplicate detection, and
referential consistency between relations and entities.
"""
from __future__ import annotations

from collections import Counter

from .pubtator import Document

VALID_TYPES = {"Chemical", "Disease"}


def check_offsets(docs: list[Document]) -> dict:
    """Verify that every entity span matches the underlying text slice."""
    mismatches = []
    total = 0
    for d in docs:
        text = d.text
        for e in d.entities:
            total += 1
            if text[e.start:e.end] != e.text:
                mismatches.append(
                    {"pmid": d.pmid, "span": (e.start, e.end),
                     "expected": e.text, "found": text[e.start:e.end]}
                )
    return {"total_entities": total, "offset_mismatches": len(mismatches),
            "examples": mismatches[:5]}


def check_label_validity(docs: list[Document]) -> dict:
    bad = [e.type for d in docs for e in d.entities if e.type not in VALID_TYPES]
    return {"invalid_type_count": len(bad), "invalid_types": dict(Counter(bad))}


def check_split_disjoint(splits: dict[str, list[Document]]) -> dict:
    pmids = {name: {d.pmid for d in docs} for name, docs in splits.items()}
    overlaps = {}
    names = list(pmids)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            common = pmids[a] & pmids[b]
            if common:
                overlaps[f"{a}&{b}"] = len(common)
    return {"disjoint": not overlaps, "overlaps": overlaps}


def check_duplicates(docs: list[Document]) -> dict:
    texts = Counter(d.text for d in docs)
    dups = {t: c for t, c in texts.items() if c > 1}
    return {"duplicate_abstract_groups": len(dups)}


def check_relation_consistency(docs: list[Document]) -> dict:
    """Each relation's chemical/disease MeSH id should appear among the doc's
    linked entities of the corresponding type."""
    dangling = 0
    total = 0
    for d in docs:
        chem_ids = {m for e in d.entities if e.type == "Chemical" for m in e.mesh_ids}
        dis_ids = {m for e in d.entities if e.type == "Disease" for m in e.mesh_ids}
        for r in d.relations:
            total += 1
            if r.chemical_mesh not in chem_ids or r.disease_mesh not in dis_ids:
                dangling += 1
    return {"total_relations": total, "dangling_relations": dangling}


def corpus_summary(docs: list[Document]) -> dict:
    n_chem = sum(1 for d in docs for e in d.entities if e.type == "Chemical")
    n_dis = sum(1 for d in docs for e in d.entities if e.type == "Disease")
    n_rel = sum(len(d.relations) for d in docs)
    n_unlinked = sum(1 for d in docs for e in d.entities if not e.mesh_ids)
    return {
        "documents": len(docs),
        "chemical_mentions": n_chem,
        "disease_mentions": n_dis,
        "cid_relations": n_rel,
        "unlinked_mentions": n_unlinked,
    }


def run_all(splits: dict[str, list[Document]]) -> dict:
    report: dict = {"per_split": {}, "global": {}}
    for name, docs in splits.items():
        report["per_split"][name] = {
            "summary": corpus_summary(docs),
            "offsets": check_offsets(docs),
            "labels": check_label_validity(docs),
            "duplicates": check_duplicates(docs),
            "relation_consistency": check_relation_consistency(docs),
        }
    report["global"]["split_disjoint"] = check_split_disjoint(splits)
    return report
