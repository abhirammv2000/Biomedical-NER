"""Parser for the BC5CDR corpus in PubTator format.

PubTator layout per document:
    PMID|t|<title>
    PMID|a|<abstract>
    PMID<TAB>start<TAB>end<TAB>mention<TAB>type<TAB>MeSH_id(s)   (one per entity)
    PMID<TAB>CID<TAB>chemical_mesh<TAB>disease_mesh               (one per relation)
    <blank line between documents>

Entity character offsets index into ``title + " " + abstract``.
MeSH ids may be ``-1`` (unlinked) or composite (e.g. ``D1|D2`` for a combined
mention); composite ids are split into a list.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Entity:
    start: int
    end: int
    text: str
    type: str            # "Chemical" | "Disease"
    mesh_ids: list[str]  # [] when unlinked ("-1")


@dataclass
class Relation:
    chemical_mesh: str
    disease_mesh: str
    type: str = "CID"


@dataclass
class Document:
    pmid: str
    title: str
    abstract: str
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Full text the entity offsets index into."""
        return f"{self.title} {self.abstract}"


def _parse_mesh(raw: str) -> list[str]:
    raw = raw.strip()
    if raw in ("-1", "", "-"):
        return []
    # composite mentions use '|' or '+' to join concept ids
    parts = raw.replace("+", "|").split("|")
    return [p.strip() for p in parts if p.strip() and p.strip() != "-1"]


def parse_pubtator(path: str | Path) -> list[Document]:
    """Parse a PubTator file into a list of Document objects."""
    path = Path(path)
    docs: dict[str, Document] = {}
    order: list[str] = []

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if "|t|" in line:
                pmid, _, title = line.split("|", 2)
                docs.setdefault(pmid, Document(pmid=pmid, title="", abstract=""))
                docs[pmid].title = title
                if pmid not in order:
                    order.append(pmid)
            elif "|a|" in line:
                pmid, _, abstract = line.split("|", 2)
                docs.setdefault(pmid, Document(pmid=pmid, title="", abstract=""))
                docs[pmid].abstract = abstract
            else:
                cols = line.split("\t")
                pmid = cols[0]
                if len(cols) >= 6 and cols[1].isdigit():
                    # entity annotation
                    start, end, mention, etype, mesh = (
                        int(cols[1]), int(cols[2]), cols[3], cols[4], cols[5],
                    )
                    docs[pmid].entities.append(
                        Entity(start, end, mention, etype, _parse_mesh(mesh))
                    )
                elif len(cols) >= 4 and cols[1] == "CID":
                    docs[pmid].relations.append(
                        Relation(chemical_mesh=cols[2].strip(), disease_mesh=cols[3].strip())
                    )

    return [docs[p] for p in order]


def documents_to_jsonl(docs: list[Document], path: str | Path) -> None:
    """Serialize documents to JSON Lines (one document per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[Document]:
    """Load documents previously written by ``documents_to_jsonl``."""
    docs: list[Document] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            docs.append(
                Document(
                    pmid=rec["pmid"],
                    title=rec["title"],
                    abstract=rec["abstract"],
                    entities=[Entity(**e) for e in rec["entities"]],
                    relations=[Relation(**r) for r in rec["relations"]],
                )
            )
    return docs
