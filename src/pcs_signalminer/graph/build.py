"""Build the chemical->disease safety knowledge graph from extracted CID signals,
load it into DuckDB (SQL layer), and run graph algorithms.

Signal source is pluggable: gold relations for development, or the model's
predicted relations in production. Each signal keeps its source PMID(s) so every
edge is auditable back to the literature.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import networkx as nx

from ..data.pubtator import Document


@dataclass
class Signal:
    chemical_mesh: str
    disease_mesh: str
    chemical_name: str
    disease_name: str
    pmids: list[str]


def build_name_map(docs: list[Document]) -> dict[tuple[str, str], str]:
    """(type, mesh_id) -> most common surface mention, for human-readable labels."""
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for d in docs:
        for e in d.entities:
            for m in e.mesh_ids:
                counts[(e.type, m)][e.text] += 1
    return {k: c.most_common(1)[0][0] for k, c in counts.items()}


def signals_from_gold(docs: list[Document]) -> list[Signal]:
    """Aggregate gold CID relations across the corpus into unique edges with
    provenance (the production path swaps this for predicted relations)."""
    name = build_name_map(docs)
    agg: dict[tuple[str, str], list[str]] = defaultdict(list)
    for d in docs:
        for r in d.relations:
            agg[(r.chemical_mesh, r.disease_mesh)].append(d.pmid)
    out = []
    for (c, dis), pmids in agg.items():
        out.append(Signal(
            chemical_mesh=c, disease_mesh=dis,
            chemical_name=name.get(("Chemical", c), c),
            disease_name=name.get(("Disease", dis), dis),
            pmids=sorted(set(pmids)),
        ))
    return out


def build_graph(signals: list[Signal]) -> nx.DiGraph:
    g = nx.DiGraph()
    for s in signals:
        g.add_node(s.chemical_mesh, kind="Chemical", label=s.chemical_name)
        g.add_node(s.disease_mesh, kind="Disease", label=s.disease_name)
        g.add_edge(s.chemical_mesh, s.disease_mesh, weight=len(s.pmids),
                   pmids=s.pmids)
    return g


def centrality(g: nx.DiGraph, top_k: int = 15) -> dict:
    label = nx.get_node_attributes(g, "label")
    out_deg = {n: g.out_degree(n) for n, k in nx.get_node_attributes(g, "kind").items()
               if k == "Chemical"}
    in_deg = {n: g.in_degree(n) for n, k in nx.get_node_attributes(g, "kind").items()
              if k == "Disease"}
    pr = nx.pagerank(g, weight="weight")
    top = lambda d: [{"mesh": n, "label": label.get(n, n), "degree": v}
                     for n, v in sorted(d.items(), key=lambda x: -x[1])[:top_k]]
    top_pr = [{"mesh": n, "label": label.get(n, n), "kind": g.nodes[n]["kind"],
               "pagerank": round(v, 5)}
              for n, v in sorted(pr.items(), key=lambda x: -x[1])[:top_k]]
    return {"top_inducer_chemicals": top(out_deg),
            "most_induced_diseases": top(in_deg),
            "top_pagerank": top_pr}


def communities(g: nx.DiGraph, max_communities: int = 10) -> list[dict]:
    """Detect toxicity clusters on the undirected projection."""
    ug = g.to_undirected()
    try:
        comms = nx.community.louvain_communities(ug, weight="weight", seed=42)
    except Exception:  # noqa: BLE001
        comms = nx.community.greedy_modularity_communities(ug, weight="weight")
    label = nx.get_node_attributes(g, "label")
    out = []
    for i, c in enumerate(sorted(comms, key=len, reverse=True)[:max_communities]):
        members = sorted(c, key=lambda n: -g.degree(n))
        out.append({
            "community": i,
            "size": len(c),
            "chemicals": [label.get(n, n) for n in members
                          if g.nodes[n]["kind"] == "Chemical"][:8],
            "diseases": [label.get(n, n) for n in members
                         if g.nodes[n]["kind"] == "Disease"][:8],
        })
    return out


def link_prediction(g: nx.DiGraph, top_k: int = 20) -> list[dict]:
    """Hypothesize novel chemical->disease signals via chemical similarity.

    Two chemicals are similar if they induce overlapping disease sets (Jaccard).
    For a chemical c and a disease d it does NOT yet induce, score d by the summed
    similarity of c to the chemicals that DO induce d. High scores = plausible
    undiscovered safety signals worth a human safety scientist's review.
    """
    label = nx.get_node_attributes(g, "label")
    chem_dis = {n: set(g.successors(n)) for n, k in nx.get_node_attributes(g, "kind").items()
                if k == "Chemical"}
    chem_dis = {c: ds for c, ds in chem_dis.items() if ds}

    def jaccard(a: set, b: set) -> float:
        u = a | b
        return len(a & b) / len(u) if u else 0.0

    inducers: dict[str, set] = defaultdict(set)
    for c, ds in chem_dis.items():
        for d in ds:
            inducers[d].add(c)

    scores: list[tuple[float, str, str]] = []
    chems = list(chem_dis)
    for c in chems:
        cd = chem_dis[c]
        # candidate diseases = those induced by similar chemicals but not by c
        cand: dict[str, float] = defaultdict(float)
        for d, cset in inducers.items():
            if d in cd:
                continue
            for c2 in cset:
                if c2 == c:
                    continue
                cand[d] += jaccard(cd, chem_dis[c2])
        for d, sc in cand.items():
            if sc > 0:
                scores.append((sc, c, d))
    scores.sort(reverse=True)
    return [{"chemical": label.get(c, c), "disease": label.get(d, d),
             "score": round(sc, 3), "chemical_mesh": c, "disease_mesh": d}
            for sc, c, d in scores[:top_k]]


def graph_summary(g: nx.DiGraph) -> dict:
    kinds = Counter(nx.get_node_attributes(g, "kind").values())
    return {"nodes": g.number_of_nodes(),
            "chemicals": kinds["Chemical"], "diseases": kinds["Disease"],
            "edges": g.number_of_edges()}
