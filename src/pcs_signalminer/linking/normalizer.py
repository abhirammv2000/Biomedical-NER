"""Entity normalization: map chemical/disease surface mentions to MeSH concept
ids.

Implements a dictionary normalizer fitted on the training split, the standard
strong baseline for BC5CDR normalization. A surface form is mapped to the most
frequent MeSH id observed for that (surface, type) in training; light
normalization (lowercasing, whitespace/punctuation trimming) improves recall on
unseen casing. Unseen mentions are left unlinked (returned as None) so the
pipeline never fabricates a concept id.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..data.pubtator import Document

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s.strip().lower())


class DictionaryNormalizer:
    def __init__(self) -> None:
        # type -> surface -> Counter(mesh_id)
        self.table: dict[str, dict[str, Counter]] = {
            "Chemical": defaultdict(Counter), "Disease": defaultdict(Counter)}

    def fit(self, docs: list[Document]) -> "DictionaryNormalizer":
        for d in docs:
            for e in d.entities:
                if not e.mesh_ids:
                    continue
                # use the primary (first) concept id for composite mentions
                self.table[e.type][_norm(e.text)][e.mesh_ids[0]] += 1
        return self

    def predict(self, surface: str, etype: str, fuzzy_cutoff: float | None = None) -> str | None:
        tbl = self.table.get(etype, {})
        cand = tbl.get(_norm(surface))
        if cand:
            return cand.most_common(1)[0][0]
        if fuzzy_cutoff is not None:
            import difflib
            match = difflib.get_close_matches(_norm(surface), tbl.keys(),
                                              n=1, cutoff=fuzzy_cutoff)
            if match:
                return tbl[match[0]].most_common(1)[0][0]
        return None

    def save(self, path: str | Path) -> None:
        obj = {t: {s: dict(c) for s, c in tbl.items()} for t, tbl in self.table.items()}
        Path(path).write_text(json.dumps(obj), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DictionaryNormalizer":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        inst = cls()
        for t, tbl in obj.items():
            for s, c in tbl.items():
                inst.table[t][s] = Counter(c)
        return inst


def evaluate(norm: DictionaryNormalizer, docs: list[Document],
             fuzzy_cutoff: float | None = None) -> dict:
    """Accuracy on gold linked spans: prediction counts as correct if it is among
    the gold concept ids for that mention. Also reports coverage (share of
    mentions that received any prediction)."""
    by_type = {"Chemical": [0, 0, 0], "Disease": [0, 0, 0]}  # [correct, predicted, total]
    for d in docs:
        for e in d.entities:
            if not e.mesh_ids:
                continue
            t = e.type
            by_type[t][2] += 1
            pred = norm.predict(e.text, t, fuzzy_cutoff=fuzzy_cutoff)
            if pred is not None:
                by_type[t][1] += 1
                if pred in set(e.mesh_ids):
                    by_type[t][0] += 1
    out = {}
    tot_c = tot_p = tot_n = 0
    for t, (c, p, n) in by_type.items():
        out[t] = {
            "accuracy": round(c / n, 4) if n else 0.0,           # over all gold
            "precision_when_predicted": round(c / p, 4) if p else 0.0,
            "coverage": round(p / n, 4) if n else 0.0,
            "n": n,
        }
        tot_c += c; tot_p += p; tot_n += n
    out["micro"] = {
        "accuracy": round(tot_c / tot_n, 4) if tot_n else 0.0,
        "coverage": round(tot_p / tot_n, 4) if tot_n else 0.0,
        "n": tot_n,
    }
    return out
