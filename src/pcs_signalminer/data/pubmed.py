"""Fetch PubMed publication years for PMIDs via NCBI E-utilities (esummary).

Adds a temporal dimension to the corpus (which ships without dates), enabling
time-series trend analysis of safety signals. Results are cached to disk so the
network fetch runs once.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def _extract_year(pubdate: str) -> int | None:
    m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", pubdate or "")
    return int(m.group(1)) if m else None


def fetch_years(pmids: list[str], cache: str | Path,
                batch: int = 200, sleep: float = 0.4) -> dict[str, int]:
    """Return {pmid: year}. Reads/writes a JSON cache; only fetches missing ids."""
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    years: dict[str, int] = {}
    if cache.exists():
        years = {k: v for k, v in json.loads(cache.read_text()).items() if v}

    missing = [p for p in pmids if p not in years]
    for i in range(0, len(missing), batch):
        chunk = missing[i:i + batch]
        try:
            r = requests.get(ESUMMARY, params={
                "db": "pubmed", "id": ",".join(chunk), "retmode": "json",
            }, timeout=30)
            r.raise_for_status()
            result = r.json().get("result", {})
            for pid in chunk:
                rec = result.get(pid, {})
                yr = _extract_year(rec.get("pubdate", "") or rec.get("epubdate", ""))
                if yr:
                    years[pid] = yr
        except Exception as e:  # noqa: BLE001
            print(f"  warn: batch {i//batch} failed ({e})")
        time.sleep(sleep)
        cache.write_text(json.dumps(years), encoding="utf-8")

    return years
