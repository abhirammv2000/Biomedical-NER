"""DuckDB persistence + SQL query layer for the safety knowledge graph."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .build import Signal


def signals_to_df(signals: list[Signal]) -> pd.DataFrame:
    return pd.DataFrame([{
        "chemical_mesh": s.chemical_mesh,
        "chemical_name": s.chemical_name,
        "disease_mesh": s.disease_mesh,
        "disease_name": s.disease_name,
        "n_pmids": len(s.pmids),
        "pmids": ";".join(s.pmids),
    } for s in signals])


def write_duckdb(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = duckdb.connect(str(path))
    con.register("df", df)
    con.execute("CREATE TABLE signals AS SELECT * FROM df")
    # concept dimension tables for the dashboard / joins
    con.execute("""
        CREATE TABLE chemicals AS
        SELECT chemical_mesh AS mesh, any_value(chemical_name) AS name,
               COUNT(*) AS n_diseases, SUM(n_pmids) AS n_evidence
        FROM signals GROUP BY chemical_mesh
    """)
    con.execute("""
        CREATE TABLE diseases AS
        SELECT disease_mesh AS mesh, any_value(disease_name) AS name,
               COUNT(*) AS n_chemicals, SUM(n_pmids) AS n_evidence
        FROM signals GROUP BY disease_mesh
    """)
    con.close()


def example_queries(path: str | Path) -> dict:
    con = duckdb.connect(str(path), read_only=True)
    q = {}
    q["top_inducer_chemicals"] = con.execute("""
        SELECT name, n_diseases, n_evidence
        FROM chemicals ORDER BY n_diseases DESC, n_evidence DESC LIMIT 10
    """).df().to_dict("records")
    q["most_implicated_diseases"] = con.execute("""
        SELECT name, n_chemicals, n_evidence
        FROM diseases ORDER BY n_chemicals DESC, n_evidence DESC LIMIT 10
    """).df().to_dict("records")
    q["best_evidenced_signals"] = con.execute("""
        SELECT chemical_name, disease_name, n_pmids
        FROM signals ORDER BY n_pmids DESC LIMIT 10
    """).df().to_dict("records")
    con.close()
    return q
