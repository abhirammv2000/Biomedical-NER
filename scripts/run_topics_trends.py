"""Stage: topic modeling / clustering of abstracts + time-series trends of
safety signals.

1. Fetch PubMed publication years (cached) to add a temporal axis.
2. Time-series: chemical-induced-disease signals per year.
3. Topic modeling: BERTopic over abstracts -> safety-relevant topics.

Run:
    python scripts/run_topics_trends.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcs_signalminer.data.pubtator import load_jsonl  # noqa: E402
from pcs_signalminer.data.pubmed import fetch_years  # noqa: E402

PROC = ROOT / "data" / "processed"
EXT = ROOT / "data" / "external"
FIG = ROOT / "reports" / "figures"
REPORT = ROOT / "reports"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})


def load_all():
    docs = []
    for n in ("train", "dev", "test"):
        docs.extend(load_jsonl(PROC / f"bc5cdr_{n}.jsonl"))
    return docs


def time_series(docs, years):
    rows = []
    for d in docs:
        y = years.get(d.pmid)
        if y:
            rows.append({"year": y, "n_signals": len(d.relations)})
    df = pd.DataFrame(rows)
    per_year = df.groupby("year")["n_signals"].sum().sort_index()

    plt.figure(figsize=(8, 4))
    per_year.plot(marker="o", color="#c0392b")
    plt.ylabel("chemical-induced-disease signals")
    plt.xlabel("publication year")
    plt.title("Drug-safety signals reported per year (BC5CDR)")
    plt.grid(alpha=0.3)
    plt.savefig(FIG / "signals_per_year.png"); plt.close()

    cum = per_year.cumsum()
    plt.figure(figsize=(8, 4))
    cum.plot(color="#2c3e50")
    plt.ylabel("cumulative signals"); plt.xlabel("publication year")
    plt.title("Cumulative drug-safety signals over time")
    plt.grid(alpha=0.3)
    plt.savefig(FIG / "signals_cumulative.png"); plt.close()

    return {"years_covered": [int(per_year.index.min()), int(per_year.index.max())],
            "total_with_year": int(df.n_signals.sum()),
            "busiest_years": per_year.sort_values(ascending=False).head(5).astype(int).to_dict()}


def topic_model(docs):
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())["topics"]
    texts = [d.text for d in docs]
    # strip stopwords + add bigrams so topic terms are clinically meaningful
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3)
    model = BERTopic(embedding_model=cfg["embedding_model"],
                     vectorizer_model=vectorizer,
                     min_topic_size=cfg["min_topic_size"], verbose=False)
    topics, _ = model.fit_transform(texts)
    info = model.get_topic_info()
    info.to_csv(REPORT / "topics.csv", index=False)

    # bar chart of top topics (excluding outlier topic -1)
    top = info[info.Topic != -1].head(10)
    labels = [f"{r.Topic}: " + ", ".join([w for w, _ in model.get_topic(r.Topic)[:3]])
              for r in top.itertuples()]
    plt.figure(figsize=(8, 5))
    plt.barh(labels[::-1], top.Count.tolist()[::-1], color="#16a085")
    plt.xlabel("documents"); plt.title("Top abstract topics (BERTopic)")
    plt.savefig(FIG / "topics_top.png"); plt.close()

    n_topics = int((info.Topic != -1).sum())
    outliers = int(info.loc[info.Topic == -1, "Count"].sum()) if (info.Topic == -1).any() else 0
    return {"n_topics": n_topics, "n_outliers": outliers,
            "top_topics": [{"topic": int(r.Topic), "count": int(r.Count),
                            "terms": [w for w, _ in model.get_topic(r.Topic)[:6]]}
                           for r in top.itertuples()]}


def main():
    docs = load_all()
    pmids = [d.pmid for d in docs]
    print(f"Fetching publication years for {len(pmids)} PMIDs (cached)...")
    years = fetch_years(pmids, EXT / "pmid_years.json")
    print(f"  resolved {len(years)}/{len(pmids)} years")

    ts = time_series(docs, years)
    print(f"Time series: {ts['years_covered']}, busiest years: {ts['busiest_years']}")

    print("Fitting BERTopic (CPU, ~1-3 min)...")
    tm = topic_model(docs)
    print(f"Topics: {tm['n_topics']} ({tm['n_outliers']} outlier docs)")
    for t in tm["top_topics"][:5]:
        print(f"  topic {t['topic']:>2} (n={t['count']}): {', '.join(t['terms'])}")

    (REPORT / "topics_trends_report.json").write_text(
        json.dumps({"time_series": ts, "topics": tm}, indent=2), encoding="utf-8")
    print(f"\nReport -> reports/topics_trends_report.json")


if __name__ == "__main__":
    main()
