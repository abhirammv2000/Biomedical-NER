"""Stage: train a document-level chemical-induced-disease (CID) relation
classifier on BC5CDR candidate pairs.

Entity-marker featurization + PubMedBERT sequence classification. Evaluated with
precision/recall/F1 on the positive (CID) class, the metric that matters for a
safety-signal pipeline.

Run:
    python scripts/run_train_relation.py
    python scripts/run_train_relation.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcs_signalminer.relation.featurize import mark_text, MARKER_TOKENS  # noqa: E402

PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models" / "relation"
REPORT = ROOT / "reports"


def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())["relation"]


def read_candidates(name: str) -> list[dict]:
    rows = []
    with (PROC / f"re_candidates_{name}.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def build_ds(rows, tokenizer, max_length):
    from datasets import Dataset

    texts = [mark_text(r["text"],
                       [tuple(o) for o in r["chemical_offsets"]],
                       [tuple(o) for o in r["disease_offsets"]]) for r in rows]
    enc = tokenizer(texts, truncation=True, max_length=max_length)
    enc["labels"] = [r["label"] for r in rows]
    return Dataset.from_dict(enc)


def make_metrics():
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    def compute(p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        pr, rc, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", pos_label=1, zero_division=0)
        return {"precision": pr, "recall": rc, "f1": f1,
                "accuracy": accuracy_score(labels, preds)}

    return compute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              DataCollatorWithPadding, Trainer, TrainingArguments)

    cfg = load_cfg()
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    tokenizer.add_special_tokens({"additional_special_tokens": MARKER_TOKENS})
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["base_model"], num_labels=2,
        id2label={0: "no_relation", 1: "CID"}, label2id={"no_relation": 0, "CID": 1})
    model.resize_token_embeddings(len(tokenizer))

    tr, dv, te = read_candidates("train"), read_candidates("dev"), read_candidates("test")
    if args.smoke:
        tr, dv, te = tr[:80], dv[:40], te[:40]

    ds_tr = build_ds(tr, tokenizer, cfg["max_length"])
    ds_dv = build_ds(dv, tokenizer, cfg["max_length"])
    ds_te = build_ds(te, tokenizer, cfg["max_length"])

    # class weighting to handle ~19% positive rate
    import collections
    counts = collections.Counter(r["label"] for r in tr)
    w = torch.tensor([1.0, counts[0] / max(counts[1], 1)], dtype=torch.float)

    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                out.logits, labels, weight=w.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    targs = TrainingArguments(
        output_dir=str(MODELS_DIR),
        learning_rate=float(cfg["lr"]),
        num_train_epochs=1 if args.smoke else cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        logging_steps=50, seed=42, fp16=torch.cuda.is_available(), report_to="none",
    )

    metrics = make_metrics()
    trainer = WTrainer(
        model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_dv,
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics)
    trainer.train()

    test_out = trainer.predict(ds_te)
    test_metrics = metrics(test_out)
    print("\n=== TEST (CID positive class) ===")
    print({k: round(float(v), 4) for k, v in test_metrics.items()})

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODELS_DIR)); tokenizer.save_pretrained(str(MODELS_DIR))
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "relation_test_metrics.json").write_text(
        json.dumps({k: float(v) for k, v in test_metrics.items()}, indent=2), encoding="utf-8")
    print(f"Saved model -> {MODELS_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
