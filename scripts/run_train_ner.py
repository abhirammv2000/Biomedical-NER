"""Stage: fine-tune PubMedBERT for chemical/disease NER on BC5CDR.

Document-level token classification with offset-based BIO alignment, evaluated
with seqeval entity-level F1 (per type + micro).

Run:
    python scripts/run_train_ner.py            # full training
    python scripts/run_train_ner.py --smoke    # 1-epoch tiny run to verify wiring
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

from pcs_signalminer.data.pubtator import load_jsonl  # noqa: E402
from pcs_signalminer.ner.dataset import LABELS, LABEL2ID, ID2LABEL, align_labels  # noqa: E402

PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models" / "ner"
REPORT_DIR = ROOT / "reports"


def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())["ner"]


def build_hf_dataset(docs, tokenizer, max_length):
    from datasets import Dataset

    texts = [d.text for d in docs]
    enc = tokenizer(texts, truncation=True, max_length=max_length,
                    return_offsets_mapping=True)
    all_labels = []
    for i, d in enumerate(docs):
        all_labels.append(align_labels(d, enc["offset_mapping"][i]))
    enc.pop("offset_mapping")
    enc["labels"] = all_labels
    return Dataset.from_dict(enc)


def make_compute_metrics():
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=2)
        labels = p.label_ids
        true_pred, true_lab = [], []
        for pred, lab in zip(preds, labels):
            tp, tl = [], []
            for pi, li in zip(pred, lab):
                if li != -100:
                    tp.append(ID2LABEL[pi])
                    tl.append(ID2LABEL[li])
            true_pred.append(tp)
            true_lab.append(tl)
        return {
            "precision": precision_score(true_lab, true_pred),
            "recall": recall_score(true_lab, true_pred),
            "f1": f1_score(true_lab, true_pred),
            "report": classification_report(true_lab, true_pred, digits=4),
        }

    return compute_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                              DataCollatorForTokenClassification, Trainer,
                              TrainingArguments)

    cfg = load_cfg()
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForTokenClassification.from_pretrained(
        cfg["base_model"], num_labels=len(LABELS),
        id2label=ID2LABEL, label2id=LABEL2ID,
    )

    train_docs = load_jsonl(PROC_DIR / "bc5cdr_train.jsonl")
    dev_docs = load_jsonl(PROC_DIR / "bc5cdr_dev.jsonl")
    test_docs = load_jsonl(PROC_DIR / "bc5cdr_test.jsonl")
    if args.smoke:
        train_docs, dev_docs, test_docs = train_docs[:40], dev_docs[:20], test_docs[:20]

    ds_train = build_hf_dataset(train_docs, tokenizer, cfg["max_length"])
    ds_dev = build_hf_dataset(dev_docs, tokenizer, cfg["max_length"])
    ds_test = build_hf_dataset(test_docs, tokenizer, cfg["max_length"])

    targs = TrainingArguments(
        output_dir=str(MODELS_DIR),
        learning_rate=float(cfg["lr"]),
        num_train_epochs=1 if args.smoke else cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        seed=42,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    cm = make_compute_metrics()
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=ds_train, eval_dataset=ds_dev,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=lambda p: {k: v for k, v in cm(p).items() if k != "report"},
    )
    trainer.train()

    test_metrics = cm(trainer.predict(ds_test))
    print("\n=== TEST (entity-level) ===")
    print(test_metrics["report"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODELS_DIR))
    tokenizer.save_pretrained(str(MODELS_DIR))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "ner_test_metrics.json").write_text(
        json.dumps({k: v for k, v in test_metrics.items()}, indent=2), encoding="utf-8")
    print(f"\nSaved model -> {MODELS_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
