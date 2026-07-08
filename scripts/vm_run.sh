#!/usr/bin/env bash
# Runs ON the GPU VM. Installs deps on top of the Deep Learning image's CUDA
# torch, then trains the NER and relation models.
set -euo pipefail

cd ~/pcs

# Non-interactive SSH doesn't activate conda; resolve the image's python explicitly.
if [ -x /opt/conda/bin/python ]; then PY=/opt/conda/bin/python
elif command -v python3 >/dev/null 2>&1; then PY=$(command -v python3)
else PY=$(command -v python); fi
echo "Using interpreter: $PY"

echo "=== GPU check ==="
nvidia-smi || { echo "No GPU visible!"; exit 1; }
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== Installing python deps (keeping image's CUDA torch) ==="
# Pin transformers<5: 5.x eagerly imports torchaudio (Parakeet RNNT loss), which
# is ABI-incompatible with the image's torch and crashes at import. 4.46+ has the
# Trainer API we use (processing_class, eval_strategy) and never imports torchaudio.
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q "transformers>=4.46,<5" datasets accelerate seqeval evaluate scikit-learn pyyaml tqdm

mkdir -p reports models   # scripts write here; dirs aren't in the payload

echo "=== Sanity: data present ==="
ls -1 data/processed/ | head

echo "=== Rebuild RE candidate dataset from base JSONL ==="
"$PY" scripts/run_build_re_dataset.py

echo "=== Train NER ==="
"$PY" scripts/run_train_ner.py

echo "=== Train Relation ==="
"$PY" scripts/run_train_relation.py

echo "=== DONE. Artifacts: ==="
ls -R models/ | head -40
cat reports/ner_test_metrics.json 2>/dev/null | head -30 || true
cat reports/relation_test_metrics.json 2>/dev/null || true

if [ -n "${BUCKET:-}" ]; then
  echo "=== Pushing models + metrics to $BUCKET ==="
  gcloud storage cp -r models/ner "$BUCKET/models/ner" -q || true
  gcloud storage cp -r models/relation "$BUCKET/models/relation" -q || true
  gcloud storage cp reports/ner_test_metrics.json "$BUCKET/reports/" -q || true
  gcloud storage cp reports/relation_test_metrics.json "$BUCKET/reports/" -q || true
fi
