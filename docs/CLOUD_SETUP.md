# Cloud GPU Setup

The training scripts auto-detect CUDA (`fp16` and device placement switch on
when `torch.cuda.is_available()`), so the same commands run on CPU or GPU.

## Option A: bare GPU VM (any provider: AWS/GCP/Azure/Lambda/RunPod)

```bash
# 1. clone + enter
git clone <your-repo-url> && cd "Biomedical NER"

# 2. python env (conda or venv)
python -m venv .venv && source .venv/bin/activate      # or conda create -n biomed-ner python=3.11

# 3. CUDA torch (match the box's CUDA; cu121 shown)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt

# 4. data (one-time)
bash scripts/fetch_corpus.sh          # downloads + extracts BC5CDR
python scripts/run_download_data.py   # parse + QC -> data/processed
python scripts/run_build_re_dataset.py

# 5. train (GPU auto-used)
python scripts/run_train_ner.py
python scripts/run_train_relation.py

# 6. pull trained models back
#   models/ner/ and models/relation/ -> copy to local for the dashboard/KG
```

Recommended instance: a single 16-24 GB GPU (T4/L4/A10) is plenty for
PubMedBERT-base on BC5CDR. Full NER + relation training is well under an hour.

## Option B: Docker

```bash
docker build -t pcs-signalminer -f Dockerfile .
docker run --gpus all -v $PWD:/workspace pcs-signalminer \
    python scripts/run_train_ner.py
```

## Option C: Colab / Kaggle
Upload the repo, `pip install -r requirements-gpu.txt` (torch already present on
those runtimes), run the same `scripts/run_*.py`. Download `models/` at the end.

## Notes
- Increase `ner.batch_size` / `relation.batch_size` in `configs/config.yaml` on
  larger GPUs (e.g. 32-64) for speed.
- Set `epochs` back to 4-5 for NER on GPU (CPU default was lowered to 3).
- All runs are seeded (`seed: 42`) for reproducibility.
