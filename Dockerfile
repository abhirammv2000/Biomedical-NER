# GPU-ready image for PCS-SignalMiner training.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /workspace

# system deps for data fetch
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl unzip git && rm -rf /var/lib/apt/lists/*

# python deps (torch already in base image)
COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

COPY . .
RUN pip install --no-cache-dir -e .

CMD ["python", "scripts/run_train_ner.py"]
