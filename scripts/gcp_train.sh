#!/usr/bin/env bash
# Orchestrate T4 training on GCP Compute Engine, staging files through a GCS
# bucket (avoids Windows pscp quirks entirely):
#   bucket upload -> create VM -> VM pulls+trains+pushes -> bucket download -> delete VM
#
# Usage: bash scripts/gcp_train.sh
set -euo pipefail

PROJECT="${PROJECT:-final-project-478101}"
VM="${VM:-pcs-t4-trainer}"
MACHINE="${MACHINE:-n1-standard-4}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUCKET="gs://${PROJECT}-pcs-staging"

ZONES="${ZONES:-us-central1-a us-central1-b us-central1-c us-central1-f us-east1-c us-east1-d us-west1-a us-west1-b us-east4-a us-east4-b us-west4-a us-west4-b}"
ZONE=""

echo ">>> Project=$PROJECT VM=$VM Machine=$MACHINE + 1x T4 | staging=$BUCKET"

cleanup() {
  if [ -n "$ZONE" ]; then
    echo ">>> Deleting VM $VM in $ZONE ..."
    gcloud compute instances delete "$VM" --project "$PROJECT" --zone "$ZONE" -q || true
  fi
}
trap cleanup EXIT

# ----- 0. staging bucket + payload upload (no pscp) -----
gcloud storage buckets create "$BUCKET" --project "$PROJECT" --location us 2>/dev/null || true

echo ">>> Packaging + uploading payload to $BUCKET ..."
TARBALL="/tmp/pcs_payload.tar.gz"   # no-space temp path
tar -czf "$TARBALL" -C "$ROOT" \
  src scripts configs requirements-gpu.txt pyproject.toml \
  data/processed/bc5cdr_train.jsonl data/processed/bc5cdr_dev.jsonl data/processed/bc5cdr_test.jsonl
gcloud storage cp "$(cygpath -w "$TARBALL")" "$BUCKET/pcs.tar.gz" -q
gcloud storage cp "$(cygpath -w "$ROOT/scripts/vm_run.sh")" "$BUCKET/vm_run.sh" -q

# ----- 1. create VM with Deep Learning image + T4, trying zones for capacity -----
for z in $ZONES; do
  echo ">>> Trying zone $z ..."
  if gcloud compute instances create "$VM" \
      --project "$PROJECT" --zone "$z" \
      --machine-type "$MACHINE" \
      --accelerator "type=nvidia-tesla-t4,count=1" \
      --image-family "pytorch-2-9-cu129-ubuntu-2204-nvidia-580" \
      --image-project "deeplearning-platform-release" \
      --maintenance-policy TERMINATE --restart-on-failure \
      --boot-disk-size 100GB --boot-disk-type pd-balanced \
      --scopes cloud-platform \
      --metadata "install-nvidia-driver=True" 2>&1 | tee /tmp/vm_create.log; then
    ZONE="$z"; echo ">>> VM created in $ZONE."; break
  fi
  if grep -q "ZONE_RESOURCE_POOL_EXHAUSTED" /tmp/vm_create.log; then
    echo ">>> $z exhausted, trying next zone..."; continue
  fi
  echo ">>> Non-capacity error creating VM; aborting."; exit 1
done
[ -z "$ZONE" ] && { echo ">>> No zone had T4 capacity. Try later."; exit 1; }

# ----- 2. wait for SSH -----
echo ">>> Waiting for SSH..."
for i in $(seq 1 30); do
  if gcloud compute ssh "$VM" --project "$PROJECT" --zone "$ZONE" --command "echo ready" -q >/dev/null 2>&1; then
    echo ">>> SSH ready."; break
  fi
  echo "   ...retry $i"; sleep 15
done

# ----- 3. VM: pull payload from bucket, train, push results back to bucket -----
echo ">>> Training on VM (pull -> train -> push). ~20 min..."
gcloud compute ssh "$VM" --project "$PROJECT" --zone "$ZONE" --command "
  set -e
  mkdir -p ~/pcs && cd ~/pcs
  gcloud storage cp $BUCKET/pcs.tar.gz . && tar -xzf pcs.tar.gz
  gcloud storage cp $BUCKET/vm_run.sh .
  BUCKET=$BUCKET bash vm_run.sh
"

# ----- 4. download trained models + metrics from bucket -----
echo ">>> Downloading models + metrics from $BUCKET ..."
mkdir -p "$ROOT/models" "$ROOT/reports"
gcloud storage cp -r "$BUCKET/models/*" "$(cygpath -w "$ROOT/models")" -q || true
gcloud storage cp "$BUCKET/reports/*" "$(cygpath -w "$ROOT/reports")" -q || true

echo ">>> Done. Models in models/ ; metrics in reports/."
