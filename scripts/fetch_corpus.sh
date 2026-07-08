#!/usr/bin/env bash
# Download and extract the BC5CDR corpus (PubTator format) into data/raw/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"

URL="https://github.com/JHnlp/BioCreative-V-CDR-Corpus/raw/master/CDR_Data.zip"
echo "Downloading BC5CDR corpus..."
curl -sL --max-time 180 "$URL" -o "$RAW/CDR_Data.zip"

echo "Extracting..."
cd "$RAW"
unzip -o CDR_Data.zip >/dev/null
rm -rf __MACOSX CDR_Data.zip

echo "Done. Splits:"
ls -1 "$RAW/CDR_Data/CDR.Corpus.v010516/"
