#!/usr/bin/env bash
set -euo pipefail

BIRD_DATA_URL="https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
DATA_DIR="data/bird"

echo "=== BIRD Mini-Dev Download Script ==="
echo ""

if [ -d "$DATA_DIR/dev_databases" ]; then
    echo "BIRD data already exists at $DATA_DIR"
    echo "To re-download, remove the directory first: rm -rf $DATA_DIR"
    exit 0
fi

echo "Downloading BIRD Mini-Dev dataset..."
echo "Source: $BIRD_DATA_URL"
echo "Target: $DATA_DIR"
echo ""

mkdir -p "$DATA_DIR"
TMP_ZIP=$(mktemp /tmp/bird_minidev.XXXXXX.zip)

echo "[1/3] Downloading (this may take a few minutes)..."
curl -L --progress-bar -o "$TMP_ZIP" "$BIRD_DATA_URL"

echo "[2/3] Extracting..."
unzip -q -o "$TMP_ZIP" -d "$DATA_DIR"

# BIRD Mini-Dev zip may extract into a subdirectory; flatten if needed
if [ -d "$DATA_DIR/mini_dev_data" ]; then
    echo "[3/3] Reorganizing files..."
    cp -r "$DATA_DIR/mini_dev_data/"* "$DATA_DIR/" 2>/dev/null || true
fi

rm -f "$TMP_ZIP"

echo ""
echo "Done! BIRD Mini-Dev data is ready at $DATA_DIR"
echo ""
echo "Directory structure:"
ls -la "$DATA_DIR/" | head -20
echo ""
echo "Next steps:"
echo "  1. cp .env.example .env"
echo "  2. Add your GROQ_API_KEY to .env"
echo "  3. uv run python -m benchmarks.run --agent react --samples 5"
