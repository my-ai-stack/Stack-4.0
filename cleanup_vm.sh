#!/bin/bash
# Cleanup script — Run on GCP VM to free up space before training
# SAFE: keeps merged model (for upload), training scripts, training data (on HF)

set -e
echo "===== VM Cleanup ====="

WORKDIR="/home/walidsobhi"

echo "--- Before ---"
df -h / | tail -1

# 1. Delete duplicate base_model in stack-space (15GB)
echo ""
echo "Removing duplicate base_model (stack-space/)..."
sudo rm -rf /home/walidsobhi/stack-space/base_model
echo "✅ Deleted stack-space/base_model (15GB)"

# 2. Delete training data (already on HF — 800MB)
echo ""
echo "Removing training data (already backed up to HF)..."
sudo rm -rf /home/walidsobhi/stack-3.0/training/data/
sudo rm -rf /home/walidsobhi/stack-3.0/training/data-expanded/
echo "✅ Deleted training/data and training/data-expanded (~800MB)"

# 3. Delete build artifacts from llama.cpp (987MB)
# Actually keep llama.cpp — it's useful for inference
# But delete its build/cache to save space
echo ""
echo "Cleaning llama.cpp build cache..."
sudo rm -rf /home/walidsobhi/llama.cpp/build/ 2>/dev/null || true
sudo rm -rf /home/walidsobhi/llama.cpp/examples/ 2>/dev/null || true
echo "✅ Cleaned llama.cpp build cache"

# 4. Delete eval logs and temp files
echo ""
echo "Cleaning temp files..."
sudo find /home/walidsobhi/stack-3.0 -name "*.log" -delete 2>/dev/null || true
sudo find /home/walidsobhi -name "*.pyc" -delete 2>/dev/null || true
echo "✅ Cleaned logs and pycache"

# 5. Show result
echo ""
echo "--- After ---"
df -h / | tail -1
echo ""
echo "Remaining large dirs:"
sudo du -sh /home/walidsobhi/* 2>/dev/null | sort -hr | head -10
