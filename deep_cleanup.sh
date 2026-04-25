#!/bin/bash
# Deep cleanup — removes GGUF duplicates, old training outputs
# SAFE: keeps merged-model (15GB for upload), Ollama GGUF (currently serving)

set -e
echo "===== VM Deep Cleanup ====="

WORKDIR="/home/walidsobhi"

echo "--- Before ---"
df -h / | tail -1

# 1. Delete base model GGUF (15GB) — already backed up on HF as safetensors
echo ""
echo "Removing base Qwen GGUF (15GB)..."
sudo rm -f /home/walidsobhi/stack-3.0/gguf/qwen25-coder-7b-f16.gguf
echo "✅ Deleted qwen25-coder-7b-f16.gguf"

# 2. Delete old training iteration outputs (~2GB)
echo ""
echo "Removing old training outputs..."
sudo rm -rf /home/walidsobhi/stack-3.0/output/stack-3.0-7b-iter2
sudo rm -rf /home/walidsobhi/stack-3.0/output/stack-3.0-hf-50k
echo "✅ Deleted old training outputs (~2GB)"

# 3. Delete stack-3.0-qwen local git objects on VM (not needed, we have GitHub)
sudo rm -rf /home/walidsobhi/stack-3.0/.git 2>/dev/null || true

echo ""
echo "--- After ---"
df -h / | tail -1
echo ""
echo "Remaining large dirs:"
sudo du -sh /home/walidsobhi/* 2>/dev/null | sort -hr | head -10
