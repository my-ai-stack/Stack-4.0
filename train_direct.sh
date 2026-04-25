#!/bin/bash
# Stack 4.0 Direct Training Script — No Axolotl needed
# Uses transformers + peft directly on V100 16GB
#
# Usage: HF_TOKEN=xxx bash train_direct.sh

set -e

echo "===== Stack 4.0 Direct Training ====="
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

export HF_TOKEN="${HF_TOKEN:-}"
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: Set HF_TOKEN env var first"
    exit 1
fi

# Login to HF
echo "$HF_TOKEN" | tr -d '\n' | huggingface-cli login --token

# Working directory on VM
WORKDIR="/home/walidsobhi/stack-4.0"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Clone repo if needed (only the small files, not LFS)
if [ ! -d ".git" ]; then
    GIT_LFS_SKIP_DOWNLOAD=1 git clone https://github.com/my-ai-stack/Stack-4.0.git .
fi

echo "===== Running Training ====="
python3 train_direct.py 2>&1

echo "===== Training Complete: $(date) ====="

# If training succeeded, upload adapter to HF
if [ -d "adapter" ]; then
    echo "===== Uploading adapter to HF ====="
    python3 << PYEOF
from huggingface_hub import HfApi, create_repo, upload_folder
from pathlib import Path

api = HfApi(token="$HF_TOKEN")
try:
    create_repo("my-ai-stack/Stack-4.0-Omni-Nexus-Agentic", exist_ok=True)
except:
    pass

api.upload_folder(
    folder_path="adapter",
    repo_id="my-ai-stack/Stack-4.0-Omni-Nexus-Agentic",
    repo_type="model",
)
print("✅ LoRA adapter uploaded to HuggingFace!")
PYEOF
else
    echo "No adapter found — training may have failed"
fi
