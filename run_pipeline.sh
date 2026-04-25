#!/bin/bash
# Stack 4.0 Full Pipeline — Run on GCP VM
# Usage:
#   1. Upload model first:  HF_TOKEN=xxx bash run_pipeline.sh --upload-model
#   2. Train:               HF_TOKEN=xxx bash run_pipeline.sh --train
#   3. Both:                HF_TOKEN=xxx bash run_pipeline.sh --all

set -e
HF_TOKEN="${HF_TOKEN:-}"

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: Set HF_TOKEN env var first:  HF_TOKEN=xxx bash run_pipeline.sh --train"
    exit 1
fi

WORKDIR="/home/walidsobhi/stack-4.0"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Login to HF
echo "$HF_TOKEN" | tr -d '\n' | huggingface-cli login --token

# Clone Stack-4.0 repo if needed (skip LFS files — we only need the scripts)
if [ ! -d ".git" ]; then
    GIT_LFS_SKIP_DOWNLOAD=1 git clone https://github.com/my-ai-stack/Stack-4.0.git "$WORKDIR"
fi

echo "===== Stack 4.0 Pipeline ====="
echo "HF_TOKEN: ${HF_TOKEN:0:8}..."

# ── Step 1: Upload merged model to HF ────────────────────────────────────
upload_model() {
    echo ""
    echo "===== Step 1: Upload Merged Qwen Model to HuggingFace ====="
    python3 upload_model_to_hf.py
}

# ── Step 2: Train Llama 3.1 LoRA ───────────────────────────────────────────
train() {
    echo ""
    echo "===== Step 2: Training Llama 3.1 8B QLoRA on V100 16GB ====="
    echo "This will take ~2-4 hours. Checkpoints saved every 100 steps."
    echo ""
    python3 train_direct.py
}

# ── Parse args ─────────────────────────────────────────────────────────────
case "${1:-}" in
    --upload-model)
        upload_model
        ;;
    --train)
        train
        ;;
    --all)
        upload_model
        train
        ;;
    *)
        echo "Usage: HF_TOKEN=xxx bash run_pipeline.sh [--upload-model|--train|--all]"
        echo "  --upload-model  Upload merged Qwen model to HF (once)"
        echo "  --train         Start training (Llama 3.1 LoRA)"
        echo "  --all           Do both in sequence"
        ;;
esac
