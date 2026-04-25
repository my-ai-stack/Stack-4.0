#!/bin/bash
# Stack 4.0 Training Script — Run on GCP VM
# Trains Llama 3.1 8B LoRA on agentic tool-use data
#
# Usage: HF_TOKEN=xxx bash train.sh
# (Set HF_TOKEN environment variable — never hardcode tokens)

set -e

echo "===== Stack 4.0 Training ====="
echo "Start time: $(date)"

# Check GPU
nvidia-smi

# Install axolotl if needed
pip install axolotl transformers datasets peft bitsandbytes scipy accelerate

# Login to HF using token from env var
echo "Logging into HuggingFace..."
echo $HF_TOKEN | tr -d '\n' | huggingface-cli login

# Run training
echo "Starting LoRA training..."
cd /home/walidsobhi/stack-4.0

accelerate launch \
    --mixed_precision=bf16 \
    axolotl/train.py \
    llama3_1_lora_config.yaml

echo "Training complete: $(date)"

# Upload adapter to HuggingFace
echo "Uploading LoRA adapter to HuggingFace..."
python3 << PYEOF
from huggingface_hub import HfApi, create_repo
from pathlib import Path

api = HfApi(token="$HF_TOKEN")

# Create model repo if needed
try:
    create_repo("my-ai-stack/Stack-4.0-Omni-Nexus-Agentic", exist_ok=True)
except:
    pass

# Upload all files from output dir
output = Path("/tmp/stack-4.0-output")
for f in output.rglob("*"):
    if f.is_file() and not f.name.startswith("."):
        rel = f.relative_to(output)
        print(f"Uploading {rel}...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=str(rel),
            repo_id="my-ai-stack/Stack-4.0-Omni-Nexus-Agentic",
        )
print("Done uploading!")
PYEOF
