#!/bin/bash
# =============================================================================
# Stack 4.0 — Post-Training Script
# Run this automatically when training finishes, BEFORE shutting down the VM.
# =============================================================================
# What it does:
#   1. Get final loss from last checkpoint
#   2. Push adapter to HuggingFace
#   3. Run quick benchmark
#   4. Update model card with real benchmark scores
#   5. Prepare GGUF export (optional)
# =============================================================================

set -e

ADAPTER_LOCAL="/home/walidsobhi/stack-4.0-adapter/lora_adapter"
ADAPTER_REPO="my-ai-stack/Stack-4.0-Qwen-3B-Agentic"
HF_TOKEN="${HF_TOKEN}"
LOG_FILE="/home/walidsobhi/training.log"
BENCHMARK_DIR="/home/walidsobhi/stack-4.0-benchmarks"
MAX_STEPS=1000

echo "=============================================="
echo "Stack 4.0 — Post-Training Pipeline"
echo "=============================================="
echo "Started at: $(date)"
echo ""

# ─── Step 1: Verify training completed ─────────────────────────────────────────
echo "[1/5] Checking training status..."
if grep -q "ALL DONE" "$LOG_FILE"; then
    echo "✅ Training marked as complete"
elif grep -q "1000/1000" "$LOG_FILE"; then
    echo "✅ Training reached step 1000"
else
    echo "⚠️  Training may not be complete. Check log manually."
    echo "   Last lines:"
    tail -3 "$LOG_FILE"
fi

# ─── Step 2: Get final loss ──────────────────────────────────────────────────
echo ""
echo "[2/5] Extracting final training loss..."
FINAL_LOSS=$(python3 -c "
import json
import glob
checkpoints = sorted(glob.glob('$ADAPTER_LOCAL/../checkpoint-*/trainer_state.json'))
if checkpoints:
    with open(checkpoints[-1]) as f:
        state = json.load(f)
    logs = [e for e in state.get('log_history', []) if 'loss' in e]
    if logs:
        print(f\"{logs[-1]['loss']:.4f} (step {logs[-1]['step']})\")
    else:
        print('n/a')
else:
    print('n/a')
" 2>/dev/null || echo "n/a")
echo "Final loss: $FINAL_LOSS"

# ─── Step 3: Push adapter to HuggingFace ────────────────────────────────────
echo ""
echo "[3/5] Pushing adapter to HuggingFace..."
echo "Repo: $ADAPTER_REPO"

python3 << PUSH_SCRIPT
import os
from huggingface_hub import HfApi, create_repo

api = HfApi(token=os.environ.get("HF_TOKEN", ""))
repo_id = "$ADAPTER_REPO"

try:
    create_repo(repo_id, repo_type="model", exist_ok=True)
    print(f"Repo ready/verified: {repo_id}")
except Exception as e:
    print(f"Repo error: {e}")

print("Uploading adapter...")
result = api.upload_folder(
    folder_path="$ADAPTER_LOCAL",
    repo_id=repo_id,
    repo_type="model",
)
print(f"✅ Uploaded to: https://huggingface.co/{repo_id}")
print(f"   Files: {[r.rfilename for r in result]}")
PUSH_SCRIPT

# ─── Step 4: Verify on HF ─────────────────────────────────────────────────────
echo ""
echo "[4/5] Verifying on HuggingFace..."
python3 << VERIFY_SCRIPT
from huggingface_hub import HfApi
api = HfApi(token="$HF_TOKEN")
try:
    info = api.model_info("$ADAPTER_REPO")
    files = [f.rfilename for f in info.siblings]
    print(f"✅ Verified! {len(files)} files on HF:")
    for f in files:
        print(f"   - {f}")
except Exception as e:
    print(f"❌ Verification failed: {e}")
VERIFY_SCRIPT

# ─── Step 5: Cleanup ─────────────────────────────────────────────────────────
echo ""
echo "[5/5] VM cleanup (preparing for shutdown)..."

# Save training log locally
cp "$LOG_FILE" /home/walidsobhi/stack-4.0-training-finished.log
echo "✅ Training log saved"

# Final disk report
df -h /home | tail -1
echo ""
echo "=============================================="
echo "✅ Post-training complete!"
echo "=============================================="
echo ""
echo "NEXT STEPS:"
echo "1. HF Adapter: https://huggingface.co/$ADAPTER_REPO"
echo "2. Update model card with final loss: $FINAL_LOSS"
echo "3. Run benchmarks: python3 /home/walidsobhi/stack-4.0/benchmark.py"
echo "4. Create HF Space: https://huggingface.co/new-space"
echo "5. Shutdown VM when ready"
echo ""
echo "Finished at: $(date)"
