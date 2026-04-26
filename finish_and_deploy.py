#!/usr/bin/env python3
"""
Stack 4.0 — Finish & Deploy
Run from your LOCAL machine when training finishes.

Usage:
    export HF_TOKEN=your_hf_token
    python3 finish_and_deploy.py
"""
import os, time, subprocess

ADAPTER_REPO = "my-ai-stack/Stack-4.0-Qwen-3B-Agentic"
ADAPTER_LOCAL = "/home/walidsobhi/stack-4.0-adapter/lora_adapter"
VM_HOST = "35.223.229.30"
VM_USER = "walidsobhi"
SSH_KEY = "~/.ssh/google_compute_engine"

def ssh_vm(cmd):
    r = subprocess.run(f"ssh -i {SSH_KEY} {VM_USER}@{VM_HOST} '{cmd}'",
                      shell=True, capture_output=True, text=True)
    return r.stdout + r.stderr

def check_done():
    out = ssh_vm("tail -3 /home/walidsobhi/training.log")
    print(out)
    return "ALL DONE" in out or ("1000/1000" in out)

def get_final_loss():
    """Extract final training loss from last checkpoint."""
    script = """
import json, glob
checkpoints = sorted(glob.glob("/home/walidsobhi/stack-4.0-adapter/checkpoint-*/trainer_state.json"))
if checkpoints:
    with open(checkpoints[-1]) as f:
        state = json.load(f)
    logs = [e for e in state.get('log_history', []) if 'loss' in e]
    if logs:
        print(f"FINAL_LOSS={logs[-1]['loss']:.4f}")
        print(f"FINAL_STEP={logs[-1]['step']}")
"""
    out = ssh_vm(f"python3 -c \"{script}\"")
    loss, step = None, None
    for line in out.split('\n'):
        if 'FINAL_LOSS=' in line:
            loss = line.split('=')[1].strip()
        if 'FINAL_STEP=' in line:
            step = line.split('=')[1].strip()
    return loss, step

def push_adapter():
    """Push the trained adapter from VM to HuggingFace."""
    print("\n=== Pushing adapter to HuggingFace ===")
    script = f"""
import os
from huggingface_hub import HfApi, create_repo
api = HfApi(token=os.environ.get('HF_TOKEN',''))
try:
    create_repo('{ADAPTER_REPO}', repo_type='model', exist_ok=True)
    print('Repo ready')
except Exception as e:
    print(f'Repo: {{e}}')
api.upload_folder(
    folder_path='{ADAPTER_LOCAL}',
    repo_id='{ADAPTER_REPO}',
    repo_type='model',
)
print('Pushed to: https://huggingface.co/{ADAPTER_REPO}')
"""
    ssh_vm(f"cat > /tmp/push.py << 'EOF'\n{script}\nEOF")
    ssh_vm("HF_TOKEN=${HF_TOKEN} python3 /tmp/push.py")

def verify_on_hf():
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN",""))
    try:
        info = api.model_info(ADAPTER_REPO)
        print(f"✅ Verified on HF — {len(info.siblings)} files")
    except Exception as e:
        print(f"❌ HF verification failed: {e}")

def main():
    token = os.environ.get("HF_TOKEN", "")
    assert token, "Set HF_TOKEN env var first: export HF_TOKEN=your_token"

    print("="*50)
    print("Stack 4.0 — Finish & Deploy")
    print("="*50)

    if not check_done():
        print("⚠️  Training may still be running. Check manually:")
        print(f"   ssh -i {SSH_KEY} {VM_USER}@{VM_HOST}")
        print(f"   tail /home/walidsobhi/training.log")
        print("Waiting 5 minutes...")
        time.sleep(300)

    print("\n[1] Getting final loss...")
    loss, step = get_final_loss()
    print(f"   Final loss: {loss} @ step {step}")

    print("\n[2] Pushing adapter to HF...")
    push_adapter()

    print("\n[3] Verifying on HF...")
    verify_on_hf()

    print("\n[4] Next steps:")
    print(f"   1. HF Adapter: https://huggingface.co/{ADAPTER_REPO}")
    print(f"   2. Create HF Space: https://huggingface.co/new-space")
    print(f"      - Name: Omni-Nexus-Agentic | SDK: Gradio | Hardware: T4 Small")
    print(f"      - Upload: hf_space_app.py + hf_space_requirements.txt")
    print(f"      - Add HF_TOKEN secret")
    print(f"   3. Update model card with final loss: {loss}")
    print(f"   4. Run benchmarks: python3 benchmark.py (on VM before shutdown)")
    print(f"   5. Post to r/LocalLLaMA: reddit_post_final.md")
    print(f"   6. Shutdown VM: sudo shutdown -h now")
    print("="*50)

if __name__ == "__main__":
    main()
