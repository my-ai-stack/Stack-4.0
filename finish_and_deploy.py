#!/usr/bin/env python3
"""
Stack 4.0 — Finish & Deploy
Run from your LOCAL machine when training finishes.

Usage:
    export HF_TOKEN=hf_xxx
    python3 finish_and_deploy.py
"""
import os, time, subprocess

ADAPTER_REPO = "my-ai-stack/Stack-4.0-Qwen-3B-Agentic"
ADAPTER_LOCAL = "/home/walidsobhi/stack-4.0-adapter/lora_adapter"
VM_HOST = "35.223.229.30"
SSH_KEY = "~/.ssh/google_compute_engine"

def ssh_vm(cmd):
    r = subprocess.run(f"ssh -i {SSH_KEY} walidsobhi@{VM_HOST} '{cmd}'",
                      shell=True, capture_output=True, text=True)
    return r.stdout + r.stderr

def check_done():
    out = ssh_vm("tail -3 /home/walidsobhi/training.log")
    print(out)
    return "ALL DONE" in out or ("100%" in out and "complete" in out)

def push_adapter():
    print("\n=== Pushing adapter to HuggingFace ===")
    script = f"""
import os
from huggingface_hub import HfApi, create_repo
api = HfApi(token=os.environ.get('HF_TOKEN',''))
try:
    create_repo('{ADAPTER_REPO}', repo_type='model', exist_ok=True)
except: pass
api.upload_folder(folder_path='{ADAPTER_LOCAL}', repo_id='{ADAPTER_REPO}', repo_type='model')
print('Pushed to: https://huggingface.co/{ADAPTER_REPO}')
"""
    ssh_vm(f"cat > /tmp/push.py << 'EOF'\n{script}\nEOF")
    ssh_vm("HF_TOKEN=${HF_TOKEN} python3 /tmp/push.py")
    print(f"✅ https://huggingface.co/{ADAPTER_REPO}")

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
    assert token, "Set HF_TOKEN env var first:  export HF_TOKEN=hf_xxx"

    print("="*50)
    print("Stack 4.0 — Finish & Deploy")
    print("="*50)

    if not check_done():
        print("⚠️  Training may still be running. Check manually:")
        print(f"   ssh -i {SSH_KEY} walidsobhi@{VM_HOST}")
        print(f"   tail /home/walidsobhi/training.log")

    print("\n[1] Pushing adapter to HF...")
    push_adapter()

    print("\n[2] Verifying on HF...")
    verify_on_hf()

    print("\n[3] Next steps:")
    print(f"   1. Go to https://huggingface.co/{ADAPTER_REPO}")
    print(f"   2. Create a HF Space (https://huggingface.co/new-space)")
    print(f"   3. Upload hf_space_app.py + requirements.txt")
    print(f"   4. Set HF_TOKEN as a secret in the Space settings")
    print(f"   5. Your app will be live at: https://huggingface.co/spaces/YOUR-USERNAME/Stack-4.0-Agentic")
    print("="*50)

if __name__ == "__main__":
    main()
