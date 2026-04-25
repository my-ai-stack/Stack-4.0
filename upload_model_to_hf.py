#!/usr/bin/env python3
"""
Upload the full 15GB Qwen merged model from VM to HuggingFace.
Run this ON the VM:  python3 upload_model_to_hf.py
"""

import os
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    logger.error("Set HF_TOKEN env var")
    exit(1)

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

api = HfApi(token=HF_TOKEN)
MODEL_DIR = Path("/home/walidsobhi/stack-3.0/merged-model")
REPO_ID = "my-ai-stack/Stack-3.0-Omni-Nexus"

logger.info(f"Source: {MODEL_DIR}")
logger.info(f"Target: {REPO_ID}")

# Create repo
try:
    create_repo(REPO_ID, repo_type="model", exist_ok=True)
    logger.info("✅ Repo ready")
except Exception as e:
    logger.info(f"Repo: {e}")

# Upload model files (safetensors + config + tokenizer)
files = [
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
]

for fname in files:
    fpath = MODEL_DIR / fname
    if not fpath.exists():
        logger.info(f"⚠️  Missing: {fname} — skipping")
        continue

    size_gb = fpath.stat().st_size / 1024**3
    logger.info(f"Uploading {fname} ({size_gb:.1f}GB)...")

    try:
        upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fname,
            repo_id=REPO_ID,
            repo_type="model",
        )
        logger.info("✅")
    except Exception as e:
        logger.warning(f"❌ {e}")

    time.sleep(3)

logger.info(f"\n✅ Model uploaded — https://huggingface.co/{REPO_ID}")
