#!/usr/bin/env python3
"""
Stack 4.0 Training — Qwen2.5-Coder-3B QLoRA on V100 16GB
Checkpoint every 100 steps. Loads dataset directly from HF JSONL files.
No bitsandbytes — uses bf16 + paged AdamW.
"""

import os, sys, gc, json, logging
from pathlib import Path
from datetime import datetime

import torch
import torch.cuda
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/walidsobhi/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
DATASET_REPO = "my-ai-stack/Stack-4.0-Dataset"
OUTPUT_DIR = "/home/walidsobhi/stack-4.0-adapter"
ADAPTER_DIR = "/home/walidsobhi/stack-4.0-adapter/lora_adapter"

EPOCHS = 2
BATCH_SIZE = 4
GRAD_ACCUM = 32
LR = 2e-4
WARMUP = 50
MAX_STEPS = 1000
SAVE_STEPS = 100
EVAL_STEPS = 100
MAX_GRAD_NORM = 0.5
WEIGHT_DECAY = 0.01
SEED = 42
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj"]

set_seed(SEED)
START = datetime.now()


def get_token():
    t = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    if not t:
        p = Path.home() / ".cache" / "huggingface" / "token"
        if p.exists():
            t = p.read_text().strip()
    if not t:
        logger.error("No HF token — set HF_TOKEN env var")
        sys.exit(1)
    return t


def check_vram(label=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"VRAM {label}: {alloc:.1f}GB alloc, {peak:.1f}GB peak")


def format_example(ex):
    msgs = ex.get("messages", [])
    out = []
    for m in msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "assistant" and not content and m.get("tool_calls"):
            tc = m["tool_calls"]
            content = f"<think>\nI'll use a tool.\n</think>\n<tool_call>\n{json.dumps(tc)}\n</tool_call>"
        if content:
            out.append({"role": role, "content": content})
    return {"messages": out}


def load_jsonl_dataset(repo_id, tokenizer, hf_token):
    """Load dataset from HF dataset repo JSONL files — avoids feature compatibility issues."""
    from huggingface_hub import hf_hub_download

    # Pick the smart subset — best quality examples
    jsonl_file = "agentic_data/tool_examples_smart_20k.jsonl"
    logger.info(f"Downloading {jsonl_file} from {repo_id}...")

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=jsonl_file,
        repo_type="dataset",
        token=hf_token,
        local_dir="/tmp/stack4ds",
    )
    logger.info(f"Dataset file: {local_path}")

    examples = []
    skipped = 0
    with open(local_path, "r") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
                formatted = format_example(ex)
                text = tokenizer.apply_chat_template(
                    formatted["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                if 30 <= len(text) <= 8000:
                    examples.append({"text": text})
                else:
                    skipped += 1
            except Exception:
                skipped += 1
            if i > 0 and i % 5000 == 0:
                logger.info(f"  Processed {i} lines, {len(examples)} kept, {skipped} skipped")

    logger.info(f"Loaded {len(examples)} examples ({skipped} skipped)")
    return Dataset.from_list(examples)


def tokenize_dataset(ds, tokenizer, max_len=512):
    """Add tokenized input_ids to dataset for DataCollatorForLanguageModeling."""
    def tok(example):
        enc = tokenizer(
            example["text"],
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": enc["input_ids"],
        }

    logger.info(f"Tokenizing dataset (max_len={max_len})...")
    ds = ds.map(tok, remove_columns=["text"], num_proc=4)
    return ds


def main():
    hf_token = get_token()
    logger.info("=" * 60)
    logger.info("Stack 4.0 Training — Qwen2.5-Coder-3B QLoRA")
    logger.info(f"Token: {hf_token[:8]}...")
    logger.info("=" * 60)

    # Tokenizer
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, use_fast=True, token=hf_token
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Dataset
    logger.info(f"Loading dataset from: {DATASET_REPO}")
    ds = load_jsonl_dataset(DATASET_REPO, tokenizer, hf_token)

    # Tokenize for trainer
    ds = tokenize_dataset(ds, tokenizer, max_len=1024)

    split = ds.train_test_split(test_size=0.05, seed=SEED)
    train_ds, eval_ds = split["train"], split["test"]
    logger.info(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # Model — bf16 on GPU (let device_map decide, no CPU offloading for training)
    logger.info(f"Loading model: {MODEL_NAME}")
    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )

    # LoRA
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    check_vram("after model+LoRA")

    # Training args
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_steps=WARMUP,
        max_grad_norm=MAX_GRAD_NORM,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=False,
        report_to="none",
        seed=SEED,
        hub_model_id="my-ai-stack/Stack-4.0-Qwen-3B-Agentic",
        hub_token=hf_token,
        push_to_hub=False,
        logging_first_step=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    logger.info(f"Effective batch: {BATCH_SIZE * GRAD_ACCUM} | Steps: {MAX_STEPS} | Checkpoint: every {SAVE_STEPS}")
    check_vram("before training")

    # Train
    logger.info("🚀 Training starting...")
    trainer.train()

    elapsed = (datetime.now() - START).total_seconds()
    logger.info(f"Training done in {elapsed/3600:.2f}h")
    check_vram("after training")

    # Save adapter
    Path(ADAPTER_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving to {ADAPTER_DIR}/")
    trainer.save_model(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    # Upload
    logger.info("Uploading adapter to HF...")
    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=hf_token)
    try:
        create_repo("my-ai-stack/Stack-4.0-Qwen-3B-Agentic", exist_ok=True, repo_type="model")
    except Exception as e:
        logger.warning(f"Repo: {e}")
    api.upload_folder(folder_path=ADAPTER_DIR, repo_id="my-ai-stack/Stack-4.0-Qwen-3B-Agentic", repo_type="model")
    logger.info("✅ https://huggingface.co/my-ai-stack/Stack-4.0-Qwen-3B-Agentic")

    # Cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    check_vram("final")
    logger.info("ALL DONE")


if __name__ == "__main__":
    main()
