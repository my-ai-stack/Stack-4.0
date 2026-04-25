#!/usr/bin/env python3
"""
Stack 4.0 Training — Qwen2.5-Coder-7B (or 3B fallback) QLoRA on V100 16GB
Uses full 55K agentic dataset. No bitsandbytes — uses native bf16 + CPU offload.
Checkpoint saving every 100 steps.
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

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
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
DATASET = "my-ai-stack/Stack-4.0-Dataset"
OUTPUT_DIR = "/home/walidsobhi/stack-4.0-adapter"
ADAPTER_DIR = "/home/walidsobhi/stack-4.0-adapter/lora_adapter"

# Training
EPOCHS = 2
BATCH_SIZE = 1
GRAD_ACCUM = 32          # effective = 32
LR = 2e-4
WARMUP = 50
MAX_STEPS = 1000
SAVE_STEPS = 100
EVAL_STEPS = 100
MAX_GRAD_NORM = 0.5
WEIGHT_DECAY = 0.01
SEED = 42

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj"]

# VRAM: use 7B if it fits, else fall back to 3B
USE_FALLBACK = os.environ.get("USE_FALLBACK_MODEL", "0") == "1"

set_seed(SEED)
START = datetime.now()


def get_token():
    token = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    if not token:
        p = Path.home() / ".cache" / "huggingface" / "token"
        if p.exists():
            token = p.read_text().strip()
    if not token:
        logger.error("No HF token found — set HF_TOKEN env var or ~/.cache/huggingface/token")
        sys.exit(1)
    return token


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
            content = f"<think>\nI'll use a tool to help.\n</think>\n<tool_call>\n{json.dumps(tc)}\n</tool_call>"
        if content:
            out.append({"role": role, "content": content})
    return {"messages": out}


def prepare_batch(ex, tokenizer):
    try:
        formatted = format_example(ex)
        text = tokenizer.apply_chat_template(
            formatted["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        if len(text) < 30:
            return {"text": ""}
        if len(text) > 8000:
            text = text[:8000]
        return {"text": text}
    except Exception:
        return {"text": ""}


def main():
    hf_token = get_token()
    logger.info("=" * 60)
    logger.info("Stack 4.0 Training — Qwen2.5-Coder QLoRA")
    logger.info(f"Token: {hf_token[:8]}...")
    logger.info("=" * 60)

    # ── Determine which model to use ─────────────────────────────────────────
    model_name = MODEL_NAME
    if USE_FALLBACK:
        model_name = FALLBACK_MODEL
        logger.info("Using 3B model (fallback mode)")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
        token=hf_token,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model — bfloat16 with CPU offload for VRAM management ─────────────────
    # Use max_memory to leave headroom for training
    max_memory = {0: "14GiB", "cpu": "80GiB"}
    
    logger.info(f"Loading model: {model_name} (bfloat16 + device_map auto)")
    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        token=hf_token,
    )

    # Attach LoRA directly — bf16 model doesn't need prepare_model_for_kbit_training
    # (that call is only for 4-bit quantized models that need fp32 conversion)
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    check_vram("after model+LoRA load")

    # ── Dataset ───────────────────────────────────────────────────────────────
    logger.info(f"Loading dataset: {DATASET}")
    ds = load_dataset(DATASET, split="train")
    logger.info(f"Total examples: {len(ds)}")

    ds = ds.map(
        lambda ex: prepare_batch(ex, tokenizer),
        remove_columns=ds.column_names,
        num_proc=4,
    )
    ds = ds.filter(lambda x: x["text"] and len(x["text"]) > 30)
    logger.info(f"After filter: {len(ds)}")

    split = ds.train_test_split(test_size=0.05, seed=SEED)
    train_ds, eval_ds = split["train"], split["test"]
    logger.info(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # ── Training Arguments ───────────────────────────────────────────────────
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
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",   # memory-efficient optimizer (no bitsandbytes needed)
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=False,
        report_to="none",
        seed=SEED,
        hub_model_id="my-ai-stack/Stack-4.0-Qwen-Agentic",
        hub_token=hf_token,
        push_to_hub=False,
        logging_first_step=True,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    # ── Trainer ─────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    eff_batch = BATCH_SIZE * GRAD_ACCUM
    logger.info(f"Model: {model_name}")
    logger.info(f"Effective batch: {eff_batch} | Steps: {MAX_STEPS} | Checkpoint: every {SAVE_STEPS}")
    check_vram("before training")

    # ── Train ────────────────────────────────────────────────────────────────
    logger.info("🚀 Starting training...")
    trainer.train()

    elapsed = (datetime.now() - START).total_seconds()
    logger.info(f"Training complete in {elapsed/3600:.2f} hours")
    check_vram("after training")

    # ── Save + Upload ────────────────────────────────────────────────────────
    Path(ADAPTER_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving adapter to {ADAPTER_DIR}/")
    trainer.save_model(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    logger.info("Uploading to HuggingFace...")
    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=hf_token)
    try:
        create_repo("my-ai-stack/Stack-4.0-Qwen-Agentic", exist_ok=True, repo_type="model")
    except Exception as e:
        logger.warning(f"Repo: {e}")

    api.upload_folder(
        folder_path=ADAPTER_DIR,
        repo_id="my-ai-stack/Stack-4.0-Qwen-Agentic",
        repo_type="model",
    )
    logger.info("✅ https://huggingface.co/my-ai-stack/Stack-4.0-Qwen-Agentic")

    # ── Cleanup ──────────────────────────────────────────────────────────────
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    check_vram("final")
    logger.info("=" * 60)
    logger.info("ALL DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
