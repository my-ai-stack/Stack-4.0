#!/usr/bin/env python3
"""
Stack 4.0 Training — Qwen2.5-Coder-7B QLoRA on V100 16GB
Uses the full 55K agentic dataset to train Qwen as a tool-using agent.
Checkpoint saving every 100 steps, optimized for 16GB VRAM.
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
import bitsandbytes as bnb

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
CFG = {
    # Use Qwen — fully accessible on HF (unlike Llama which is gated)
    "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "dataset": "my-ai-stack/Stack-4.0-Dataset",
    "output_dir": "/home/walidsobhi/stack-4.0-adapter",
    "adapter_dir": "/home/walidsobhi/stack-4.0-adapter/lora_adapter",

    # LoRA — moderate rank for 16GB V100
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target": ["q_proj", "k_proj", "v_proj", "o_proj"],

    # Training
    "epochs": 2,
    "batch_size": 1,
    "grad_accum": 32,          # effective = 32
    "lr": 2e-4,
    "warmup": 50,
    "max_steps": 1000,
    "save_steps": 100,          # checkpoint every 100 steps
    "eval_steps": 100,
    "max_grad_norm": 0.5,
    "weight_decay": 0.01,
    "seed": 42,

    # VRAM survival
    "load_in_4bit": True,
    "bnb_compute": "bfloat16",
    "gradient_checkpointing": True,
}

set_seed(CFG["seed"])
START = datetime.now()

# ─── Helpers ────────────────────────────────────────────────────────────────

def format_example(ex):
    """Convert agentic example to Qwen chat format."""
    msgs = ex.get("messages", [])
    out = []
    for m in msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "assistant" and not content and m.get("tool_calls"):
            tc = m["tool_calls"]
            # Format as Qwen tool call block
            content = f"<think>\nI'll use a tool to help answer this.\n</think>\n<tool_call>\n{json.dumps(tc)}\n</tool_call>"
        if content:
            out.append({"role": role, "content": content})
    return {"messages": out}


def prepare_batch(ex, tokenizer):
    """Format one example for Qwen training."""
    try:
        formatted = format_example(ex)
        # Qwen chat template
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
    except Exception as e:
        return {"text": ""}


def check_vram(label=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"VRAM {label}: {alloc:.1f}GB allocated, peak {peak:.1f}GB")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    hf_token = (
        os.environ.get("HF_TOKEN", "")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    )
    if not hf_token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            hf_token = token_path.read_text().strip()
    if not hf_token:
        logger.error("Set HF_TOKEN env var")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Stack 4.0 Training — Qwen2.5-Coder-7B QLoRA")
    logger.info(f"V100 16GB | save every {CFG['save_steps']} steps")
    logger.info(f"Token: {hf_token[:8]}...")
    logger.info("=" * 60)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {CFG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        CFG["model_name"],
        trust_remote_code=True,
        use_fast=True,
        token=hf_token,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model (4-bit QLoRA) ──────────────────────────────────────────────────
    logger.info("Loading model with 4-bit BitsAndBytes QLoRA...")
    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        CFG["model_name"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        quantization_config=bnb.BitsAndBytesConfig(
            load_in_4bit=CFG["load_in_4bit"],
            bnb_4bit_compute_dtype=getattr(torch, CFG["bnb_compute"]),
            bnb_4bit_quant_type="nf4",
        ),
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=CFG["gradient_checkpointing"],
    )

    # LoRA
    lora_cfg = LoraConfig(
        r=CFG["lora_r"],
        lora_alpha=CFG["lora_alpha"],
        lora_dropout=CFG["lora_dropout"],
        target_modules=CFG["lora_target"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    check_vram("after model load")

    # ── Dataset ───────────────────────────────────────────────────────────────
    logger.info(f"Loading dataset: {CFG['dataset']}")
    ds = load_dataset(CFG["dataset"], split="train")
    logger.info(f"Total examples: {len(ds)}")

    # Use full 55K — more data = better agentic behavior
    ds = ds.map(
        lambda ex: prepare_batch(ex, tokenizer),
        remove_columns=ds.column_names,
        num_proc=4,
    )
    ds = ds.filter(lambda x: x["text"] and len(x["text"]) > 30)
    logger.info(f"After filter: {len(ds)}")

    split = ds.train_test_split(test_size=0.05, seed=CFG["seed"])
    train_ds, eval_ds = split["train"], split["test"]
    logger.info(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # ── Training Arguments ───────────────────────────────────────────────────
    Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=CFG["output_dir"],
        max_steps=CFG["max_steps"],
        per_device_train_batch_size=CFG["batch_size"],
        gradient_accumulation_steps=CFG["grad_accum"],
        learning_rate=CFG["lr"],
        warmup_steps=CFG["warmup"],
        max_grad_norm=CFG["max_grad_norm"],
        weight_decay=CFG["weight_decay"],
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=CFG["eval_steps"],
        save_strategy="steps",
        save_steps=CFG["save_steps"],
        save_total_limit=3,         # keep only last 3 checkpoints
        load_best_model_at_end=False,
        report_to="none",
        seed=CFG["seed"],
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

    eff_batch = CFG["batch_size"] * CFG["grad_accum"]
    logger.info(f"Effective batch: {eff_batch} | Steps: {CFG['max_steps']} | Checkpoint: every {CFG['save_steps']}")

    # ── Train ────────────────────────────────────────────────────────────────
    logger.info("🚀 Starting training...")
    trainer.train()

    elapsed = (datetime.now() - START).total_seconds()
    logger.info(f"Training complete in {elapsed/3600:.2f} hours")
    check_vram("after training")

    # ── Save ─────────────────────────────────────────────────────────────────
    Path(CFG["adapter_dir"]).mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving adapter to {CFG['adapter_dir']}/")
    trainer.save_model(CFG["adapter_dir"])
    tokenizer.save_pretrained(CFG["adapter_dir"])

    # ── Push to HF ───────────────────────────────────────────────────────────
    logger.info("Uploading adapter to HuggingFace...")
    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=hf_token)
    try:
        create_repo("my-ai-stack/Stack-4.0-Qwen-Agentic", exist_ok=True, repo_type="model")
    except Exception as e:
        logger.warning(f"Repo create: {e}")

    api.upload_folder(
        folder_path=CFG["adapter_dir"],
        repo_id="my-ai-stack/Stack-4.0-Qwen-Agentic",
        repo_type="model",
    )
    logger.info("✅ Adapter uploaded: https://huggingface.co/my-ai-stack/Stack-4.0-Qwen-Agentic")

    # ── Cleanup VRAM ─────────────────────────────────────────────────────────
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    check_vram("final")

    logger.info("=" * 60)
    logger.info("ALL DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
