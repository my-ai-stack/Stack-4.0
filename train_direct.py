#!/usr/bin/env python3
"""
Stack 4.0 Training — Llama 3.1 8B QLoRA on V100 16GB
Checkpoint saving every 100 steps, optimized for 16GB VRAM.
Run on GCP VM:  HF_TOKEN=xxx python3 train_direct.py
"""

import os
import sys
import json
import logging
import gc
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.cuda
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import bitsandbytes as bnb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────
@dataclass
class Config:
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    dataset_path: str = "my-ai-stack/Stack-4.0-Dataset"
    adapter_dir: str = "adapter"
    checkpoint_dir: str = "checkpoints"
    
    # LoRA — smaller rank for 16GB VRAM
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])
    
    # Training — conservative for 16GB V100
    num_epochs: int = 2
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 32   # effective batch = 32
    learning_rate: float = 2e-4
    warmup_steps: int = 50
    max_grad_norm: float = 0.5
    weight_decay: float = 0.01
    max_steps: int = 1000                  # stop after N steps (2 epochs ≈ 800 steps)
    save_steps: int = 100                  # checkpoint every 100 steps
    eval_steps: int = 100
    
    # Quantization — 4-bit for VRAM survival
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    
    seed: int = 42
    output_dir: str = "stack4_output"


@dataclass
class BitsAndBytesConfig:
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"


def format_example(example):
    """Convert agentic example to chat format for Llama 3.1."""
    messages = example.get("messages", [])
    formatted = []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # If assistant has tool_calls but no content, encode as tool call text
        if role == "assistant" and not content and msg.get("tool_calls"):
            tc = msg["tool_calls"]
            content = f"[TOOL_CALLS]\n{json.dumps(tc, indent=2)}"
        
        if content and content != "null":
            formatted.append({"role": role, "content": content})
    
    return {"messages": formatted}


def chat_to_text(messages, tokenizer):
    """Apply Llama 3.1 chat template."""
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return text
    except Exception as e:
        logger.warning(f"Template failed: {e}")
        # Fallback: simple concatenation
        text = ""
        for m in messages:
            text += f"<|{m['role']}|>{m['content']}<|end|>\n"
        return text


def prepare_example(example, tokenizer):
    try:
        formatted = format_example(example)
        text = chat_to_text(formatted["messages"], tokenizer)
        return {"text": text}
    except Exception as e:
        return {"text": ""}


def main():
    cfg = Config()
    set_seed(cfg.seed)
    start_time = datetime.now()
    
    logger.info("=" * 50)
    logger.info("Stack 4.0 Training — Llama 3.1 8B QLoRA")
    logger.info(f"V100 16GB — checkpoint every {cfg.save_steps} steps")
    logger.info("=" * 50)
    
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        logger.error("Set HF_TOKEN env var first")
        sys.exit(1)
    
    # ── Tokenizer ────────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # ── Model (4-bit QLoRA) ──────────────────────────────────────────────────
    logger.info(f"Loading model with 4-bit quantization...")
    logger.info("VRAM before model load:")
    torch.cuda.reset_peak_memory_stats()
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb.BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        ),
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # Add LoRA adapters
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    
    # Count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    logger.info(f"Peak VRAM after model load: {peak_vram:.1f}GB")
    
    # ── Dataset ──────────────────────────────────────────────────────────────
    logger.info(f"Loading dataset: {cfg.dataset_path}")
    ds = load_dataset(cfg.dataset_path, split="train")
    logger.info(f"Raw examples: {len(ds)}")
    
    # Format subset for speed (first 20K examples — enough for meaningful training)
    if len(ds) > 20000:
        ds = ds.select(range(20000))
        logger.info(f"Using subset: {len(ds)} examples")
    
    def safe_prepare(example):
        result = prepare_example(example, tokenizer)
        if not result["text"] or len(result["text"]) < 30:
            return {"text": ""}
        if len(result["text"]) > 6000:  # Truncate long examples
            result["text"] = result["text"][:6000]
        return result
    
    ds = ds.map(safe_prepare, remove_columns=ds.column_names, num_proc=4)
    ds = ds.filter(lambda x: x["text"] and len(x["text"]) > 30)
    logger.info(f"After filtering: {len(ds)} examples")
    
    # Train/eval split
    split = ds.train_test_split(test_size=0.05, seed=cfg.seed)
    train_ds = split["train"]
    eval_ds = split["test"]
    logger.info(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    
    # ── Data Collator ────────────────────────────────────────────────────────
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # ── Training Arguments ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        max_grad_norm=cfg.max_grad_norm,
        weight_decay=cfg.weight_decay,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=3,          # Keep only last 3 checkpoints
        load_best_model_at_end=False,
        report_to="none",
        seed=cfg.seed,
        hub_model_id="my-ai-stack/Stack-4.0-Omni-Nexus-Agentic",
        hub_token=hf_token,
        push_to_hub=False,            # Push manually after training
        hub_private_repo=False,
        logging_first_step=True,
        remove_unused_columns=False,
    )
    
    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )
    
    # Print training summary
    effective_batch = cfg.per_device_batch_size * cfg.gradient_accumulation_steps
    logger.info("=" * 50)
    logger.info("TRAINING SUMMARY")
    logger.info(f"Effective batch size: {effective_batch}")
    logger.info(f"Steps per epoch: ~{len(train_ds) // cfg.per_device_batch_size // cfg.gradient_accumulation_steps}")
    logger.info(f"Max steps: {cfg.max_steps}")
    logger.info(f"Checkpoint every: {cfg.save_steps} steps")
    logger.info(f"LoRA rank: {cfg.lora_r}, alpha: {cfg.lora_alpha}")
    logger.info("=" * 50)
    
    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting training...")
    trainer.train()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Training complete in {elapsed/3600:.1f} hours")
    
    # ── Save Adapter ────────────────────────────────────────────────────────────
    adapter_path = Path(cfg.adapter_dir)
    adapter_path.mkdir(exist_ok=True)
    
    logger.info(f"Saving adapter to {adapter_path}/")
    trainer.save_model(str(adapter_path))
    
    # Save tokenizer for easy loading
    tokenizer.save_pretrained(str(adapter_path))
    
    logger.info("=" * 50)
    logger.info("TRAINING DONE")
    logger.info(f"Adapter saved to: {adapter_path}/")
    logger.info(f"Next: HF_TOKEN={hf_token[:8]}... python3 push_adapter_to_hf.py")
    logger.info("=" * 50)
    
    # Auto-upload to HF
    logger.info("Uploading adapter to HuggingFace...")
    push_cmd = f'''
import os
from huggingface_hub import HfApi, create_repo
api = HfApi(token=os.environ.get("HF_TOKEN",""))
try:
    create_repo("my-ai-stack/Stack-4.0-Omni-Nexus-Agentic", exist_ok=True)
except:
    pass
api.upload_folder(
    folder_path="{adapter_path}",
    repo_id="my-ai-stack/Stack-4.0-Omni-Nexus-Agentic",
    repo_type="model",
)
print("Done uploading!")
'''
    os.system(f"HF_TOKEN={hf_token} python3 -c \"{push_cmd.replace(chr(10),';')}\"")
    
    # Clear VRAM
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("VRAM cleared")


if __name__ == "__main__":
    main()
