#!/usr/bin/env python3
"""
Stack 4.0 Direct Training — Llama 3.1 8B QLoRA
Trains on agentic tool-use examples from HuggingFace dataset
Optimized for V100 16GB
"""

import os
import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────
@dataclass
class Config:
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    dataset_path: str = "my-ai-stack/Stack-4.0-Dataset"
    adapter_dir: str = "adapter"
    
    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    
    # Training
    num_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16  # effective batch = 16
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    
    output_dir: str = "adapter"
    seed: int = 42


def format_tool_example(example):
    """Convert agentic example to chat format for Llama 3.1."""
    messages = example.get("messages", [])
    formatted = []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # Skip tool_calls from assistant (we're training it to produce tool calls via format)
        if role == "assistant" and not content and msg.get("tool_calls"):
            # Store tool_calls in the content for the model to learn
            tc_json = json.dumps(msg["tool_calls"], indent=2)
            content = f"[TOOL_CALLS]\n{tc_json}"
        
        if content:
            formatted.append({"role": role, "content": content})
    
    return {"messages": formatted}


def chat_to_text(messages, tokenizer):
    """Convert chat messages to text using tokenizer's chat template."""
    # Use tokenizer's built-in chat template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return formatted


def load_and_format_dataset(cfg: Config, tokenizer):
    """Load dataset from HF and format for training."""
    logger.info(f"Loading dataset: {cfg.dataset_path}")
    
    # Load all JSONL files from the dataset
    ds = load_dataset(cfg.dataset_path, split="train")
    logger.info(f"Loaded {len(ds)} examples")
    
    def prepare_example(example):
        # Format the example
        formatted = format_tool_example(example)
        text = chat_to_text(formatted["messages"], tokenizer)
        return {"text": text}
    
    ds = ds.map(prepare_example, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["text"]) > 20 and len(x["text"]) < 8192)
    logger.info(f"After filtering: {len(ds)} examples")
    
    return ds


def create_model_and_tokenizer(cfg: Config):
    """Create quantized model with LoRA adapters."""
    logger.info(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    logger.info(f"Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        ) if cfg.load_in_4bit else None,
    )
    
    # Prepare for k-bit training
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Add LoRA
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    
    return model, tokenizer


@dataclass  
class BitsAndBytesConfig:
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"


def main():
    cfg = Config()
    set_seed(cfg.seed)
    
    logger.info("===== Stack 4.0 Training =====")
    logger.info(f"Model: {cfg.model_name}")
    logger.info(f"Dataset: {cfg.dataset_path}")
    logger.info(f"VRAM: check with nvidia-smi")
    
    # Load model and tokenizer
    model, tokenizer = create_model_and_tokenizer(cfg)
    
    # Load and format dataset
    dataset = load_and_format_dataset(cfg, tokenizer)
    
    # Split for train/eval
    split_ds = dataset.train_test_split(test_size=0.05, seed=cfg.seed)
    train_ds = split_ds["train"]
    eval_ds = split_ds["test"]
    
    logger.info(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        max_grad_norm=cfg.max_grad_norm,
        weight_decay=cfg.weight_decay,
        fp16=False,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        report_to="none",
        seed=cfg.seed,
        hub_model_id="my-ai-stack/Stack-4.0-Omni-Nexus-Agentic",
        hub_token=os.environ.get("HF_TOKEN", ""),
        push_to_hub=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )
    
    logger.info("===== Starting Training =====")
    trainer.train()
    
    logger.info("===== Saving Adapter =====")
    trainer.save_model(cfg.adapter_dir)
    logger.info(f"Adapter saved to: {cfg.adapter_dir}")
    
    logger.info("===== Training Complete =====")


if __name__ == "__main__":
    main()
