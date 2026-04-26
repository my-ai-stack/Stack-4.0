#!/usr/bin/env python3
"""
Stack 4.0 — Inference Server
Serves the agentic model via FastAPI + Groq client (or standalone).
Run this on Modal.com (free GPU) or as a simple FastAPI server.

Usage:
    # Option 1: Modal.com (FREE GPU)
    modal deploy serve_model.py

    # Option 2: Local / any server
    python serve_model.py
"""

import os
import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ─── Config ─────────────────────────────────────────────────────────────────
ADAPTER_REPO = "my-ai-stack/Stack-4.0-Qwen-3B-Agentic"
BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

app = FastAPI(title="Stack 4.0 Agentic Model")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ─── Tools registry (for agentic responses) ─────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from the filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Full path to the file"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
]


# ─── Request/Response models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    messages: list[dict]  # [{"role": "user", "content": "..."}]
    temperature: float = 0.7
    max_tokens: int = 1024
    tools: bool = True  # If True, model can output tool calls


class ChatResponse(BaseModel):
    content: str
    tool_calls: Optional[list] = None
    model: str


# ─── Model loading ────────────────────────────────────────────────────────────
model = None
tokenizer = None


def load_model():
    global model, tokenizer
    if model is not None:
        return

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model with bf16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    print(f"Loading LoRA adapter: {ADAPTER_REPO}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_REPO, token=HF_TOKEN)
    model.eval()
    print("✅ Model ready!")


@app.on_event("startup")
def startup():
    load_model()


@app.get("/")
def root():
    return {"model": ADAPTER_REPO, "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": ADAPTER_REPO}


@app.post("/v1/chat")
def chat(req: ChatRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    # Build conversation text
    text = tokenizer.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Parse tool calls if present
    tool_calls = None
    if "<tool_call>" in response_text:
        try:
            start = response_text.index("<tool_call>") + len("<tool_call>")
            end = response_text.index("</tool_call>")
            tool_calls = json.loads(response_text[start:end])
            response_text = response_text[:start - len("<tool_call>")]
        except Exception:
            pass

    return ChatResponse(
        content=response_text.strip(),
        tool_calls=tool_calls,
        model=ADAPTER_REPO,
    )


@app.get("/v1/tools")
def list_tools():
    return {"tools": TOOLS}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
