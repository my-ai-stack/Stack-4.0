"""
Stack 4.0 — Modal.com Deployment (FREE GPU)

Setup:
1. pip install modal
2. modal auth login
3. modal deploy modal_serve.py

Then call the endpoint:
   curl -X POST https://your-app.modal.com/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
"""

import modal

# ─── Image definition ────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "transformers>=4.36",
        "peft>=0.14.0",
        "accelerate>=0.25",
        "fastapi",
        "uvicorn[standard]",
        "pydantic",
    )
)

app = modal.App("stack-4-agentic")

# ─── Volume for model cache ────────────────────────────────────────────────────
volume = modal.Volume.from_name("stack4-model-cache", create=True)

# ─── Model serving ───────────────────────────────────────────────────────────
@app.cls(gpu="T4",  # Free T4 GPU!
         image=image,
         volumes={"/model_cache": volume},
         timeout=600,
         container_idle_timeout=300,
         retries=2)
class AgenticModel:
    def __enter__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import os

        HF_TOKEN = os.environ.get("HF_TOKEN", "")
        ADAPTER_REPO = "my-ai-stack/Stack-4.0-Qwen-3B-Agentic"
        BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=True, token=HF_TOKEN
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
        )

        print("Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(
            self.base_model, ADAPTER_REPO, token=HF_TOKEN
        )
        self.model.eval()
        print("✅ Model ready!")

    @modal.method()
    def generate(self, messages: list, temperature: float = 0.7, max_tokens: int = 1024):
        import torch
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return {"content": response_text.strip(), "model": ADAPTER_REPO}


# ─── Web endpoint ────────────────────────────────────────────────────────────
@app.function(image=image, timeout=30)
@modal.web_endpoint(method="POST")
def chat(req: dict):
    """FastAPI-style endpoint callable as HTTP POST."""
    model_instance = AgenticModel()
    result = model_instance.generate.remote(
        messages=req.get("messages", []),
        temperature=req.get("temperature", 0.7),
        max_tokens=req.get("max_tokens", 1024),
    )
    return result


@app.function(image=image)
@modal.web_endpoint(method="GET")
def health():
    return {"status": "ok", "model": "my-ai-stack/Stack-4.0-Qwen-3B-Agentic"}


# ─── CLI for local testing ────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    print("Testing model endpoint...")
    result = AgenticModel().generate.remote(
        messages=[{"role": "user", "content": "What is 2+2? Use a tool if needed."}],
        temperature=0.7,
        max_tokens=256,
    )
    print("Response:", result)
