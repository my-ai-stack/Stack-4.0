"""
Stack 4.0 — HF Space for permanent FREE hosting
Gradio web UI — anyone can access at: https://huggingface.co/spaces/my-ai-stack/Stack-4.0-Agentic

Setup:
1. Create HF Space: https://huggingface.co/new-space (select Gradio SDK)
2. Upload these files to the Space repo
3. Add HF_TOKEN secret in Space settings

Requires: transformers, peft, gradio, torch
"""

import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "my-ai-stack/Stack-4.0-Qwen-3B-Agentic")
BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

print(f"Loading: {BASE_MODEL} + adapter {ADAPTER_REPO}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True, token=HF_TOKEN,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO, token=HF_TOKEN)
model.eval()
print("✅ Model ready!")


def respond(message, history, system_prompt, temperature, max_tokens):
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": message})
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=int(max_tokens),
                             temperature=temperature, do_sample=temperature > 0,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return reply.strip()


with gr.Blocks(title="Stack 4.0 Omni-Nexus-Agentic", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 🧠 Stack 4.0 Omni-Nexus-Agentic\nModel: `{ADAPTER_REPO}`")
    with gr.Row():
        chat = gr.Chatbot(height=500, show_copy_button=True)
        with gr.Column():
            gr.Markdown("### ⚙️ Settings")
            system = gr.Textbox(label="System prompt", value="You are a helpful AI assistant that can use tools.", lines=3)
            temp = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature")
            maxtok = gr.Slider(64, 2048, value=512, step=64, label="Max new tokens")
    msg = gr.Textbox(label="Message", placeholder="Ask me anything!")
    send = gr.Button("Send", variant="primary")
    send.click(respond, [msg, chat, system, temp, maxtok], [msg, chat])
    msg.submit(respond, [msg, chat, system, temp, maxtok], [msg, chat])

demo.launch()
