---
base_model: Qwen/Qwen2.5-Coder-3B-Instruct
datasets:
- my-ai-stack/Stack-4.0-Dataset
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
---

# Stack 4.0 Omni-Nexus-Agentic

**Model ID:** `my-ai-stack/Stack-4.0-Qwen-3B-Agentic`

A 3-billion parameter instruction-tuned coding model, fine-tuned from Qwen2.5-Coder-3B-Instruct on 55,000 agentic tool-use conversations. Designed for developers who want a fast, local-friendly AI that can reason about when and how to use external tools.

## Training Results

| Metric | Value |
|--------|-------|
| Final training loss | **0.1411** |
| Training steps | 1,000 |
| Total training time | ~10 hours |
| Hardware | GCP Tesla V100 16GB |

**Loss curve:** 2.73 (step 1) → 0.14 (step 1000) — clean convergence, no instability.

## Key Differences from Stack 3.0

| | Stack 3.0 | Stack 4.0 |
|--|-----------|-----------|
| Parameters | 7B | **3B** |
| VRAM needed | ~14GB | **~6GB** |
| Speed | Moderate | **Fast** |
| Training data | 55K tool-use | **55K tool-use (cleaner)** |
| Tool-calling | Good | **Improved** |

Stack 4.0 is lighter, faster, and easier to run locally — while maintaining most of the capability.

## Benchmark Results

| Benchmark | Score | Notes |
|-----------|-------|-------|
| HumanEval | Pending | Full eval pending |
| ARC-C | Pending | Full eval pending |
| MMLU | Pending | Full eval pending |
| GSM8K | Pending | Full eval pending |

*Note: 3B models typically score lower than 7B on standard benchmarks. Tool-use reasoning is the focus.*

## Training Details

| Parameter | Value |
|-----------|-------|
| Method | LoRA (QLoRA) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Trainable params | 7.3M / 3.1B (0.24%) |
| Batch size | 1 |
| Grad accumulation | 16 |
| Max length | 512 |
| Learning rate | 2e-4 |
| Final loss | 0.1411 |
| Hardware | GCP Tesla V100 16GB |
| Training time | ~10 hours |

## Tool Use

Stack 4.0 is trained on agentic conversations and supports these tools:

- **`search_web(query)`** — Search the web for current information
- **`read_file(path)`** — Read files from the local filesystem
- **`run_command(command)`** — Execute shell commands (git, python, ls, grep, etc.)
- **`calculator(expression)`** — Safe mathematical evaluation
- **`ask_user(question)`** — Request clarification from the user

### Example Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
ADAPTER = "my-ai-stack/Stack-4.0-Qwen-3B-Agentic"

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()

# Tool-use prompt
messages = [
    {"role": "user", "content": "Read the file at /tmp/test.txt"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
# → <tool_call>{"name": "read_file", "arguments": {"path": "/tmp/test.txt"}}</tool_call>
```

## Local Inference

### Ollama
```bash
# GGUF conversion in progress
ollama run my-ai-stack/stack-4-0-omninexus-agentic
```

### transformers + peft
```bash
pip install transformers peft accelerate
```

## Limitations

- **3B model** — smaller than 7B/13B models; less capable on complex reasoning tasks
- **Tool definitions** — model is trained to call tools but the actual tool execution must be implemented in your code
- **No real-world file access** — tool calls must be handled by the calling application
- **Egyptian Arabic** — not optimized; English recommended

## Citation

```bibtex
@misc{stack-4-0,
  title={Stack 4.0 Omni-Nexus-Agentic},
  url={https://huggingface.co/my-ai-stack/Stack-4.0-Qwen-3B-Agentic},
  author={AI Stack},
  year={2026}
}
```

## See Also

- [Training script + dataset](https://github.com/my-ai-stack/Stack-4.0)
- [Agent loop implementation](https://github.com/my-ai-stack/Stack-4.0/blob/main/agent_loop.py)
- [Stack 3.0 (7B version)](https://huggingface.co/my-ai-stack/Stack-3.0-Omni-Nexus)
