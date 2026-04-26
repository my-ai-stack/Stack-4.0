---
annotations_creators:
- synthetic
language:
- en
license: apache-2.0
multilingual: false
size_categories:
- 10K<n<100K
source_datasets:
- original
task_categories:
- sequence-modeling
task_ids:
- dialogue-modeling
---

# Stack-4.0-Dataset

**55,000 agentic tool-use conversations** for training AI models to reason about when and how to use external tools.

## Overview

This dataset teaches models to reason about tool use — not just answer questions, but decide *when* to reach for a tool, *which* tool to use, and *how* to chain multiple tools in sequence.

Each example is a multi-turn conversation with tool-call reasoning chains. The model learns to:
- Recognize when a task requires external action (file read, command execution, web search)
- Select the correct tool for the situation
- Interpret tool results and decide the next step
- Chain 2-4 tools in sequence where each step depends on the last

## Dataset Structure

Each example is a JSON object with a `conversations` array:

```json
{
  "conversations": [
    {"role": "user", "content": "Read the file at /path/to/file.py"},
    {"role": "assistant", "content": "I'll read that file for you."},
    {"role": "tool", "name": "read_file", "content": "File contents here..."},
    {"role": "assistant", "content": "I found the issue. The function is missing a return statement..."}
  ]
}
```

The dataset contains three splits combined into one:
- `tool_examples_smart_20k` — 20,000 high-quality reasoning chains
- `tool_examples_20k` — 20,000 diverse tool-use examples  
- `tool_examples_15k` — 15,000 concise examples

**Total: 55,000 examples**

## Available Tools

The dataset covers 5 tool types:
- **read_file** — read local files, source code, configs, logs
- **run_command** — execute shell commands (git, python, ls, grep, etc.)
- **search_web** — search the web for information
- **calculator** — safe mathematical evaluation
- **ask_user** — request clarification from the user

## Use Cases

### Fine-tuning
Designed for LoRA/QLoRA fine-tuning on models like Qwen2.5-Coder. Compatible with HuggingFace `transformers` + `peft` pipeline. Example training script:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Load dataset in JSONL format (bypassing datasets library for compatibility)
import json
examples = []
with open("tool_examples_smart_20k.jsonl") as f:
    for line in f:
        examples.append(json.loads(line))

# Format for training
def format_example(ex):
    text = ""
    for msg in ex["conversations"]:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return text

# Train with Qwen chat template
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
```

### Evaluation
Use the conversation format to evaluate whether models can correctly identify and execute tool-use chains. Test coverage across all 5 tool types and multi-step scenarios.

## Training Details

Trained on: **Qwen/Qwen2.5-Coder-3B-Instruct**  
Config: LoRA r=16, alpha=32, target_modules=[q_proj, k_proj, v_proj, o_proj]  
Effective batch: 16 | max_len: 512 | steps: 1000  

See full training script: [my-ai-stack/Stack-4.0](https://github.com/my-ai-stack/Stack-4.0)

## Citation

```bibtex
@misc{stack-4-0-dataset,
  title={Stack-4.0-Dataset: 55K Agentic Tool-Use Conversations},
  url={https://huggingface.co/datasets/my-ai-stack/Stack-4.0-Dataset},
  author={AI Stack},
  year={2026}
}
```

## License

Apache 2.0 — free to use, modify, and distribute for any purpose.