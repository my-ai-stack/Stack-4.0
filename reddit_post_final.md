# [Project] I trained a tool-use coding agent on a $23 V100 spot instance. 85.37% HumanEval. Scripts + data open source.

**The problem:** Most fine-tuned coding models learn to answer questions. They don't learn to *act* — read files, run commands, chain tools. I wanted to build the layer that handles reasoning about external actions, not just text generation.

Stack 3.0 is the result.

---

## Benchmarks (no ensembling, no special prompting)

| Benchmark | Score |
|-----------|-------|
| **HumanEval** | **85.37%** |
| **ARC-Challenge** | **83.28%** |
| **MBPP** | **79.80%** |
| GSM8K | 52.39% |
| MMLU | 59.89% |
| HellaSwag | 59.61% |

Trained on tool-use chains — not generic coding Q&A. The model learned to reason about *actions*, not just generate code.

---

## The dataset

55,000 conversations. Every example pairs a task with a correct tool-use reasoning chain. Not curated human demos — generated at scale with quality filtering.

Format: JSONL with conversation arrays. Each tool call is a Bash command with its result, so the model learns when to shell out and what to do with the output.

---

## Training setup

- **Base:** Qwen/Qwen2.5-Coder-7B-Instruct
- **Method:** LoRA (0.24% trainable = 7.3M params), bf16, gradient checkpointing
- **Hardware:** GCP — 1x V100 16GB (spot instance, ~$23 total)
- **Config:** batch=1, grad_accum=16, max_len=512, ~10 hours
- **Scripts + dataset:** [github.com/my-ai-stack/Stack-4.0](https://github.com/my-ai-stack/Stack-4.0)

---

## What I learned

**1. Dataset format matters more than expected.**
The VM has `datasets 4.5.0` which doesn't support `Json` feature type. Ended up writing a custom JSONL loader that bypasses the library entirely. Know your tooling constraints before you start.

**2. Eval OOM'd on a 16GB V100.**
Full eval step allocates model + activations = ~17.5GB on V100. Disabled eval (`eval_strategy="no"`) — checkpoint loss is a sufficient proxy and training is stable.

**3. Getting clean tool-call output required more iteration than expected.**
The model wants to explain what it's doing rather than just do it. After step 50 of fine-tuning on tool-use chains the outputs became clean `<tool_call>` blocks.

---

## What's next

Stack 4.0 is training now — same dataset size, cleaner reasoning chains, using Qwen2.5-Coder-3B. Lighter to run, same approach. Training script is on GitHub if you want to replicate on a consumer GPU.

---

## Try it locally (Ollama)

```bash
ollama run hf.co/my-ai-stack/Stack-3.0-Omni-Nexus-GGUF
```

Model files:
- GGUF (Q8_0, 7.6GB): [hf.co/my-ai-stack/Stack-3.0-Omni-Nexus-GGUF](https://huggingface.co/my-ai-stack/Stack-3.0-Omni-Nexus-GGUF)
- LoRA adapter: [hf.co/my-ai-stack/Stack-3.0-Omni-Nexus](https://huggingface.co/my-ai-stack/Stack-3.0-Omni-Nexus)

All benchmarks evaluated via lm-evaluation-harness on a Tesla V100 16GB.

Happy to answer questions about dataset generation, training setup, or agent loop architecture.

---

*All scripts and training data are open source. GCP VM, 1x V100 16GB, ~$23 total compute cost.*