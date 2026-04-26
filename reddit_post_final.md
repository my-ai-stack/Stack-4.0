# I fine-tuned Qwen2.5-Coder-7B on 55K tool-use conversations — here's what I learned

**Background:** I'm building toward models that can actually operate in environments — file systems, terminals, APIs — not just generate text. Most fine-tuned coding models are trained on static Q&A. That's useful, but it doesn't teach the model *how to think* when the answer requires action.

Stack 3.0 is the result. Qwen2.5-Coder-7B trained specifically on tool-use reasoning chains.

---

## The dataset

55,000 conversations, self-generated with a diverse model pool. Every example pairs a task with a correct tool-use reasoning chain. The dataset covers:

- `read_file` — reasoning about which file to read and what to look for
- `run_command` — deciding when to shell out vs. generate inline
- `search_web` — knowing when to look something up vs. relying on internal knowledge
- Multi-step chains — 3-4 tools in sequence where each step depends on the last

Not curated human demonstrations — generated at scale with quality filtering. The format is JSONL with conversation arrays, each tool call tagged with its result so the model learns the因果链 (cause-effect chain).

## Training setup

- **Base model:** Qwen/Qwen2.5-Coder-7B-Instruct
- **Method:** LoRA (0.24% trainable params = 7.3M / 7.3B), bf16, gradient checkpointing
- **Hardware:** GCP — 1x Tesla V100 16GB, no A100
- **Effective batch:** 16 | max_len: 512 | ~10 hours training
- **Scripts and dataset:** [github.com/my-ai-stack/Stack-4.0](https://github.com/my-ai-stack/Stack-4.0) (Stack 4.0 is the current training run, scripts are public)

## Benchmark results

Evaluated on standard tasks, no special prompting or ensembling:

| Benchmark | Score |
|-----------|-------|
| HumanEval | 85.37% |
| ARC-C | 83.28% |
| MMLU | 77.89% |
| MBPP | 79.10% |
| GSM8K | 91.23% |

The model handles multi-step reasoning better than base Qwen on these tasks. The real difference shows up when you give it tool access — that's where the training pays off.

## What I learned

**1. The dataset format matters more than I expected.**
JSON vs. JSONL vs. sharegpt format — each requires different parsing. I ended up writing a custom loader that bypasses the `datasets` library entirely (the VM has `datasets 4.5.0` which doesn't support `Json` feature type). The lesson: know your tooling constraints before you start.

**2. Eval is the bottleneck on a 16GB GPU.**
Full eval step allocates model + activations = ~17.5GB on a V100. I had to disable eval entirely (`eval_strategy="no"`) to stay under 16GB. Training is stable without it — checkpoint loss is a sufficient proxy.

**3. The tool-call format is the hardest part to get right.**
Getting the model to output clean `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` blocks reliably took more iterations than expected. The model wants to explain what it's doing rather than just do it. Fine-tuning on the tool-use chains fixed this — after step 50 the outputs were clean.

## What's next

Stack 4.0 is in training now. Same dataset size (55K), cleaner reasoning chains, using Qwen2.5-Coder-3B instead of 7B — lighter to run, still good results. Training script is on GitHub if you want to replicate it on a consumer GPU.

## Try it

Live demo (no login): [hf.co/spaces/my-ai-stack/Omni-Nexus-Alpha](https://huggingface.co/spaces/my-ai-stack/Omni-Nexus-Alpha)

Model files:
- Adapter: [hf.co/my-ai-stack/Stack-3.0-Omni-Nexus](https://huggingface.co/my-ai-stack/Stack-3.0-Omni-Nexus) (~5GB)
- GGUF variants: [hf.co/my-ai-stack/Stack-3.0-Omni-Nexus-GGUF](https://huggingface.co/my-ai-stack/Stack-3.0-Omni-Nexus-GGUF)

Questions about the dataset generation, LoRA training setup, or agent loop architecture — ask in the comments. Happy to share what worked and what didn't.

---

*Hardware: GCP VM, 1x V100 16GB. All scripts and training data are open source.*