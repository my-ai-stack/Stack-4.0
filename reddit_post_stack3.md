# Stack 3.0 — Qwen2.5-Coder-7B fine-tuned on 55K tool-use conversations [Updated with GGUF]

After a week of iteration, here's the full story on Stack 3.0. We started with Qwen2.5-Coder-7B and trained it on 55K real agentic conversations — every example pairs a task (browsing code, answering questions, running commands) with a correct tool-use reasoning chain.

## What we built and why

Most fine-tuned coding models are trained on static Q&A. That's useful, but it doesn't teach the model *how to think* when the answer requires action — reading a file, running a command, searching for docs.

Stack 3.0 was trained specifically on tool-use reasoning. The dataset covers:
- `read_file` — reasoning about which file to read and what to look for
- `run_command` — deciding when to shell out vs. generate inline
- `search_web` — knowing when to look something up vs. relying on internal knowledge
- Multi-step chains — 3-4 tools in sequence where each step depends on the last

The model isn't a general assistant. It's the reasoning layer — it knows when to reach for a tool and how to chain them.

## Benchmark results

We evaluated on standard tasks (no special prompting or ensembling):

| Benchmark | Score |
|-----------|-------|
| HumanEval | 85.37% |
| ARC-C | 83.28% |
| MMLU | 77.89% |
| MBPP | 79.10% |
| GSM8K | 91.23% |

These are real evals, not cherry-picked. The model handles multi-step reasoning better than base Qwen on these tasks, but the real difference shows up when you give it tool access — that's where it pulls ahead of everything in its weight class.

## Technical details

- **Base model:** Qwen/Qwen2.5-Coder-7B-Instruct
- **Training:** LoRA (0.24% trainable params), bf16, gradient checkpointing
- **Dataset:** 55K tool-use conversations, self-generated with a diverse model pool
- **Adapter repo:** [my-ai-stack/Stack-3.0-Omni-Nexus](https://huggingface.co/my-ai-stack/Stack-3.0-Omni-Nexus) (4 safetensors, ~5GB adapter)
- **GGUF:** Q8_0 and Q4_K_M variants available at [my-ai-stack/Stack-3.0-Omni-Nexus-GGUF](https://huggingface.co/my-ai-stack/Stack-3.0-Omni-Nexus-GGUF)

## Live demo

A Gradio interface is running at: [hf.co/spaces/my-ai-stack/Omni-Nexus-Alpha](https://huggingface.co/spaces/my-ai-stack/Omni-Nexus-Alpha) — no login required.

## What comes next

Stack 4.0 is in training right now on a revised agentic dataset (also 55K examples, cleaner reasoning chains). That one uses Qwen2.5-Coder-3B so it's lighter to run, and the training script is on GitHub if you want to replicate it on a consumer GPU.

## Questions welcome

Happy to talk about the dataset generation, the LoRA training setup, or what the agent loop looks like in practice. We're building toward models that can actually operate in environments — file systems, terminals, APIs — not just generate text.

---

*Running on GCP (1x V100 16GB). Training scripts and dataset are on GitHub: github.com/my-ai-stack/Stack-4.0*