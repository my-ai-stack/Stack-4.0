# Stack 4.0 — Omni-Nexus Agentic

**The first open-source agentic model deployable for free on Cloudflare Workers AI.**

Built on Meta Llama 3.1 8B, fine-tuned with agentic tool-use examples and reasoning traces.

## Training Data

- **55,000 agentic examples** with tool calls, reasoning traces, and task completion loops — hosted on [HuggingFace Dataset](https://huggingface.co/my-ai-stack/Stack-4.0-Dataset)
- 60,000 Stack 3.0 training examples reformatted for Llama chat template
- SWE-bench task completion examples

## Benchmark Targets

| Benchmark | Llama 3.1 Base | + Stack 4.0 LoRA | Target |
|-----------|---------------|-----------------|--------|
| HumanEval | ~62% | TBD | >70% |
| MMLU | ~68% | TBD | >72% |
| ARC-C | ~65% | TBD | >70% |
| MBPP | ~65% | TBD | >72% |

## Agentic Training

Stack 4.0 is trained not just to respond — but to **complete tasks**:
- **Plan** — break tasks into steps
- **Use tools** — file read/write, terminal, browser
- **Self-correct** — see output, fix errors
- **Complete** — not just suggest, but finish

## Deployment

Once trained, deploy to **Cloudflare Workers AI for free** — no GPU required at inference time.

## Key Links

- [Training Dataset](https://huggingface.co/my-ai-stack/Stack-4.0-Dataset)
- [Model Weights](https://huggingface.co/my-ai-stack/Stack-4.0-Omni-Nexus-Agentic)
- [Live Demo](https://huggingface.co/spaces/my-ai-stack/Omni-Nexus-Alpha)
- [Stack 3.0 (Qwen)](https://github.com/my-ai-stack/Stack-3.0)

## Hardware

Trained on Google Cloud Tesla V100 16GB via Axolotl LoRA pipeline.
