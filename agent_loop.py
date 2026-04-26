"""
Stack 4.0 — Agentic Inference Loop
==================================
The actual product. This is what makes Stack 4.0 a working agent, not just weights.

The loop:
    User message → Model → [Tool call?] → Execute → Loop back → Final response

Supports 5 tools out of the box:
    • read_file     — read any file
    • run_command   — execute shell commands  
    • search_web    — search the web ( Brave API)
    • ask_user      — ask the user a follow-up question
    • calculator    — safe math evaluation

Usage:
    # Basic
    python3 agent_loop.py

    # With custom model
    python3 agent_loop.py --adapter my-ai-stack/Stack-4.0-Qwen-3B-Agentic

    # With Groq API key (bypass local model)
    GROQ_API_KEY=xxx python3 agent_loop.py --mode groq
"""

import os
import sys
import json
import re
import math
import subprocess
import argparse
from typing import Optional, Literal
from dataclasses import dataclass, field

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ─── Config ──────────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "my-ai-stack/Stack-4.0-Qwen-3B-Agentic")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MAX_LOOP_ITERS = 10
MAX_TOKENS = 512
TEMPERATURE = 0.7

# ─── Tool definitions ─────────────────────────────────────────────────────────
TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the complete contents of a file from the local filesystem. Use for source code, configs, logs, or any text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Full path to the file to read"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command and return the stdout + stderr output. Use for git, python, ls, grep, or any terminal command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"},
                    "cwd": {"type": "string", "description": "Working directory for the command (optional)"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information using Brave Search API. Returns titles, URLs and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "count": {"type": "integer", "description": "Number of results (default 5, max 10)"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Ask the user a follow-up question or for clarification. Use when you need additional information to proceed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to ask the user"}
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression safely. Use for arithmetic, algebra, or any computation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The math expression to evaluate, e.g. '2+2', 'sqrt(16)', '3**2'"}
                },
                "required": ["expression"]
            }
        }
    },
]

TOOL_MAP = {d["function"]["name"]: d for d in TOOL_DEFS}


# ─── Tool Implementations ─────────────────────────────────────────────────────
def tool_read_file(path: str) -> dict:
    """Read a file from the filesystem."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"ok": True, "content": content[:3000]}  # truncate long files
    except FileNotFoundError:
        return {"ok": False, "error": f"File not found: {path}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_run_command(command: str, cwd: Optional[str] = None) -> dict:
    """Execute a shell command."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=60, cwd=cwd
        )
        return {
            "ok": True,
            "returncode": result.returncode,
            "stdout": result.stdout[:2000],
            "stderr": result.stderr[:500],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Command timed out after 60 seconds"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_search_web(query: str, count: int = 5) -> dict:
    """Search the web using Brave API."""
    try:
        import requests
        api_key = os.environ.get("BRAVE_API_KEY", "")
        if not api_key:
            return {"ok": False, "error": "BRAVE_API_KEY not set. Get one at https://brave.com/search/api/"}
        
        resp = requests.get(
            "https://api.search.brave.com/rest/v1/search",
            params={"q": query, "count": count},
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
            timeout=10,
        )
        data = resp.json()
        results = [
            {"title": r.get("title",""), "url": r.get("url",""), "snippet": r.get("description","")}
            for r in data.get("web", {}).get("results", [])[:count]
        ]
        return {"ok": True, "results": results, "query": query}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_ask_user(question: str) -> dict:
    """Ask the user a question. Handled specially in the loop."""
    return {"ok": True, "question": question, "type": "user_input_required"}


def tool_calculator(expression: str) -> dict:
    """Evaluate a math expression safely."""
    try:
        # Only allow math operations - no arbitrary code
        allowed = set("0123456789+-*/().e ,")
        if not all(c in allowed for c in expression.replace(" ", "")):
            return {"ok": False, "error": "Invalid characters in expression"}
        result = eval(expression, {"__builtins__": {}, "sqrt": math.sqrt, "pi": math.pi, "e": math.e})
        return {"ok": True, "result": float(result), "expression": expression}
    except Exception as e:
        return {"ok": False, "error": f"Evaluation error: {e}"}


TOOL_IMPLS = {
    "read_file": tool_read_file,
    "run_command": tool_run_command,
    "search_web": tool_search_web,
    "ask_user": tool_ask_user,
    "calculator": tool_calculator,
}


# ─── Agent Loop Core ──────────────────────────────────────────────────────────
@dataclass
class Message:
    role: str
    content: str
    tool_calls: Optional[list] = None


@dataclass 
class ToolCall:
    id: str
    name: str
    arguments: dict


class AgentLoop:
    """
    The core agentic inference loop.
    Loads model + adapter, then handles multi-turn tool-use conversations.
    """

    def __init__(self, mode: str = "local", adapter_repo: str = ADAPTER_REPO, base_model: str = BASE_MODEL):
        self.mode = mode
        self.adapter_repo = adapter_repo
        self.base_model = base_model
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.mode == "groq":
            print("Using Groq API for inference")
            return
        print(f"Loading {self.base_model} + adapter {self.adapter_repo}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=True, token=HF_TOKEN
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
        )
        self.model = PeftModel.from_pretrained(base_model, self.adapter_repo, token=HF_TOKEN)
        self.model.eval()
        print("✅ Model ready!")

    def generate(self, messages: list[dict], max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE) -> str:
        """Generate a response given conversation history."""
        if self.mode == "groq":
            return self._groq_generate(messages, max_tokens, temperature)

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def _groq_generate(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        """Use Groq API for inference."""
        import requests
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=30,
        )
        return resp.json()["choices"][0]["message"]["content"]

    def parse_tool_calls(self, text: str) -> list[ToolCall]:
        """Parse tool calls from model output. Supports multiple formats."""
        calls = []
        
        # Try JSON format first: {"name": "...", "arguments": {...}}
        try:
            # Handle <tool_call>...</tool_call> blocks
            for block in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
                data = json.loads(block.strip())
                calls.append(ToolCall(
                    id=f"call_{len(calls)}",
                    name=data["name"],
                    arguments=data.get("arguments", {})
                ))
        except (json.JSONDecodeError, KeyError):
            pass

        # Try markdown code block format
        try:
            for block in re.findall(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL):
                data = json.loads(block.strip())
                if isinstance(data, dict) and "name" in data:
                    calls.append(ToolCall(
                        id=f"call_{len(calls)}",
                        name=data["name"],
                        arguments=data.get("arguments", {})
                    ))
        except json.JSONDecodeError:
            pass

        return calls

    def execute_tool(self, tool_call: ToolCall) -> dict:
        """Execute a single tool call and return the result."""
        name = tool_call.name
        args = tool_call.arguments
        
        if name not in TOOL_IMPLS:
            return {"ok": False, "error": f"Unknown tool: {name}"}
        
        impl = TOOL_IMPLS[name]
        
        # Special case: ask_user needs interactive input
        if name == "ask_user":
            print(f"\n🤖 Agent asks: {args.get('question','')}")
            answer = input("\n👤 Your answer: ").strip()
            return {"ok": True, "answer": answer}
        
        try:
            result = impl(**args)
            return result
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def format_tool_result(self, tool_call: ToolCall, result: dict) -> str:
        """Format a tool result as a string for the conversation."""
        if result.get("type") == "user_input_required":
            return f"User answered: {result.get('answer','')}"
        
        if not result.get("ok"):
            return f"[Error: {result.get('error', 'unknown')}]"
        
        # Format based on tool type
        name = tool_call.name
        if name == "read_file":
            return result.get("content", "")[:1500] + ("..." if len(str(result.get("content",""))) > 1500 else "")
        elif name == "run_command":
            out = result.get("stdout","") or result.get("stderr","")
            return out[:1500]
        elif name == "search_web":
            items = result.get("results", [])
            if not items:
                return "No results found."
            lines = [f"- {r['title']}: {r['url']}" for r in items[:5]]
            return "\n".join(lines)
        elif name == "calculator":
            return str(result.get("result", "error"))
        else:
            return str(result)[:500]

    def run(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """
        Run the full agent loop for a user message.
        Returns the final response string.
        """
        # Build conversation
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        loop_count = 0
        while loop_count < MAX_LOOP_ITERS:
            loop_count += 1

            # Generate
            response_text = self.generate(messages)
            print(f"\n[Loop {loop_count}] Model output:\n{response_text[:300]}")

            # Parse tool calls
            tool_calls = self.parse_tool_calls(response_text)

            if not tool_calls:
                # No tool calls — this is the final response
                return response_text.strip()

            # Remove the assistant message with tool calls and append separately
            assistant_msg = {"role": "assistant", "content": response_text}
            messages.append(assistant_msg)

            # Execute each tool call and add results
            for tc in tool_calls:
                print(f"\n🔧 Executing tool: {tc.name}({tc.arguments})")
                result = self.execute_tool(tc)
                formatted = self.format_tool_result(tc, result)
                
                # Add tool result as a system/tool message
                messages.append({
                    "role": "tool",
                    "content": f"Tool '{tc.name}' returned: {formatted}",
                    "name": tc.name,
                    "tool_call_id": tc.id,
                })

            # Continue loop — model will see tool results and respond

        # Max iterations reached
        return messages[-1]["content"] if messages else "Max iterations reached."


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stack 4.0 Agentic Loop")
    parser.add_argument("--mode", choices=["local", "groq"], default=os.environ.get("AGENT_MODE", "local"),
                        help="local = use LoRA adapter; groq = use Groq API")
    parser.add_argument("--adapter", default=ADAPTER_REPO, help="HF repo for LoRA adapter")
    parser.add_argument("--base", default=BASE_MODEL, help="Base model name")
    parser.add_argument("--system", default=None, help="System prompt")
    parser.add_argument("prompt", nargs="*", help="Initial user message")
    args = parser.parse_args()

    agent = AgentLoop(mode=args.mode, adapter_repo=args.adapter, base_model=args.base)

    if args.mode == "local":
        agent.load()
    elif args.mode == "groq" and not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set. Get one at https://console.groq.com")
        sys.exit(1)

    print("\n" + "="*60)
    print("🧠 Stack 4.0 — Agentic Loop")
    print("="*60)
    print(f"Mode: {'Groq API' if args.mode == 'groq' else f'Local: {args.adapter}'}")
    print(f"Tools: {list(TOOL_MAP.keys())}")
    print("Type 'quit' or Ctrl+C to exit\n")

    system_prompt = args.system or (
        "You are Stack 4.0, an AI assistant trained on 55K agentic examples. "
        "You have access to tools. Use them when helpful. "
        "When you need to use a tool, output it as JSON inside <tool_call>...</tool_call> tags. "
        "After receiving tool results, continue the conversation with your analysis or next action."
    )

    if args.prompt:
        prompt = " ".join(args.prompt)
        print(f"\n👤 {prompt}")
        result = agent.run(prompt, system_prompt)
        print(f"\n🤖 {result}")
        return

    # Interactive mode
    print("\n📌 Example prompts to try:")
    print("  • 'Read the file /etc/hosts'")
    print("  • 'What files are in the current directory?'")
    print("  • 'Search the web for best practices for LLM fine-tuning'")
    print("  • 'Calculate 2**10'")
    print()

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            result = agent.run(user_input, system_prompt)
            print(f"\n🤖 Agent: {result}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    main()