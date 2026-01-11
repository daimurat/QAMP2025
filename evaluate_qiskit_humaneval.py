#!/usr/bin/env python3
"""
Evaluate an OpenAI model on the QiskitHumanEval benchmark with RAG support.

This script integrates CLAPP's RAG backend with the QiskitHumanEval evaluation framework.
- Loads tasks from Hugging Face: Qiskit/qiskit_humaneval (default) or qiskit_humaneval_hard
- Uses RAG to retrieve relevant Qiskit documentation for context
- For each task:
  * Retrieves context from vector store (if RAG enabled)
  * Sends the prompt with context to the model
  * Combines the original prompt, the model completion, and the dataset's test code
  * Executes tests in an isolated subprocess with a timeout
- Produces pass@1, a CSV of per-task outcomes, and saves raw generations.

⚠️ This executes LLM-generated Python. Run only in an isolated, throwaway environment.

References:
- Dataset: https://huggingface.co/datasets/Qiskit/qiskit_humaneval
- Paper: "Qiskit HumanEval" (arXiv:2406.14712): https://arxiv.org/abs/2406.14712
- CLAPP: Multi-agent system with RAG for Qiskit code generation
"""

from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List

# Third-party
from datasets import load_dataset
from dotenv import load_dotenv

# RAG
from rag import RAGRetriever

# Groq for LLM
from groq import Groq

# ----------------------------
# CLI & defaults
# ----------------------------
def build_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate an OpenAI model on QiskitHumanEval with RAG support.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=os.getenv("GROQ_EVAL_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct"),
                   help="Groq model id (e.g., moonshotai/kimi-k2-instruct-0905).")
    p.add_argument("--dataset", default="Qiskit/qiskit_humaneval",
                   choices=["Qiskit/qiskit_humaneval", "Qiskit/qiskit_humaneval_hard"],
                   help="Which dataset variant to use.")
    p.add_argument("--split", default="test", help="Dataset split.")
    p.add_argument("--max-items", type=int, default=None, help="Limit number of tasks for a quick run.")
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    p.add_argument("--max-output-tokens", type=int, default=800, help="Max tokens to generate.")
    p.add_argument("--timeout-sec", type=int, default=45, help="Per-test execution timeout (seconds).")
    p.add_argument("--use-rag", action="store_true", default=True, help="Use RAG for context retrieval.")
    p.add_argument("--no-rag", dest="use_rag", action="store_false", help="Disable RAG.")
    p.add_argument("--rag-top-k", type=int, default=5, help="Number of documents to retrieve for RAG.")
    p.add_argument("--db-path", default="QAMP/data/qamp.db", help="Path to SQLite vector database.")
    p.add_argument("--difficulty", default=None,
                   choices=["basic", "intermediate", "advanced"],
                   help="Filter tasks by difficulty level.")
    p.add_argument("--outdir", default="out", help="Directory to write logs/artifacts.")
    p.add_argument("--dry-run", action="store_true", help="Skip model calls and reuse previous generations if present.")
    return p.parse_args()

# ----------------------------
# Data types
# ----------------------------
@dataclass
class Task:
    idx: int
    task_id: str
    entry_point: str
    prompt: str
    test: str
    difficulty_scale: Optional[str] = None

@dataclass
class Result:
    task_id: str
    entry_point: str
    passed: bool
    error: Optional[str]
    gen_tokens: Optional[int]
    prompt_chars: int
    completion_chars: int
    latency_s: float
    difficulty_scale: Optional[str]
    model: str
    file_path: str
    rag_used: bool

# ----------------------------
# Utility: file safe writing
# ----------------------------
def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Prompting helpers
# ----------------------------
SYSTEM_INSTRUCTIONS = """You are a senior Qiskit+Python developer.
Given a prompt that already includes imports and a function signature docstring, 
return a FULL, correct Python function implementation that matches the signature.

Requirements:
- Output ONLY Python code. No Markdown, no ``` fences, no prose.
- Define exactly one function named as specified by the signature.
- Assume imports present in the prompt are available; avoid extra imports unless necessary.
- Avoid network calls or file I/O.
- Focus on correctness and completeness.
"""

USER_SUFFIX = """

Implement the required function now.
IMPORTANT: Output ONLY the function definition (no imports, no tests, no comments above the def).
"""

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)

def extract_code_only(text: str) -> str:
    """
    Try to robustly extract Python code from an LLM response:
    - Prefer fenced code blocks if present.
    - Otherwise, return the raw text (already instructed to be code-only).
    """
    m = CODE_BLOCK_RE.search(text)
    return m.group(1).strip() if m else text.strip()

# ----------------------------
# LLM call
# ----------------------------
def call_llm(
    client: Groq,
    model: str,
    prompt: str,
    context: str = "",
    temperature: float = 0.2,
    max_output_tokens: int = 800,
) -> Tuple[str, Optional[int], float]:
    """
    Call LLM to generate code. Returns (text, output_token_count|None, latency).
    """
    # Build user message
    if context:
        user_content = f"""Context from Qiskit documentation:
{context}

Task:
{prompt}{USER_SUFFIX}"""
    else:
        user_content = f"{prompt}{USER_SUFFIX}"
    
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": user_content},
    ]
    
    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_output_tokens,
            top_p=1,
            stream=False,
            stop=None,
        )
        latency = time.time() - t0
        
        # Extract text from response
        text = response.choices[0].message.content.strip()
        
        # Get token count
        output_tokens = None
        if response.usage:
            output_tokens = response.usage.completion_tokens
        
        return text, output_tokens, latency
    except Exception as e:
        latency = time.time() - t0
        raise RuntimeError(f"LLM API error: {e}") from e

# ----------------------------
# Execution harness
# ----------------------------
EXEC_TEMPLATE = """\
# === BEGIN PROMPT (dataset) ===
{prompt}

# === BEGIN MODEL COMPLETION ===
{completion}

# === BEGIN TEST CODE (dataset) ===
{test_code}

# === HARNESS ===
def __run_check():
    # Import the solution function by name and run dataset's check()
    return check({entry_point})

if __name__ == "__main__":
    try:
        __run_check()
        print("___QHE_PASS___")
    except Exception as e:
        print("___QHE_FAIL___:" + repr(e))
"""

def run_in_subprocess(code: str, timeout_sec: int) -> Tuple[bool, Optional[str]]:
    """
    Execute the provided code string in a fresh Python subprocess with a timeout.
    Returns (passed, error_str_if_any).

    We deliberately avoid importing this code in the current process.
    """
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "eval_task.py"
        src.write_text(code, encoding="utf-8")

        try:
            # -I: isolate (no user site); -B: no .pyc
            p = subprocess.run(
                [sys.executable, "-I", "-B", str(src)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                text=True,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            return False, f"Timeout({timeout_sec}s)"
        except Exception as e:
            return False, f"SubprocessError: {e!r}"

        out = (p.stdout or "").strip()
        if "___QHE_PASS___" in out:
            return True, None
        # Try to capture explicit failure reason
        m = re.search(r"___QHE_FAIL___:(.*)$", out, flags=re.M | re.S)
        return False, (m.group(1).strip() if m else f"RuntimeError: {out[:5000]}")

# ----------------------------
# Main evaluation loop
# ----------------------------
def load_tasks(dataset: str, split: str, limit: Optional[int], difficulty: Optional[str] = None) -> List[Task]:
    ds = load_dataset(dataset, split=split)
    tasks: List[Task] = []
    for i, row in enumerate(ds):
        # Filter by difficulty if specified
        if difficulty is not None:
            task_difficulty = row.get("difficulty_scale", "").lower()
            if task_difficulty != difficulty.lower():
                continue
        
        if limit is not None and len(tasks) >= limit:
            break
        tasks.append(Task(
            idx=len(tasks),
            task_id=row.get("task_id", f"{i}"),
            entry_point=row["entry_point"],
            prompt=row["prompt"],
            test=row["test"],
            difficulty_scale=row.get("difficulty_scale"),
        ))
    return tasks

def evaluate(args: argparse.Namespace) -> None:
    run_ts = now_stamp()
    rag_suffix = "_rag" if args.use_rag else "_norag"
    out_root = Path(args.outdir) / f"{Path(args.dataset).name}_{run_ts}_{args.model.replace('/', '_')}{rag_suffix}"
    gens_dir = out_root / "generations"
    ensure_dir(gens_dir)

    # Initialize Groq client
    client = Groq()
    print(f"✓ Initialized Groq client with model: {args.model}")

    # Initialize RAG retriever
    retriever = None
    if args.use_rag:
        try:
            retriever = RAGRetriever(db_path=args.db_path)
            print(f"✓ Loaded RAG from {args.db_path}")
        except Exception as e:
            print(f"⚠️  RAG requested but failed to load: {e}")
            args.use_rag = False

    # Load tasks
    tasks = load_tasks(args.dataset, args.split, args.max_items, args.difficulty)
    difficulty_str = f" (difficulty: {args.difficulty})" if args.difficulty else ""
    print(f"✓ Loaded {len(tasks)} tasks from {args.dataset}:{args.split}{difficulty_str}")
    print(f"✓ RAG: {'enabled' if args.use_rag else 'disabled'}")
    print(f"✓ Output directory: {out_root}\n")

    results: List[Result] = []

    for t in tasks:
        print(f"=== [{t.idx+1}/{len(tasks)}] {t.task_id} :: {t.entry_point} ===")
        gen_path = gens_dir / f"{t.idx:03d}_{t.entry_point}.py"

        # 1) Get / reuse generation
        if args.dry_run and gen_path.exists():
            completion_text = gen_path.read_text(encoding="utf-8")
            output_tokens = None
            latency = 0.0
            print("  (dry-run) Loaded cached completion.")
        else:
            # Retrieve context if RAG enabled
            context = ""
            if args.use_rag and retriever:
                print("  Retrieving context...")
                context = retriever.retrieve_context(t.prompt, top_k=args.rag_top_k)
                print(f"  Retrieved {len(context)} chars of context")

            # Generate code
            try:
                print("  Generating code...")
                raw_text, output_tokens, latency = call_llm(
                    client=client,
                    model=args.model,
                    prompt=t.prompt,
                    context=context,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
            except RuntimeError as e:
                print(f"  LLM error: {e}")
                raw_text, output_tokens, latency = "", None, 0.0

            completion_text = extract_code_only(raw_text)
            gen_path.write_text(completion_text, encoding="utf-8")
            print(f"  Generated {len(completion_text)} chars in {latency:.2f}s")

        # 2) Build executable combined program
        program = EXEC_TEMPLATE.format(
            prompt=t.prompt,
            completion=completion_text,
            test_code=t.test,
            entry_point=t.entry_point,
        )

        # 3) Execute tests
        print("  Executing tests...")
        t0 = time.time()
        passed, err = run_in_subprocess(program, timeout_sec=args.timeout_sec)
        exec_latency = time.time() - t0

        # 4) Record result
        res = Result(
            task_id=t.task_id,
            entry_point=t.entry_point,
            passed=passed,
            error=err,
            gen_tokens=output_tokens,
            prompt_chars=len(t.prompt),
            completion_chars=len(completion_text),
            latency_s=latency,
            difficulty_scale=t.difficulty_scale,
            model=args.model,
            file_path=str(gen_path),
            rag_used=args.use_rag,
        )
        results.append(res)
        status = "✓ PASS" if passed else f"✗ FAIL"
        print(f"  {status} (exec: {exec_latency:.2f}s)")
        if not passed and err:
            print(f"  Error: {err[:200]}")
        print()

    # ---------------- Summary & persistence ----------------
    passed_n = sum(1 for r in results if r.passed)
    total_n = len(results)
    pass_at_1 = passed_n / total_n if total_n else 0.0

    # Save CSV
    csv_path = out_root / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))

    # Save summary JSON
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "timestamp": run_ts,
        "rag_enabled": args.use_rag,
        "rag_top_k": args.rag_top_k if args.use_rag else None,
        "temperature": args.temperature,
        "pass_at_1": pass_at_1,
        "passed": passed_n,
        "total": total_n,
        "by_difficulty": {
            k: {
                "passed": sum(1 for r in results if r.passed and r.difficulty_scale == k),
                "total": sum(1 for r in results if r.difficulty_scale == k),
            }
            for k in sorted({r.difficulty_scale for r in results if r.difficulty_scale})
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(json.dumps(summary, indent=2))
    print(f"\n✓ Artifacts written to: {out_root}")
    print(f"✓ Results CSV: {csv_path}")
    print(f"✓ Summary JSON: {out_root / 'summary.json'}")

if __name__ == "__main__":
    load_dotenv()
    args = build_cli()
    evaluate(args)

# Made with Bob
