#!/usr/bin/env python3
"""
Evaluate an OpenAI model on the QiskitHumanEval benchmark.

- Loads tasks from Hugging Face: Qiskit/qiskit_humaneval (default) or qiskit_humaneval_hard
- For each task:
  * Sends the provided prompt to the model
  * Combines the original prompt, the model completion, and the dataset's test code
  * Executes tests in an isolated subprocess with a timeout
- Produces pass@1, a CSV of per-task outcomes, and saves raw generations.

⚠️ This executes LLM-generated Python. Run only in an isolated, throwaway environment.

References:
- Dataset (151 tasks; fields: prompt, entry_point, test, canonical_solution, difficulty_scale): https://huggingface.co/datasets/Qiskit/qiskit_humaneval
- Paper: "Qiskit HumanEval" (arXiv:2406.14712): https://arxiv.org/abs/2406.14712
- OpenAI Python SDK (Responses API): https://github.com/openai/openai-python
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
import textwrap
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# Third-party
from datasets import load_dataset  # pip install datasets
from openai import OpenAI, APIError, APIStatusError, APIConnectionError  # pip install openai

# ----------------------------
# CLI & defaults
# ----------------------------
def build_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate an OpenAI model on QiskitHumanEval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=os.getenv("OPENAI_EVAL_MODEL", "gpt-4.1-mini"),
                   help="OpenAI model id (e.g., gpt-4.1-mini, gpt-4o-mini).")
    p.add_argument("--dataset", default="Qiskit/qiskit_humaneval",
                   choices=["Qiskit/qiskit_humaneval", "Qiskit/qiskit_humaneval_hard"],
                   help="Which dataset variant to use.")
    p.add_argument("--split", default="test", help="Dataset split.")
    p.add_argument("--max-items", type=int, default=None, help="Limit number of tasks for a quick run.")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    p.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top_p.")
    p.add_argument("--max-output-tokens", type=int, default=800, help="Max tokens to generate.")
    p.add_argument("--seed", type=int, default=42, help="Sampling seed (for determinism where supported).")
    p.add_argument("--parallel", type=int, default=1, help="(Sequential driver for reliability) Number of concurrent gens (>=1).")
    p.add_argument("--timeout-sec", type=int, default=45, help="Per-test execution timeout (seconds).")
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
SYSTEM_INSTRUCTIONS = (
    "You are a senior Qiskit+Python developer.\n"
    "Given a prompt that already includes imports and a function signature docstring, "
    "return a FULL, correct Python function implementation that matches the signature. "
    "Requirements:\n"
    "- Output ONLY Python code. No Markdown, no ``` fences, no prose.\n"
    "- Define exactly one function named as specified by the signature.\n"
    "- Assume imports present in the prompt are available; avoid extra imports unless necessary.\n"
    "- Avoid network calls or file I/O.\n"
)

USER_SUFFIX = (
    "\n\n# Implement the required function now.\n"
    "# IMPORTANT: Output ONLY the function definition (no imports, no tests, no comments above the def).\n"
)

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
# OpenAI call
# ----------------------------
def call_openai(client: OpenAI, model: str, prompt: str, temperature: float, top_p: float, max_output_tokens: int, seed: int) -> Tuple[str, Optional[int]]:
    """
    Use the Responses API to generate code. Returns (text, output_token_count|None).
    """
    # The dataset's 'prompt' contains imports + signature; we append user suffix to nudge correct format.
    input_text = prompt + USER_SUFFIX
    t0 = time.time()
    try:
        resp = client.responses.create(
            model=model,
            # responses API accepts either `input=...` or `messages-like` structures
            input=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": input_text},
            ],
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            # seed parameter removed as it's not supported by Responses API
        )
        latency = time.time() - t0
    except (APIStatusError, APIConnectionError, APIError) as e:
        raise RuntimeError(f"OpenAI API error: {e}") from e

    text = (resp.output_text or "").strip()
    # token usage may or may not be present depending on plan/endpoint
    out_tok = None
    try:
        # Prefer resp.usage.output_tokens if available
        if getattr(resp, "usage", None):
            out_tok = getattr(resp.usage, "output_tokens", None)
    except Exception:
        pass

    return text, out_tok

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
            # -I: isolate (no user site); -B: no .pyc; -S: don't import site (reduce side effects)
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
def load_tasks(dataset: str, split: str, limit: Optional[int]) -> List[Task]:
    ds = load_dataset(dataset, split=split)  # requires internet on first run
    tasks: List[Task] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        tasks.append(Task(
            idx=i,
            task_id=row.get("task_id", f"{i}"),
            entry_point=row["entry_point"],
            prompt=row["prompt"],
            test=row["test"],
            difficulty_scale=row.get("difficulty_scale"),
        ))
    return tasks

def evaluate(args: argparse.Namespace) -> None:
    run_ts = now_stamp()
    out_root = Path(args.outdir) / f"{Path(args.dataset).name}_{run_ts}_{args.model.replace('/', '_')}"
    gens_dir = out_root / "generations"
    ensure_dir(gens_dir)

    client = OpenAI()  # reads OPENAI_API_KEY from env

    tasks = load_tasks(args.dataset, args.split, args.max_items)
    print(f"Loaded {len(tasks)} tasks from {args.dataset}:{args.split}")

    results: List[Result] = []

    for t in tasks:
        print(f"\n=== [{t.idx+1}/{len(tasks)}] {t.task_id} :: {t.entry_point} ===")
        gen_path = gens_dir / f"{t.idx:03d}_{t.entry_point}.py"

        # 1) Get / reuse generation
        if args.dry_run and gen_path.exists():
            completion_text = gen_path.read_text(encoding="utf-8")
            output_tokens = None
            print("  (dry-run) Loaded cached completion.")
        else:
            try:
                raw_text, output_tokens = call_openai(
                    client=client,
                    model=args.model,
                    prompt=t.prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_output_tokens=args.max_output_tokens,
                    seed=args.seed,
                )
            except RuntimeError as e:
                print(f"  OpenAI error: {e}")
                raw_text, output_tokens = "", None

            completion_text = extract_code_only(raw_text)
            gen_path.write_text(completion_text, encoding="utf-8")

        # 2) Build executable combined program
        program = EXEC_TEMPLATE.format(
            prompt=t.prompt,
            completion=completion_text,
            test_code=t.test,
            entry_point=t.entry_point,
        )

        # 3) Execute tests
        t0 = time.time()
        passed, err = run_in_subprocess(program, timeout_sec=args.timeout_sec)
        latency = time.time() - t0

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
        )
        results.append(res)
        print(f"  -> {'PASS' if passed else 'FAIL'}{'' if passed else f' | {err}'}")

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
        "pass_at_1": pass_at_1,
        "passed": passed_n,
        "total": total_n,
        "by_difficulty": {
            k: {
                "passed": sum(1 for r in results if r.passed and r.difficulty_scale == k),
                "total": sum(1 for r in results if r.difficulty_scale == k),
            }
            for k in sorted({r.difficulty_scale for r in results})
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts written to: {out_root}")

if __name__ == "__main__":
    args = build_cli()
    evaluate(args)
