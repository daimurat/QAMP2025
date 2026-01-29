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
from typing import Optional, Tuple, List, Dict

# Third-party
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

from src.application.workflows.deep_thought import run_deep_thought_mode
from config.constants import OPENROUTER_BASE_URL, OPENROUTER_MODELS, OPENROUTER_PROVIDER_BODY
from src.utils.trace_logger import TraceLogger, ModelConfig

# ----------------------------
# CLI & defaults
# ----------------------------

def build_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate an OpenAI model on QiskitHumanEval with RAG support.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default=os.getenv("GROQ_EVAL_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct"),
        help="Groq model id (e.g., moonshotai/kimi-k2-instruct-0905).",
    )
    p.add_argument(
        "--dataset",
        default="Qiskit/qiskit_humaneval",
        choices=["Qiskit/qiskit_humaneval", "Qiskit/qiskit_humaneval_hard"],
        help="Which dataset variant to use.",
    )
    p.add_argument("--split", default="test", help="Dataset split.")
    p.add_argument("--max-items", type=int, default=None, help="Limit number of tasks for a quick run.")
    p.add_argument(
        "--mode",
        choices=["deep", "fast"],
        default="deep",
        help="Generation mode: deep=multi-agent (default), fast=single-turn no-RAG.",
    )
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    p.add_argument("--max-output-tokens", type=int, default=800, help="Max tokens to generate.")
    p.add_argument("--timeout-sec", type=int, default=45, help="Per-test execution timeout (seconds).")
    p.add_argument("--use-rag", action="store_true", default=True, help="Use RAG for context retrieval.")
    p.add_argument("--no-rag", dest="use_rag", action="store_false", help="Disable RAG.")
    p.add_argument("--rag-top-k", type=int, default=5, help="Number of documents to retrieve for RAG.")
    p.add_argument(
        "--rag-min-score",
        type=float,
        default=0.0,
        help="Minimum similarity score for RAG chunks (0.0-1.0). Chunks below this are discarded.",
    )
    p.add_argument("--db-path", default="data/qamp.db", help="Path to SQLite vector database.")
    p.add_argument(
        "--task-id",
        default=None,
        help="Run only a specific task by ID (e.g., '47' or 'qiskitHumanEval/47').",
    )
    p.add_argument(
        "--difficulty",
        default=None,
        choices=["basic", "intermediate", "difficult"],
        help="Filter tasks by difficulty level.",
    )
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
    prompt_chars: int
    completion_chars: int
    latency_s: float
    difficulty_scale: Optional[str]
    model: str
    file_path: str
    mode: str
    rag_used: bool
    rag_chunks_used: int


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
# Fast mode helpers (no RAG, single-turn)
# ----------------------------

def _resolve_chat_settings(selected_model: str, api_key_openai: Optional[str], api_key_openrouter: Optional[str]) -> Dict:
    """
    Build OpenAI client settings for OpenAI or OpenRouter models.
    """
    use_openrouter = selected_model in OPENROUTER_MODELS or selected_model.startswith("openai/")
    if use_openrouter:
        key = api_key_openrouter or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY missing for OpenRouter model.")
        return {
            "api_key": key,
            "base_url": OPENROUTER_BASE_URL,
            "extra_body": OPENROUTER_PROVIDER_BODY,
        }

    key = api_key_openai or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing for OpenAI model.")
    return {
        "api_key": key,
        "base_url": None,
        "extra_body": None,
    }



def _build_fast_messages(task: Task) -> List[Dict[str, str]]:
    """
    Compose a simple, focused prompt for single-turn generation.
    """
    system_msg = (
        "You are an expert Python+Qiskit engineer. "
        "Write a correct, self-contained implementation for the requested entry point. "
        "Respond with a single Python code block only—no explanations."
    )
    user_msg = (
        f"Problem statement:\n{task.prompt}\n\n"
        f"Implement the function `{task.entry_point}`. "
        "Include any needed imports or helper functions."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]



def generate_fast_completion(
    task: Task,
    selected_model: str,
    temperature: float,
    max_output_tokens: int,
    api_key_openai: Optional[str],
    api_key_openrouter: Optional[str],
) -> Tuple[str, float]:
    """
    Single-turn generation without RAG or multi-agent orchestration.
    """
    settings = _resolve_chat_settings(selected_model, api_key_openai, api_key_openrouter)
    client_kwargs = {"api_key": settings["api_key"]}
    if settings["base_url"]:
        client_kwargs["base_url"] = settings["base_url"]
    client = OpenAI(**client_kwargs)

    messages = _build_fast_messages(task)
    start = time.time()
    resp = client.chat.completions.create(
        model=selected_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_output_tokens,
        extra_body=settings["extra_body"],
    )
    latency = time.time() - start
    content = resp.choices[0].message.content if resp and resp.choices else ""
    return extract_code_only(content or ""), latency


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

def load_tasks(
    dataset: str,
    split: str,
    limit: Optional[int],
    difficulty: Optional[str] = None,
    task_id: Optional[str] = None,
) -> List[Task]:
    ds = load_dataset(dataset, split=split)
    tasks: List[Task] = []
    for i, row in enumerate(ds):
        row_task_id = row.get("task_id", f"{i}")

        # Filter by task_id if specified (supports "47" or "qiskitHumanEval/47")
        if task_id is not None:
            # Normalize: if task_id is just a number, match the suffix
            if task_id.isdigit():
                if not row_task_id.endswith(f"/{task_id}"):
                    continue
            elif row_task_id != task_id:
                continue

        # Filter by difficulty if specified
        if difficulty is not None:
            task_difficulty = row.get("difficulty_scale", "").lower()
            if task_difficulty != difficulty.lower():
                continue

        if limit is not None and len(tasks) >= limit:
            break
        tasks.append(
            Task(
                idx=len(tasks),
                task_id=row_task_id,
                entry_point=row["entry_point"],
                prompt=row["prompt"],
                test=row["test"],
                difficulty_scale=row.get("difficulty_scale"),
            )
        )
    return tasks


def evaluate(args: argparse.Namespace) -> None:
    run_ts = now_stamp()
    rag_enabled = args.use_rag and args.mode == "deep"
    rag_suffix = "_rag" if rag_enabled else "_norag"
    out_root = Path(args.outdir) / f"{Path(args.dataset).name}_{run_ts}_{args.model.replace('/', '_')}_{args.mode}{rag_suffix}"
    gens_dir = out_root / "generations"
    traces_dir = out_root / "traces"
    ensure_dir(gens_dir)
    ensure_dir(traces_dir)

    # Load tasks
    tasks = load_tasks(args.dataset, args.split, args.max_items, args.difficulty, args.task_id)
    difficulty_str = f" (difficulty: {args.difficulty})" if args.difficulty else ""
    print(f"✓ Loaded {len(tasks)} tasks from {args.dataset}:{args.split}{difficulty_str}")
    print(f"✓ Mode: {args.mode}")
    print(f"✓ RAG: {'enabled' if rag_enabled else 'disabled'}")
    if rag_enabled:
        print(f"✓ RAG debug log: {out_root / 'rag_debug.log'}")
    elif args.mode == "fast" and args.use_rag:
        print("  (note) Fast mode ignores RAG even if --use-rag is set.")
    print(f"✓ Traces directory: {traces_dir}")
    print(f"✓ Output directory: {out_root}\n")

    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    results: List[Result] = []

    # Initialize progress tracker
    progress_path = out_root / "progress.log"
    with progress_path.open("w", encoding="utf-8") as pf:
        pf.write(f"# Evaluation Progress - {run_ts}\n")
        pf.write(f"# Model: {args.model}\n")
        pf.write(f"# Dataset: {args.dataset} ({len(tasks)} tasks)\n")
        pf.write(f"# Mode: {args.mode}\n")
        pf.write(f"# RAG: {'enabled' if rag_enabled else 'disabled'}\n")
        pf.write("=" * 60 + "\n\n")
    print(f"✓ Progress tracker: {progress_path}\n")

    for t in tasks:
        print(f"=== [{t.idx+1}/{len(tasks)}] {t.task_id} :: {t.entry_point} ===")
        gen_path = gens_dir / f"{t.idx:03d}_{t.entry_point}.py"

        # 1) Get / reuse generation
        chunks_used = 0  # Track how many RAG chunks were used
        trace_logger = None  # Will be created for non-dry runs (deep mode)

        if args.dry_run and gen_path.exists():
            completion_text = gen_path.read_text(encoding="utf-8")
            latency = 0.0
            print("  (dry-run) Loaded cached completion.")
        else:
            # Create TraceLogger for deep mode only
            if args.mode == "deep":
                model_config = ModelConfig(
                    model=args.model,
                    temperature=args.temperature,
                    api_type="openai",
                )
                trace_logger = TraceLogger(
                    task_id=t.task_id,
                    entry_point=t.entry_point,
                    prompt=t.prompt,
                    model_config=model_config,
                )

            # Generate code with retry logic
            raw_text = None
            latency = 0.0
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    print(f"  Generating code (attempt {attempt + 1}/{max_retries})...")
                    if args.mode == "fast":
                        raw_text, latency = generate_fast_completion(
                            t,
                            selected_model=args.model,
                            temperature=args.temperature,
                            max_output_tokens=args.max_output_tokens,
                            api_key_openai=openai_key,
                            api_key_openrouter=openrouter_key,
                        )
                    else:
                        raw_text, latency = run_deep_thought_mode(
                            t.prompt,
                            selected_model=args.model,
                            api_key_openai=openai_key,
                            api_key_openrouter=openrouter_key,
                            trace_logger=trace_logger,
                            use_rag=rag_enabled,
                        )
                    # Check if we got valid output
                    if raw_text and raw_text.strip():
                        break
                    print("  Empty response, retrying...")
                except RuntimeError as e:
                    print(f"  LLM error (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        print(f"  Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        raw_text = ""

            # Handle None from failed extraction
            if raw_text is None:
                raw_text = ""
                print("  WARNING: No code generated after retries")

            completion_text = extract_code_only(raw_text) if raw_text else ""
            gen_path.write_text(completion_text, encoding="utf-8")
            print(f"  Generated {len(completion_text)} chars in {latency:.2f}s")

            # Update trace logger with chunks used count (deep mode only)
            if trace_logger:
                chunks_used = len(trace_logger.trace.rag_queries)

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
            prompt_chars=len(t.prompt),
            completion_chars=len(completion_text),
            latency_s=latency,
            difficulty_scale=t.difficulty_scale,
            model=args.model,
            file_path=str(gen_path),
            mode=args.mode,
            rag_used=rag_enabled,
            rag_chunks_used=chunks_used,
        )
        results.append(res)
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} (exec: {exec_latency:.2f}s)")
        if not passed and err:
            print(f"  Error: {err[:200]}")
        print()

        # Append to progress tracker
        passed_so_far = sum(1 for r in results if r.passed)
        failed_so_far = len(results) - passed_so_far
        with progress_path.open("a", encoding="utf-8") as pf:
            pf.write(f"[{t.task_id}] ({t.idx+1}/{len(tasks)}) {status} | {t.entry_point}\n")
            if not passed and err:
                pf.write(f"         Error: {err[:100]}...\n")
            pf.write(
                f"         Running: {passed_so_far} passed, {failed_so_far} failed "
                f"({100*passed_so_far/len(results):.1f}% pass rate)\n\n"
            )

        # Save trace if available
        if trace_logger:
            trace_logger.set_result(
                final_code=completion_text,
                latency_s=latency,
                passed=passed,
                error=err,
            )
            trace_dir = trace_logger.save(traces_dir)
            print(f"  Trace saved: {trace_dir}")

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
        "mode": args.mode,
        "rag_enabled": rag_enabled,
        "rag_top_k": args.rag_top_k if rag_enabled else None,
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

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\n✓ Artifacts written to: {out_root}")
    print(f"✓ Results CSV: {csv_path}")
    print(f"✓ Summary JSON: {out_root / 'summary.json'}")
    if rag_enabled:
        print(f"✓ RAG debug log: {out_root / 'rag_debug.log'}")


if __name__ == "__main__":
    load_dotenv()
    args = build_cli()
    evaluate(args)

# Made with Bob
