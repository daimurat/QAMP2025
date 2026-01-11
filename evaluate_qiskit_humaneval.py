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

# LangChain for LLM and RAG
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage

from workflows.deep_thought import run_deep_thought_mode

# ----------------------------
# CLI & defaults
# ----------------------------
def build_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate an OpenAI model on QiskitHumanEval with RAG support.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=os.getenv("OPENAI_EVAL_MODEL", "gpt-4o-mini"),
                   help="OpenAI model id (e.g., gpt-4o-mini, gpt-4o).")
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
    p.add_argument("--vector-store-path", default="faiss_index", help="Path to FAISS vector store.")
    p.add_argument("--index-name", default="qiskit_docs_index", help="Name of FAISS index.")
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
    rag_used: bool

# ----------------------------
# Utility: file safe writing
# ----------------------------
def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# ----------------------------
# RAG: Vector Store initialization
# ----------------------------
def load_vector_store(vector_store_path: str, index_name: str) -> Optional[FAISS]:
    """Load FAISS vector store for RAG."""
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local(
            folder_path=vector_store_path,
            index_name=index_name,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✓ Loaded vector store from {vector_store_path}")
        return vector_store
    except Exception as e:
        print(f"⚠️  Failed to load vector store: {e}")
        return None

def retrieve_context(vector_store: Optional[FAISS], query: str, top_k: int = 5) -> str:
    """Retrieve relevant context from vector store."""
    if vector_store is None:
        return ""
    
    try:
        docs = vector_store.similarity_search(query, k=top_k)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    except Exception as e:
        print(f"⚠️  RAG retrieval failed: {e}")
        return ""

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
def load_tasks(dataset: str, split: str, limit: Optional[int]) -> List[Task]:
    ds = load_dataset(dataset, split=split)
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
    rag_suffix = "_rag" if args.use_rag else "_norag"
    out_root = Path(args.outdir) / f"{Path(args.dataset).name}_{run_ts}_{args.model.replace('/', '_')}{rag_suffix}"
    gens_dir = out_root / "generations"
    ensure_dir(gens_dir)

    # Initialize LLM
    llm = ChatOpenAI(
        model=args.model,
        temperature=args.temperature,
        max_completion_tokens=args.max_output_tokens,
    )
    print(f"✓ Initialized LLM: {args.model}")

    # Initialize vector store for RAG
    vector_store = None
    if args.use_rag:
        vector_store = load_vector_store(args.vector_store_path, args.index_name)
        if vector_store is None:
            print("⚠️  RAG requested but vector store not available. Continuing without RAG.")
            args.use_rag = False

    # Load tasks
    tasks = load_tasks(args.dataset, args.split, args.max_items)
    print(f"✓ Loaded {len(tasks)} tasks from {args.dataset}:{args.split}")
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
            try:
                raw_text,latency = run_deep_thought_mode(t.prompt)
            except RuntimeError as e:
                print(f"  LLM error: {e}")
                raw_text, latency = "", 0.0

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
