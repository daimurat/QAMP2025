"""
Verification harness: merge model code with dataset test code and run in subprocess.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple


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


def build_executable_code(prompt: str, completion: str, test_code: str, entry_point: str) -> str:
    """Combine dataset prompt, model completion, and test code into an executable script."""
    return EXEC_TEMPLATE.format(
        prompt=prompt,
        completion=completion,
        test_code=test_code,
        entry_point=entry_point,
    )


def _get_venv_python() -> str:
    """Return the Python executable from the active virtual environment."""
    # Check if running inside a virtualenv
    if hasattr(sys, "real_prefix"):
        # Old-style virtualenv
        base = sys.prefix
    elif sys.base_prefix != sys.prefix:
        # Modern venv
        base = sys.prefix
    else:
        # Not in a virtual environment, use sys.executable
        return sys.executable

    # Construct path to the venv Python
    if sys.platform == "win32":
        python_path = Path(base) / "Scripts" / "python.exe"
    else:
        python_path = Path(base) / "bin" / "python"

    if python_path.exists():
        return str(python_path)
    return sys.executable


def run_in_subprocess(
    code: str,
    timeout_sec: int,
    python_executable: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Execute code in a fresh Python subprocess with a timeout.
    
    Args:
        code: The Python code to execute.
        timeout_sec: Maximum seconds to wait for execution.
        python_executable: Optional path to Python interpreter. If None,
            uses the virtual environment Python if available.
    """
    if python_executable is None:
        python_executable = _get_venv_python()

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "eval_task.py"
        src.write_text(code, encoding="utf-8")

        try:
            p = subprocess.run(
                [python_executable, "-I", "-B", str(src)],
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
        m = re.search(r"___QHE_FAIL___:(.*)$", out, flags=re.M | re.S)
        return False, (m.group(1).strip() if m else f"RuntimeError: {out[:5000]}")

