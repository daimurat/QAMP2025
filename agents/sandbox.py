"""
Local sandbox for executing generated code in a subprocess.

This is a lightweight, best-effort sandbox: it enforces allowed imports and
wall-clock timeouts, and runs code in an isolated working directory.
"""
from __future__ import annotations

import asyncio
import re
import sys
import time
from pathlib import Path
from typing import List

from models import ExecutionResult


class LocalCodeSandbox:
    def __init__(self, python_executable: str = sys.executable):
        self.python_executable = python_executable

    async def execute(
        self,
        code: str,
        timeout_seconds: int,
        allowed_imports: List[str],
        working_directory: str,
        max_memory_mb: int = 512,
    ) -> ExecutionResult:
        work_dir = Path(working_directory)
        work_dir.mkdir(parents=True, exist_ok=True)

        # Basic import whitelist check
        disallowed = self._find_disallowed_imports(code, allowed_imports)
        if disallowed:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Disallowed imports: {', '.join(sorted(disallowed))}",
                return_value=None,
                artifacts={},
                execution_time_ms=0,
                memory_usage_mb=0.0,
                error_type="ImportError",
                error_traceback=None,
            )

        script_path = work_dir / "code.py"
        script_path.write_text(code, encoding="utf-8")

        start = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_exec(
                self.python_executable,
                "-u",
                str(script_path),
                cwd=str(work_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=f"Execution timed out after {timeout_seconds}s",
                    return_value=None,
                    artifacts={},
                    execution_time_ms=int((time.perf_counter() - start) * 1000),
                    memory_usage_mb=0.0,
                    error_type="TimeoutError",
                    error_traceback=None,
                )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_value=None,
                artifacts={},
                execution_time_ms=int((time.perf_counter() - start) * 1000),
                memory_usage_mb=0.0,
                error_type="ExecutionError",
                error_traceback=None,
            )

        stdout = (stdout_bytes or b"").decode("utf-8", errors="replace")
        stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")

        success = proc.returncode == 0
        return ExecutionResult(
            success=success,
            stdout=stdout,
            stderr=stderr,
            return_value=None,
            artifacts={},
            execution_time_ms=int((time.perf_counter() - start) * 1000),
            memory_usage_mb=0.0,
            error_type=None if success else "RuntimeError",
            error_traceback=None if success else stderr,
        )

    def _find_disallowed_imports(
        self, code: str, allowed_imports: List[str]
    ) -> List[str]:
        allowed = set(allowed_imports)
        disallowed = set()
        import_pattern = re.compile(r"^\s*(import|from)\s+([a-zA-Z0-9_\.]+)", re.MULTILINE)
        for match in import_pattern.finditer(code):
            module = match.group(2).split(".")[0]
            if module not in allowed:
                disallowed.add(module)
        return list(disallowed)


__all__ = ["LocalCodeSandbox"]
