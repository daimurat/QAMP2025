"""
Executor agent implementation wrapping a sandbox.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from agents.base import AgentError, ExecutorAgent
from models import ExecutionResult, ExecutorInput, ExecutorOutput


class ExecutorAgentImpl(ExecutorAgent):
    def __init__(self, sandbox):
        """
        Args:
            sandbox: Object with an async execute(code, timeout_seconds, allowed_imports, working_directory) -> ExecutionResult
        """
        self.sandbox = sandbox

    def get_system_prompt(self) -> str:
        return ""  # Executor is not an LLM

    async def invoke(self, input: ExecutorInput) -> ExecutorOutput:
        work_dir = Path(input.working_directory)
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            result: ExecutionResult = await self.sandbox.execute(
                code=input.code,
                timeout_seconds=input.timeout_seconds,
                allowed_imports=input.allowed_imports,
                working_directory=str(work_dir),
                max_memory_mb=input.max_memory_mb,
            )
        except asyncio.TimeoutError:
            result = ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {input.timeout_seconds}s",
                return_value=None,
                artifacts={},
                execution_time_ms=input.timeout_seconds * 1000,
                memory_usage_mb=0.0,
                error_type="TimeoutError",
                error_traceback=None,
            )
        except Exception as e:
            raise AgentError(f"Sandbox execution failed: {e}")

        return ExecutorOutput(result=result, execution_id=f"exec_{work_dir.name}")


__all__ = ["ExecutorAgentImpl"]
