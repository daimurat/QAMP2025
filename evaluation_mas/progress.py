"""
Lightweight console progress reporting.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class EvaluationProgress:
    total_tasks: int
    completed_tasks: int = 0
    passed_tasks: int = 0
    failed_tasks: int = 0
    error_tasks: int = 0
    timeout_tasks: int = 0
    start_time: float = field(default_factory=time.time)
    current_task: str | None = None
    current_phase: str | None = None

    def update(self, status: str):
        self.completed_tasks += 1
        if status == "passed":
            self.passed_tasks += 1
        elif status == "failed":
            self.failed_tasks += 1
        elif status == "error":
            self.error_tasks += 1
        elif status == "timeout":
            self.timeout_tasks += 1

    def summary_line(self) -> str:
        elapsed = time.time() - self.start_time
        remaining = (elapsed / max(self.completed_tasks, 1)) * (
            self.total_tasks - self.completed_tasks
        )
        phase = f" | Phase: {self.current_phase}" if self.current_phase else ""
        task = f" | Task: {self.current_task}" if self.current_task else ""
        return (
            f"Progress: {self.completed_tasks}/{self.total_tasks} | "
            f"Passed: {self.passed_tasks} | Failed: {self.failed_tasks} | "
            f"Error: {self.error_tasks} | Timeout: {self.timeout_tasks} | "
            f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s{task}{phase}"
        )
