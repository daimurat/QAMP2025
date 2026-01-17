from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    orchestrator_success: bool
    iterations_used: int
    termination_reason: str
    test_passed: bool
    test_error: Optional[str]
    total_time_seconds: float
    orchestrator_time_seconds: float
    verification_time_seconds: float
    final_code: Optional[str]
    session_id: Optional[str]
    error_message: Optional[str]
    traceback: Optional[str]
    session_state: Optional[Any] = None  # full SessionState for persistence
