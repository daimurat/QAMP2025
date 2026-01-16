from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .state import TerminationReason


@dataclass
class OrchestratorConfig:
    max_iterations: int = 5
    iteration_timeout_seconds: int = 120
    code_execution_timeout_seconds: int = 60
    max_rag_queries_per_iteration: int = 3
    max_total_rag_queries: int = 10
    planner_model: str = "gpt-4.1"
    code_model: str = "gpt-4.1"
    evaluator_model: str = "gpt-4.1"
    enable_code_execution: bool = True
    enable_rag: bool = True
    persist_state: bool = True


@dataclass
class OrchestratorResult:
    success: bool
    final_answer: Optional[str]
    final_code: Optional[str]
    session_id: str
    iterations_used: int
    termination_reason: TerminationReason
    total_time_seconds: float
    session_state: any


@dataclass
class RetryPolicy:
    max_retries: int = 3
    initial_delay_seconds: float = 0.1
    max_delay_seconds: float = 5.0
    exponential_base: float = 2.0
    jitter: bool = False

