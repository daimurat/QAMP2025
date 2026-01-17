"""
Agent input/output dataclasses aligned with the multi-agent specification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .state import (
    Decision,
    Evaluation,
    ExecutionResult,
    Plan,
    RetrievedDocument,
)


@dataclass
class PlannerInput:
    user_question: str
    iteration_number: int
    previous_feedback: Optional[str] = None
    previous_plan: Optional[Plan] = None
    previous_code: Optional[str] = None
    previous_execution_result: Optional[ExecutionResult] = None
    all_retrieved_docs: List[RetrievedDocument] = field(default_factory=list)


@dataclass
class PlannerOutput:
    plan: Plan
    raw_response: str


@dataclass
class CodeAgentInput:
    user_question: str  # Original user question containing function signature
    plan: Plan
    code_requirements: List[str]
    retrieved_context: str
    previous_code: Optional[str] = None
    previous_error: Optional[str] = None
    previous_feedback: Optional[str] = None


@dataclass
class CodeAgentOutput:
    code: str
    language: str
    raw_response: str


@dataclass
class ExecutorInput:
    code: str
    timeout_seconds: int = 60
    max_memory_mb: int = 512
    allowed_imports: List[str] = field(
        default_factory=lambda: [
            "qiskit",
            "qiskit_aer",
            "qiskit_ibm_runtime",
            "numpy",
            "matplotlib",
            "scipy",
            "math",
            "json",
            "datetime",
            "collections",
        ]
    )
    working_directory: str = "work"


@dataclass
class ExecutorOutput:
    result: ExecutionResult
    execution_id: str


@dataclass
class EvaluatorInput:
    user_question: str
    plan: Plan
    code: str
    execution_result: ExecutionResult
    iteration_number: int
    max_iterations: int
    previous_evaluations: List[Evaluation]
    test_result: Optional[Any] = None  # TestResult from test verification


@dataclass
class EvaluatorOutput:
    evaluation: Evaluation
    decision: Decision
    feedback: Optional[str]
    final_answer: Optional[str]
    raw_response: str


__all__ = [
    "PlannerInput",
    "PlannerOutput",
    "CodeAgentInput",
    "CodeAgentOutput",
    "ExecutorInput",
    "ExecutorOutput",
    "EvaluatorInput",
    "EvaluatorOutput",
]
