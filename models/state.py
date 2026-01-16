"""
Core dataclasses and enums for the multi-agent orchestrator.

Implements serialization helpers so state can be persisted and rehydrated
without losing enum or datetime fidelity.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


def _serialize_value(value: Any) -> Any:
    """Convert values into JSON-serializable forms."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    return value


def _deserialize_bytes(value: Any) -> Optional[bytes]:
    """Decode bytes stored by _serialize_value."""
    if isinstance(value, dict) and "__bytes__" in value:
        return base64.b64decode(value["__bytes__"].encode("ascii"))
    return None


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(value) if value else None


class Decision(Enum):
    SUCCESS = "success"
    RETRY = "retry"
    ABORT = "abort"


class TerminationReason(Enum):
    SUCCESS = "success"
    MAX_ITERATIONS = "max_iterations_reached"
    ABORT_REQUESTED = "evaluator_abort"
    FATAL_ERROR = "fatal_error"
    TIMEOUT = "timeout"
    USER_CANCELLED = "user_cancelled"


class TaskType(Enum):
    RETRIEVAL = "retrieval"
    COMPUTATION = "computation"
    VALIDATION = "validation"


@dataclass
class SubTask:
    task_id: str
    description: str
    task_type: TaskType
    dependencies: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "task_type": self.task_type.value,
            "dependencies": list(self.dependencies),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubTask":
        return cls(
            task_id=data["task_id"],
            description=data["description"],
            task_type=TaskType(data["task_type"]),
            dependencies=data.get("dependencies", []),
        )


@dataclass
class Plan:
    plan_id: str
    version: int
    sub_tasks: List[SubTask]
    rag_needed: bool
    rag_queries: List[str]
    code_needed: bool
    code_requirements: List[str]
    acceptance_criteria: List[str]
    uses_previous_context: bool
    referenced_iterations: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "version": self.version,
            "sub_tasks": [st.to_dict() for st in self.sub_tasks],
            "rag_needed": self.rag_needed,
            "rag_queries": list(self.rag_queries),
            "code_needed": self.code_needed,
            "code_requirements": list(self.code_requirements),
            "acceptance_criteria": list(self.acceptance_criteria),
            "uses_previous_context": self.uses_previous_context,
            "referenced_iterations": list(self.referenced_iterations),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Plan":
        return cls(
            plan_id=data["plan_id"],
            version=int(data["version"]),
            sub_tasks=[SubTask.from_dict(st) for st in data.get("sub_tasks", [])],
            rag_needed=bool(data.get("rag_needed", False)),
            rag_queries=data.get("rag_queries", []),
            code_needed=bool(data.get("code_needed", True)),
            code_requirements=data.get("code_requirements", []),
            acceptance_criteria=data.get("acceptance_criteria", []),
            uses_previous_context=bool(data.get("uses_previous_context", False)),
            referenced_iterations=data.get("referenced_iterations", []),
        )


@dataclass
class RetrievedDocument:
    doc_id: str
    text: str
    source: str
    url: str
    relevance_score: float
    retrieved_at: datetime
    query_used: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "source": self.source,
            "url": self.url,
            "relevance_score": float(self.relevance_score),
            "retrieved_at": self.retrieved_at.isoformat(),
            "query_used": self.query_used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievedDocument":
        return cls(
            doc_id=data["doc_id"],
            text=data["text"],
            source=data["source"],
            url=data["url"],
            relevance_score=float(data["relevance_score"]),
            retrieved_at=_parse_datetime(data.get("retrieved_at")) or datetime.now(),
            query_used=data["query_used"],
        )


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    return_value: Any
    artifacts: Dict[str, bytes] = field(default_factory=dict)
    execution_time_ms: int = 0
    memory_usage_mb: float = 0.0
    error_type: Optional[str] = None
    error_traceback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        serialized_artifacts = {
            k: _serialize_value(v) for k, v in self.artifacts.items()
        }
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_value": self.return_value,
            "artifacts": serialized_artifacts,
            "execution_time_ms": self.execution_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "error_type": self.error_type,
            "error_traceback": self.error_traceback,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionResult":
        artifacts: Dict[str, bytes] = {}
        for key, value in data.get("artifacts", {}).items():
            decoded = _deserialize_bytes(value)
            artifacts[key] = decoded if decoded is not None else value  # type: ignore[arg-type]

        return cls(
            success=bool(data["success"]),
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            return_value=data.get("return_value"),
            artifacts=artifacts,
            execution_time_ms=int(data.get("execution_time_ms", 0)),
            memory_usage_mb=float(data.get("memory_usage_mb", 0.0)),
            error_type=data.get("error_type"),
            error_traceback=data.get("error_traceback"),
        )


@dataclass
class Evaluation:
    evaluation_id: str
    answers_question: bool
    code_executes: bool
    output_valid: bool
    criteria_met: Dict[str, bool]
    correctness_score: float
    completeness_score: float
    code_quality_score: float
    reasoning: str
    identified_issues: List[str]
    suggested_improvements: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "answers_question": self.answers_question,
            "code_executes": self.code_executes,
            "output_valid": self.output_valid,
            "criteria_met": dict(self.criteria_met),
            "correctness_score": self.correctness_score,
            "completeness_score": self.completeness_score,
            "code_quality_score": self.code_quality_score,
            "reasoning": self.reasoning,
            "identified_issues": list(self.identified_issues),
            "suggested_improvements": list(self.suggested_improvements),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evaluation":
        return cls(
            evaluation_id=data["evaluation_id"],
            answers_question=bool(data.get("answers_question", False)),
            code_executes=bool(data.get("code_executes", False)),
            output_valid=bool(data.get("output_valid", False)),
            criteria_met=data.get("criteria_met", {}),
            correctness_score=float(data.get("correctness_score", 0.0)),
            completeness_score=float(data.get("completeness_score", 0.0)),
            code_quality_score=float(data.get("code_quality_score", 0.0)),
            reasoning=data.get("reasoning", ""),
            identified_issues=data.get("identified_issues", []),
            suggested_improvements=data.get("suggested_improvements", []),
        )


@dataclass
class IterationState:
    iteration_id: int
    timestamp: datetime
    plan: Plan
    rag_queries: List[str]
    retrieved_documents: List[RetrievedDocument]
    generated_code: str
    code_version: int
    execution_result: ExecutionResult
    evaluation: Optional[Evaluation]
    decision: Optional[Decision]
    feedback: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration_id": self.iteration_id,
            "timestamp": self.timestamp.isoformat(),
            "plan": self.plan.to_dict(),
            "rag_queries": list(self.rag_queries),
            "retrieved_documents": [doc.to_dict() for doc in self.retrieved_documents],
            "generated_code": self.generated_code,
            "code_version": self.code_version,
            "execution_result": self.execution_result.to_dict(),
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "decision": self.decision.value if self.decision else None,
            "feedback": self.feedback,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterationState":
        return cls(
            iteration_id=int(data["iteration_id"]),
            timestamp=_parse_datetime(data.get("timestamp")) or datetime.now(),
            plan=Plan.from_dict(data["plan"]),
            rag_queries=data.get("rag_queries", []),
            retrieved_documents=[
                RetrievedDocument.from_dict(doc)
                for doc in data.get("retrieved_documents", [])
            ],
            generated_code=data.get("generated_code", ""),
            code_version=int(data.get("code_version", 0)),
            execution_result=ExecutionResult.from_dict(data["execution_result"]),
            evaluation=(
                Evaluation.from_dict(data["evaluation"])
                if data.get("evaluation") is not None
                else None
            ),
            decision=Decision(data["decision"]) if data.get("decision") else None,
            feedback=data.get("feedback"),
        )


@dataclass
class SessionState:
    session_id: str
    user_question: str
    start_time: datetime
    end_time: Optional[datetime] = None
    max_iterations: int = 0
    current_iteration: int = 0
    iterations: List[IterationState] = field(default_factory=list)
    all_retrieved_docs: Dict[str, RetrievedDocument] = field(default_factory=dict)
    final_answer: Optional[str] = None
    final_code: Optional[str] = None
    termination_reason: Optional[TerminationReason] = None
    total_rag_calls: int = 0
    total_llm_calls: int = 0
    total_code_executions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_question": self.user_question,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "max_iterations": self.max_iterations,
            "current_iteration": self.current_iteration,
            "iterations": [it.to_dict() for it in self.iterations],
            "all_retrieved_docs": {
                doc_id: doc.to_dict()
                for doc_id, doc in self.all_retrieved_docs.items()
            },
            "final_answer": self.final_answer,
            "final_code": self.final_code,
            "termination_reason": (
                self.termination_reason.value if self.termination_reason else None
            ),
            "total_rag_calls": self.total_rag_calls,
            "total_llm_calls": self.total_llm_calls,
            "total_code_executions": self.total_code_executions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        retrieved_docs = {
            doc_id: RetrievedDocument.from_dict(doc_dict)
            for doc_id, doc_dict in data.get("all_retrieved_docs", {}).items()
        }
        return cls(
            session_id=data["session_id"],
            user_question=data["user_question"],
            start_time=_parse_datetime(data.get("start_time")) or datetime.now(),
            end_time=_parse_datetime(data.get("end_time")),
            max_iterations=int(data.get("max_iterations", 0)),
            current_iteration=int(data.get("current_iteration", 0)),
            iterations=[
                IterationState.from_dict(it) for it in data.get("iterations", [])
            ],
            all_retrieved_docs=retrieved_docs,
            final_answer=data.get("final_answer"),
            final_code=data.get("final_code"),
            termination_reason=(
                TerminationReason(data["termination_reason"])
                if data.get("termination_reason")
                else None
            ),
            total_rag_calls=int(data.get("total_rag_calls", 0)),
            total_llm_calls=int(data.get("total_llm_calls", 0)),
            total_code_executions=int(data.get("total_code_executions", 0)),
        )
