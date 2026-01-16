import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is importable when running pytest from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.state import (  # noqa: E402
    Decision,
    Evaluation,
    ExecutionResult,
    IterationState,
    Plan,
    RetrievedDocument,
    SessionState,
    SubTask,
    TaskType,
    TerminationReason,
)


def _sample_plan() -> Plan:
    return Plan(
        plan_id="plan_0",
        version=1,
        sub_tasks=[
            SubTask(
                task_id="task_1",
                description="Retrieve docs",
                task_type=TaskType.RETRIEVAL,
                dependencies=[],
            ),
            SubTask(
                task_id="task_2",
                description="Generate code",
                task_type=TaskType.COMPUTATION,
                dependencies=["task_1"],
            ),
        ],
        rag_needed=True,
        rag_queries=["sampler v2 qiskit"],
        code_needed=True,
        code_requirements=["Use SamplerV2"],
        acceptance_criteria=["Code runs", "Uses retrieved info"],
        uses_previous_context=False,
        referenced_iterations=[],
    )


def _sample_retrieved_doc() -> RetrievedDocument:
    return RetrievedDocument(
        doc_id="doc_1",
        text="Sample document content",
        source="docs/api",
        url="https://example.com",
        relevance_score=0.9,
        retrieved_at=datetime(2024, 1, 1, 12, 0, 0),
        query_used="sampler v2 qiskit",
    )


def _sample_execution_result() -> ExecutionResult:
    return ExecutionResult(
        success=True,
        stdout="ok",
        stderr="",
        return_value={"result": 1},
        artifacts={"plot.png": b"binarydata"},
        execution_time_ms=1200,
        memory_usage_mb=32.5,
        error_type=None,
        error_traceback=None,
    )


def _sample_evaluation() -> Evaluation:
    return Evaluation(
        evaluation_id="eval_1",
        answers_question=True,
        code_executes=True,
        output_valid=True,
        criteria_met={"Code runs": True, "Uses retrieved info": True},
        correctness_score=0.9,
        completeness_score=0.85,
        code_quality_score=0.8,
        reasoning="All checks passed.",
        identified_issues=[],
        suggested_improvements=["Add docstring"],
    )


def test_plan_round_trip():
    plan = _sample_plan()
    reconstructed = Plan.from_dict(plan.to_dict())
    assert reconstructed == plan


def test_execution_result_serializes_artifacts():
    exec_result = _sample_execution_result()
    as_dict = exec_result.to_dict()
    json.dumps(as_dict)  # should be JSON serializable
    recovered = ExecutionResult.from_dict(as_dict)
    assert recovered == exec_result


def test_iteration_and_session_round_trip():
    plan = _sample_plan()
    retrieved_doc = _sample_retrieved_doc()
    exec_result = _sample_execution_result()
    evaluation = _sample_evaluation()
    now = datetime(2024, 1, 2, 8, 30, 0)

    iteration = IterationState(
        iteration_id=0,
        timestamp=now,
        plan=plan,
        rag_queries=["sampler v2 qiskit"],
        retrieved_documents=[retrieved_doc],
        generated_code="# sample",
        code_version=0,
        execution_result=exec_result,
        evaluation=evaluation,
        decision=Decision.SUCCESS,
        feedback=None,
    )

    session = SessionState(
        session_id="sess_123",
        user_question="How to use SamplerV2?",
        start_time=now,
        end_time=None,
        max_iterations=3,
        current_iteration=1,
        iterations=[iteration],
        all_retrieved_docs={retrieved_doc.doc_id: retrieved_doc},
        final_answer="Use SamplerV2 like this...",
        final_code="# sample",
        termination_reason=TerminationReason.SUCCESS,
        total_rag_calls=1,
        total_llm_calls=3,
        total_code_executions=1,
    )

    as_dict = session.to_dict()
    json.dumps(as_dict)  # ensure serializable
    reconstructed = SessionState.from_dict(as_dict)
    assert reconstructed == session


def test_optional_fields_handle_none():
    minimal_exec = ExecutionResult(
        success=False,
        stdout="",
        stderr="err",
        return_value=None,
        artifacts={},
        error_type=None,
        error_traceback=None,
    )
    round_trip = ExecutionResult.from_dict(minimal_exec.to_dict())
    assert round_trip == minimal_exec


def test_enum_value_parsing():
    assert Decision("success") is Decision.SUCCESS
    assert TerminationReason("fatal_error") is TerminationReason.FATAL_ERROR
    assert TaskType("retrieval") is TaskType.RETRIEVAL
