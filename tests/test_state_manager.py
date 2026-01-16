import sys
from datetime import datetime
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import (  # noqa: E402
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
from orchestrator import StateManager  # noqa: E402


class DummyConfig:
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations


def _sample_plan() -> Plan:
    return Plan(
        plan_id="p1",
        version=1,
        sub_tasks=[
            SubTask(
                task_id="t1",
                description="Retrieve docs",
                task_type=TaskType.RETRIEVAL,
                dependencies=[],
            )
        ],
        rag_needed=True,
        rag_queries=["sampler v2"],
        code_needed=True,
        code_requirements=["Use SamplerV2"],
        acceptance_criteria=["Code runs"],
        uses_previous_context=False,
        referenced_iterations=[],
    )


def _sample_iteration(iteration_id: int = 0) -> IterationState:
    now = datetime(2024, 1, 1, 12, 0, 0)
    retrieved = RetrievedDocument(
        doc_id="doc1",
        text="content",
        source="docs",
        url="https://example.com",
        relevance_score=0.9,
        retrieved_at=now,
        query_used="sampler v2",
    )
    exec_result = ExecutionResult(
        success=True,
        stdout="ok",
        stderr="",
        return_value=None,
        artifacts={},
        execution_time_ms=10,
        memory_usage_mb=1.0,
    )
    evaluation = Evaluation(
        evaluation_id="e1",
        answers_question=True,
        code_executes=True,
        output_valid=True,
        criteria_met={"Code runs": True},
        correctness_score=1.0,
        completeness_score=1.0,
        code_quality_score=1.0,
        reasoning="good",
        identified_issues=[],
        suggested_improvements=[],
    )
    return IterationState(
        iteration_id=iteration_id,
        timestamp=now,
        plan=_sample_plan(),
        rag_queries=["sampler v2"],
        retrieved_documents=[retrieved],
        generated_code="# code",
        code_version=iteration_id,
        execution_result=exec_result,
        evaluation=evaluation,
        decision=Decision.SUCCESS,
        feedback=None,
    )


def test_create_and_get_session(tmp_path: Path):
    manager = StateManager(base_dir=tmp_path)
    session = manager.create_session("What is SamplerV2?", DummyConfig(max_iterations=5))
    assert session.session_id
    assert session.max_iterations == 5
    loaded = manager.get_session(session.session_id)
    assert loaded.session_id == session.session_id
    assert loaded.user_question == "What is SamplerV2?"


def test_save_iteration_and_persist(tmp_path: Path):
    manager = StateManager(base_dir=tmp_path)
    session = manager.create_session("Q", DummyConfig())
    iteration = _sample_iteration(0)
    manager.save_iteration(session.session_id, iteration)
    loaded = manager.get_session(session.session_id)
    assert len(loaded.iterations) == 1
    assert loaded.iterations[0].iteration_id == 0

    # Replace same iteration id
    iteration2 = _sample_iteration(0)
    manager.save_iteration(session.session_id, iteration2)
    loaded2 = manager.get_session(session.session_id)
    assert len(loaded2.iterations) == 1
    assert loaded2.iterations[0].iteration_id == 0


def test_update_session_fields(tmp_path: Path):
    manager = StateManager(base_dir=tmp_path)
    session = manager.create_session("Q", DummyConfig())
    manager.update_session(
        session.session_id,
        {"final_answer": "done", "termination_reason": TerminationReason.SUCCESS},
    )
    loaded = manager.get_session(session.session_id)
    assert loaded.final_answer == "done"
    assert loaded.termination_reason == TerminationReason.SUCCESS


def test_add_retrieved_docs_dedup(tmp_path: Path):
    manager = StateManager(base_dir=tmp_path)
    session = manager.create_session("Q", DummyConfig())
    doc = RetrievedDocument(
        doc_id="doc1",
        text="A",
        source="s",
        url="u",
        relevance_score=0.5,
        retrieved_at=datetime.now(),
        query_used="q",
    )
    manager.add_retrieved_docs(session.session_id, [doc, doc])
    loaded = manager.get_session(session.session_id)
    assert len(loaded.all_retrieved_docs) == 1
    assert "doc1" in loaded.all_retrieved_docs


def test_persistence_across_manager_instances(tmp_path: Path):
    manager1 = StateManager(base_dir=tmp_path)
    session = manager1.create_session("Q", DummyConfig())
    iteration = _sample_iteration(0)
    manager1.save_iteration(session.session_id, iteration)

    # New manager should read existing state
    manager2 = StateManager(base_dir=tmp_path)
    loaded = manager2.get_session(session.session_id)
    assert len(loaded.iterations) == 1
    assert loaded.iterations[0].plan.plan_id == "p1"
