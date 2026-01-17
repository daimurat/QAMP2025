import json
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
from observability import QueryReportGenerator, ReportConfig  # noqa: E402


def _sample_plan() -> Plan:
    return Plan(
        plan_id="plan_0",
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
        rag_queries=["qiskit sampler usage", "aer backend"],
        code_needed=True,
        code_requirements=["Return heads/tails counts"],
        acceptance_criteria=["Returns dict"],
        uses_previous_context=False,
        referenced_iterations=[],
    )


def _iteration(iter_id: int, doc_suffix: str) -> IterationState:
    now = datetime(2025, 1, 1, 12, 0, 0)
    retrieved = [
        RetrievedDocument(
            doc_id=f"doc{doc_suffix}",
            text="Line 1\nLine 2\nLine 3\nLine 4",
            source="docs/api",
            url="https://example.com/doc",
            relevance_score=0.8,
            retrieved_at=now,
            query_used="qiskit sampler usage",
        ),
    ]
    exec_result = ExecutionResult(
        success=True,
        stdout="ok",
        stderr="",
        return_value=None,
        artifacts={},
        execution_time_ms=120,
        memory_usage_mb=2.5,
        error_type=None,
        error_traceback=None,
    )
    evaluation = Evaluation(
        evaluation_id=f"eval_{iter_id}",
        answers_question=iter_id > 0,
        code_executes=True,
        output_valid=iter_id > 0,
        criteria_met={"Returns dict": iter_id > 0},
        correctness_score=1.0 if iter_id > 0 else 0.5,
        completeness_score=1.0 if iter_id > 0 else 0.5,
        code_quality_score=0.8,
        reasoning="Looks good" if iter_id > 0 else "Needs fixes",
        identified_issues=[] if iter_id > 0 else ["Missing counts"],
        suggested_improvements=["Add tests"],
    )
    return IterationState(
        iteration_id=iter_id,
        timestamp=now,
        plan=_sample_plan(),
        rag_queries=["qiskit sampler usage", "aer backend"],
        retrieved_documents=retrieved,
        generated_code=f"# code v{iter_id}",
        code_version=iter_id,
        execution_result=exec_result,
        evaluation=evaluation,
        decision=Decision.SUCCESS if iter_id > 0 else Decision.RETRY,
        feedback=None if iter_id > 0 else "Try again",
    )


def _session_state() -> SessionState:
    it0 = _iteration(0, "a")
    it1 = _iteration(1, "b")
    now = datetime(2025, 1, 1, 12, 0, 0)
    all_docs = {doc.doc_id: doc for doc in it1.retrieved_documents}
    all_docs.update({doc.doc_id: doc for doc in it0.retrieved_documents})
    return SessionState(
        session_id="session123",
        user_question="Explain how to use qiskit sampler.",
        start_time=now,
        end_time=now,
        max_iterations=3,
        current_iteration=2,
        iterations=[it0, it1],
        all_retrieved_docs=all_docs,
        final_answer="done",
        final_code=it1.generated_code,
        termination_reason=TerminationReason.SUCCESS,
        total_rag_calls=2,
        total_llm_calls=3,
        total_code_executions=2,
    )


def test_report_generator_writes_files(tmp_path: Path):
    session = _session_state()
    cfg = ReportConfig(
        output_dir=tmp_path / "queries",
        max_chunk_preview_lines=2,
    )
    generator = QueryReportGenerator(cfg)
    query_dir = generator.generate_from_session(
        session=session,
        task_id="qiskitHumanEval/47",
        test_passed=False,
        test_error="TypeError",
    )

    assert query_dir is not None
    assert (query_dir / "report.md").exists()
    assert (query_dir / "prompt.txt").exists()
    assert (query_dir / "final_code.py").exists()
    for idx in (0, 1):
        iter_dir = query_dir / f"iteration_{idx}"
        assert (iter_dir / "01_planner.md").exists()
        assert (iter_dir / "02_rag.md").exists()
        assert (iter_dir / "03_code.md").exists()
        assert (iter_dir / "03_code.py").exists()
        assert (iter_dir / "04_execution.md").exists()
        assert (iter_dir / "05_evaluation.md").exists()

    rag_text = (query_dir / "iteration_0" / "02_rag.md").read_text(encoding="utf-8")
    assert "... [truncated]" in rag_text
    report_text = (query_dir / "report.md").read_text(encoding="utf-8")
    assert "FAILED" in report_text
    assert "TypeError" in report_text


def test_generate_from_json_file(tmp_path: Path):
    session = _session_state()
    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps(session.to_dict()), encoding="utf-8")

    cfg = ReportConfig(output_dir=tmp_path / "queries")
    generator = QueryReportGenerator(cfg)
    query_dir = generator.generate_from_json_file(
        session_path=session_path,
        task_id="qiskitHumanEval/47",
        test_passed=True,
        test_error=None,
    )

    assert query_dir is not None
    assert query_dir.name == "qiskitHumanEval_047"
    assert (query_dir / "report.md").exists()


def test_write_run_index(tmp_path: Path):
    cfg = ReportConfig(output_dir=tmp_path / "queries")
    generator = QueryReportGenerator(cfg)
    tasks = [
        {
            "task_id": "qiskitHumanEval/47",
            "display_id": "047",
            "status": "failed",
            "iterations": 2,
            "time_seconds": 10.0,
            "test_error": "TypeError",
            "link": "./qiskitHumanEval_047/report.md",
        }
    ]
    index_path = generator.write_run_index(run_id="run_123", task_summaries=tasks)
    assert index_path is not None
    text = index_path.read_text(encoding="utf-8")
    assert "run_123" in text
    assert "047" in text
