"""
Evaluation runner for the multi-agent system against Qiskit HumanEval.
"""
from __future__ import annotations

import asyncio
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Python 3.11+ has asyncio.timeout; older versions need async_timeout
if sys.version_info >= (3, 11):
    from asyncio import timeout as async_timeout_ctx
else:
    try:
        from async_timeout import timeout as async_timeout_ctx
    except ImportError:
        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def async_timeout_ctx(delay):
            yield

from agents import (
    CodeAgentImpl,
    EvaluatorAgentImpl,
    ExecutorAgentImpl,
    LocalCodeSandbox,
    PlannerAgentImpl,
)
from clients.chat_factory import create_chat_client
from evaluation_mas.config import EvaluationConfig
from evaluation_mas.tasks import EvaluationTask, load_tasks
from evaluation_mas.verification import build_executable_code, run_in_subprocess
from evaluation_mas.output import write_run_outputs
from evaluation_mas.progress import EvaluationProgress
from evaluation_mas.logging_setup import setup_logging
from observability import QueryReportGenerator, ReportConfig
from models import OrchestratorConfig
from orchestrator import MultiAgentOrchestrator, MultiQueryRAGRetriever, StateManager, TestResult
from rag import RAGRetriever

from .results import TaskResult, TaskStatus
from .output import ensure_dir
from orchestrator.session_persistence import save_session_state


class TaskTestVerifier:
    """Verifier that runs actual test assertions against generated code."""

    def __init__(self, task: EvaluationTask, timeout_sec: int):
        self.task = task
        self.timeout_sec = timeout_sec

    def verify(self, code: str) -> TestResult:
        """Run test code against the generated code."""
        merged = build_executable_code(
            prompt=self.task.prompt,
            completion=code,
            test_code=self.task.test_code,
            entry_point=self.task.entry_point,
        )
        passed, error = run_in_subprocess(merged, self.timeout_sec)
        return TestResult(passed=passed, error=error)


class EvaluationRunner:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    async def run(self) -> List[TaskResult]:
        tasks = load_tasks(
            dataset=self.config.evaluation.dataset,
            split=self.config.evaluation.split,
            difficulty=self.config.evaluation.difficulty,
            max_tasks=self.config.evaluation.max_tasks,
            task_id=(self.config.evaluation.task_ids or [None])[0]
            if self.config.evaluation.task_ids
            else None,
            cache_dir=Path(self.config.evaluation.output_dir) / "hf_cache",
        )

        if self.config.evaluation.dry_run:
            for t in tasks:
                print(f"{t.task_id} ({t.difficulty})")
            return []

        results: List[TaskResult] = []
        run_id = f"run_{int(time.time())}"
        progress = EvaluationProgress(total_tasks=len(tasks))
        log_dir = ensure_dir(Path(self.config.evaluation.output_dir) / run_id)
        setup_logging(log_dir / "evaluation.log", verbose=self.config.evaluation.verbose, quiet=self.config.evaluation.quiet)
        logger = logging.getLogger("evaluation_mas")
        report_generator = self._build_report_generator(log_dir)
        report_summaries: List[Dict[str, Any]] = []

        # Resume support: skip tasks already completed if resume path provided
        completed_ids = set()
        if self.config.evaluation.resume:
            prev_results_path = Path(self.config.evaluation.resume) / "results.csv"
            if prev_results_path.exists():
                with prev_results_path.open("r", encoding="utf-8") as f:
                    for line in f.readlines()[1:]:
                        parts = line.strip().split(",")
                        if len(parts) >= 2:
                            task_id = parts[0]
                            status = parts[1]
                            if status == "passed":
                                completed_ids.add(task_id)
            if completed_ids and not self.config.evaluation.quiet:
                print(f"Resuming; skipping {len(completed_ids)} completed tasks")
            if completed_ids:
                logger.info("Resume: skipping completed tasks", extra={"skipped": list(completed_ids)})

        sessions_dir = ensure_dir(Path(self.config.evaluation.output_dir) / run_id / "sessions") if self.config.evaluation.save_sessions else None
        generations_dir = ensure_dir(Path(self.config.evaluation.output_dir) / run_id / "generations") if self.config.evaluation.save_generations else None

        for task in tasks:
            if task.task_id in completed_ids:
                if not self.config.evaluation.quiet:
                    print(f"Skipping completed task {task.task_id}")
                continue
            progress.current_task = task.task_id
            progress.current_phase = "orchestrator"
            res = await self._evaluate_task(task, progress)
            # Retry failed tasks if enabled
            retries_left = self.config.evaluation.max_retries_per_task if self.config.evaluation.retry_failed_tasks else 0
            while retries_left > 0 and not res.test_passed:
                retries_left -= 1
                if not self.config.evaluation.quiet:
                    print(f"Retrying task {task.task_id}, attempts left: {retries_left}")
                res = await self._evaluate_task(task, progress)
            results.append(res)
            self._maybe_save_generation(res, task, generations_dir)
            self._maybe_save_session(res, task, sessions_dir)
            report_dir: Optional[Path] = None
            if report_generator:
                try:
                    report_dir = self._maybe_generate_report(report_generator, task, res)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Report generation failed", exc_info=exc)
                report_summaries.append(self._build_task_summary(task, res, report_dir))
            progress.update(res.status.value)
            if not self.config.evaluation.quiet:
                print(progress.summary_line())
            logger.info(
                "Task completed",
                extra={
                    "task_id": task.task_id,
                    "status": res.status.value,
                    "test_passed": res.test_passed,
                    "iterations": res.iterations_used,
                    "termination": res.termination_reason,
                    "total_time": res.total_time_seconds,
                },
            )
        write_run_outputs(results, self.config, run_id, sessions_dir=sessions_dir, generations_dir=generations_dir)
        if report_generator and report_summaries:
            try:
                report_generator.write_run_index(run_id=run_id, task_summaries=report_summaries)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Report index generation failed", exc_info=exc)
        return results

    async def _evaluate_task(self, task: EvaluationTask, progress: EvaluationProgress) -> TaskResult:
        start = time.time()
        orch_time = 0.0
        verification_time = 0.0
        try:
            orchestrator = self._build_orchestrator(progress, task)
            # Run orchestrator with task timeout
            try:
                async with async_timeout_ctx(self.config.evaluation.timeout_per_task_seconds):
                    t0 = time.time()
                    # Orchestrator phases are internal; we track overall here
                    orch_result = await orchestrator.run(
                        user_question=task.prompt,
                        config=self.config.orchestrator,
                    )
                    orch_time = time.time() - t0
            except asyncio.TimeoutError:
                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.TIMEOUT,
                    orchestrator_success=False,
                    iterations_used=0,
                    termination_reason="timeout",
                    test_passed=False,
                    test_error="orchestrator timeout",
                    total_time_seconds=time.time() - start,
                    orchestrator_time_seconds=orch_time,
                    verification_time_seconds=verification_time,
                    final_code=None,
                    session_id=None,
                    error_message=None,
                    traceback=None,
                )

            # Verification
            verify_t0 = time.time()
            if orch_result.final_code:
                merged_code = build_executable_code(
                    prompt=task.prompt,
                    completion=orch_result.final_code,
                    test_code=task.test_code,
                    entry_point=task.entry_point,
                )
                test_passed, test_error = run_in_subprocess(
                    merged_code,
                    timeout_sec=self.config.evaluation.timeout_per_task_seconds,
                )
            else:
                test_passed, test_error = False, "No code generated"
            verification_time = time.time() - verify_t0

            status = TaskStatus.PASSED if test_passed else TaskStatus.FAILED

            return TaskResult(
                task_id=task.task_id,
                status=status,
                orchestrator_success=orch_result.success,
                iterations_used=orch_result.iterations_used,
                termination_reason=orch_result.termination_reason.value
                if orch_result.termination_reason
                else "",
                test_passed=test_passed,
                test_error=test_error,
                total_time_seconds=time.time() - start,
                orchestrator_time_seconds=orch_time,
                verification_time_seconds=verification_time,
                final_code=orch_result.final_code,
                session_id=orch_result.session_id,
                error_message=None,
                traceback=None,
                session_state=orch_result.session_state,
            )

        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.ERROR,
                orchestrator_success=False,
                iterations_used=0,
                termination_reason="error",
                test_passed=False,
                test_error=str(e),
                total_time_seconds=time.time() - start,
                orchestrator_time_seconds=orch_time,
                verification_time_seconds=verification_time,
                final_code=None,
                session_id=None,
                error_message=str(e),
                traceback=None,
            )

    def _build_orchestrator(self, progress: EvaluationProgress, task: Optional[EvaluationTask] = None) -> MultiAgentOrchestrator:
        def progress_cb(phase: str, iteration: int):
            progress.current_phase = phase

        cfg: OrchestratorConfig = self.config.orchestrator
        provider = self.config.models.provider
        planner_llm = create_chat_client(provider, model=self.config.models.planner_model)
        code_llm = create_chat_client(provider, model=self.config.models.code_model)
        evaluator_llm = create_chat_client(provider, model=self.config.models.evaluator_model)

        planner = PlannerAgentImpl(planner_llm)
        code_agent = CodeAgentImpl(code_llm)
        evaluator = EvaluatorAgentImpl(evaluator_llm)
        sandbox = LocalCodeSandbox()
        executor = ExecutorAgentImpl(sandbox=sandbox)

        rag_base = RAGRetriever(db_path=self.config.rag.db_path)
        rag = MultiQueryRAGRetriever(rag_base)

        state_manager = StateManager()

        # Create test verifier if task is provided
        test_verifier = None
        if task is not None:
            test_verifier = TaskTestVerifier(
                task=task,
                timeout_sec=self.config.evaluation.timeout_per_task_seconds,
            )

        orchestrator = MultiAgentOrchestrator(
            planner=planner,
            code_agent=code_agent,
            executor=executor,
            evaluator=evaluator,
            rag_retriever=rag,
            state_manager=state_manager,
            config=cfg,
            progress_callback=progress_cb,
            test_verifier=test_verifier,
        )
        return orchestrator

    def _maybe_save_generation(self, result: TaskResult, task: EvaluationTask, generations_dir: Path | None):
        if generations_dir is None:
            return
        if not result.final_code:
            return
        fname = f"{task.task_number:03d}_{task.entry_point}.py"
        (generations_dir / fname).write_text(result.final_code, encoding="utf-8")

    def _maybe_save_session(self, result: TaskResult, task: EvaluationTask, sessions_dir: Path | None):
        if sessions_dir is None:
            return
        if not getattr(result, "session_state", None):
            return
        path = sessions_dir / f"{task.task_number:03d}_{task.entry_point}.json"
        save_session_state(result.session_state, path)

    def _build_report_generator(self, log_dir: Path) -> Optional[QueryReportGenerator]:
        if not getattr(self.config, "reports", None) or not self.config.reports.enabled:
            return None
        queries_dir = ensure_dir(log_dir / self.config.reports.output_subdir)
        report_cfg = ReportConfig(
            output_dir=queries_dir,
            include_rag_chunks=self.config.reports.include_rag_chunks,
            include_raw_responses=self.config.reports.include_raw_responses,
            max_chunk_preview_lines=self.config.reports.max_chunk_preview_lines,
            save_code_files=self.config.reports.save_code_files,
            generate_analysis=self.config.reports.generate_analysis,
            failed_only=self.config.reports.failed_only,
        )
        return QueryReportGenerator(report_cfg)

    def _maybe_generate_report(
        self,
        generator: QueryReportGenerator,
        task: EvaluationTask,
        result: TaskResult,
    ) -> Optional[Path]:
        if result.session_state is None:
            return None
        return generator.generate_from_session(
            session=result.session_state,
            task_id=task.task_id,
            test_passed=result.test_passed,
            test_error=result.test_error,
            prompt=task.prompt,
            final_code=result.final_code,
        )

    def _build_task_summary(
        self,
        task: EvaluationTask,
        result: TaskResult,
        report_dir: Optional[Path],
    ) -> Dict[str, Any]:
        return {
            "task_id": task.task_id,
            "display_id": f"{task.task_number:03d}",
            "status": result.status.value,
            "iterations": result.iterations_used or 0,
            "time_seconds": result.total_time_seconds,
            "test_error": result.test_error or "",
            "link": f"./{report_dir.name}/report.md" if report_dir else None,
        }
