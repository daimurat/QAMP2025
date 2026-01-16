"""
Multi-agent orchestrator implementation (async).
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import List, Optional

from agents import CodeAgent, EvaluatorAgent, ExecutorAgent, PlannerAgent
from models import (
    CodeAgentInput,
    Decision,
    EvaluatorInput,
    ExecutorInput,
    IterationState,
    OrchestratorConfig,
    OrchestratorResult,
    Plan,
    RetrievedDocument,
    RetryPolicy,
    SessionState,
    TerminationReason,
)
from orchestrator.retry import with_retry
from orchestrator.state_manager import StateManager
from orchestrator.timeout_manager import TimeoutManager


class MultiAgentOrchestrator:
    def __init__(
        self,
        planner: PlannerAgent,
        code_agent: CodeAgent,
        executor: ExecutorAgent,
        evaluator: EvaluatorAgent,
        rag_retriever: any,
        state_manager: StateManager,
        config: OrchestratorConfig,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        self.planner = planner
        self.code_agent = code_agent
        self.executor = executor
        self.evaluator = evaluator
        self.rag = rag_retriever
        self.state_manager = state_manager
        self.config = config
        self.retry_policy = retry_policy or RetryPolicy()
        self.timeout_manager = TimeoutManager(
            iteration_timeout=config.iteration_timeout_seconds,
            execution_timeout=config.code_execution_timeout_seconds,
        )
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def get_state(self) -> SessionState:
        raise NotImplementedError("State retrieval requires session_id")

    async def run(
        self,
        user_question: str,
        config: Optional[OrchestratorConfig] = None,
    ) -> OrchestratorResult:
        config = config or self.config
        start_time = time.time()
        session = self.state_manager.create_session(user_question, config)

        try:
            while session.current_iteration < config.max_iterations:
                if self._cancelled:
                    return self._create_cancelled_result(session, start_time)

                iteration_result = await self._run_iteration(
                    session=session,
                    user_question=user_question,
                    config=config,
                )
                self.state_manager.save_iteration(session.session_id, iteration_result)
                session.iterations.append(iteration_result)
                session.current_iteration += 1

                if iteration_result.decision == Decision.SUCCESS:
                    return self._create_success_result(
                        session=session,
                        final_answer=iteration_result.evaluation.reasoning
                        if iteration_result.evaluation
                        else "",
                        final_code=iteration_result.generated_code,
                        start_time=start_time,
                    )
                if iteration_result.decision == Decision.ABORT:
                    return self._create_abort_result(
                        session=session,
                        reason=iteration_result.feedback or "",
                        start_time=start_time,
                    )

            return self._create_max_iterations_result(session, start_time)

        except Exception:
            return self._create_error_result(session, start_time)
        finally:
            session.end_time = datetime.now()
            self.state_manager.update_session(
                session.session_id, {"end_time": session.end_time}
            )

    async def _run_iteration(
        self,
        session: SessionState,
        user_question: str,
        config: OrchestratorConfig,
    ) -> IterationState:
        iteration_id = session.current_iteration
        prev_iteration = session.iterations[-1] if session.iterations else None

        async with self.timeout_manager.iteration_scope():
            # Planner
            planner_input = self._build_planner_input(
                user_question=user_question,
                iteration_id=iteration_id,
                prev_iteration=prev_iteration,
                session=session,
            )
            planner_output = await with_retry(
                lambda: self.planner.invoke(planner_input),
                self.retry_policy,
                context=f"planner-{iteration_id}",
            )
            plan: Plan = planner_output.plan

            # RAG
            retrieved_docs: List[RetrievedDocument] = []
            if plan.rag_needed and plan.rag_queries:
                queries = plan.rag_queries[: config.max_rag_queries_per_iteration]
                rag_results = await with_retry(
                    lambda: self.rag.retrieve(queries, top_k=5),
                    self.retry_policy,
                    context=f"rag-{iteration_id}",
                )
                retrieved_docs = rag_results or []
                if retrieved_docs:
                    self.state_manager.add_retrieved_docs(session.session_id, retrieved_docs)
                    for doc in retrieved_docs:
                        session.all_retrieved_docs[doc.doc_id] = doc
                    session.total_rag_calls += len(queries)

            # Code Agent
            context = self._format_context(list(session.all_retrieved_docs.values()))
            code_input = CodeAgentInput(
                plan=plan,
                code_requirements=plan.code_requirements,
                retrieved_context=context,
                previous_code=prev_iteration.generated_code if prev_iteration else None,
                previous_error=(
                    prev_iteration.execution_result.error_traceback
                    if prev_iteration
                    and prev_iteration.execution_result
                    and not prev_iteration.execution_result.success
                    else None
                ),
                previous_feedback=prev_iteration.feedback if prev_iteration else None,
            )
            code_output = await with_retry(
                lambda: self.code_agent.invoke(code_input),
                self.retry_policy,
                context=f"code-{iteration_id}",
            )

            session.total_llm_calls += 2  # planner + code agent

            # Executor
            execution_result = None
            if config.enable_code_execution and plan.code_needed:
                exec_input = ExecutorInput(
                    code=code_output.code,
                    timeout_seconds=config.code_execution_timeout_seconds,
                    working_directory=f"work/session_{session.session_id}",
                )
                async with self.timeout_manager.execution_scope():
                    executor_output = await self.executor.invoke(exec_input)
                    execution_result = executor_output.result
                session.total_code_executions += 1
            else:
                execution_result = None

            if execution_result is None:
                from models import ExecutionResult

                execution_result = ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="Code execution disabled",
                    return_value=None,
                    artifacts={},
                    execution_time_ms=0,
                    memory_usage_mb=0.0,
                    error_type=None,
                    error_traceback=None,
                )

            # Evaluator
            evaluator_input = EvaluatorInput(
                user_question=user_question,
                plan=plan,
                code=code_output.code,
                execution_result=execution_result,
                iteration_number=iteration_id,
                max_iterations=config.max_iterations,
                previous_evaluations=[
                    it.evaluation for it in session.iterations if it.evaluation
                ],
            )
            evaluator_output = await with_retry(
                lambda: self.evaluator.invoke(evaluator_input),
                self.retry_policy,
                context=f"eval-{iteration_id}",
            )
            session.total_llm_calls += 1

            return IterationState(
                iteration_id=iteration_id,
                timestamp=datetime.now(),
                plan=plan,
                rag_queries=plan.rag_queries if plan.rag_needed else [],
                retrieved_documents=retrieved_docs,
                generated_code=code_output.code,
                code_version=iteration_id,
                execution_result=execution_result,
                evaluation=evaluator_output.evaluation,
                decision=evaluator_output.decision,
                feedback=evaluator_output.feedback,
            )

    def _build_planner_input(
        self,
        user_question: str,
        iteration_id: int,
        prev_iteration: Optional[IterationState],
        session: SessionState,
    ):
        from models import PlannerInput

        return PlannerInput(
            user_question=user_question,
            iteration_number=iteration_id,
            previous_feedback=prev_iteration.feedback if prev_iteration else None,
            previous_plan=prev_iteration.plan if prev_iteration else None,
            previous_code=prev_iteration.generated_code if prev_iteration else None,
            previous_execution_result=prev_iteration.execution_result
            if prev_iteration
            else None,
            all_retrieved_docs=list(session.all_retrieved_docs.values()),
        )

    def _format_context(self, docs: List[RetrievedDocument]) -> str:
        if not docs:
            return ""
        sorted_docs = sorted(docs, key=lambda d: d.relevance_score, reverse=True)
        parts = []
        for i, doc in enumerate(sorted_docs[:10]):
            parts.append(
                f"[Doc {i+1}] (score={doc.relevance_score:.2f}) {doc.text}"
            )
        return "\n\n".join(parts)

    def _create_success_result(
        self,
        session: SessionState,
        final_answer: str,
        final_code: str,
        start_time: float,
    ) -> OrchestratorResult:
        session.final_answer = final_answer
        session.final_code = final_code
        session.termination_reason = TerminationReason.SUCCESS
        return OrchestratorResult(
            success=True,
            final_answer=final_answer,
            final_code=final_code,
            session_id=session.session_id,
            iterations_used=session.current_iteration,
            termination_reason=TerminationReason.SUCCESS,
            total_time_seconds=time.time() - start_time,
            session_state=session,
        )

    def _create_abort_result(
        self, session: SessionState, reason: str, start_time: float
    ) -> OrchestratorResult:
        session.termination_reason = TerminationReason.ABORT_REQUESTED
        return OrchestratorResult(
            success=False,
            final_answer=reason,
            final_code=None,
            session_id=session.session_id,
            iterations_used=session.current_iteration,
            termination_reason=TerminationReason.ABORT_REQUESTED,
            total_time_seconds=time.time() - start_time,
            session_state=session,
        )

    def _create_max_iterations_result(
        self, session: SessionState, start_time: float
    ) -> OrchestratorResult:
        session.termination_reason = TerminationReason.MAX_ITERATIONS
        best_iteration = None
        if session.iterations:
            best_iteration = max(
                session.iterations,
                key=lambda it: it.evaluation.correctness_score
                if it.evaluation
                else 0.0,
            )
        return OrchestratorResult(
            success=False,
            final_answer=None,
            final_code=best_iteration.generated_code if best_iteration else None,
            session_id=session.session_id,
            iterations_used=session.current_iteration,
            termination_reason=TerminationReason.MAX_ITERATIONS,
            total_time_seconds=time.time() - start_time,
            session_state=session,
        )

    def _create_error_result(
        self, session: SessionState, start_time: float
    ) -> OrchestratorResult:
        session.termination_reason = TerminationReason.FATAL_ERROR
        return OrchestratorResult(
            success=False,
            final_answer="Error occurred",
            final_code=None,
            session_id=session.session_id,
            iterations_used=session.current_iteration,
            termination_reason=TerminationReason.FATAL_ERROR,
            total_time_seconds=time.time() - start_time,
            session_state=session,
        )

    def _create_cancelled_result(
        self, session: SessionState, start_time: float
    ) -> OrchestratorResult:
        session.termination_reason = TerminationReason.USER_CANCELLED
        return OrchestratorResult(
            success=False,
            final_answer=None,
            final_code=None,
            session_id=session.session_id,
            iterations_used=session.current_iteration,
            termination_reason=TerminationReason.USER_CANCELLED,
            total_time_seconds=time.time() - start_time,
            session_state=session,
        )


__all__ = ["MultiAgentOrchestrator"]
