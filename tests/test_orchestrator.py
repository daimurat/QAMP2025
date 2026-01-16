import asyncio
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents import (
    CodeAgentImpl,
    EvaluatorAgentImpl,
    ExecutorAgentImpl,
    PlannerAgentImpl,
)
from models import (
    Decision,
    ExecutionResult,
    OrchestratorConfig,
    Plan,
    PlannerInput,
    PlannerOutput,
    RetrievedDocument,
    SessionState,
    SubTask,
    TaskType,
    TerminationReason,
)
from orchestrator import MultiAgentOrchestrator, StateManager


class StubLLM:
    def __init__(self, content):
        self.content = content

    async def chat(self, messages, **kwargs):
        return type("Resp", (), {"content": self.content})


class StubPlanner(PlannerAgentImpl):
    def __init__(self, plan: Plan):
        self.plan_to_return = plan
        super().__init__(llm_client=StubLLM("{}"), prompt_path="prompts/planner_agent.txt")

    async def invoke(self, input: PlannerInput) -> PlannerOutput:
        return PlannerOutput(plan=self.plan_to_return, raw_response="{}")


class StubCodeAgent(CodeAgentImpl):
    def __init__(self, code: str):
        self.code = code
        super().__init__(llm_client=StubLLM(code), prompt_path="prompts/code_agent.txt")

    async def invoke(self, input):
        return type("CodeOut", (), {"code": self.code, "language": "python", "raw_response": self.code})


class StubSandbox:
    def __init__(self, success: bool = True):
        self.success = success

    async def execute(self, code, timeout_seconds, allowed_imports, working_directory, max_memory_mb):
        return ExecutionResult(
            success=self.success,
            stdout="ok" if self.success else "",
            stderr="" if self.success else "err",
            return_value=None,
            artifacts={},
            execution_time_ms=1,
            memory_usage_mb=1.0,
            error_type=None if self.success else "RuntimeError",
            error_traceback=None if self.success else "trace",
        )


class StubEvaluator(EvaluatorAgentImpl):
    def __init__(self, decision: Decision, feedback=None):
        self._decision = decision
        self._feedback = feedback
        super().__init__(llm_client=StubLLM("{}"), prompt_path="prompts/evaluator_agent.txt")

    async def invoke(self, input):
        from models import Evaluation

        eval_obj = Evaluation(
            evaluation_id="e1",
            answers_question=True,
            code_executes=True,
            output_valid=True,
            criteria_met={"c": True},
            correctness_score=1.0,
            completeness_score=1.0,
            code_quality_score=1.0,
            reasoning="answer",
            identified_issues=[],
            suggested_improvements=[],
        )
        return type("EvalOut", (), {
            "evaluation": eval_obj,
            "decision": self._decision,
            "feedback": self._feedback,
            "final_answer": "answer",
            "raw_response": "{}",
        })()


class StubRAG:
    async def retrieve(self, queries, top_k=5):
        return [
            RetrievedDocument(
                doc_id="doc1",
                text="t",
                source="s",
                url="u",
                relevance_score=0.9,
                retrieved_at=None,
                query_used=queries[0],
            )
        ]


def _simple_plan(rag_needed: bool = False) -> Plan:
    return Plan(
        plan_id="p",
        version=0,
        sub_tasks=[
            SubTask(
                task_id="t1",
                description="d",
                task_type=TaskType.RETRIEVAL if rag_needed else TaskType.COMPUTATION,
                dependencies=[],
            )
        ],
        rag_needed=rag_needed,
        rag_queries=["q"] if rag_needed else [],
        code_needed=True,
        code_requirements=[],
        acceptance_criteria=[],
        uses_previous_context=False,
        referenced_iterations=[],
    )


@pytest.mark.asyncio
async def test_single_iteration_success(tmp_path: Path):
    planner = StubPlanner(plan=_simple_plan())
    code_agent = StubCodeAgent(code="# code")
    executor = ExecutorAgentImpl(sandbox=StubSandbox(success=True))
    evaluator = StubEvaluator(decision=Decision.SUCCESS)
    rag = StubRAG()
    state_manager = StateManager(base_dir=tmp_path)
    orchestrator = MultiAgentOrchestrator(
        planner=planner,
        code_agent=code_agent,
        executor=executor,
        evaluator=evaluator,
        rag_retriever=rag,
        state_manager=state_manager,
        config=OrchestratorConfig(max_iterations=2),
    )

    result = await orchestrator.run("Q")
    assert result.success is True
    assert result.termination_reason == TerminationReason.SUCCESS
    assert result.iterations_used == 1
    assert len(result.session_state.iterations) == 1


@pytest.mark.asyncio
async def test_retry_then_success(tmp_path: Path):
    planner = StubPlanner(plan=_simple_plan())
    code_agent = StubCodeAgent(code="# code")
    executor = ExecutorAgentImpl(sandbox=StubSandbox(success=True))

    first_eval = StubEvaluator(decision=Decision.RETRY, feedback="fix")
    second_eval = StubEvaluator(decision=Decision.SUCCESS)

    class SwitchEval:
        def __init__(self):
            self.calls = 0

        async def invoke(self, input):
            self.calls += 1
            if self.calls == 1:
                return await first_eval.invoke(input)
            return await second_eval.invoke(input)

        @property
        def name(self):
            return "evaluator"

        def get_system_prompt(self):
            return ""

    evaluator = SwitchEval()
    rag = StubRAG()
    state_manager = StateManager(base_dir=tmp_path)
    orchestrator = MultiAgentOrchestrator(
        planner=planner,
        code_agent=code_agent,
        executor=executor,
        evaluator=evaluator,  # type: ignore[arg-type]
        rag_retriever=rag,
        state_manager=state_manager,
        config=OrchestratorConfig(max_iterations=3),
    )

    result = await orchestrator.run("Q")
    assert len(result.session_state.iterations) == 2
    assert result.iterations_used == 2
    assert result.termination_reason == TerminationReason.SUCCESS


@pytest.mark.asyncio
async def test_max_iterations(tmp_path: Path):
    planner = StubPlanner(plan=_simple_plan())
    code_agent = StubCodeAgent(code="# code")
    executor = ExecutorAgentImpl(sandbox=StubSandbox(success=True))
    evaluator = StubEvaluator(decision=Decision.RETRY)
    rag = StubRAG()
    state_manager = StateManager(base_dir=tmp_path)
    orchestrator = MultiAgentOrchestrator(
        planner=planner,
        code_agent=code_agent,
        executor=executor,
        evaluator=evaluator,
        rag_retriever=rag,
        state_manager=state_manager,
        config=OrchestratorConfig(max_iterations=2),
    )

    result = await orchestrator.run("Q")
    assert result.success is False
    assert result.termination_reason == TerminationReason.MAX_ITERATIONS
    assert len(result.session_state.iterations) == 2


@pytest.mark.asyncio
async def test_rag_retrieval_added_to_state(tmp_path: Path):
    planner = StubPlanner(plan=_simple_plan(rag_needed=True))
    code_agent = StubCodeAgent(code="# code")
    executor = ExecutorAgentImpl(sandbox=StubSandbox(success=True))
    evaluator = StubEvaluator(decision=Decision.SUCCESS)
    rag = StubRAG()
    state_manager = StateManager(base_dir=tmp_path)
    orchestrator = MultiAgentOrchestrator(
        planner=planner,
        code_agent=code_agent,
        executor=executor,
        evaluator=evaluator,
        rag_retriever=rag,
        state_manager=state_manager,
        config=OrchestratorConfig(max_iterations=1),
    )

    result = await orchestrator.run("Q")
    assert result.session_state.total_rag_calls == 1
    assert len(result.session_state.all_retrieved_docs) == 1

