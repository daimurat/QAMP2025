import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Ensure project root is importable when running pytest from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents import (  # noqa: E402
    Agent,
    AgentError,
    CodeAgent,
    CodeAgentImpl,
    EvaluatorAgent,
    EvaluatorAgentImpl,
    ExecutorAgent,
    ExecutorAgentImpl,
    PlannerAgent,
    PlannerAgentImpl,
)
from models import (  # noqa: E402
    CodeAgentInput,
    CodeAgentOutput,
    Decision,
    Evaluation,
    ExecutionResult,
    EvaluatorInput,
    EvaluatorOutput,
    ExecutorInput,
    ExecutorOutput,
    Plan,
    PlannerInput,
    PlannerOutput,
    SubTask,
    TaskType,
)


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


def test_agent_is_abstract():
    with pytest.raises(TypeError):
        Agent()  # type: ignore[abstract]


def test_planner_agent_requires_impl():
    with pytest.raises(TypeError):
        PlannerAgent()  # type: ignore[abstract]


class DummyPlanner(PlannerAgent):
    def get_system_prompt(self) -> str:
        return "prompt"

    async def invoke(self, input: PlannerInput) -> PlannerOutput:
        return PlannerOutput(plan=_sample_plan(), raw_response="ok")


@pytest.mark.asyncio
async def test_concrete_planner_invocation():
    agent = DummyPlanner()
    assert agent.name == "planner"
    out = await agent.invoke(
        PlannerInput(user_question="q", iteration_number=0)
    )
    assert isinstance(out, PlannerOutput)
    assert out.plan.plan_id == "p1"


class DummyCodeAgent(CodeAgent):
    def get_system_prompt(self) -> str:
        return "prompt"

    async def invoke(self, input: CodeAgentInput) -> CodeAgentOutput:
        return CodeAgentOutput(code="# code", language="python", raw_response="ok")


@pytest.mark.asyncio
async def test_concrete_code_agent_invocation():
    agent = DummyCodeAgent()
    assert agent.name == "code_agent"
    out = await agent.invoke(
        CodeAgentInput(
            user_question="def sample_func(): pass",
            plan=_sample_plan(),
            code_requirements=[],
            retrieved_context="",
        )
    )
    assert out.language == "python"


class DummyExecutor(ExecutorAgent):
    def get_system_prompt(self) -> str:
        return ""

    async def invoke(self, input: ExecutorInput) -> ExecutorOutput:
        return ExecutorOutput(
            result=ExecutionResult(
                success=True,
                stdout="",
                stderr="",
                return_value=None,
                artifacts={},
                execution_time_ms=0,
                memory_usage_mb=0.0,
            ),
            execution_id="exec1",
        )


@pytest.mark.asyncio
async def test_concrete_executor_invocation():
    agent = DummyExecutor()
    assert agent.name == "executor"
    out = await agent.invoke(ExecutorInput(code="print('hi')"))
    assert out.result.success is True


class DummyEvaluator(EvaluatorAgent):
    def get_system_prompt(self) -> str:
        return "prompt"

    async def invoke(self, input: EvaluatorInput) -> EvaluatorOutput:
        eval_obj = Evaluation(
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
        return EvaluatorOutput(
            evaluation=eval_obj,
            decision=Decision.SUCCESS,
            feedback=None,
            final_answer="ok",
            raw_response="ok",
        )


@pytest.mark.asyncio
async def test_concrete_evaluator_invocation():
    agent = DummyEvaluator()
    assert agent.name == "evaluator"
    exec_result = ExecutionResult(
        success=True,
        stdout="",
        stderr="",
        return_value=None,
        artifacts={},
        execution_time_ms=0,
        memory_usage_mb=0.0,
    )
    input_obj = EvaluatorInput(
        user_question="q",
        plan=_sample_plan(),
        code="# code",
        execution_result=exec_result,
        iteration_number=0,
        max_iterations=1,
        previous_evaluations=[],
    )

    out = await agent.invoke(input_obj)
    assert out.decision is Decision.SUCCESS


class StubLLM:
    """Simple stub that echoes canned content."""

    def __init__(self, content: str):
        self.content = content

    async def chat(self, messages, **kwargs):
        return type("Resp", (), {"content": self.content})


@pytest.mark.asyncio
async def test_planner_agent_impl_parses_plan():
    llm = StubLLM(
        content=json.dumps(
            {
                "analysis": "ok",
                "sub_tasks": [
                    {
                        "task_id": "t1",
                        "description": "Retrieve docs",
                        "task_type": "retrieval",
                        "dependencies": [],
                    }
                ],
                "rag_needed": True,
                "rag_queries": ["q"],
                "code_needed": True,
                "code_requirements": ["req"],
                "acceptance_criteria": ["crit"],
                "uses_previous_context": False,
                "referenced_iterations": [],
            }
        )
    )
    agent = PlannerAgentImpl(llm_client=llm, prompt_path="prompts/planner_agent.txt")
    out = await agent.invoke(PlannerInput(user_question="q", iteration_number=0))
    assert out.plan.rag_needed is True
    assert out.plan.rag_queries == ["q"]


@pytest.mark.asyncio
async def test_code_agent_impl_extracts_code():
    llm = StubLLM(content="```python\nprint('hi')\n```")
    agent = CodeAgentImpl(llm_client=llm, prompt_path="prompts/code_agent.txt")
    out = await agent.invoke(
        CodeAgentInput(
            user_question="def sample_func(): pass",
            plan=_sample_plan(),
            code_requirements=[],
            retrieved_context="",
        )
    )
    assert out.code.strip() == "print('hi')"


class DummySandbox:
    async def execute(
        self,
        code: str,
        timeout_seconds: int,
        allowed_imports,
        working_directory: str,
        max_memory_mb: int,
    ):
        return ExecutionResult(
            success=True,
            stdout="ok",
            stderr="",
            return_value=None,
            artifacts={},
            execution_time_ms=1,
            memory_usage_mb=1.0,
        )


@pytest.mark.asyncio
async def test_executor_agent_impl_runs_sandbox(tmp_path: Path):
    sandbox = DummySandbox()
    agent = ExecutorAgentImpl(sandbox=sandbox)
    out = await agent.invoke(
        ExecutorInput(code="print('ok')", working_directory=str(tmp_path))
    )
    assert out.result.success is True


@pytest.mark.asyncio
async def test_evaluator_agent_impl_parses_decision():
    llm = StubLLM(
        content=json.dumps(
            {
                "assessment": {
                    "answers_question": True,
                    "code_executes": True,
                    "output_valid": True,
                    "criteria_met": {"c": True},
                },
                "scores": {"correctness": 1.0, "completeness": 1.0, "code_quality": 1.0},
                "reasoning": "ok",
                "issues": [],
                "decision": "success",
                "feedback": None,
                "final_answer": "done",
            }
        )
    )
    agent = EvaluatorAgentImpl(llm_client=llm, prompt_path="prompts/evaluator_agent.txt")
    exec_result = ExecutionResult(
        success=True,
        stdout="",
        stderr="",
        return_value=None,
        artifacts={},
        execution_time_ms=0,
        memory_usage_mb=0.0,
    )
    input_obj = EvaluatorInput(
        user_question="q",
        plan=_sample_plan(),
        code="# code",
        execution_result=exec_result,
        iteration_number=0,
        max_iterations=1,
        previous_evaluations=[],
    )
    out = await agent.invoke(input_obj)
    assert out.decision is Decision.SUCCESS
