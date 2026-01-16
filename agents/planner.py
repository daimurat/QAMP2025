"""
Planner agent implementation.

Loads a system prompt from file, calls an injected LLM client, and parses the
returned JSON into a Plan dataclass.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from agents.base import AgentError, PlannerAgent
from models import Plan, PlannerInput, PlannerOutput, SubTask, TaskType


class PlannerAgentImpl(PlannerAgent):
    def __init__(
        self,
        llm_client: any,
        prompt_path: str = "prompts/planner_agent.txt",
    ):
        self.llm = llm_client
        self.prompt_path = prompt_path
        self._system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        path = Path(self.prompt_path)
        if path.exists():
            return path.read_text(encoding="utf-8")
        # Fallback minimal prompt
        return "You are the PLANNER agent for Qiskit. Return JSON per schema."

    def get_system_prompt(self) -> str:
        return self._system_prompt

    async def invoke(self, input: PlannerInput) -> PlannerOutput:
        user_message = self._build_user_message(input)
        response = await self.llm.chat(
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = getattr(response, "content", "")
        plan = self._parse_plan(raw, input.iteration_number)
        return PlannerOutput(plan=plan, raw_response=raw)

    def _build_user_message(self, input: PlannerInput) -> str:
        parts = [f"## User Question\n{input.user_question}"]
        if input.iteration_number > 0:
            parts.append(f"\n## Iteration\n{input.iteration_number}")
            if input.previous_feedback:
                parts.append(f"\n## Feedback\n{input.previous_feedback}")
        if input.all_retrieved_docs:
            parts.append(
                f"\n## Retrieved Docs\nYou have {len(input.all_retrieved_docs)} docs available."
            )
        return "\n".join(parts)

    def _parse_plan(self, raw: str, iteration: int) -> Plan:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise AgentError(f"Planner returned invalid JSON: {e}")

        try:
            sub_tasks = [
                SubTask(
                    task_id=t["task_id"],
                    description=t["description"],
                    task_type=TaskType(t["task_type"].lower())
                    if isinstance(t["task_type"], str)
                    else TaskType(t["task_type"]),
                    dependencies=t.get("dependencies", []),
                )
                for t in data.get("sub_tasks", [])
            ]
            return Plan(
                plan_id=data.get("plan_id", f"plan_{iteration}"),
                version=iteration,
                sub_tasks=sub_tasks,
                rag_needed=bool(data.get("rag_needed", False)),
                rag_queries=data.get("rag_queries", []),
                code_needed=bool(data.get("code_needed", True)),
                code_requirements=data.get("code_requirements", []),
                acceptance_criteria=data.get("acceptance_criteria", []),
                uses_previous_context=bool(data.get("uses_previous_context", False)),
                referenced_iterations=data.get("referenced_iterations", []),
            )
        except KeyError as e:
            raise AgentError(f"Planner response missing field: {e}")


__all__ = ["PlannerAgentImpl"]
