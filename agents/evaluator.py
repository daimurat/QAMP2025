"""
Evaluator agent implementation.

Calls an injected LLM client to assess execution results and decide next action.
"""
from __future__ import annotations

import json
from pathlib import Path

from agents.base import AgentError, EvaluatorAgent
from models import Decision, EvaluatorInput, EvaluatorOutput, Evaluation


class EvaluatorAgentImpl(EvaluatorAgent):
    def __init__(
        self,
        llm_client: any,
        prompt_path: str = "prompts/evaluator_agent.txt",
    ):
        self.llm = llm_client
        self.prompt_path = prompt_path
        self._system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        path = Path(self.prompt_path)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return "You are the EVALUATOR agent. Return JSON per schema."

    def get_system_prompt(self) -> str:
        return self._system_prompt

    async def invoke(self, input: EvaluatorInput) -> EvaluatorOutput:
        user_message = self._build_user_message(input)
        response = await self.llm.chat(
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = getattr(response, "content", "")
        return self._parse_output(raw)

    def _build_user_message(self, input: EvaluatorInput) -> str:
        parts = [
            f"## User Question\n{input.user_question}",
            "\n## Plan\n",
            str(input.plan.to_dict()),
            "\n## Code\n",
            input.code,
            "\n## Execution Result\n",
            f"success: {input.execution_result.success}\nstdout:\n{input.execution_result.stdout}\nstderr:\n{input.execution_result.stderr}\nerror:{input.execution_result.error_traceback}",
        ]
        # Add test verification result if available
        if input.test_result is not None:
            parts.append("\n## Test Verification\n")
            parts.append(f"test_passed: {input.test_result.passed}")
            if input.test_result.error:
                parts.append(f"\ntest_error: {input.test_result.error}")
        parts.append(f"\n## Iteration {input.iteration_number}/{input.max_iterations}")
        return "\n".join(parts)

    def _parse_output(self, raw: str) -> EvaluatorOutput:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise AgentError(f"Evaluator returned invalid JSON: {e}")

        assessment = data.get("assessment", {})
        scores = data.get("scores", {})
        decision_str = data.get("decision", "").lower()
        if decision_str not in Decision._value2member_map_:
            raise AgentError(f"Evaluator decision invalid: {decision_str}")
        decision = Decision(decision_str)

        eval_obj = Evaluation(
            evaluation_id=data.get("evaluation_id", ""),
            answers_question=bool(assessment.get("answers_question", False)),
            code_executes=bool(assessment.get("code_executes", False)),
            output_valid=bool(assessment.get("output_valid", False)),
            criteria_met=assessment.get("criteria_met", {}),
            correctness_score=float(scores.get("correctness", 0.0)),
            completeness_score=float(scores.get("completeness", 0.0)),
            code_quality_score=float(scores.get("code_quality", 0.0)),
            reasoning=data.get("reasoning", ""),
            identified_issues=data.get("issues", []),
            suggested_improvements=data.get("suggested_improvements", []),
        )

        return EvaluatorOutput(
            evaluation=eval_obj,
            decision=decision,
            feedback=data.get("feedback"),
            final_answer=data.get("final_answer"),
            raw_response=raw,
        )


__all__ = ["EvaluatorAgentImpl"]
