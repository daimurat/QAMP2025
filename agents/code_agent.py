"""
Code agent implementation.

Calls an injected LLM client and extracts Python code from the response.
"""
from __future__ import annotations

import re
from pathlib import Path

from agents.base import AgentError, CodeAgent
from models import CodeAgentInput, CodeAgentOutput


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


class CodeAgentImpl(CodeAgent):
    def __init__(
        self,
        llm_client: any,
        prompt_path: str = "prompts/code_agent.txt",
    ):
        self.llm = llm_client
        self.prompt_path = prompt_path
        self._system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        path = Path(self.prompt_path)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return "You are the CODE agent. Return Python code inside one code block."

    def get_system_prompt(self) -> str:
        return self._system_prompt

    async def invoke(self, input: CodeAgentInput) -> CodeAgentOutput:
        user_message = self._build_user_message(input)
        response = await self.llm.chat(
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
        )
        raw = getattr(response, "content", "")
        code = self._extract_code(raw)
        if not code:
            raise AgentError("Code agent returned empty code")
        return CodeAgentOutput(code=code, language="python", raw_response=raw)

    def _build_user_message(self, input: CodeAgentInput) -> str:
        parts = [
            "## Plan",
            str(input.plan.to_dict()),
            "\n## Requirements",
            "\n".join(input.code_requirements),
        ]
        if input.retrieved_context:
            parts.append("\n## Retrieved Context\n" + input.retrieved_context)
        if input.previous_error:
            parts.append("\n## Previous Error\n" + input.previous_error)
        if input.previous_feedback:
            parts.append("\n## Feedback\n" + input.previous_feedback)
        return "\n".join(parts)

    def _extract_code(self, raw: str) -> str:
        match = CODE_BLOCK_RE.search(raw)
        if match:
            return match.group(1).strip()
        return raw.strip()


__all__ = ["CodeAgentImpl"]
