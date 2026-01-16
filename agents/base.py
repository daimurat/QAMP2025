"""
Agent base interfaces for the multi-agent system.

Defines the generic Agent ABC and concrete interfaces for planner, code,
executor, and evaluator agents.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from models import (
    CodeAgentInput,
    CodeAgentOutput,
    EvaluatorInput,
    EvaluatorOutput,
    ExecutorInput,
    ExecutorOutput,
    PlannerInput,
    PlannerOutput,
)


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class AgentError(Exception):
    """Raised when an agent fails to process a request."""


class Agent(ABC, Generic[InputT, OutputT]):
    """Base interface for all agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent identifier."""
        raise NotImplementedError

    @abstractmethod
    async def invoke(self, input: InputT) -> OutputT:
        """
        Process input and produce output.

        Args:
            input: Agent-specific input dataclass

        Returns:
            Agent-specific output dataclass

        Raises:
            AgentError: On processing failure
        """
        raise NotImplementedError

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the agent's system prompt."""
        raise NotImplementedError


class PlannerAgent(Agent[PlannerInput, PlannerOutput]):
    """Interface for the Planner agent."""

    @property
    def name(self) -> str:
        return "planner"


class CodeAgent(Agent[CodeAgentInput, CodeAgentOutput]):
    """Interface for the Code agent."""

    @property
    def name(self) -> str:
        return "code_agent"


class ExecutorAgent(Agent[ExecutorInput, ExecutorOutput]):
    """Interface for the Executor agent."""

    @property
    def name(self) -> str:
        return "executor"


class EvaluatorAgent(Agent[EvaluatorInput, EvaluatorOutput]):
    """Interface for the Evaluator agent."""

    @property
    def name(self) -> str:
        return "evaluator"


__all__ = [
    "Agent",
    "AgentError",
    "PlannerAgent",
    "CodeAgent",
    "ExecutorAgent",
    "EvaluatorAgent",
]
