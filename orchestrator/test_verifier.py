"""
Test verification abstraction for the orchestrator.

Allows injecting actual test execution into the orchestrator loop
so test failures can trigger retries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class TestResult:
    """Result of running test assertions against generated code."""
    passed: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {"passed": self.passed, "error": self.error}

    @classmethod
    def from_dict(cls, data: dict) -> "TestResult":
        return cls(passed=data.get("passed", False), error=data.get("error"))


class TestVerifier(Protocol):
    """Protocol for test verification implementations."""

    def verify(self, code: str) -> TestResult:
        """
        Run test assertions against generated code.
        
        Args:
            code: The generated code to test.
            
        Returns:
            TestResult with pass/fail status and any error message.
        """
        ...


__all__ = ["TestResult", "TestVerifier"]
