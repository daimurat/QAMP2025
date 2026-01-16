"""
State manager with JSON persistence.

Implements the interface from the specification: create sessions, save
iterations, update fields, and manage retrieved documents with deduplication.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from models import IterationState, RetrievedDocument, SessionState, TerminationReason


class StateManager:
    """
    JSON-backed state manager.

    Stores one JSON file per session under base_dir.
    """

    def __init__(self, base_dir: str = "state"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.json"

    def _write_session(self, state: SessionState) -> None:
        path = self._session_path(state.session_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=True, indent=2)

    def _read_session_dict(self, session_id: str) -> Dict[str, Any]:
        path = self._session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session {session_id} not found at {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def create_session(self, user_question: str, config: Any) -> SessionState:
        """
        Create a new session, persisted immediately.

        Args:
            user_question: Original user question.
            config: OrchestratorConfig-like object with max_iterations.
        """
        session_id = uuid.uuid4().hex
        max_iterations = getattr(config, "max_iterations", 0)

        state = SessionState(
            session_id=session_id,
            user_question=user_question,
            start_time=datetime.now(),
            end_time=None,
            max_iterations=max_iterations,
            current_iteration=0,
            iterations=[],
            all_retrieved_docs={},
            final_answer=None,
            final_code=None,
            termination_reason=None,
            total_rag_calls=0,
            total_llm_calls=0,
            total_code_executions=0,
        )
        self._write_session(state)
        return state

    def save_iteration(self, session_id: str, iteration: IterationState) -> None:
        """
        Persist an iteration. If the iteration ID exists, replace it; else append.
        """
        state = self.get_session(session_id)

        existing_ids = [it.iteration_id for it in state.iterations]
        if iteration.iteration_id in existing_ids:
            state.iterations = [
                iteration if it.iteration_id == iteration.iteration_id else it
                for it in state.iterations
            ]
        else:
            state.iterations.append(iteration)
        self._write_session(state)

    def get_session(self, session_id: str) -> SessionState:
        """Retrieve session state by ID."""
        data = self._read_session_dict(session_id)
        return SessionState.from_dict(data)

    def update_session(self, session_id: str, updates: Dict[str, Any]) -> None:
        """
        Update top-level fields on SessionState and persist.
        """
        state = self.get_session(session_id)
        for key, value in updates.items():
            if not hasattr(state, key):
                continue
            setattr(state, key, value)
        self._write_session(state)

    def add_retrieved_docs(
        self, session_id: str, docs: List[RetrievedDocument]
    ) -> None:
        """Add retrieved documents with deduplication by doc_id."""
        state = self.get_session(session_id)
        for doc in docs:
            state.all_retrieved_docs[doc.doc_id] = doc
        self._write_session(state)


__all__ = ["StateManager"]
