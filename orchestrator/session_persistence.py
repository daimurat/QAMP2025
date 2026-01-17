"""
Helpers to persist session state to disk.
"""
from __future__ import annotations

import json
from pathlib import Path

from models import SessionState


def save_session_state(session: SessionState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")


__all__ = ["save_session_state"]
