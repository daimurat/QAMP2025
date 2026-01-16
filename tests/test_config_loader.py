import os
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import load_config  # noqa: E402


def test_load_config_from_yaml(tmp_path: Path):
    cfg_path = tmp_path / "multi_agent_config.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "orchestrator": {
                    "max_iterations": 3,
                    "iteration_timeout_seconds": 10,
                    "enable_code_execution": False,
                },
                "models": {
                    "planner": {"model": "gpt-x"},
                    "code_agent": {"model": "gpt-y"},
                    "evaluator": {"model": "gpt-z"},
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_path))
    assert cfg.max_iterations == 3
    assert cfg.iteration_timeout_seconds == 10
    assert cfg.enable_code_execution is False
    assert cfg.planner_model == "gpt-x"
    assert cfg.code_model == "gpt-y"
    assert cfg.evaluator_model == "gpt-z"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_MAX_ITERATIONS", "2")
    monkeypatch.setenv("CODE_EXECUTION_ENABLED", "false")
    cfg = load_config(path="nonexistent.yaml")
    assert cfg.max_iterations == 2
    assert cfg.enable_code_execution is False
