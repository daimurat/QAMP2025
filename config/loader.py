"""
Config loader for orchestrator and agents.

Loads from a YAML file if present, then applies environment variable overrides.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

from models import OrchestratorConfig


ENV_OVERRIDES = {
    "ORCHESTRATOR_MAX_ITERATIONS": "max_iterations",
    "ORCHESTRATOR_TIMEOUT": "iteration_timeout_seconds",
    "CODE_EXECUTION_ENABLED": "enable_code_execution",
    "RAG_ENABLED": "enable_rag",
}


def load_config(path: str = "config/multi_agent_config.yaml") -> OrchestratorConfig:
    cfg = OrchestratorConfig()
    p = Path(path)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        orch = data.get("orchestrator", {})
        cfg.max_iterations = int(orch.get("max_iterations", cfg.max_iterations))
        cfg.iteration_timeout_seconds = int(
            orch.get("iteration_timeout_seconds", cfg.iteration_timeout_seconds)
        )
        cfg.code_execution_timeout_seconds = int(
            orch.get(
                "code_execution_timeout_seconds", cfg.code_execution_timeout_seconds
            )
        )
        cfg.max_rag_queries_per_iteration = int(
            orch.get("max_rag_queries_per_iteration", cfg.max_rag_queries_per_iteration)
        )
        cfg.max_total_rag_queries = int(
            orch.get("max_total_rag_queries", cfg.max_total_rag_queries)
        )
        models = data.get("models", {})
        cfg.planner_model = models.get("planner", {}).get("model", cfg.planner_model)
        cfg.code_model = models.get("code_agent", {}).get("model", cfg.code_model)
        cfg.evaluator_model = models.get("evaluator", {}).get(
            "model", cfg.evaluator_model
        )
        features = data.get("orchestrator", {})
        cfg.enable_code_execution = bool(
            features.get("enable_code_execution", cfg.enable_code_execution)
        )
        cfg.enable_rag = bool(features.get("enable_rag", cfg.enable_rag))
        cfg.persist_state = bool(features.get("persist_state", cfg.persist_state))

    # Environment overrides
    for env_key, attr in ENV_OVERRIDES.items():
        if env_key in os.environ:
            value = os.environ[env_key]
            if value.lower() in {"true", "false"}:
                setattr(cfg, attr, value.lower() == "true")
            elif value.isdigit():
                setattr(cfg, attr, int(value))
            else:
                setattr(cfg, attr, value)

    return cfg


__all__ = ["load_config"]
