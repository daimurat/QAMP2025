"""
Evaluation configuration loader.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from models import OrchestratorConfig


@dataclass
class RagEvalConfig:
    db_path: str = "QAMP/data/qamp.db"
    top_k: int = 5
    min_score: float = 0.0


@dataclass
class EvaluationSettings:
    dataset: str = "Qiskit/qiskit_humaneval"
    split: str = "test"
    difficulty: Optional[str] = None
    max_tasks: Optional[int] = None
    task_ids: Optional[List[str]] = None
    timeout_per_task_seconds: int = 300
    retry_failed_tasks: bool = False
    max_retries_per_task: int = 1
    output_dir: str = "evaluation_runs"
    save_sessions: bool = True
    save_generations: bool = True
    verbose: bool = False
    quiet: bool = False
    dry_run: bool = False
    resume: Optional[str] = None


@dataclass
class ModelSettings:
    provider: str = "groq"  # "groq" or "openrouter"
    planner_model: str = "moonshotai/kimi-k2-instruct-0905"
    code_model: str = "moonshotai/kimi-k2-instruct-0905"
    evaluator_model: str = "moonshotai/kimi-k2-instruct-0905"


@dataclass
class ReportSettings:
    enabled: bool = False
    output_subdir: str = "queries"
    failed_only: bool = False
    include_rag_chunks: bool = True
    include_raw_responses: bool = False
    max_chunk_preview_lines: int = 30
    save_code_files: bool = True
    generate_analysis: bool = False


@dataclass
class EvaluationConfig:
    evaluation: EvaluationSettings
    orchestrator: OrchestratorConfig
    models: ModelSettings
    rag: RagEvalConfig
    reports: ReportSettings


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_evaluation_config(path: str = "config/evaluation_config.yaml") -> EvaluationConfig:
    data = _load_yaml(Path(path))

    eval_data = data.get("evaluation", {})
    evaluation = EvaluationSettings(
        dataset=eval_data.get("dataset", EvaluationSettings.dataset),
        split=eval_data.get("split", EvaluationSettings.split),
        difficulty=eval_data.get("difficulty"),
        max_tasks=eval_data.get("max_tasks"),
        task_ids=eval_data.get("task_ids"),
        timeout_per_task_seconds=eval_data.get(
            "timeout_per_task_seconds", EvaluationSettings.timeout_per_task_seconds
        ),
        retry_failed_tasks=eval_data.get(
            "retry_failed_tasks", EvaluationSettings.retry_failed_tasks
        ),
        max_retries_per_task=eval_data.get(
            "max_retries_per_task", EvaluationSettings.max_retries_per_task
        ),
        output_dir=eval_data.get("output_dir", EvaluationSettings.output_dir),
        save_sessions=eval_data.get("save_sessions", EvaluationSettings.save_sessions),
        save_generations=eval_data.get(
            "save_generations", EvaluationSettings.save_generations
        ),
        verbose=eval_data.get("verbose", EvaluationSettings.verbose),
        quiet=eval_data.get("quiet", EvaluationSettings.quiet),
        dry_run=eval_data.get("dry_run", EvaluationSettings.dry_run),
        resume=eval_data.get("resume", EvaluationSettings.resume),
    )

    orch_data = data.get("orchestrator", {})
    orchestrator = OrchestratorConfig(
        max_iterations=orch_data.get("max_iterations", OrchestratorConfig.max_iterations),
        iteration_timeout_seconds=orch_data.get(
            "iteration_timeout_seconds", OrchestratorConfig.iteration_timeout_seconds
        ),
        code_execution_timeout_seconds=orch_data.get(
            "code_execution_timeout_seconds",
            OrchestratorConfig.code_execution_timeout_seconds,
        ),
        max_rag_queries_per_iteration=orch_data.get(
            "max_rag_queries_per_iteration",
            OrchestratorConfig.max_rag_queries_per_iteration,
        ),
        max_total_rag_queries=orch_data.get(
            "max_total_rag_queries", OrchestratorConfig.max_total_rag_queries
        ),
        planner_model=orch_data.get("planner_model", OrchestratorConfig.planner_model),
        code_model=orch_data.get("code_model", OrchestratorConfig.code_model),
        evaluator_model=orch_data.get(
            "evaluator_model", OrchestratorConfig.evaluator_model
        ),
        enable_code_execution=orch_data.get(
            "enable_code_execution", OrchestratorConfig.enable_code_execution
        ),
        enable_rag=orch_data.get("enable_rag", OrchestratorConfig.enable_rag),
        persist_state=orch_data.get("persist_state", OrchestratorConfig.persist_state),
    )

    models_data = data.get("models", {})
    models = ModelSettings(
        provider=models_data.get("provider", ModelSettings.provider),
        planner_model=models_data.get(
            "planner", {}
        ).get("model", ModelSettings.planner_model),
        code_model=models_data.get("code_agent", {}).get(
            "model", ModelSettings.code_model
        ),
        evaluator_model=models_data.get("evaluator", {}).get(
            "model", ModelSettings.evaluator_model
        ),
    )

    rag_data = data.get("rag", {})
    rag = RagEvalConfig(
        db_path=rag_data.get("db_path", RagEvalConfig.db_path),
        top_k=rag_data.get("top_k", RagEvalConfig.top_k),
        min_score=rag_data.get("min_score", RagEvalConfig.min_score),
    )

    reports_data = data.get("reports", {})
    reports = ReportSettings(
        enabled=reports_data.get("enabled", ReportSettings.enabled),
        output_subdir=reports_data.get("output_subdir", ReportSettings.output_subdir),
        failed_only=reports_data.get("failed_only", ReportSettings.failed_only),
        include_rag_chunks=reports_data.get(
            "include_rag_chunks", ReportSettings.include_rag_chunks
        ),
        include_raw_responses=reports_data.get(
            "include_raw_responses", ReportSettings.include_raw_responses
        ),
        max_chunk_preview_lines=reports_data.get(
            "max_chunk_preview_lines", ReportSettings.max_chunk_preview_lines
        ),
        save_code_files=reports_data.get(
            "save_code_files", ReportSettings.save_code_files
        ),
        generate_analysis=reports_data.get(
            "generate_analysis", ReportSettings.generate_analysis
        ),
    )

    _apply_env_overrides(evaluation, orchestrator, models, reports)

    return EvaluationConfig(
        evaluation=evaluation,
        orchestrator=orchestrator,
        models=models,
        rag=rag,
        reports=reports,
    )


def _apply_env_overrides(
    eval_settings: EvaluationSettings,
    orch: OrchestratorConfig,
    models: ModelSettings,
    reports: ReportSettings,
):
    # Evaluation envs
    if os.getenv("EVAL_DATASET"):
        eval_settings.dataset = os.getenv("EVAL_DATASET", eval_settings.dataset)
    if os.getenv("EVAL_DIFFICULTY"):
        eval_settings.difficulty = os.getenv("EVAL_DIFFICULTY")
    if os.getenv("EVAL_MAX_TASKS"):
        eval_settings.max_tasks = int(os.getenv("EVAL_MAX_TASKS"))
    if os.getenv("EVAL_OUTPUT_DIR"):
        eval_settings.output_dir = os.getenv("EVAL_OUTPUT_DIR", eval_settings.output_dir)
    if os.getenv("EVAL_VERBOSE"):
        eval_settings.verbose = os.getenv("EVAL_VERBOSE", "").lower() == "true"
    if os.getenv("EVAL_QUIET"):
        eval_settings.quiet = os.getenv("EVAL_QUIET", "").lower() == "true"
    if os.getenv("EVAL_TIMEOUT_PER_TASK"):
        eval_settings.timeout_per_task_seconds = int(
            os.getenv("EVAL_TIMEOUT_PER_TASK")
        )

    # Model/provider env overrides
    if os.getenv("EVAL_PROVIDER"):
        models.provider = os.getenv("EVAL_PROVIDER", models.provider)

    # Orchestrator env overrides
    if os.getenv("ORCHESTRATOR_MAX_ITERATIONS"):
        orch.max_iterations = int(os.getenv("ORCHESTRATOR_MAX_ITERATIONS"))
    if os.getenv("ORCHESTRATOR_TIMEOUT"):
        orch.iteration_timeout_seconds = int(os.getenv("ORCHESTRATOR_TIMEOUT"))
    if os.getenv("CODE_EXECUTION_ENABLED"):
        orch.enable_code_execution = (
            os.getenv("CODE_EXECUTION_ENABLED", "").lower() == "true"
        )
    if os.getenv("RAG_ENABLED"):
        orch.enable_rag = os.getenv("RAG_ENABLED", "").lower() == "true"

    # Report generation env overrides
    if os.getenv("MAS_OBSERVABILITY_ENABLED"):
        reports.enabled = os.getenv("MAS_OBSERVABILITY_ENABLED", "").lower() == "true"
    if os.getenv("MAS_REPORT_FAILED_ONLY"):
        reports.failed_only = os.getenv("MAS_REPORT_FAILED_ONLY", "").lower() == "true"
    if os.getenv("MAS_REPORT_DIR"):
        reports.output_subdir = os.getenv("MAS_REPORT_DIR", reports.output_subdir)
