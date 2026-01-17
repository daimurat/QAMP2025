from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List

from evaluation_mas.config import EvaluationConfig
from evaluation_mas.results import TaskResult, TaskStatus


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_run_outputs(
    results: List[TaskResult],
    cfg: EvaluationConfig,
    run_id: str,
    sessions_dir: Path | None = None,
    generations_dir: Path | None = None,
) -> Path:
    """
    Persist run artifacts: config snapshot, results.csv, summary.json.
    """
    root = ensure_dir(Path(cfg.evaluation.output_dir) / run_id)

    # Save config snapshot
    (root / "config_used.json").write_text(
        json.dumps(_config_to_json(cfg), indent=2), encoding="utf-8"
    )

    write_results_csv(root / "results.csv", results)
    write_summary_json(root / "summary.json", results, cfg, run_id)

    # Save session states and generations if provided
    if sessions_dir:
        ensure_dir(root / "sessions")
    if generations_dir:
        ensure_dir(root / "generations")

    return root


def write_results_csv(path: Path, results: List[TaskResult]) -> None:
    fieldnames = [
        "task_id",
        "status",
        "orchestrator_success",
        "iterations_used",
        "termination_reason",
        "test_passed",
        "test_error",
        "total_time_seconds",
        "orchestrator_time_seconds",
        "verification_time_seconds",
        "final_code",
        "session_id",
        "error_message",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "task_id": r.task_id,
                    "status": r.status.value,
                    "orchestrator_success": r.orchestrator_success,
                    "iterations_used": r.iterations_used,
                    "termination_reason": r.termination_reason,
                    "test_passed": r.test_passed,
                    "test_error": r.test_error or "",
                    "total_time_seconds": f"{r.total_time_seconds:.3f}",
                    "orchestrator_time_seconds": f"{r.orchestrator_time_seconds:.3f}",
                    "verification_time_seconds": f"{r.verification_time_seconds:.3f}",
                    "final_code": r.final_code or "",
                    "session_id": r.session_id or "",
                    "error_message": r.error_message or "",
                }
            )


def write_summary_json(
    path: Path, results: List[TaskResult], cfg: EvaluationConfig, run_id: str
) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.test_passed)
    failed = sum(1 for r in results if r.status == TaskStatus.FAILED)
    errors = sum(1 for r in results if r.status == TaskStatus.ERROR)
    timeouts = sum(1 for r in results if r.status == TaskStatus.TIMEOUT)

    summary = {
        "metadata": {
            "run_id": run_id,
            "end_time": datetime.utcnow().isoformat(),
        },
        "configuration": {
            "dataset": cfg.evaluation.dataset,
            "difficulty": cfg.evaluation.difficulty,
            "max_tasks": cfg.evaluation.max_tasks,
            "model_planner": cfg.models.planner_model,
            "model_code": cfg.models.code_model,
            "model_evaluator": cfg.models.evaluator_model,
            "rag_enabled": cfg.orchestrator.enable_rag,
            "max_iterations": cfg.orchestrator.max_iterations,
        },
        "results": {
            "total_tasks": total,
            "passed": passed,
            "failed": failed,
            "error": errors,
            "timeout": timeouts,
            "pass_rate": (passed / total) if total else 0.0,
        },
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _config_to_json(cfg: EvaluationConfig):
    return {
        "evaluation": asdict(cfg.evaluation),
        "orchestrator": asdict(cfg.orchestrator),
        "models": {
            "planner_model": cfg.models.planner_model,
            "code_model": cfg.models.code_model,
            "evaluator_model": cfg.models.evaluator_model,
        },
        "rag": asdict(cfg.rag),
        "reports": asdict(cfg.reports),
    }
