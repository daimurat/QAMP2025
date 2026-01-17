import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from evaluation_mas.config import EvaluationConfig, load_evaluation_config
from evaluation_mas.runner import EvaluationRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate multi-agent system on Qiskit HumanEval")
    p.add_argument(
        "--config",
        default="config/evaluation_config.yaml",
        help="Path to evaluation config YAML",
    )
    p.add_argument("--task-id", default=None, help="Run single task by ID (e.g., 47)")
    p.add_argument(
        "--difficulty",
        default=None,
        choices=["basic", "intermediate", "difficult"],
        help="Filter by difficulty",
    )
    p.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks")
    p.add_argument("--dry-run", action="store_true", help="List tasks without running")
    p.add_argument("--quiet", action="store_true", help="Suppress extra output")
    p.add_argument(
        "--generate-reports",
        action="store_true",
        help="Generate human-readable query reports",
    )
    p.add_argument(
        "--reports-failed-only",
        action="store_true",
        help="Only generate reports for failed tasks",
    )
    p.add_argument(
        "--reports-subdir",
        default=None,
        help="Directory name for reports under the run folder",
    )
    return p.parse_args()


def apply_cli_overrides(cfg: EvaluationConfig, args: argparse.Namespace) -> None:
    if args.task_id:
        cfg.evaluation.task_ids = [args.task_id]
    if args.difficulty:
        cfg.evaluation.difficulty = args.difficulty
    if args.max_tasks is not None:
        cfg.evaluation.max_tasks = args.max_tasks
    if args.dry_run:
        cfg.evaluation.dry_run = True
    if args.quiet:
        cfg.evaluation.quiet = True
    if args.generate_reports:
        cfg.reports.enabled = True
    if args.reports_failed_only:
        cfg.reports.failed_only = True
        cfg.reports.enabled = True
    if args.reports_subdir:
        cfg.reports.output_subdir = args.reports_subdir


def main():
    project_root = Path(__file__).resolve().parents[1]
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)

    args = parse_args()
    cfg = load_evaluation_config(args.config)
    apply_cli_overrides(cfg, args)

    runner = EvaluationRunner(cfg)
    results = asyncio.run(runner.run())
    if cfg.evaluation.dry_run:
        return
    if not cfg.evaluation.quiet:
        passed = sum(1 for r in results if r.test_passed)
        print(f"Completed {len(results)} tasks. Passed: {passed}/{len(results)}")


if __name__ == "__main__":
    main()
