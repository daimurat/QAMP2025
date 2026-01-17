from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from jinja2 import Environment, FileSystemLoader

from models import IterationState, SessionState


def _default_template_dir() -> Path:
    return Path(__file__).resolve().parent / "templates"


def _format_task_dirname(task_id: str) -> str:
    """
    Convert a task id like ``qiskitHumanEval/47`` into a filesystem-friendly
    directory name such as ``qiskitHumanEval_047``.
    """
    cleaned = task_id.replace("/", "_").replace(" ", "_")
    suffix = "".join(ch for ch in cleaned.split("_")[-1] if ch.isdigit())
    if suffix:
        try:
            return f"{cleaned.rsplit('_', 1)[0]}_{int(suffix):03d}"
        except Exception:
            return cleaned
    return cleaned


def _format_seconds(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}s"


@dataclass
class ReportConfig:
    """
    Configuration for observability reports.

    Attributes mirror the specification with pragmatic defaults so reports
    can be enabled without custom templates or analysis.
    """

    output_dir: Path
    create_index: bool = True
    include_rag_chunks: bool = True
    include_raw_responses: bool = False
    max_chunk_preview_lines: int = 30
    save_code_files: bool = True
    generate_analysis: bool = False
    template_dir: Optional[Path] = None
    failed_only: bool = False


class QueryReportGenerator:
    """
    Generate human-readable Markdown reports from a SessionState.
    """

    def __init__(self, config: ReportConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tmpl_dir = Path(config.template_dir) if config.template_dir else _default_template_dir()
        self.env = Environment(
            loader=FileSystemLoader(str(tmpl_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.env.filters["truncate_lines"] = self._truncate_lines
        self.env.filters["basename"] = lambda value: Path(value).name

    def generate_from_session(
        self,
        session: SessionState | Dict[str, Any] | Path,
        task_id: str,
        test_passed: bool,
        test_error: Optional[str] = None,
        prompt: Optional[str] = None,
        final_code: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Generate the full report tree for a single session.

        Args:
            session: SessionState object, raw session dict, or path to a JSON session file.
            task_id: Unique task identifier (e.g., ``qiskitHumanEval/47``).
            test_passed: Whether the verification step passed.
            test_error: Optional verification error message.
            prompt: Optional original prompt; falls back to session.user_question.
            final_code: Optional final code override.

        Returns:
            Path to the query directory, or None if generation was skipped.
        """
        if self.config.failed_only and test_passed:
            return None

        session_state = self._normalize_session(session)
        query_dir = self._create_query_directory(task_id)

        prompt_text = prompt or session_state.user_question
        if prompt_text:
            (query_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")

        final_code_text = (
            final_code
            or session_state.final_code
            or (session_state.iterations[-1].generated_code if session_state.iterations else None)
        )
        if self.config.save_code_files and final_code_text:
            (query_dir / "final_code.py").write_text(final_code_text, encoding="utf-8")

        self._write_report(
            query_dir=query_dir,
            session=session_state,
            task_id=task_id,
            test_passed=test_passed,
            test_error=test_error,
        )
        for iteration in session_state.iterations:
            self._generate_iteration_reports(query_dir, iteration)

        return query_dir

    def generate_from_json_file(
        self,
        session_path: Path,
        task_id: str,
        test_passed: bool,
        test_error: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Convenience wrapper to load a session from disk and generate reports.
        """
        session_data = json.loads(Path(session_path).read_text(encoding="utf-8"))
        return self.generate_from_session(
            session=session_data,
            task_id=task_id,
            test_passed=test_passed,
            test_error=test_error,
            prompt=prompt,
        )

    def write_run_index(
        self,
        run_id: str,
        task_summaries: Sequence[Dict[str, Any]],
    ) -> Optional[Path]:
        """
        Write queries/index.md summarizing task statuses for a run.
        """
        if not self.config.create_index:
            return None
        index_template = self.env.get_template("index.md.jinja2")
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        normalized_tasks = [self._normalize_task_summary(t) for t in task_summaries]
        total = len(normalized_tasks)
        passed = sum(1 for t in normalized_tasks if t.get("status") == "passed")
        failed = sum(1 for t in normalized_tasks if t.get("status") == "failed")
        content = index_template.render(
            run_id=run_id,
            generated_at=now,
            tasks=normalized_tasks,
            totals={
                "total": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": (passed / total) if total else 0.0,
            },
        )
        index_path = self.output_dir / "index.md"
        index_path.write_text(content, encoding="utf-8")
        return index_path

    def _create_query_directory(self, task_id: str) -> Path:
        query_dir = self.output_dir / _format_task_dirname(task_id)
        query_dir.mkdir(parents=True, exist_ok=True)
        return query_dir

    def _write_report(
        self,
        query_dir: Path,
        session: SessionState,
        task_id: str,
        test_passed: bool,
        test_error: Optional[str],
    ) -> None:
        report_template = self.env.get_template("report.md.jinja2")
        iterations = sorted(session.iterations, key=lambda it: it.iteration_id)
        iteration_links = [
            {
                "iteration_id": it.iteration_id,
                "decision": it.decision.value if it.decision else "unknown",
                "path": f"./iteration_{it.iteration_id}/",
            }
            for it in iterations
        ]
        end_time = session.end_time or datetime.now()
        total_seconds = (end_time - session.start_time).total_seconds() if session.start_time else None
        content = report_template.render(
            task_id=task_id,
            status="passed" if test_passed else "failed",
            iterations=len(iterations),
            total_time=_format_seconds(total_seconds),
            termination=session.termination_reason.value if session.termination_reason else "n/a",
            prompt=session.user_question,
            test_error=test_error,
            iteration_links=iteration_links,
        )
        (query_dir / "report.md").write_text(content, encoding="utf-8")

    def _generate_iteration_reports(
        self,
        query_dir: Path,
        iteration: IterationState,
    ) -> None:
        iter_dir = query_dir / f"iteration_{iteration.iteration_id}"
        iter_dir.mkdir(exist_ok=True)

        self._render_to_file(
            "planner.md.jinja2",
            {
                "iteration": iteration,
                "plan": iteration.plan,
            },
            iter_dir / "01_planner.md",
        )

        query_docs = self._group_docs_by_query(iteration)
        self._render_to_file(
            "rag.md.jinja2",
            {
                "iteration": iteration,
                "query_docs": query_docs,
                "rag_summary": self._build_rag_summary(query_docs),
                "include_chunks": self.config.include_rag_chunks,
                "max_lines": self.config.max_chunk_preview_lines,
            },
            iter_dir / "02_rag.md",
        )

        self._render_to_file(
            "code.md.jinja2",
            {"iteration": iteration},
            iter_dir / "03_code.md",
        )
        if self.config.save_code_files and iteration.generated_code:
            (iter_dir / "03_code.py").write_text(iteration.generated_code, encoding="utf-8")

        self._render_to_file(
            "execution.md.jinja2",
            {"iteration": iteration},
            iter_dir / "04_execution.md",
        )
        self._render_to_file(
            "evaluation.md.jinja2",
            {"iteration": iteration},
            iter_dir / "05_evaluation.md",
        )

    def _render_to_file(self, template_name: str, context: Dict[str, Any], path: Path) -> None:
        template = self.env.get_template(template_name)
        rendered = template.render(**context)
        path.write_text(rendered, encoding="utf-8")

    def _group_docs_by_query(self, iteration: IterationState) -> Dict[str, List[Any]]:
        grouped: Dict[str, List[Any]] = {}
        for doc in iteration.retrieved_documents:
            grouped.setdefault(doc.query_used, []).append(doc)
        return grouped

    def _build_rag_summary(self, grouped_docs: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for query, docs in grouped_docs.items():
            max_score = max((getattr(doc, "relevance_score", 0.0) or 0.0) for doc in docs) if docs else 0.0
            summary.append(
                {
                    "query": query,
                    "docs": len(docs),
                    "max_score": f"{max_score:.2f}",
                }
            )
        return summary

    def _truncate_lines(self, value: Any, max_lines: int = 30) -> str:
        text = "" if value is None else str(value)
        lines = text.splitlines()
        if max_lines <= 0 or len(lines) <= max_lines:
            return text
        trimmed = lines[:max_lines] + ["... [truncated]"]
        return "\n".join(trimmed)

    def _normalize_task_summary(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        task_id = entry.get("task_id") or entry.get("display_id") or "task"
        return {
            "task_id": task_id,
            "display_id": entry.get("display_id") or task_id,
            "status": entry.get("status", "unknown"),
            "iterations": entry.get("iterations", 0),
            "time_seconds": entry.get("time_seconds", 0.0),
            "test_error": entry.get("test_error") or "",
            "link": entry.get("link"),
        }

    def _normalize_session(self, session: SessionState | Dict[str, Any] | Path) -> SessionState:
        if isinstance(session, SessionState):
            return session
        if isinstance(session, Path):
            data = json.loads(session.read_text(encoding="utf-8"))
            return SessionState.from_dict(data)
        if isinstance(session, dict):
            return SessionState.from_dict(session)
        raise TypeError("session must be SessionState, dict, or Path to JSON file")


__all__ = ["QueryReportGenerator", "ReportConfig"]
