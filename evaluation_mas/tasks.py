"""
Task selection and filtering for evaluation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class EvaluationTask:
    task_id: str
    task_number: int
    entry_point: str
    prompt: str
    test_code: str
    difficulty: Optional[str]


def load_tasks(
    dataset: str,
    split: str,
    difficulty: Optional[str] = None,
    max_tasks: Optional[int] = None,
    task_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> List[EvaluationTask]:
    """Load and filter tasks from HuggingFace dataset."""
    from datasets import load_dataset  # Lazy import to avoid dependency at import time

    cache_path = Path(cache_dir) if cache_dir else Path.cwd() / ".hf_cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_path))
    ds = load_dataset(dataset, split=split, cache_dir=str(cache_path))
    tasks: List[EvaluationTask] = []

    for row in ds:
        row_task_id = row.get("task_id") or row.get("id")
        if row_task_id is None:
            continue

        # Normalize numeric task_id suffix
        task_number = _extract_task_number(row_task_id)

        # Single task filter
        if task_id:
            if task_id.isdigit():
                if task_number != int(task_id):
                    continue
            elif row_task_id != task_id:
                continue

        # Difficulty filter
        if difficulty:
            row_diff = (row.get("difficulty_scale") or "").lower()
            if row_diff != difficulty.lower():
                continue

        tasks.append(
            EvaluationTask(
                task_id=row_task_id,
                task_number=task_number,
                entry_point=row["entry_point"],
                prompt=row["prompt"],
                test_code=row["test"],
                difficulty=row.get("difficulty_scale"),
            )
        )

    # Deterministic ordering by task number
    tasks.sort(key=lambda t: t.task_number)

    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    return tasks


def _extract_task_number(task_id: str) -> int:
    try:
        suffix = task_id.split("/")[-1]
        return int(suffix)
    except Exception:
        return 0
