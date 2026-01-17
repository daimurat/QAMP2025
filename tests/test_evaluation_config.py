import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_mas.config import load_evaluation_config  # noqa: E402


def test_load_evaluation_config(tmp_path: Path):
    cfg_path = tmp_path / "evaluation_config.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "evaluation": {
                    "dataset": "Qiskit/qiskit_humaneval",
                    "difficulty": "basic",
                    "max_tasks": 2,
                },
                "orchestrator": {"max_iterations": 3},
                "models": {
                    "planner": {"model": "m1"},
                    "code_agent": {"model": "m2"},
                    "evaluator": {"model": "m3"},
                },
                "reports": {"enabled": True, "output_subdir": "queries_debug"},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_evaluation_config(str(cfg_path))
    assert cfg.evaluation.dataset == "Qiskit/qiskit_humaneval"
    assert cfg.evaluation.max_tasks == 2
    assert cfg.orchestrator.max_iterations == 3
    assert cfg.models.planner_model == "m1"
    assert cfg.models.code_model == "m2"
    assert cfg.models.evaluator_model == "m3"
    assert cfg.reports.enabled is True
    assert cfg.reports.output_subdir == "queries_debug"
