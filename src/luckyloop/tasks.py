from __future__ import annotations

import json
from pathlib import Path

from .schemas import TaskSpec


ROOT = Path(__file__).resolve().parents[2]


def load_task(path: str | Path | None) -> TaskSpec:
    if path is None:
        path = ROOT / "configs" / "tasks" / "breast_cancer_accuracy.json"
    task_path = Path(path)
    if not task_path.is_absolute():
        task_path = ROOT / task_path
    data = json.loads(task_path.read_text(encoding="utf-8"))
    return TaskSpec(**data)
