from __future__ import annotations

import json
from pathlib import Path

from .schemas import LabNotebookEntry


def append_notebook_entry(workspace: Path, entry: LabNotebookEntry) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / "notebook.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry.model_dump_json() + "\n")


def write_predictions(workspace: Path, step: int, predictions: list[dict]) -> None:
    out_dir = workspace / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "world_model_predictions.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for prediction in predictions:
            payload = {"step": step, **prediction}
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
