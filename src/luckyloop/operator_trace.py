from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from luckyloop.tasks import ROOT


DEFAULT_TRACE_PATH = ROOT / "reports" / "operator_trace" / "codex_operator_trace.jsonl"
DEFAULT_SUMMARY_PATH = ROOT / "reports" / "operator_trace" / "codex_operator_summary.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_operator_event(
    *,
    event_type: str,
    goal: str,
    action: str,
    operator: str = "codex_operator",
    status: str = "started",
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    rationale: str = "",
    trace_path: Path = DEFAULT_TRACE_PATH,
) -> dict[str, Any]:
    event = {
        "timestamp_utc": _now(),
        "operator": operator,
        "event_type": event_type,
        "goal": goal,
        "action": action,
        "status": status,
        "rationale": rationale,
        "inputs": inputs or {},
        "outputs": outputs or {},
    }
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")
    return event


def load_operator_events(trace_path: Path = DEFAULT_TRACE_PATH) -> list[dict[str, Any]]:
    if not trace_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def write_operator_summary(
    *,
    events: list[dict[str, Any]] | None = None,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
) -> None:
    events = events if events is not None else load_operator_events()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    completed = [event for event in events if event.get("status") == "completed"]
    failed = [event for event in events if event.get("status") == "failed"]
    lines = [
        "# Operator Trace Summary",
        "",
        "This file records repo-level autoresearch operations triggered by the agent operator. It complements experiment traces: experiment traces show prediction/run/verification evidence; operator traces show which benchmark or validation actions were launched and why.",
        "",
        f"- Events logged: {len(events)}",
        f"- Completed events: {len(completed)}",
        f"- Failed events: {len(failed)}",
        "",
        "## Recent Events",
        "",
        "| Time UTC | Operator | Event | Action | Status | Rationale |",
        "|---|---|---|---|---|---|",
    ]
    for event in events[-30:]:
        rationale = str(event.get("rationale") or "").replace("|", "/")
        action = str(event.get("action") or "").replace("|", "/")
        lines.append(
            f"| {event.get('timestamp_utc', '')} | {event.get('operator', '')} | "
            f"{event.get('event_type', '')} | {action} | {event.get('status', '')} | {rationale} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
