#!/usr/bin/env python3
"""CLI: project a finished lab workspace into a single `Run` run.json.

    PYTHONPATH=src .venv/bin/python scripts/export_run.py \
        --workspace reports/lab/<slug> \
        --out reports/run_export/run.json

With no --workspace it picks the most recently modified reports/lab/* workspace.
The emitted run.json matches the frontend `Run` type field-for-field; the
frontend serves it (copy to public/run.json) and getRun() fetches "/run.json".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from luckyloop.run_export import build_run, latest_workspace  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=None, help="reports/lab/<slug> (default: latest)")
    parser.add_argument("--out", default="reports/run_export/run.json")
    args = parser.parse_args()

    workspace = Path(args.workspace) if args.workspace else latest_workspace(ROOT)
    if not workspace.is_absolute():
        workspace = ROOT / workspace

    run = build_run(workspace)
    warnings = run.pop("_warnings", [])

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {out.relative_to(ROOT)} from {workspace.relative_to(ROOT)}")
    print(
        f"  papers={len(run['papers'])} predictions={len(run['predictions'])} "
        f"diffs={len(run['diffs'])} findings={len(run['findings'])} "
        f"verdict={run['verdict']['state']}"
    )
    for w in warnings:
        print(f"  WARNING: {w}")


if __name__ == "__main__":
    main()
