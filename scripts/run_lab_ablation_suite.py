#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from luckyloop.lab import run_lab


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTION = "Do random train/test splits overstate performance on sequential sensor datasets?"


def qwen_configured() -> bool:
    return bool(os.getenv("LUCKYWORLD_SIMULATOR_BASE_URL") and os.getenv("LUCKYWORLD_SIMULATOR_MODEL"))


def agent_configured() -> bool:
    return bool(
        os.getenv("LUCKYLOOP_AGENT_BASE_URL")
        and os.getenv("LUCKYLOOP_AGENT_MODEL")
        and os.getenv("LUCKYLOOP_AGENT_API_KEY")
    )


def summarize_result(result) -> dict:
    claims = [claim.model_dump() for claim in result.claims]
    return {
        "workspace": result.workspace,
        "claim_count": len(claims),
        "blocked_claims": sum(1 for claim in claims if claim["verdict"] == "blocked"),
        "supported_claims": sum(1 for claim in claims if claim["verdict"] in {"supported", "weakly_supported"}),
        "observation_or_inconclusive": sum(1 for claim in claims if claim["verdict"] in {"observation_only", "inconclusive"}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--study", default="split_validity_sensor")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--require-qwen", action="store_true")
    parser.add_argument("--planner", choices=["deterministic", "llm"], default="deterministic")
    parser.add_argument("--require-agent", action="store_true")
    args = parser.parse_args()

    if args.require_qwen and not qwen_configured():
        raise SystemExit(
            "--require-qwen was set, but LUCKYWORLD_SIMULATOR_BASE_URL and "
            "LUCKYWORLD_SIMULATOR_MODEL are not configured."
        )
    if args.require_agent and not agent_configured():
        raise SystemExit(
            "--require-agent was set, but LUCKYLOOP_AGENT_BASE_URL, "
            "LUCKYLOOP_AGENT_MODEL, and LUCKYLOOP_AGENT_API_KEY are not configured."
        )

    rows = []
    for policy in ["classic_score_chaser", "classic_verified", "lucky_loop_lab"]:
        result = run_lab(
            question=args.question,
            study_id=args.study,
            budget=args.budget,
            require_qwen=args.require_qwen and policy == "lucky_loop_lab",
            policy=policy,
            planner=args.planner if policy == "lucky_loop_lab" else "deterministic",
            require_agent=args.require_agent and policy == "lucky_loop_lab",
        )
        rows.append({"policy": policy, **summarize_result(result)})

    out_dir = ROOT / "reports" / "lab_ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "question": args.question,
        "study": args.study,
        "require_qwen": args.require_qwen,
        "require_agent": args.require_agent,
        "rows": rows,
        "summary": {
            "policies": [row["policy"] for row in rows],
            "lucky_loop_has_qwen": args.require_qwen,
        },
    }
    (out_dir / "lab_ablation_suite.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Lab Ablation Suite",
        "",
        f"Question: {args.question}",
        "",
        "| Policy | Workspace | Claims | Blocked | Supported | Observation/Inconclusive |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['policy']}` | `{row['workspace']}` | {row['claim_count']} | "
            f"{row['blocked_claims']} | {row['supported_claims']} | {row['observation_or_inconclusive']} |"
        )
    (out_dir / "lab_ablation_suite.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
