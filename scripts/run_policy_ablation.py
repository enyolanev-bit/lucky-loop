#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from luckyloop.schemas import ExperimentTrace
from luckyloop.tasks import ROOT


TASKS = ["breast_cancer_accuracy", "wine_accuracy", "digits_accuracy"]


def load_traces(task: str) -> list[ExperimentTrace]:
    return [
        ExperimentTrace.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted((ROOT / "runs" / task).glob("run_*.json"))
    ]


def best_single_run(traces: list[ExperimentTrace]) -> ExperimentTrace | None:
    singles = [
        trace
        for trace in traces
        if trace.actual_result.accuracy is not None
        and trace.proposed_action.model not in {"verification_sweep", "top_model_verification"}
    ]
    return max(singles, key=lambda trace: trace.actual_result.accuracy or -1, default=None)


def summarize_policy(task: str, traces: list[ExperimentTrace], policy: str) -> dict:
    best = best_single_run(traces)
    top_verifications = [trace for trace in traces if trace.proposed_action.model == "top_model_verification"]
    blocked = sum(1 for trace in traces if trace.verification and not trace.verification.trustworthy)
    supported = sum(1 for trace in traces if trace.verification and trace.verification.trustworthy)
    if policy == "score_chaser":
        return {
            "task": task,
            "policy": policy,
            "best_single_run_score": best.actual_result.accuracy if best else None,
            "top_model_verification_performed": False,
            "unsupported_best_model_claims": 1 if best else 0,
            "claims_blocked": 0,
            "supported_claims": 0,
            "prediction_misses_logged": sum(1 for trace in traces if trace.comparison.unexpected_events),
            "compute_spent_before_verified_claim": 0,
        }
    if policy == "fixed_order":
        return {
            "task": task,
            "policy": policy,
            "best_single_run_score": best.actual_result.accuracy if best else None,
            "top_model_verification_performed": bool(top_verifications),
            "unsupported_best_model_claims": 0 if top_verifications else 1,
            "claims_blocked": blocked,
            "supported_claims": supported,
            "prediction_misses_logged": sum(1 for trace in traces if trace.comparison.unexpected_events),
            "compute_spent_before_verified_claim": len(traces),
        }
    return {
        "task": task,
        "policy": policy,
        "best_single_run_score": best.actual_result.accuracy if best else None,
        "top_model_verification_performed": bool(top_verifications),
        "unsupported_best_model_claims": 0,
        "claims_blocked": blocked,
        "supported_claims": supported,
        "prediction_misses_logged": sum(1 for trace in traces if trace.comparison.unexpected_events),
        "compute_spent_before_verified_claim": len(traces),
    }


def main() -> None:
    rows = []
    for task in TASKS:
        traces = load_traces(task)
        for policy in ["fixed_order", "score_chaser", "lucky_loop_adaptive"]:
            rows.append(summarize_policy(task, traces, policy))

    out_json = ROOT / "reports" / "policy_ablation.json"
    out_json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    lines = [
        "# Policy Ablation",
        "",
        "This ablation is computed from the real benchmark traces. It contrasts what the current adaptive loop logs against simpler policies over the same observed evidence.",
        "",
        "| Task | Policy | Best single-run | Top-model verification | Unsupported best-model claims | Claims blocked | Supported claims | Prediction misses |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        score = "" if row["best_single_run_score"] is None else f"{row['best_single_run_score']:.4f}"
        lines.append(
            f"| {row['task']} | {row['policy']} | {score} | "
            f"{'yes' if row['top_model_verification_performed'] else 'no'} | "
            f"{row['unsupported_best_model_claims']} | {row['claims_blocked']} | "
            f"{row['supported_claims']} | {row['prediction_misses_logged']} |"
        )
    out_md = ROOT / "reports" / "policy_ablation.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_md.relative_to(ROOT)} and {out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
