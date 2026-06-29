#!/usr/bin/env python3
"""Honest analysis of a world-model ON/OFF ablation.

Reads the ON and OFF run workspaces (each arm repeated >=3x), computes the same
metrics symmetrically for both arms FROM THE ACTUALS, and reports signed deltas.
It does NOT use the counterfactual `saved_remaining_runs` metric (verifier-
attributed, one-sided — see HQ ablation spec T3#4). It also fingerprints the
world-model's STRUCTURED signal so a null result (Qwen == fallback) is detected
and reported honestly rather than hidden.

Inputs (either form):
  --ablation-dir reports/lab_ablations/<study>     # auto-discovers on_run*/off_run*
  --on  "glob/to/on_run*"  --off "glob/to/off_run*"

Output: a markdown table + JSON next to --out (default reports/lab_ablations/analysis).

Arm = one workspace dir containing notebook.jsonl + claim_ledger.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Kinds that actually spend experiment compute (exclude cheap probe + the no-op stop).
EXPERIMENT_KINDS = {"run_protocol", "run_baseline", "run_replication", "run_ablation"}
SUPPORTED = {"supported", "weakly_supported", "strongly_supported"}
# A structured signal that actually steers select_action away from "always run".
# `stop_and_report` is excluded: every run ends on a terminal stop action whose
# prediction is structurally "stop_and_report" regardless of the world model, so
# counting it would falsely flag the fallback as a non-trivial signal.
NONTRIVIAL_RECS = {"skip", "modify", "verify"}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def metrics_for_workspace(ws: Path) -> dict | None:
    """All metrics for one run. Identical computation for ON and OFF arms."""
    notebook = _read_jsonl(ws / "notebook.jsonl")
    if not notebook:
        return None
    claims = _read_json(ws / "claim_ledger.json", []) or []

    experiment_runs = 0
    total_runtime = 0.0
    steps = 0
    runs_to_first_supported = None
    sources: dict[str, int] = {}
    signal_fingerprint: set[tuple] = set()
    nontrivial_signal = False

    for entry in notebook:
        steps += 1
        sa = entry.get("selected_action") or {}
        ob = entry.get("actual_observation") or {}
        kind = sa.get("kind")
        if kind in EXPERIMENT_KINDS and ob.get("status") == "success":
            experiment_runs += 1
        total_runtime += float(ob.get("runtime_seconds") or 0.0)

        if runs_to_first_supported is None:
            for cu in entry.get("claim_updates") or []:
                if cu.get("verdict") in SUPPORTED:
                    runs_to_first_supported = entry.get("step", steps)
                    break

        for pred in entry.get("qwen_predictions") or []:
            src = pred.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
            rec = pred.get("recommendation")
            cwr = pred.get("compute_waste_risk")
            voi = pred.get("value_of_information")
            ecd = pred.get("expected_claim_delta")
            signal_fingerprint.add((rec, cwr, voi, ecd))
            if rec in NONTRIVIAL_RECS or (isinstance(cwr, (int, float)) and cwr > 0):
                nontrivial_signal = True

    verdicts = [c.get("verdict") for c in claims]
    dominant_source = max(sources, key=sources.get) if sources else "none"
    return {
        "workspace": str(ws.relative_to(ROOT)) if ws.is_absolute() else str(ws),
        "wm_source": dominant_source,  # qwen_agentworld (ON) | plumbing_not_called (OFF)
        "steps": steps,
        "experiment_runs": experiment_runs,
        "total_runtime_s": round(total_runtime, 4),
        "claims_total": len(verdicts),
        "claims_supported": sum(1 for v in verdicts if v in SUPPORTED),
        "claims_blocked": sum(1 for v in verdicts if v == "blocked"),
        "claims_observation_inconclusive": sum(1 for v in verdicts if v in {"observation_only", "inconclusive"}),
        "runs_to_first_supported": runs_to_first_supported,
        "nontrivial_signal": nontrivial_signal,
        "signal_fingerprint": sorted(str(t) for t in signal_fingerprint),
    }


def _agg(runs: list[dict], key: str) -> dict:
    vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": round(statistics.fmean(vals), 4), "std": round(statistics.pstdev(vals), 4), "n": len(vals)}


def summarize_arm(runs: list[dict]) -> dict:
    sources = {r["wm_source"] for r in runs}
    fingerprints = {tuple(r["signal_fingerprint"]) for r in runs}
    return {
        "n_runs": len(runs),
        "wm_source": sorted(sources),
        "any_nontrivial_signal": any(r["nontrivial_signal"] for r in runs),
        "distinct_signal_fingerprints": len(fingerprints),
        "signal_fingerprint": sorted({f for fp in fingerprints for f in fp}),
        "experiment_runs": _agg(runs, "experiment_runs"),
        "total_runtime_s": _agg(runs, "total_runtime_s"),
        "claims_supported": _agg(runs, "claims_supported"),
        "claims_blocked": _agg(runs, "claims_blocked"),
        "runs_to_first_supported": _agg(runs, "runs_to_first_supported"),
    }


def _discover(ablation_dir: Path | None, on_glob: str | None, off_glob: str | None) -> tuple[list[Path], list[Path]]:
    if ablation_dir:
        on = sorted(ablation_dir.glob("on_run*")) or sorted(ablation_dir.glob("on*"))
        off = sorted(ablation_dir.glob("off_run*")) or sorted(ablation_dir.glob("off*"))
        return on, off
    on = [Path(p) for p in sorted(glob.glob(on_glob or ""))]
    off = [Path(p) for p in sorted(glob.glob(off_glob or ""))]
    return on, off


def build_report(on_runs: list[dict], off_runs: list[dict]) -> dict:
    on, off = summarize_arm(on_runs), summarize_arm(off_runs)

    def delta(metric: str):
        a, b = off[metric]["mean"], on[metric]["mean"]
        return round(a - b, 4) if (a is not None and b is not None) else None

    # Signed effect of the world model: OFF - ON (positive runtime/runs delta = WM saved compute).
    effect = {
        "compute_runs_saved_by_wm": delta("experiment_runs"),
        "runtime_s_saved_by_wm": delta("total_runtime_s"),
        "claims_supported_delta_on_minus_off": (
            round((on["claims_supported"]["mean"] or 0) - (off["claims_supported"]["mean"] or 0), 4)
        ),
    }

    # Honest null detection: did the world-model's structured signal actually diverge?
    on_fp = set(on["signal_fingerprint"])
    off_fp = set(off["signal_fingerprint"])
    signal_diverged = (on_fp != off_fp) or on["any_nontrivial_signal"]
    metrics_moved = any(effect[k] not in (None, 0) for k in ("compute_runs_saved_by_wm", "runtime_s_saved_by_wm"))

    if not signal_diverged and not metrics_moved:
        verdict = (
            "NULL (honest) → Option B. The world-model's STRUCTURED signal is identical to the fallback "
            "(same recommendation / compute_waste_risk / value_of_information) and arm metrics do not move. "
            "Qwen produces richer text but exerts no measured decision influence on these tasks. Present the "
            "system + M1 with this as a stated limitation — do not claim a world-model compute win."
        )
    elif signal_diverged and metrics_moved:
        verdict = "EFFECT MEASURED. Qwen's structured signal diverged from the fallback and arm metrics moved — report the signed deltas."
    else:
        verdict = "PARTIAL / inconclusive. Signal or metrics moved but not both — inspect per-run detail; likely LLM noise, add repeats."

    return {
        "schema_version": "1.0",
        "arms": {"on": on, "off": off},
        "world_model_effect_off_minus_on": effect,
        "verdict": verdict,
        "integrity_notes": [
            "Metrics are ACTUALS (real runs executed, summed runtime), computed identically for both arms.",
            "Counterfactual 'saved_remaining_runs' is deliberately NOT used (verifier-attributed, one-sided).",
            "wm_source proves the arm: qwen_agentworld (ON) vs plumbing_not_called (OFF).",
            "Deltas are signed (OFF - ON); a world-model win is positive, a regression is negative — both reported.",
        ],
    }


def render_md(report: dict, on_runs: list[dict], off_runs: list[dict]) -> str:
    on, off = report["arms"]["on"], report["arms"]["off"]
    e = report["world_model_effect_off_minus_on"]

    def cell(arm, k):
        m = arm[k]
        return f"{m['mean']} ± {m['std']} (n={m['n']})" if m["mean"] is not None else "n/a"

    lines = [
        "# Ablation world-model ON/OFF — analyse honnête",
        "",
        f"**Verdict : {report['verdict']}**",
        "",
        f"- ON arm: {on['n_runs']} runs · source `{on['wm_source']}` · signal non-trivial: {on['any_nontrivial_signal']}",
        f"- OFF arm: {off['n_runs']} runs · source `{off['wm_source']}` · signal non-trivial: {off['any_nontrivial_signal']}",
        "",
        "| Métrique (mean ± std) | ON (world-model) | OFF (fallback) | Δ (OFF − ON) |",
        "|---|---|---|---|",
        f"| Runs d'expérience réels | {cell(on,'experiment_runs')} | {cell(off,'experiment_runs')} | {e['compute_runs_saved_by_wm']} |",
        f"| Runtime total (s) | {cell(on,'total_runtime_s')} | {cell(off,'total_runtime_s')} | {e['runtime_s_saved_by_wm']} |",
        f"| Claims supported | {cell(on,'claims_supported')} | {cell(off,'claims_supported')} | — |",
        f"| Claims blocked | {cell(on,'claims_blocked')} | {cell(off,'claims_blocked')} | — |",
        f"| Runs jusqu'au 1er supported | {cell(on,'runs_to_first_supported')} | {cell(off,'runs_to_first_supported')} | — |",
        "",
        f"**Signal structuré Qwen divergent du fallback ?** {'OUI' if (set(on['signal_fingerprint']) != set(off['signal_fingerprint']) or on['any_nontrivial_signal']) else 'NON (== fallback)'}",
        "",
        "## Intégrité",
        *[f"- {n}" for n in report["integrity_notes"]],
        "",
        "## Détail par run",
        "| arm | workspace | source | exp_runs | runtime_s | supported | blocked | non-trivial |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for arm, runs in (("ON", on_runs), ("OFF", off_runs)):
        for r in runs:
            lines.append(
                f"| {arm} | `{r['workspace']}` | {r['wm_source']} | {r['experiment_runs']} | "
                f"{r['total_runtime_s']} | {r['claims_supported']} | {r['claims_blocked']} | {r['nontrivial_signal']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ablation-dir", default=None, help="dir with on_run*/off_run* subdirs")
    ap.add_argument("--on", default=None, help="glob to ON run workspaces")
    ap.add_argument("--off", default=None, help="glob to OFF run workspaces")
    ap.add_argument("--out", default="reports/lab_ablations/analysis")
    args = ap.parse_args()

    ablation_dir = Path(args.ablation_dir) if args.ablation_dir else None
    if ablation_dir and not ablation_dir.is_absolute():
        ablation_dir = ROOT / ablation_dir
    on_paths, off_paths = _discover(ablation_dir, args.on, args.off)
    if not on_paths or not off_paths:
        raise SystemExit(f"Need both arms. Found ON={len(on_paths)} OFF={len(off_paths)}. Check --ablation-dir / --on / --off.")

    on_runs = [m for p in on_paths if (m := metrics_for_workspace(p))]
    off_runs = [m for p in off_paths if (m := metrics_for_workspace(p))]
    if not on_runs or not off_runs:
        raise SystemExit(f"No usable workspaces (need notebook.jsonl). ON={len(on_runs)} OFF={len(off_runs)}.")

    report = build_report(on_runs, off_runs)
    report["per_run"] = {"on": on_runs, "off": off_runs}

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.with_suffix(".json")).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out.with_suffix(".md")).write_text(render_md(report, on_runs, off_runs), encoding="utf-8")

    print(f"Wrote {out.with_suffix('.md').relative_to(ROOT)} and .json")
    print(f"  ON n={len(on_runs)} OFF n={len(off_runs)}")
    print(f"  VERDICT: {report['verdict']}")


if __name__ == "__main__":
    main()
