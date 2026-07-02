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

    # Count UNIQUE claims, not raw ledger rows: a duplicated supported/blocked row
    # must not move the deltas and trigger a spurious EFFECT.
    seen_claims: set = set()
    unique_claims: list[dict] = []
    for c in claims:
        key = c.get("claim_id") or (c.get("verdict"), c.get("claim"))
        if key in seen_claims:
            continue
        seen_claims.add(key)
        unique_claims.append(c)
    verdicts = [c.get("verdict") for c in unique_claims]
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


# A behavioural delta below this is treated as zero. Experiment-run counts and
# claim counts are discrete, so any genuine difference is >= 1/n_runs (well above
# this); only float dust falls under it.
BEHAVIOURAL_EPS = 1e-6

# Each arm must be repeated at least this many times before a delta is trustworthy.
# Below it, LLM/sampling noise can manufacture a one-off "effect".
MIN_RUNS = 3


def build_report(on_runs: list[dict], off_runs: list[dict]) -> dict:
    on, off = summarize_arm(on_runs), summarize_arm(off_runs)

    def off_minus_on(metric: str):
        a, b = off[metric]["mean"], on[metric]["mean"]
        return round(a - b, 4) if (a is not None and b is not None) else None

    def on_minus_off(metric: str):
        a, b = on[metric]["mean"], off[metric]["mean"]
        return round((a or 0) - (b or 0), 4)

    compute_runs_saved = off_minus_on("experiment_runs")          # discrete -> real if != 0
    runtime_saved = off_minus_on("total_runtime_s")               # timing jitter -> informational ONLY
    claims_supported_delta = on_minus_off("claims_supported")
    claims_blocked_delta = on_minus_off("claims_blocked")

    # ── Verdict drivers: ONLY behavioural deltas (decisions/claims), never runtime, never text. ──
    runs_moved = compute_runs_saved is not None and abs(compute_runs_saved) > BEHAVIOURAL_EPS
    claims_moved = abs(claims_supported_delta) > BEHAVIOURAL_EPS or abs(claims_blocked_delta) > BEHAVIOURAL_EPS
    behavioural_effect = runs_moved or claims_moved

    # Informational only — Qwen's free text/fields may differ while the decision stays "run".
    # This NEVER drives the verdict (renamed from the old "signal_diverged" to avoid confusion).
    qwen_text_differs = set(on["signal_fingerprint"]) != set(off["signal_fingerprint"])

    # Runtime: jitter. Flag whether it is within timing noise (max across-run std), never an effect.
    rt_std = max(on["total_runtime_s"]["std"] or 0.0, off["total_runtime_s"]["std"] or 0.0)
    runtime_within_noise = runtime_saved is None or abs(runtime_saved) <= max(rt_std, 0.1)

    # Sample size: a delta from <MIN_RUNS runs per arm can be LLM/sampling noise.
    underpowered = min(on["n_runs"], off["n_runs"]) < MIN_RUNS

    # Arm-source sanity: ON must be the real world model, OFF the fallback stub.
    # If not, the arms may be swapped/mislabeled and the signed delta is untrustworthy.
    arm_sources_ok = on["wm_source"] == ["qwen_agentworld"] and off["wm_source"] == ["plumbing_not_called"]
    source_warning = (
        ""
        if arm_sources_ok
        else (
            f"⚠ ARM SOURCES UNEXPECTED (ON={on['wm_source']} should be ['qwen_agentworld'], "
            f"OFF={off['wm_source']} should be ['plumbing_not_called']) — arms may be mislabeled, do not trust the sign. "
        )
    )

    effect = {
        "compute_runs_saved_by_wm": compute_runs_saved,
        "claims_supported_delta_on_minus_off": claims_supported_delta,
        "claims_blocked_delta_on_minus_off": claims_blocked_delta,
        "runtime_s_delta_off_minus_on": runtime_saved,
        "runtime_within_noise": runtime_within_noise,
        "qwen_text_differs": qwen_text_differs,
        "underpowered": underpowered,
        "min_runs_required": MIN_RUNS,
        "arm_sources_ok": arm_sources_ok,
    }

    if not behavioural_effect:
        verdict = source_warning + (
            "NULL → Option B (no measurable world-model effect). Experiment runs identical "
            "(compute_runs_saved=0) and claim verdicts identical (supported & blocked deltas=0). "
            "Runtime differences are timing jitter, not an effect"
            + ("; Qwen's free text differs from the fallback but its structured recommendation stays 'run' (same decisions)" if qwen_text_differs else "")
            + ". Present system + M1 with this as a stated limitation — do NOT claim a world-model compute win."
        )
        if underpowered:
            verdict += f" (Note: <{MIN_RUNS} runs/arm — under-powered, but NULL is the conservative read.)"
    else:
        parts = []
        if runs_moved:
            parts.append(f"compute_runs_saved(OFF-ON)={compute_runs_saved}")
        if abs(claims_supported_delta) > BEHAVIOURAL_EPS:
            parts.append(f"claims_supported_delta(ON-OFF)={claims_supported_delta}")
        if abs(claims_blocked_delta) > BEHAVIOURAL_EPS:
            parts.append(f"claims_blocked_delta(ON-OFF)={claims_blocked_delta}")
        body = ", ".join(parts)
        if underpowered:
            verdict = source_warning + (
                f"EFFECT (UNDER-POWERED — fewer than {MIN_RUNS} runs/arm; this delta may be LLM/sampling "
                f"noise, re-run ≥{MIN_RUNS}x to confirm): {body}. (Runtime excluded.)"
            )
        else:
            verdict = source_warning + f"EFFECT MEASURED on real behavioural deltas: {body}. (Runtime is informational jitter, excluded.)"

    return {
        "schema_version": "1.2",
        "arms": {"on": on, "off": off},
        "world_model_effect": effect,
        "verdict": verdict,
        "integrity_notes": [
            "Verdict driven ONLY by behavioural deltas: experiment runs + claim verdicts. Runtime jitter and Qwen free-text differences NEVER trigger an effect.",
            "Metrics are ACTUALS computed identically for both arms; counterfactual 'saved_remaining_runs' is NOT used; claims are de-duplicated before counting.",
            f"Each arm must be repeated >= {MIN_RUNS}x; a delta from fewer runs is flagged UNDER-POWERED (could be noise).",
            "Arm sources are validated: ON must be qwen_agentworld, OFF plumbing_not_called — otherwise the verdict is prefixed with a mislabel warning.",
            "qwen_text_differs is informational: the world model can emit different prose while the structured recommendation stays 'run' (same decision) -> that is NOT an effect.",
        ],
    }


def render_md(report: dict, on_runs: list[dict], off_runs: list[dict]) -> str:
    on, off = report["arms"]["on"], report["arms"]["off"]
    e = report["world_model_effect"]

    def cell(arm, k):
        m = arm[k]
        return f"{m['mean']} ± {m['std']} (n={m['n']})" if m["mean"] is not None else "n/a"

    rt_label = f"{e['runtime_s_delta_off_minus_on']} (jitter, hors verdict)" + (" — within noise" if e["runtime_within_noise"] else "")
    lines = [
        "# Ablation world-model ON/OFF — analyse honnête",
        "",
        f"**Verdict : {report['verdict']}**",
        "",
        f"- ON arm: {on['n_runs']} runs · source `{on['wm_source']}`",
        f"- OFF arm: {off['n_runs']} runs · source `{off['wm_source']}`",
        "",
        "| Métrique (mean ± std) | ON (world-model) | OFF (fallback) | Δ | compte pour le verdict ? |",
        "|---|---|---|---|---|",
        f"| **Runs d'expérience réels** | {cell(on,'experiment_runs')} | {cell(off,'experiment_runs')} | {e['compute_runs_saved_by_wm']} (OFF−ON) | ✅ OUI |",
        f"| **Claims supported** | {cell(on,'claims_supported')} | {cell(off,'claims_supported')} | {e['claims_supported_delta_on_minus_off']} (ON−OFF) | ✅ OUI |",
        f"| **Claims blocked** | {cell(on,'claims_blocked')} | {cell(off,'claims_blocked')} | {e['claims_blocked_delta_on_minus_off']} (ON−OFF) | ✅ OUI |",
        f"| Runtime total (s) | {cell(on,'total_runtime_s')} | {cell(off,'total_runtime_s')} | {rt_label} | ❌ non (jitter) |",
        f"| Runs jusqu'au 1er supported | {cell(on,'runs_to_first_supported')} | {cell(off,'runs_to_first_supported')} | — | indicatif |",
        "",
        f"**Texte/champs Qwen ≠ fallback ?** {'oui' if e['qwen_text_differs'] else 'non'} — _informatif uniquement, NE compte PAS comme effet_ (la `recommendation` reste `run` → même décision).",
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

    md_path = out.with_suffix(".md")
    shown = md_path.relative_to(ROOT) if md_path.is_relative_to(ROOT) else md_path
    print(f"Wrote {shown} and .json")
    print(f"  ON n={len(on_runs)} OFF n={len(off_runs)}")
    print(f"  VERDICT: {report['verdict']}")


if __name__ == "__main__":
    main()
