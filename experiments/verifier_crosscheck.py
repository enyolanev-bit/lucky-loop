#!/usr/bin/env python3
"""Additive verifier cross-check: paired 95% CI best-vs-second.

This script intentionally does not import or modify luckyloop.verifier. It reads the
claim ledger plus the recorded run artifacts, recomputes an independent paired
confidence interval on the observed sweep metric, and writes a markdown agreement
report.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLAIM_LEDGER = ROOT / "reports" / "claim_ledger.json"
RUNS_DIR = ROOT / "runs"
REPORT_PATH = ROOT / "reports" / "verifier_crosscheck.md"

# Two-sided t critical values for 95% CI by degrees of freedom.
# Keeps the script self-contained for the hackathon environment.
T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}

SUPPORTING_TRUST_STATUSES = {"allowed", "weakly_supported", "supported", "strongly_supported"}


@dataclass(frozen=True)
class CrossCheckResult:
    claim_id: str
    claim: str
    run_id: str | None
    trust_verdict: str
    ic95_verdict: str
    agreement: bool
    note: str
    method_note: str
    best_label: str | None = None
    second_label: str | None = None
    metric: str | None = None
    n_pairs: int = 0
    mean_diff: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def t_critical_95(df: int) -> float:
    if df <= 0:
        raise ValueError("df must be positive")
    return T_CRIT_95.get(df, 1.96)


def config_label(row: dict[str, Any]) -> str:
    return str(row.get("label") or row.get("config_key") or row.get("value") or row.get("model") or "unknown")


def config_key(row: dict[str, Any]) -> str:
    return str(row.get("config_key") or row.get("value") or row.get("model") or config_label(row))


def grouped_metric_values(runs: list[dict[str, Any]], metric: str) -> dict[str, dict[Any, float]]:
    grouped: dict[str, dict[Any, float]] = {}
    for idx, row in enumerate(runs):
        value = row.get(metric)
        if value is None:
            continue
        key = config_key(row)
        # Prefer explicit seeds. Fall back to within-config row index only if needed.
        pair_id = row.get("seed", len(grouped.get(key, {})) if "seed" not in row else idx)
        grouped.setdefault(key, {})[pair_id] = float(value)
    return grouped


def labels_by_key(runs: list[dict[str, Any]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in runs:
        labels[config_key(row)] = config_label(row)
    return labels


def paired_ci95(best: dict[Any, float], second: dict[Any, float]) -> tuple[int, float, float, float, str]:
    common_pairs = sorted(set(best) & set(second), key=lambda x: str(x))
    diffs = [best[pair] - second[pair] for pair in common_pairs]
    if len(diffs) < 2:
        raise ValueError("Need at least two paired observations for an IC95.")
    avg = mean(diffs)
    if len(diffs) == 2:
        # stdev is defined but df=1 makes the interval intentionally very wide.
        sd = stdev(diffs)
    else:
        sd = stdev(diffs)
    half_width = 0.0 if sd == 0 else t_critical_95(len(diffs) - 1) * sd / math.sqrt(len(diffs))
    low = avg - half_width
    high = avg + half_width
    pair_note = "paired by explicit seed" if common_pairs else "paired by row order fallback"
    return len(diffs), avg, low, high, pair_note


def run_for_entry(entry: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None, str | None]:
    for run_id in entry.get("evidence_run_ids", []):
        path = RUNS_DIR / f"{run_id}.json"
        if path.exists():
            return run_id, load_json(path), None
    return None, None, "no matching run artifact found under runs/"


def crosscheck_entry(entry: dict[str, Any]) -> CrossCheckResult:
    claim_id = str(entry.get("claim_id", "unknown"))
    claim = str(entry.get("claim", ""))
    trust_verdict = str(entry.get("status", "missing"))
    trust_supports = trust_verdict in SUPPORTING_TRUST_STATUSES

    run_id, run_data, missing_reason = run_for_entry(entry)
    if run_data is None:
        return CrossCheckResult(
            claim_id=claim_id,
            claim=claim,
            run_id=run_id,
            trust_verdict=trust_verdict,
            ic95_verdict="INCONCLUSIVE",
            agreement=not trust_supports,
            note=missing_reason or "missing run data",
            method_note="not recalculable",
        )

    raw = (run_data.get("actual_result") or {}).get("raw") or {}
    sweep_runs = raw.get("runs") or []
    metric = raw.get("metric") or (run_data.get("verification") or {}).get("metric") or "accuracy"
    if not sweep_runs:
        return CrossCheckResult(
            claim_id=claim_id,
            claim=claim,
            run_id=run_id,
            trust_verdict=trust_verdict,
            ic95_verdict="INCONCLUSIVE",
            agreement=not trust_supports,
            note="run artifact has no sweep rows",
            method_note="not recalculable",
            metric=metric,
        )

    grouped = grouped_metric_values(sweep_runs, metric)
    if len(grouped) < 2:
        return CrossCheckResult(
            claim_id=claim_id,
            claim=claim,
            run_id=run_id,
            trust_verdict=trust_verdict,
            ic95_verdict="INCONCLUSIVE",
            agreement=not trust_supports,
            note="fewer than two configs with numeric metric values",
            method_note="not recalculable",
            metric=metric,
        )

    means = {key: mean(values.values()) for key, values in grouped.items() if values}
    ranked = sorted(means, key=lambda key: means[key], reverse=True)
    best_key, second_key = ranked[0], ranked[1]
    labels = labels_by_key(sweep_runs)
    best_label = labels.get(best_key, best_key)
    second_label = labels.get(second_key, second_key)

    try:
        n_pairs, avg, low, high, pair_note = paired_ci95(grouped[best_key], grouped[second_key])
    except ValueError as exc:
        return CrossCheckResult(
            claim_id=claim_id,
            claim=claim,
            run_id=run_id,
            trust_verdict=trust_verdict,
            ic95_verdict="INCONCLUSIVE",
            agreement=not trust_supports,
            note=str(exc),
            method_note="not recalculable",
            best_label=best_label,
            second_label=second_label,
            metric=metric,
        )

    ic95_verdict = "PASS" if low > 0 else "FAIL"
    ic95_supports = ic95_verdict == "PASS"
    agreement = trust_supports == ic95_supports
    protocol_warning = raw.get("protocol_warning")
    note = (
        f"{best_label} vs {second_label}; n={n_pairs}; mean diff={avg:.6f}; "
        f"IC95=[{low:.6f}, {high:.6f}]"
    )
    if protocol_warning:
        note += f"; protocol warning in trust ladder: {protocol_warning}"

    return CrossCheckResult(
        claim_id=claim_id,
        claim=claim,
        run_id=run_id,
        trust_verdict=trust_verdict,
        ic95_verdict=ic95_verdict,
        agreement=agreement,
        note=note,
        method_note=pair_note,
        best_label=best_label,
        second_label=second_label,
        metric=metric,
        n_pairs=n_pairs,
        mean_diff=avg,
        ci_low=low,
        ci_high=high,
    )


def md_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def build_report(results: list[CrossCheckResult]) -> str:
    checked = len(results)
    concordant = sum(1 for result in results if result.agreement)
    lines = [
        "# Verifier cross-check: paired IC95 best-vs-2e",
        "",
        f"Résumé: {concordant}/{checked} verdicts concordent.",
        "",
        "| claim | verdict trust-ladder | verdict IC95 | accord oui/non | note |",
        "|---|---|---|---|---|",
    ]
    for result in results:
        claim_label = f"{result.claim_id}: {result.claim}"
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(claim_label),
                    md_escape(result.trust_verdict),
                    md_escape(result.ic95_verdict),
                    "oui" if result.agreement else "non",
                    md_escape(result.note),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Méthode",
            "",
            f"- Source claims: `{CLAIM_LEDGER.relative_to(ROOT)}`.",
            "- Source métriques: artifacts `runs/<run_id>.json` référencés par `evidence_run_ids`.",
            "- Unité appariée: seed explicite quand disponible, sinon fallback par ordre intra-config.",
            "- Comparaison: meilleur candidat vs deuxième meilleur candidat par moyenne de la métrique du sweep.",
            "- Différence appariée: score(best) - score(second) sur les seeds communes.",
            "- IC95: intervalle t bilatéral sur les différences appariées, self-contained via table t critique.",
            "- Verdict IC95: PASS si la borne basse de l'IC95 est strictement > 0, sinon FAIL. INCONCLUSIVE si données insuffisantes.",
            "- Accord: les statuts trust-ladder `allowed`, `weakly_supported`, `supported`, `strongly_supported` sont traités comme supportifs; `blocked`/`inconclusive` comme non-supportifs.",
            "",
            "## Limitations",
            "",
        ]
    )
    limitations = [result for result in results if result.ic95_verdict == "INCONCLUSIVE"]
    mismatches = [result for result in results if not result.agreement]
    if not limitations and not mismatches:
        lines.append("- Aucun claim exclu, inconclusive ou divergent.")
    for result in limitations:
        lines.append(f"- `{result.claim_id}` non recalculable: {result.note}.")
    for result in mismatches:
        if "protocol warning" in result.note:
            reason = (
                "l'IC95 teste seulement la significativité métrique; "
                "le trust ladder bloque aussi les warnings de protocole"
            )
        else:
            reason = (
                "l'IC95 apparié détecte un effet strictement positif, tandis que le trust ladder "
                "reste plus conservateur car il compare l'effet à la noise inter-seed"
            )
        lines.append(f"- `{result.claim_id}` diverge: {reason}.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    if not CLAIM_LEDGER.exists():
        raise FileNotFoundError(f"Missing claim ledger: {CLAIM_LEDGER}")
    ledger = load_json(CLAIM_LEDGER)
    entries = ledger.get("entries") or []
    if not entries:
        raise ValueError("claim ledger has no entries")

    results = [crosscheck_entry(entry) for entry in entries]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(results))

    concordant = sum(1 for result in results if result.agreement)
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    print(f"{concordant}/{len(results)} verdicts concordent")
    for result in results:
        print(
            f"{result.claim_id}: trust={result.trust_verdict} ic95={result.ic95_verdict} "
            f"accord={'oui' if result.agreement else 'non'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
