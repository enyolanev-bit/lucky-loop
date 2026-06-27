from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
REPORTS_DIR = ROOT / "reports"


st.set_page_config(page_title="Lucky Loop", layout="wide", initial_sidebar_state="collapsed")


st.markdown(
    """
    <style>
    :root {
      --ink: #1f2937;
      --muted: #64748b;
      --line: #d7dde6;
      --surface: #f8fafc;
      --panel: #ffffff;
      --accent: #0f766e;
      --blocked: #b42318;
      --supported: #047857;
    }
    .block-container {
      max-width: 1420px;
      padding-top: 2rem;
      padding-bottom: 4rem;
    }
    .hero {
      border-bottom: 1px solid var(--line);
      padding-bottom: 1.25rem;
      margin-bottom: 1.5rem;
    }
    .hero h1 {
      color: var(--ink);
      font-size: 2.7rem;
      line-height: 1.02;
      letter-spacing: 0;
      margin: 0;
      font-weight: 760;
    }
    .hero p {
      color: var(--muted);
      font-size: 1.02rem;
      margin: .45rem 0 0 0;
      max-width: 760px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: .95rem;
      min-height: 13.5rem;
    }
    .panel h4 {
      margin: 0 0 .55rem 0;
      font-size: .82rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .05em;
      font-weight: 700;
    }
    .panel p {
      color: var(--ink);
      margin: .35rem 0;
      font-size: .92rem;
      line-height: 1.35;
    }
    .callout {
      border-left: 4px solid var(--accent);
      background: #f8fafc;
      padding: .95rem 1rem;
      border-radius: 0 8px 8px 0;
      margin: .75rem 0;
    }
    .callout strong {
      color: var(--ink);
    }
    .badge {
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: .13rem .52rem;
      margin: .08rem .2rem .08rem 0;
      font-size: .78rem;
      color: var(--ink);
      background: #fff;
      white-space: nowrap;
    }
    .badge-supported {
      border-color: #a7f3d0;
      background: #ecfdf5;
      color: var(--supported);
    }
    .badge-blocked {
      border-color: #fecaca;
      background: #fff1f2;
      color: var(--blocked);
    }
    .badge-neutral {
      border-color: #cbd5e1;
      background: #f8fafc;
      color: #334155;
    }
    .tiny {
      color: var(--muted);
      font-size: .82rem;
      line-height: 1.35;
    }
    div[data-testid="stMetric"] {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: .85rem .95rem;
      background: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON in {path.name}: {exc}")
        return None


def _load_traces() -> list[dict[str, Any]]:
    traces = []
    for path in sorted(RUNS_DIR.glob("run_*.json")):
        data = _load_json(path)
        if data:
            traces.append(data)
    return traces


def _short(text: Any, limit: int = 190) -> str:
    value = "" if text is None else str(text).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _actual_metric(trace: dict[str, Any]) -> tuple[str, float | None, str]:
    actual = trace.get("actual_result") or {}
    raw = actual.get("raw") or {}
    if actual.get("accuracy") is not None:
        return "accuracy", float(actual["accuracy"]), f"{float(actual['accuracy']):.4f}"
    metric = raw.get("metric", "accuracy")
    best = raw.get("best") or {}
    key = f"mean_{metric}"
    if best.get(key) is not None:
        value = float(best[key])
        return metric, value, f"best mean {value:.4f}"
    return metric, None, "not recorded"


def _trace_label(trace: dict[str, Any]) -> str:
    action = trace.get("proposed_action") or {}
    raw = (trace.get("actual_result") or {}).get("raw") or {}
    return raw.get("scenario") or action.get("model") or trace.get("run_id", "run")


def _verification_badge(verification: dict[str, Any] | None) -> str:
    if not verification:
        return '<span class="badge badge-neutral">no claim gate</span>'
    status = verification.get("status", "missing_data")
    klass = "badge-supported" if status in {"supported", "strongly_supported"} else "badge-blocked"
    return f'<span class="badge {klass}">{_esc(status)}</span>'


def _match_badges(trace: dict[str, Any]) -> str:
    comparison = trace.get("comparison") or {}
    metric = "metric hit" if comparison.get("metric_match") else "metric miss"
    runtime = "runtime hit" if comparison.get("runtime_match") else "runtime miss"
    metric_class = "badge-supported" if comparison.get("metric_match") else "badge-blocked"
    runtime_class = "badge-supported" if comparison.get("runtime_match") else "badge-blocked"
    return (
        f'<span class="badge {metric_class}">{metric}</span>'
        f'<span class="badge {runtime_class}">{runtime}</span>'
    )


def _panel(title: str, body: list[str]) -> None:
    content = "".join(f"<p>{line}</p>" for line in body if line)
    st.markdown(f'<div class="panel"><h4>{_esc(title)}</h4>{content}</div>', unsafe_allow_html=True)


def _render_run(trace: dict[str, Any]) -> None:
    prediction = trace.get("world_model_prediction") or {}
    actual = trace.get("actual_result") or {}
    comparison = trace.get("comparison") or {}
    decision = trace.get("decision_trace") or {}
    verification = trace.get("verification") or {}
    metric_name, _metric_value, metric_display = _actual_metric(trace)
    action = trace.get("proposed_action") or {}
    rejected = decision.get("rejected_candidates") or []
    rejected_text = "; ".join(
        f"{(item.get('action') or {}).get('model', 'candidate')}: {item.get('reason', '')}" for item in rejected[:3]
    )
    if len(rejected) > 3:
        rejected_text += f"; +{len(rejected) - 3} more"

    cols = st.columns(4)
    with cols[0]:
        _panel(
            "World model said",
            [
                f"<strong>Metric</strong>: {_esc(prediction.get('expected_metric'))}",
                f"<strong>Runtime</strong>: {_esc(prediction.get('expected_runtime_seconds'))}",
                f"<strong>Recommendation</strong>: {_esc(prediction.get('recommendation'))}",
                f"<span class='tiny'>{_esc('; '.join(prediction.get('risks') or []) or 'No risk recorded')}</span>",
            ],
        )
    with cols[1]:
        _panel(
            "Planner did",
            [
                f"<strong>Action</strong>: {_esc(action.get('model'))}",
                f"<strong>Command</strong>: <span class='tiny'>{_esc(action.get('command'))}</span>",
                f"<span class='tiny'>{_esc(_short(decision.get('causal_reason') or trace.get('next_decision'), 240))}</span>",
                f"<span class='tiny'>Deferred: {_esc(_short(rejected_text, 220))}</span>" if rejected_text else "",
            ],
        )
    with cols[2]:
        events = comparison.get("unexpected_events") or []
        _panel(
            "Reality showed",
            [
                f"<strong>{_esc(metric_name)}</strong>: {_esc(metric_display)}",
                f"<strong>Runtime</strong>: {_esc(actual.get('runtime_seconds'))}s",
                _match_badges(trace),
                f"<span class='tiny'>{_esc(_short('; '.join(events) or comparison.get('lesson'), 220))}</span>",
            ],
        )
    with cols[3]:
        ratio = verification.get("effect_to_noise_ratio")
        ratio_text = "n/a" if ratio is None else f"{float(ratio):.2f}"
        _panel(
            "Verifier allowed / blocked",
            [
                _verification_badge(verification),
                f"<strong>Effect/noise</strong>: {_esc(ratio_text)}",
                f"<span class='tiny'><strong>Blocked</strong>: {_esc(_short(verification.get('blocked_claim'), 160))}</span>"
                if verification.get("blocked_claim")
                else "",
                f"<span class='tiny'><strong>Allowed</strong>: {_esc(_short(verification.get('allowed_claim'), 210))}</span>"
                if verification.get("allowed_claim")
                else "<span class='tiny'>No sweep claim was evaluated for this run.</span>",
            ],
        )


def _summary_rows(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for trace in traces:
        metric_name, _metric_value, metric_display = _actual_metric(trace)
        verification = trace.get("verification") or {}
        comparison = trace.get("comparison") or {}
        decision = trace.get("decision_trace") or {}
        rows.append(
            {
                "run": trace.get("run_id"),
                "action": _trace_label(trace),
                "metric": metric_name,
                "actual": metric_display,
                "metric_hit": comparison.get("metric_match"),
                "runtime_hit": comparison.get("runtime_match"),
                "verifier": verification.get("status") or "",
                "effect_noise": verification.get("effect_to_noise_ratio"),
                "decision": _short(decision.get("causal_reason") or trace.get("next_decision"), 150),
            }
        )
    return rows


def _find_run(traces: list[dict[str, Any]], run_id: str) -> dict[str, Any] | None:
    return next((trace for trace in traces if trace.get("run_id") == run_id), None)


traces = _load_traces()
ledger = _load_json(REPORTS_DIR / "claim_ledger.json") or {"entries": [], "summary": {}}

st.markdown(
    """
    <div class="hero">
      <h1>Lucky Loop</h1>
      <p>World-model-guided autonomous research. Predict before compute. Verify before claim.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not traces:
    st.warning("No traces found. Run `PYTHONPATH=src python -m luckyloop.loop --max-experiments 10` first.")
    st.stop()

metric_count = len([t for t in traces if _actual_metric(t)[1] is not None])
metric_hits = sum(1 for t in traces if (t.get("comparison") or {}).get("metric_match"))
runtime_hits = sum(1 for t in traces if (t.get("comparison") or {}).get("runtime_match"))
misses = sum(1 for t in traces if (t.get("comparison") or {}).get("unexpected_events"))
supported = (ledger.get("summary") or {}).get("strongly_supported", 0) + (ledger.get("summary") or {}).get("supported", 0)
blocked = (ledger.get("summary") or {}).get("blocked", 0)

top = st.columns(5)
top[0].metric("Predictions logged", len(traces))
top[1].metric("Metric coverage", f"{metric_hits}/{metric_count}")
top[2].metric("Runtime coverage", f"{runtime_hits}/{len(traces)}")
top[3].metric("Prediction misses", misses)
top[4].metric("Claims gated", f"{supported} allowed / {blocked} blocked")

st.markdown(
    """
    <div class="callout">
      <strong>Demo line:</strong> Qwen-AgentWorld predicts candidate outcomes first. Lucky Loop runs the real experiment,
      compares prediction with reality, then the verifier decides which scientific claims survive.
    </div>
    """,
    unsafe_allow_html=True,
)

tab_timeline, tab_moments, tab_calibration, tab_ledger, tab_raw = st.tabs(
    ["Timeline", "Demo moments", "Calibration", "Claim ledger", "Raw evidence"]
)

with tab_timeline:
    st.subheader("Experiment timeline")
    st.dataframe(pd.DataFrame(_summary_rows(traces)), width="stretch", hide_index=True)

    st.subheader("World model -> planner -> reality -> verifier")
    for trace in traces:
        label = f"{trace.get('run_id')} - {_trace_label(trace)}"
        with st.expander(label, expanded=trace.get("run_id") in {"run_002", "run_004", "run_005", "run_007", "run_008"}):
            _render_run(trace)

with tab_moments:
    st.subheader("The 60 second story")
    moments = [
        ("Prediction helps", "run_002", "The planner tests scaling right after the unscaled baseline."),
        ("Prediction miss", "run_010", "The result lands outside the predicted accuracy interval and stays visible."),
        ("Verifier blocks overclaim", "run_004", "A tempting hyperparameter claim is blocked because effect < seed noise."),
        ("Trust ladder accepts a real effect", "run_005", "A controlled real-effect case reaches strongly_supported."),
        ("Leakage trap", "run_007", "A near-perfect score is blocked by a protocol warning."),
        ("Metric misuse trap", "run_008", "Accuracy-only reasoning is rejected in favor of balanced accuracy/F1 framing."),
    ]
    for title, run_id, explanation in moments:
        trace = _find_run(traces, run_id)
        if not trace:
            continue
        st.markdown(f"#### {title}")
        st.markdown(f'<p class="tiny">{_esc(explanation)}</p>', unsafe_allow_html=True)
        _render_run(trace)

with tab_calibration:
    st.subheader("World model calibration")
    calibration_path = REPORTS_DIR / "world_model_calibration.md"
    if calibration_path.exists():
        st.markdown(calibration_path.read_text(encoding="utf-8"))
    else:
        st.info("Calibration report has not been generated yet.")

with tab_ledger:
    st.subheader("Claim ledger")
    entries = ledger.get("entries") or []
    if entries:
        ledger_rows = []
        for entry in entries:
            metrics = entry.get("metrics") or {}
            ledger_rows.append(
                {
                    "claim_id": entry.get("claim_id"),
                    "status": entry.get("status"),
                    "claim": entry.get("claim"),
                    "allowed_rewrite": entry.get("allowed_rewrite"),
                    "evidence": ", ".join(entry.get("evidence_run_ids") or []),
                    "effect_noise": metrics.get("effect_to_noise_ratio"),
                }
            )
        st.dataframe(pd.DataFrame(ledger_rows), width="stretch", hide_index=True)
    else:
        st.info("No claim ledger entries found.")

with tab_raw:
    st.subheader("Audit trail")
    selected = st.selectbox("Trace", [trace.get("run_id") for trace in traces])
    trace = _find_run(traces, selected)
    if trace:
        st.json(trace)

    report_path = REPORTS_DIR / "final_report.md"
    if report_path.exists():
        with st.expander("Final report markdown"):
            st.markdown(report_path.read_text(encoding="utf-8"))
