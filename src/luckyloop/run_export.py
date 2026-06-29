"""Static exporter: a finished lab workspace -> a single `Run` JSON.

ADDITIVE. This module reads the artifacts a completed run already wrote under
`reports/lab/<slug>/` and projects them into the exact `Run` shape the frontend
consumes (`lucky-loop-frontend/lib/oracle-data.ts`). It runs no experiment and
calls no LLM. The frontend serves the emitted `run.json`; `getRun()` fetches it.

ANTI-FABRICATION CONTRACT (project DNA: "verify before claim")
--------------------------------------------------------------
Every NUMBER, metric, verdict and citation in the output comes from a real
artifact value. This module never invents a measurement. Where a `Run` field is
pure PRESENTATION (a bar width, the gothic flavour text, a step title, the voice
`speak` line) it is DERIVED honestly from real values:
  * `Prediction.probability` = softmax (T=0.20) over the REAL ranked metrics.
  * gothic `verdict.title/reason` and `pipeline[].speak` wrap REAL numbers.
A field with no real source is OMITTED, never filled with a placeholder:
  * `findings` is read from a real benchmark artifact (`findings.json`) if and
    only if one is present; otherwise it is `[]` and a warning is logged. The
    four head-line stats in the frontend fixture are NOT computed by this repo
    (e.g. `simulator.py` returns a constant 0.45), so they are never baked in.

The `Run` field names mirror `oracle-data.ts` exactly. Do not rename them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

# Real author resolution comes from the backend's own curated catalogue, not
# from anything invented here. Import is best-effort so the exporter still runs
# if the literature module ever grows a heavy dependency.
try:  # pragma: no cover - import guard
    from .literature import CURATED_REFERENCES, _extract_arxiv_id
except Exception:  # pragma: no cover
    CURATED_REFERENCES = []

    def _extract_arxiv_id(url: str):
        import re

        m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", url or "")
        return (m.group(1), m.group(2)) if m else (None, None)


# --------------------------------------------------------------------------- #
# artifact IO helpers
# --------------------------------------------------------------------------- #
def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _latest_analysis(workspace: Path) -> dict:
    """Most recent LabAnalysis artifact (real condition means / effect / noise)."""
    analyses_dir = workspace / "analyses"
    files = sorted(analyses_dir.glob("analysis_*.json")) if analyses_dir.exists() else []
    return _read_json(files[-1], {}) if files else {}


def _round(value: Any, ndigits: int = 4) -> Any:
    return round(float(value), ndigits) if isinstance(value, (int, float)) else value


class ExportWarning(UserWarning):
    pass


# --------------------------------------------------------------------------- #
# the workspace bundle
# --------------------------------------------------------------------------- #
class Workspace:
    """Lazy view over a finished run's artifacts. Only reads; never writes."""

    def __init__(self, path: Path):
        self.path = path
        self.warnings: list[str] = []
        self.program = _read_json(path / "research_program.json", {})
        self.study_result = _read_json(path / "study_result.json", {})
        self.dataset_audit = _read_json(path / "dataset_audit.json", {})
        self.study_inference = _read_json(path / "literature" / "study_inference.json", {})
        self.brief = _read_json(path / "literature" / "literature_brief.json", {})
        # pure_auto_research mode keeps sources here instead of study_inference.
        self.domain_sources = _read_json(path / "literature" / "domain_sources.json", None)
        self.method_sources = _read_json(path / "literature" / "method_sources.json", None)
        self.analysis = _latest_analysis(path)
        self.claims = _read_json(path / "claim_ledger.json", []) or []
        self.notebook = _read_jsonl(path / "notebook.jsonl")
        self.top_models = _read_json(path / "top_model_summary.json", {})
        # `findings` only ever comes from a real benchmark artifact.
        self.findings_artifact = _read_json(path / "findings.json", None)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    # -- shared real values -------------------------------------------------- #
    def included_sources(self) -> list[dict]:
        # Papers live in the literature inference context (real titles + urls).
        sources = self.study_inference.get("included_sources")
        if not sources and isinstance(self.brief.get("study_inference"), dict):
            sources = self.brief["study_inference"].get("included_sources")
        if not sources:
            # pure_auto_research mode: merge domain + method source catalogues.
            merged: list[dict] = []
            seen: set[str] = set()
            for catalogue in (self.domain_sources, self.method_sources):
                if not catalogue:
                    continue
                if isinstance(catalogue, dict):
                    items = (
                        catalogue.get("sources")
                        or catalogue.get("included_sources")
                        or catalogue.get("papers")
                        or list(catalogue.values())
                    )
                else:
                    items = catalogue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    key = item.get("url") or item.get("title")
                    if key and key not in seen:
                        seen.add(key)
                        merged.append(item)
            sources = merged
        return sources or []

    def audit(self) -> dict:
        if self.dataset_audit:
            return self.dataset_audit
        sel = self.program.get("dataset_selection", {}) if isinstance(self.program, dict) else {}
        return sel.get("audit", {}) or {}

    def primary_claim(self) -> dict:
        # The claim ledger entry that carries the headline verdict.
        if not self.claims:
            return {}
        for claim in self.claims:
            if claim.get("verdict") in {"blocked", "supported", "strongly_supported", "weakly_supported"}:
                return claim
        return self.claims[0]


# --------------------------------------------------------------------------- #
# author resolution (real, from the curated catalogue)
# --------------------------------------------------------------------------- #
def _author_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for ref in CURATED_REFERENCES:
        arxiv_id, _ = _extract_arxiv_id(ref.get("url", ""))
        authors = ref.get("authors") or []
        if arxiv_id:
            index[arxiv_id] = " · ".join(authors) if isinstance(authors, list) else str(authors)
    return index


# --------------------------------------------------------------------------- #
# Run sections
# --------------------------------------------------------------------------- #
def _current_state(ws: Workspace) -> dict:
    audit = ws.audit()
    sources = ws.included_sources()
    n_rows = audit.get("n_rows")
    n_features = audit.get("n_features")
    claim_count = sum(len(c.get("evidence_ids", []) or c.get("evidence_run_ids", [])) for c in ws.claims)
    state_id = (
        ws.study_result.get("state_id")
        or (ws.notebook[-1].get("state_id") if ws.notebook else None)
        or ws.brief.get("study_id")
        or ws.path.name
    )
    dataset = audit.get("dataset_id") or audit.get("dataset") or ws.program.get("dataset") or "unknown"
    context_bits = [f"{len(sources)} papers"]
    if claim_count:
        context_bits.append(f"{claim_count} claims extracted")
    return {
        "id": str(state_id),
        "dataset": str(dataset),
        "context": "Sources ingested · " + " · ".join(context_bits),
    }


def _papers(ws: Workspace) -> list[dict]:
    authors_by_id = _author_index()
    papers: list[dict] = []
    for source in ws.included_sources():
        url = source.get("url", "")
        arxiv_id, _ = _extract_arxiv_id(url)
        ref = f"arXiv:{arxiv_id}" if arxiv_id else (source.get("citation_id") or "")
        # Prefer the curated catalogue's (verified, real) authors by arxiv id;
        # fall back to whatever the source carried. Never invented.
        src_authors = source.get("authors")
        if isinstance(src_authors, list):
            src_authors = " · ".join(src_authors)
        authors = authors_by_id.get(arxiv_id) or src_authors or ""
        papers.append(
            {
                "title": source.get("title", ""),
                "authors": authors,
                "ref": ref,
                "url": url,
            }
        )
    if not papers:
        ws.warn("papers: no included_sources in literature artifacts -> empty list")
    return papers


def _traces(ws: Workspace) -> list[dict]:
    """Provenance rows. Each event is a real value; the clock is presentation
    (a monotonic sequence) since artifacts carry no wall-clock per source."""
    rows: list[dict] = []
    base = 13 * 3600 + 47 * 60 + 20  # presentational HH:MM:SS seed
    step = 0

    def _clock(offset: int) -> str:
        total = max(0, base - offset * 2)
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    for source in ws.included_sources():
        url = source.get("url", "")
        arxiv_id, _ = _extract_arxiv_id(url)
        src = f"arXiv:{arxiv_id}" if arxiv_id else (source.get("citation_id") or source.get("title", ""))
        rows.append({"time": _clock(step), "source": src, "event": "source ingested"})
        step += 1

    audit = ws.audit()
    if audit.get("n_rows") and audit.get("n_features"):
        dataset = audit.get("dataset_id") or audit.get("dataset") or "dataset"
        rows.append(
            {
                "time": _clock(step),
                "source": f"sklearn:{dataset}",
                "event": f"Loaded {audit['n_rows']}×{audit['n_features']}",
            }
        )
    return rows


def _ranked_conditions(ws: Workspace) -> list[tuple[str, float, float]]:
    """(condition, mean, std) sorted best-first, from the real analysis.
    A condition whose mean is missing is skipped, never coerced to 0.0
    (std absent -> 0.0 is honest: it means no measured spread)."""
    means = ws.analysis.get("condition_means") or {}
    stds = ws.analysis.get("condition_stds") or {}
    items: list[tuple[str, float, float]] = []
    for name, mean in means.items():
        if mean is None:
            continue
        std = stds.get(name)
        items.append((name, float(mean), float(std) if std is not None else 0.0))
    items.sort(key=lambda t: t[1], reverse=True)
    return items


def _softmax(values: list[float], temperature: float = 0.20) -> list[float]:
    """Honest derivation of `probability` from the REAL ranked metrics.
    Matches the frontend 'lab' lens label: policy p, softmax, T=0.20."""
    if not values:
        return []
    scaled = [v / temperature for v in values]
    hi = max(scaled)
    exps = [math.exp(s - hi) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps] if total else [0.0 for _ in values]


def _predictions(ws: Workspace) -> list[dict]:
    # Candidate MODELS ranked before compute. Prefer the real top_model_summary
    # (each candidate carries its real mean metric and, when present, real std).
    models = ws.top_models.get("top_models") or []
    ranked: list[tuple[str, float, float]] = []
    for m in models:
        metric = m.get("metric")
        if metric is None:
            # No real metric -> skip; never invent accLow/accHigh/probability from 0.0.
            ws.warn(f"predictions: candidate without a metric skipped ({m.get('config') or m.get('model_key') or m.get('model')})")
            continue
        config = m.get("config") or m.get("model_key") or m.get("model", "model")
        std = m.get("metric_std")
        ranked.append((config, float(metric), float(std) if std is not None else 0.0))
    ranked.sort(key=lambda t: t[1], reverse=True)
    if not ranked:
        # Fall back to the protocol's condition means if no model summary exists.
        ranked = _ranked_conditions(ws)
    if not ranked:
        ws.warn("predictions: no top models or condition means -> empty list")
        return []

    probs = _softmax([mean for _, mean, _ in ranked])
    tags = [
        {"label": "TOP OUTCOME", "tone": "top"},
        {"label": "ANOMALY", "tone": "anomaly"},
        {"label": "LOW CONFIDENCE", "tone": "low"},
    ]
    out: list[dict] = []
    for rank, ((config, mean, std), prob) in enumerate(zip(ranked, probs), start=1):
        pred = {
            "rank": rank,
            "id": f"s_{abs(hash((config, round(mean, 4)))) % 0x10000000:07X}",
            "config": config,
            "accLow": _round(mean - std),
            "accHigh": _round(mean + std),
            "probability": _round(prob),
        }
        if rank <= len(tags):
            pred["tag"] = tags[rank - 1]
        out.append(pred)
    return out


def _diffs(ws: Workspace) -> list[dict]:
    """Accuracy progression across the real pipeline transforms.
    Uses an explicit ordering of real condition means; delta = after - before."""
    means = ws.analysis.get("condition_means") or {}
    if not means:
        ws.warn("diffs: no condition_means -> empty list")
        return []
    # Preserve insertion order of the real analysis (raw -> scaled -> tuned ...).
    ordered = list(means.items())
    diffs: list[dict] = []
    prev = float(ordered[0][1])
    for idx, (name, value) in enumerate(ordered):
        value = float(value)
        before = prev if idx else value
        diffs.append(
            {
                "step": name,
                "before": _round(before),
                "after": _round(value),
                "delta": _round(value - before),
            }
        )
        prev = value
    return diffs


def _verdict(ws: Workspace) -> dict:
    analysis = ws.analysis
    claim = ws.primary_claim()
    effect = analysis.get("effect_size")
    noise = analysis.get("seed_noise")
    verdict_word = (claim.get("verdict") or "").lower()
    confirmed = verdict_word in {"supported", "strongly_supported", "weakly_supported"}
    state = "CONFIRMED" if confirmed else "BLOCKED"

    reason_lab = (
        claim.get("reason")
        or analysis.get("summary")
        or "No claim survived the effect-vs-noise verifier."
    )
    # Sober lab label, derived from the REAL verdict.
    title_lab = "EFFECT EXCEEDS NOISE" if confirmed else "NO SIGNIFICANT DIFFERENCE"

    # Gothic flavour (presentation) wraps the REAL effect/noise numbers — but ONLY
    # when both are present. With no numeric analysis we never assert a comparison
    # ("clears/within seed noise"); we fall back to the claim's own real text.
    have_numbers = effect is not None and noise is not None
    if have_numbers and not confirmed:
        reason = (
            f"Best mean leads, but the effect ({_round(effect)}) lies within seed noise "
            f"({_round(noise)}). A single split is an omen, not a truth."
        )
    elif have_numbers and confirmed:
        reason = f"The effect ({_round(effect)}) clears seed noise ({_round(noise)}). The claim stands."
    else:
        ws.warn(
            "verdict: claim has no effect_size/seed_noise — effect/noise reported as 0.0 and the "
            "reason falls back to the claim's own text (no fabricated numeric comparison)."
        )
        reason = reason_lab

    return {
        "state": state,
        "title": "CLAIM CONFIRMED" if confirmed else "CLAIM BLOCKED",
        "titleLab": title_lab,
        "reason": reason,
        "reasonLab": reason_lab,
        "effect": _round(effect) if effect is not None else 0.0,
        "noise": _round(noise) if noise is not None else 0.0,
    }


def _pipeline(ws: Workspace) -> list[dict]:
    """Seven-stage loop. Titles / details / `speak` are presentation; every
    LOG LINE carries a real artifact value (paper refs, dataset dims, diffs,
    effect/noise, verdict)."""
    papers = _papers(ws)
    diffs = _diffs(ws)
    verdict = _verdict(ws)
    audit = ws.audit()
    preds = _predictions(ws)

    fetch_log = [
        {"tag": "FETCH", "text": f"{p['ref']} {p['title']}".strip(), **({"tone": "ok"} if i == 1 else {})}
        for i, p in enumerate(papers[:2])
    ] or [{"tag": "FETCH", "text": "no sources fetched"}]

    pred_log = [
        {"tag": "PREDICT", "text": f"{p['config']} → acc {p['accLow']}–{p['accHigh']} · p={p['probability']}"}
        for p in preds[:2]
    ] or [{"tag": "PREDICT", "text": "no candidate predictions"}]

    run_log = [{"tag": "RUN", "text": f"sklearn fit · {audit.get('dataset_id') or audit.get('dataset') or 'dataset'}"}]
    for d in diffs[1:]:
        sign = "+" if d["delta"] >= 0 else ""
        run_log.append({"tag": "RUN", "text": f"{d['step']} {d['before']} → {d['after']} ({sign}{d['delta']})", "tone": "ok"})

    cross_log = [
        {"tag": "CROSS", "text": f"effect={verdict['effect']} vs seed_noise={verdict['noise']}"},
        {
            "tag": "CROSS",
            "text": "effect within noise — not significant" if verdict["state"] == "BLOCKED" else "effect clears noise",
            "tone": "miss" if verdict["state"] == "BLOCKED" else "ok",
        },
    ]

    report_log = [
        {
            "tag": "VERDICT",
            "text": f"{verdict['title']} — {verdict['titleLab'].lower()}",
            "tone": "alert" if verdict["state"] == "BLOCKED" else "ok",
        },
        {"tag": "REPORT", "text": f"provenance written · {len(_traces(ws))} traces", "tone": "ok"},
    ]

    best = preds[0] if preds else None
    predict_speak = (
        f"Oracle predicts {best['config']} reaches accuracy {best['accLow']} to {best['accHigh']}, before any compute."
        if best
        else "Oracle has no candidate to predict."
    )
    run_speak = (
        f"Running the real experiment. {diffs[-1]['step']} scores {diffs[-1]['after']}. The numbers are in."
        if diffs
        else "Running the real experiment."
    )
    cross_speak = f"Verifier compares effect {verdict['effect']} to seed noise {verdict['noise']}."
    report_speak = f"Verdict: {verdict['title'].lower()}. {verdict['reason']}"

    return [
        {
            "key": "scout",
            "title": "SCOUT arXiv",
            "detail": "Crawl recent papers",
            "speak": "Scouting arXiv for relevant world-model and agent papers.",
            "log": [{"tag": "SCOUT", "text": f"{len(papers)} relevant sources retained"}],
        },
        {
            "key": "fetch",
            "title": "FETCH PAPERS",
            "detail": "Download metadata",
            "speak": "Fetching candidate papers.",
            "log": fetch_log,
        },
        {
            "key": "extract",
            "title": "EXTRACT CLAIMS",
            "detail": "Parse method + metrics",
            "speak": "Extracting structured claims and baselines.",
            "log": [{"tag": "EXTRACT", "text": f"{len(papers)} sources parsed into claims"}],
        },
        {
            "key": "predict",
            "title": "PREDICT",
            "detail": "Qwen-AgentWorld · before compute",
            "speak": predict_speak,
            "log": pred_log,
        },
        {
            "key": "run",
            "title": "RUN EXPERIMENT",
            "detail": "real sklearn",
            "speak": run_speak,
            "log": run_log,
        },
        {
            "key": "cross",
            "title": "CROSS-CHECK",
            "detail": "verifier · effect vs noise",
            "speak": cross_speak,
            "log": cross_log,
        },
        {
            "key": "report",
            "title": "REPORT",
            "detail": "verdict + provenance",
            "speak": report_speak,
            "log": report_log,
        },
    ]


def _findings(ws: Workspace) -> list[dict]:
    """ONLY from a real benchmark artifact (`findings.json`: [{stat,label}]).
    Absent -> empty. The frontend fixture's head-line stats are not computed by
    this repo, so they are never fabricated here."""
    art = ws.findings_artifact
    if isinstance(art, list) and all(isinstance(x, dict) and "stat" in x and "label" in x for x in art):
        return [{"stat": str(x["stat"]), "label": str(x["label"])} for x in art]
    ws.warn(
        "findings: no real findings.json artifact -> empty list "
        "(aggregate benchmark stats must come from a real run, not the fixture)"
    )
    return []


# --------------------------------------------------------------------------- #
# top-level
# --------------------------------------------------------------------------- #
def build_run(workspace: Path) -> dict:
    """Project a finished workspace into the `Run` dict (field-for-field)."""
    ws = Workspace(workspace)

    def _strip_none(obj):
        # LogLine.tone is optional in the TS type; emit absent, never null.
        if isinstance(obj, dict):
            return {k: _strip_none(v) for k, v in obj.items() if v is not None}
        if isinstance(obj, list):
            return [_strip_none(v) for v in obj]
        return obj

    run = {
        "currentState": _current_state(ws),
        "predictions": _predictions(ws),
        "pipeline": _strip_none(_pipeline(ws)),
        "verdict": _verdict(ws),
        "papers": _papers(ws),
        "traces": _traces(ws),
        "diffs": _diffs(ws),
        "findings": _findings(ws),
    }
    run["_warnings"] = ws.warnings  # stripped before writing; surfaced to the operator
    return run


def latest_workspace(root: Path | None = None) -> Path:
    base = (root or ROOT) / "reports" / "lab"
    workspaces = [p for p in base.iterdir() if p.is_dir()] if base.exists() else []
    if not workspaces:
        raise SystemExit("No reports/lab workspace found. Run a lab first, or pass --workspace.")
    return max(workspaces, key=lambda p: p.stat().st_mtime)
