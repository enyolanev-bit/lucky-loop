from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_digits, load_wine

from .schemas import DatasetAudit, DatasetCandidate, DatasetSearchPlan, DatasetSelectionRationale


HF_API = "https://huggingface.co/api/datasets"
OPENML_SEARCH_API = "https://www.openml.org/api/v1/json/data/list/data_name/{query}/limit/{limit}/status/active"
OPENML_FALLBACKS = [
    ("sonar", "OpenML Sonar binary classification benchmark", "https://www.openml.org/d/40"),
    ("eeg_eye_state", "OpenML EEG eye state sensor classification benchmark", "https://www.openml.org/d/1471"),
    ("har", "OpenML human activity recognition classification benchmark", "https://www.openml.org/d/1478"),
]
OPENML_DATASETS = {"sonar": 40, "eeg_eye_state": 1471, "har": 1478}


def dataset_queries(question: str, literature_context: dict, search_plan: DatasetSearchPlan | None = None) -> list[str]:
    if search_plan and search_plan.queries:
        return _expand_queries(search_plan.queries)
    text = f"{question} " + " ".join(
        str(gap.get("claim", "")) for gap in literature_context.get("gap_findings", [])
    )
    tokens = [
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 3 and token not in {"with", "that", "from", "this", "can", "does", "dataset", "research"}
    ]
    joined = " ".join(tokens[:8])
    queries = [
        f"{joined} classification",
        "machine learning classification tabular",
        "medical classification",
        "sensor classification",
    ]
    return _expand_queries(queries)


def _expand_queries(values: list[str]) -> list[str]:
    base = _dedupe(values)
    expanded = []
    for query in base:
        expanded.append(query)
        q = query.lower()
        if "human activity" in q or "har" in q:
            expanded.extend(["human activity recognition", "wearable sensor classification", "accelerometer gyroscope classification"])
        if "sensor" in q:
            expanded.extend(["human activity recognition", "eeg eye state classification", "wearable sensor classification"])
        if "tabular" in q or "classification" in q:
            expanded.extend(["tabular classification", "tabular-classification"])
    return _dedupe(expanded)


def _dedupe(values: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for query in values:
        key = query.strip().lower()
        if key and key not in seen:
            deduped.append(query)
            seen.add(key)
    return deduped


def _get_json(url: str, timeout: int = 20):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _record_error(errors: list[dict], source: str, query: str, exc: Exception) -> None:
    errors.append({"source": source, "query": query, "error": f"{type(exc).__name__}: {exc}"})


def search_huggingface(query: str, limit: int = 8, errors: list[dict] | None = None) -> list[DatasetCandidate]:
    params = {"search": query, "limit": str(limit), "full": "true"}
    if "tabular" in query.lower():
        params.pop("search")
        params["filter"] = "tabular-classification"
    url = f"{HF_API}?{urllib.parse.urlencode(params)}"
    try:
        rows = _get_json(url)
    except Exception as exc:
        if errors is not None:
            _record_error(errors, "huggingface", query, exc)
        return []
    candidates = []
    for rank, row in enumerate(rows[:limit], start=1):
        dataset_id = str(row.get("id") or row.get("name") or "")
        if not dataset_id:
            continue
        tags = [str(item) for item in row.get("tags", [])]
        card = row.get("cardData") if isinstance(row.get("cardData"), dict) else {}
        license_value = card.get("license") or row.get("license")
        score, reasons, risks = _score_candidate(query, dataset_id, " ".join(tags), license_value)
        candidates.append(
            DatasetCandidate(
                dataset_id=dataset_id,
                source="huggingface",
                name=dataset_id,
                url=f"https://huggingface.co/datasets/{dataset_id}",
                description=str(row.get("description") or ""),
                license=str(license_value) if license_value else None,
                task_tags=tags,
                score=score,
                score_reasons=["Dynamic Hugging Face API candidate", *reasons],
                risk_flags=risks,
                fallback_rank=rank,
            )
        )
    return candidates


def search_openml(query: str, limit: int = 10, errors: list[dict] | None = None) -> list[DatasetCandidate]:
    candidates: dict[str, DatasetCandidate] = {}
    for variant in _openml_query_variants(query):
        url = OPENML_SEARCH_API.format(query=urllib.parse.quote(variant), limit=limit)
        try:
            payload = _get_json(url)
        except Exception as exc:
            if errors is not None:
                _record_error(errors, "openml", variant, exc)
            continue
        rows = payload.get("data", {}).get("dataset", [])
        if isinstance(rows, dict):
            rows = [rows]
        for rank, row in enumerate(rows[:limit], start=1):
            data_id = str(row.get("did") or "")
            name = str(row.get("name") or data_id)
            if not data_id or not name:
                continue
            description = " ".join(
                str(row.get(key) or "") for key in ["name", "format", "NumberOfInstances", "NumberOfFeatures"]
            )
            score, reasons, risks = _score_candidate(f"{query} {variant}", name, description, "unknown")
            candidate = DatasetCandidate(
                dataset_id=data_id,
                source="openml",
                name=name,
                url=f"https://www.openml.org/d/{data_id}",
                description=description,
                license="unknown",
                task_tags=["classification", "tabular"],
                score=score + 8 - rank * 0.1,
                score_reasons=[f"Dynamic OpenML API candidate via query `{variant}`", *reasons],
                risk_flags=risks,
                fallback_rank=rank,
            )
            if data_id not in candidates or candidate.score > candidates[data_id].score:
                candidates[data_id] = candidate
    return sorted(candidates.values(), key=lambda item: item.score, reverse=True)


def _openml_query_variants(query: str) -> list[str]:
    q = query.lower()
    variants = [query]
    if "human activity" in q or "activity recognition" in q or "wearable" in q:
        variants.append("har")
    if "eeg" in q or "eye state" in q:
        variants.append("eeg-eye-state")
    if "sonar" in q:
        variants.append("sonar")
    if "sensor" in q:
        variants.extend(["har", "eeg-eye-state", "sonar"])
    return _dedupe(variants)


def external_registry_candidates(query_text: str) -> list[DatasetCandidate]:
    candidates = []
    for index, (name, description, url) in enumerate(OPENML_FALLBACKS, start=1):
        score, reasons, risks = _score_candidate(query_text, name, description, "unknown")
        score += 15
        candidates.append(
            DatasetCandidate(
                dataset_id=name,
                source="openml",
                name=name,
                url=url,
                description=description,
                license="unknown",
                task_tags=["classification", "tabular"],
                score=score - index * 0.1,
                score_reasons=["External OpenML registry candidate with known classification target", *reasons],
                risk_flags=[*risks, "curated_fallback"],
                fallback_rank=index,
            )
        )
    return candidates


def discover_dataset_candidates(
    question: str,
    literature_context: dict,
    workspace: Path,
    max_hf: int = 12,
    search_plan: DatasetSearchPlan | None = None,
) -> list[DatasetCandidate]:
    queries = dataset_queries(question, literature_context, search_plan)
    candidates: list[DatasetCandidate] = []
    errors: list[dict] = []
    for query in queries:
        candidates.extend(search_huggingface(query, max(2, max_hf // max(1, len(queries))), errors))
        candidates.extend(search_openml(query, limit=8, errors=errors))
    candidates.extend(external_registry_candidates(" ".join([*queries, *(search_plan.required_properties if search_plan else [])])))
    deduped: dict[tuple[str, str], DatasetCandidate] = {}
    for candidate in candidates:
        key = (candidate.source, candidate.dataset_id)
        if key not in deduped or candidate.score > deduped[key].score:
            deduped[key] = candidate
    ordered = sorted(
        deduped.values(),
        key=lambda item: ("curated_fallback" not in item.risk_flags, item.score, -item.fallback_rank),
        reverse=True,
    )
    out_dir = workspace / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "search_queries.json").write_text(json.dumps(queries, indent=2), encoding="utf-8")
    if search_plan:
        (out_dir / "search_plan.json").write_text(json.dumps(search_plan.model_dump(), indent=2), encoding="utf-8")
    (out_dir / "search_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    (out_dir / "candidates.json").write_text(
        json.dumps([item.model_dump() for item in ordered], indent=2),
        encoding="utf-8",
    )
    return ordered


def _score_candidate(query: str, dataset_id: str, tags: str, license_value) -> tuple[float, list[str], list[str]]:
    haystack = f"{dataset_id} {tags}".lower()
    score = 0.0
    reasons = []
    risks = []
    for token in re.findall(r"[a-z0-9]+", query.lower()):
        if len(token) > 3 and token in haystack:
            score += 4
    if "classification" in haystack:
        score += 8
        reasons.append("classification tag/name match")
    if any(term in haystack for term in ["tabular", "csv", "medical", "sensor", "benchmark"]):
        score += 5
        reasons.append("likely usable for empirical ML")
    if not license_value:
        score -= 3
        risks.append("missing_license")
    if any(term in haystack for term in ["image", "audio", "video", "text", "nlp"]):
        score -= 2
        risks.append("may_need_non_tabular_pipeline")
    return score, reasons or ["metadata match"], risks


def select_and_materialize_dataset(
    candidates: list[DatasetCandidate],
    workspace: Path,
    max_rows: int = 50000,
    search_plan: DatasetSearchPlan | None = None,
    hypothesis_text: str = "",
) -> tuple[DatasetCandidate, DatasetAudit]:
    audit_dir = workspace / "datasets" / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    rejected = []
    accepted = []
    ordered_candidates = sorted(
        candidates,
        key=lambda item: ("curated_fallback" not in item.risk_flags, item.score, -item.fallback_rank),
        reverse=True,
    )
    for candidate in ordered_candidates:
        audit = audit_candidate(candidate, workspace, max_rows=max_rows)
        (audit_dir / f"{_safe_name(candidate.source + '_' + candidate.dataset_id)}.json").write_text(
            json.dumps(audit.model_dump(), indent=2),
            encoding="utf-8",
        )
        if audit.status == "accepted":
            accepted.append((candidate, audit))
        else:
            rejected.append(audit.model_dump())
    (workspace / "datasets" / "rejected_candidates.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")
    if accepted:
        dynamic = [
            (c, a)
            for c, a in accepted
            if "curated_fallback" not in c.risk_flags and _fits_domain(c, a, search_plan, hypothesis_text)
        ]
        domain_fit = [(c, a) for c, a in accepted if _fits_domain(c, a, search_plan, hypothesis_text)]
        candidate, audit = max(dynamic or domain_fit or accepted, key=lambda item: item[0].score)
        rationale = DatasetSelectionRationale(
            selected_dataset_id=candidate.dataset_id,
            selected_source=candidate.source,
            selected_reason=f"{audit.reason}; score={candidate.score:.2f}; reasons={', '.join(candidate.score_reasons)}",
            rejected_summary=[f"{item['source']}:{item['dataset_id']} -> {item['reason']}" for item in rejected[:8]],
            fit_to_hypothesis=_dataset_fit_text(candidate, audit, search_plan, hypothesis_text),
            risks=[*candidate.risk_flags, *audit.warnings],
        )
        (workspace / "datasets" / "selected_dataset.json").write_text(
            json.dumps({"candidate": candidate.model_dump(), "audit": audit.model_dump()}, indent=2),
            encoding="utf-8",
        )
        (workspace / "datasets" / "selection_rationale.json").write_text(
            json.dumps(rationale.model_dump(), indent=2),
            encoding="utf-8",
        )
        return candidate, audit
    raise RuntimeError(f"No usable dataset found. Rejected audits: {json.dumps(rejected[:5], indent=2)}")


def _fits_domain(
    candidate: DatasetCandidate,
    audit: DatasetAudit,
    search_plan: DatasetSearchPlan | None,
    hypothesis_text: str,
) -> bool:
    context = " ".join(
        [
            hypothesis_text,
            " ".join(search_plan.queries if search_plan else []),
            " ".join(search_plan.required_properties if search_plan else []),
        ]
    ).lower()
    domain_terms = {
        "sensor": ["sensor", "eeg", "har", "human activity", "wearable", "accelerometer", "gyroscope"],
        "medical": ["medical", "diagnosis", "cancer", "clinical", "patient"],
    }
    required = [terms for trigger, terms in domain_terms.items() if any(term in context for term in terms)]
    if not required:
        return True
    haystack = " ".join(
        [
            candidate.dataset_id,
            candidate.name,
            candidate.description,
            " ".join(candidate.task_tags),
            audit.target_column or "",
            " ".join(audit.feature_columns[:30]),
        ]
    ).lower()
    return any(any(_contains_term(haystack, term) for term in terms) for terms in required)


def _contains_term(haystack: str, term: str) -> bool:
    if len(term) <= 3:
        return re.search(rf"\b{re.escape(term)}\b", haystack) is not None
    return term in haystack


def _dataset_fit_text(
    candidate: DatasetCandidate,
    audit: DatasetAudit,
    search_plan: DatasetSearchPlan | None,
    hypothesis_text: str,
) -> str:
    requirements = ", ".join(search_plan.required_properties) if search_plan else "supervised classification table"
    return (
        f"{candidate.source}:{candidate.dataset_id} provides {audit.n_rows} rows, {audit.n_features} features, "
        f"target `{audit.target_column}`, and is being used to test: {hypothesis_text or 'the selected hypothesis'}. "
        f"Required properties considered: {requirements}."
    )


def audit_candidate(candidate: DatasetCandidate, workspace: Path, max_rows: int = 50000) -> DatasetAudit:
    try:
        if candidate.source == "huggingface":
            frame, target = _load_hf_frame(candidate.dataset_id, max_rows)
        elif candidate.source == "openml":
            frame, target = _load_openml_frame(candidate.dataset_id, max_rows)
        else:
            frame, target = _load_sklearn_frame(candidate.dataset_id)
    except Exception as exc:
        return DatasetAudit(
            dataset_id=candidate.dataset_id,
            source=candidate.source,
            status="rejected",
            reason=f"load_failed: {type(exc).__name__}: {exc}",
        )
    return _audit_frame(candidate, frame, target, workspace)


def _load_hf_frame(dataset_id: str, max_rows: int) -> tuple[pd.DataFrame, str]:
    from datasets import get_dataset_split_names, load_dataset

    splits = get_dataset_split_names(dataset_id)
    split = "train" if "train" in splits else splits[0]
    rows = list(load_dataset(dataset_id, split=split, streaming=True).take(max_rows))
    if not rows:
        raise ValueError("empty dataset split")
    frame = pd.DataFrame(rows)
    target = _infer_target(frame)
    return frame, target


def _load_openml_frame(dataset_id: str, max_rows: int) -> tuple[pd.DataFrame, str]:
    data_id = OPENML_DATASETS[dataset_id] if dataset_id in OPENML_DATASETS else int(dataset_id)
    d = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    frame = d.frame.copy()
    target = str(d.target_names[0]) if d.target_names else _infer_target(frame)
    if len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=123)
    return frame, target


def _load_sklearn_frame(dataset_id: str) -> tuple[pd.DataFrame, str]:
    loaders = {"breast_cancer": load_breast_cancer, "wine": load_wine, "digits": load_digits}
    d = loaders[dataset_id]()
    frame = pd.DataFrame(d.data, columns=[str(name) for name in d.feature_names] if hasattr(d, "feature_names") else None)
    frame["target"] = d.target
    return frame, "target"


def _infer_target(frame: pd.DataFrame) -> str:
    preferred = ["label", "labels", "target", "class", "activity", "survived", "survival", "churn", "diagnosis", "outcome", "y"]
    lower = {str(col).lower(): str(col) for col in frame.columns}
    for name in preferred:
        if name in lower:
            return lower[name]
    candidates = []
    for col in frame.columns:
        series = frame[col]
        nunique = series.nunique(dropna=True)
        if 2 <= nunique <= min(50, max(2, len(series) // 10)):
            candidates.append((nunique, str(col)))
    if not candidates:
        raise ValueError("no plausible target column found")
    candidates.sort()
    return candidates[0][1]


def _audit_frame(candidate: DatasetCandidate, frame: pd.DataFrame, target: str, workspace: Path) -> DatasetAudit:
    if target not in frame.columns:
        return DatasetAudit(dataset_id=candidate.dataset_id, source=candidate.source, status="rejected", reason="target_missing")
    frame = frame.dropna(axis=0, subset=[target]).copy()
    y = frame[target]
    counts = y.astype(str).value_counts().to_dict()
    warnings = []
    if len(counts) < 2:
        return DatasetAudit(dataset_id=candidate.dataset_id, source=candidate.source, status="rejected", reason="target_has_less_than_two_classes")
    if min(counts.values()) < 8:
        return DatasetAudit(dataset_id=candidate.dataset_id, source=candidate.source, status="rejected", reason="minority_class_too_small")
    features = []
    for col in frame.columns:
        if col == target:
            continue
        series = frame[col]
        if pd.api.types.is_numeric_dtype(series) or series.nunique(dropna=True) <= 30:
            features.append(str(col))
    if len(features) < 2:
        return DatasetAudit(dataset_id=candidate.dataset_id, source=candidate.source, status="rejected", reason="not_enough_model_features")
    if len(frame) < 80:
        warnings.append("small_dataset")
    out_dir = workspace / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    materialized = out_dir / "selected_dataset.csv"
    frame[[*features, target]].to_csv(materialized, index=False)
    return DatasetAudit(
        dataset_id=candidate.dataset_id,
        source=candidate.source,
        status="accepted",
        reason="usable supervised classification table",
        local_path=str(materialized),
        target_column=target,
        feature_columns=features,
        n_rows=int(len(frame)),
        n_features=len(features),
        class_counts={str(key): int(value) for key, value in counts.items()},
        warnings=warnings + candidate.risk_flags,
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")[:120] or "dataset"
