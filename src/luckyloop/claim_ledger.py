from __future__ import annotations

import json
from pathlib import Path

from .schemas import ClaimLedgerEntry, ExperimentTrace


def entries_from_verification(run_id: str, verification) -> list[ClaimLedgerEntry]:
    if verification is None:
        return []

    metrics = {
        "effect_size": verification.effect_size,
        "seed_noise": verification.seed_noise,
        "effect_to_noise_ratio": verification.effect_to_noise_ratio,
        "min_seed_count": verification.min_seed_count,
        "low_n_warning": verification.low_n_warning,
    }

    if verification.trustworthy:
        claim = verification.allowed_claim or "; ".join(verification.supported_claims)
        return [
            ClaimLedgerEntry(
                claim_id=f"{run_id}_claim_001",
                claim=claim,
                status=verification.status,
                evidence_run_ids=[run_id],
                allowed_rewrite=claim,
                metrics=metrics,
            )
        ]

    blocked = verification.blocked_claim or "A robust sweep winner is not supported by the current evidence."
    allowed = verification.allowed_claim or "; ".join(verification.inconclusive_findings)
    return [
        ClaimLedgerEntry(
            claim_id=f"{run_id}_claim_001",
            claim=blocked,
            status="blocked",
            evidence_run_ids=[run_id],
            blocked_reason=verification.rationale,
            allowed_rewrite=allowed,
            metrics=metrics,
        )
    ]


def entries_from_trace(trace: ExperimentTrace) -> list[ClaimLedgerEntry]:
    return entries_from_verification(trace.run_id, trace.verification)


def build_claim_ledger(traces: list[ExperimentTrace]) -> list[ClaimLedgerEntry]:
    entries: list[ClaimLedgerEntry] = []
    for trace in traces:
        entries.extend(entries_from_trace(trace))
    return entries


def write_claim_ledger(traces: list[ExperimentTrace], path: Path) -> list[ClaimLedgerEntry]:
    entries = build_claim_ledger(traces)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "entries": [entry.model_dump() for entry in entries],
        "summary": {
            "total": len(entries),
            "blocked": sum(1 for e in entries if e.status == "blocked"),
            "weakly_supported": sum(1 for e in entries if e.status == "weakly_supported"),
            "supported": sum(1 for e in entries if e.status == "supported"),
            "strongly_supported": sum(1 for e in entries if e.status == "strongly_supported"),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return entries
