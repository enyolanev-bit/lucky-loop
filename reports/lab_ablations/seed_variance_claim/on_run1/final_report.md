# Auto-Research Report

## Abstract

This report summarizes an autonomous ML research loop: literature review, dataset selection, generated protocol, real execution, claim verification, and follow-up decision.

## Research Question

Does feature scaling improve accuracy?

## Literature Review Summary

- Related work: `literature/related_work.md`
- Sources: `literature/sources.json`

## Hypotheses

### H1_single_run_winner_not_robust

- Candidate claim: A single-run winner is not a robust best-model claim when the effect is smaller than seed noise.
- Falsification condition: Repeated seeds show an effect-to-noise ratio above the support threshold.
- Minimum evidence: Multi-seed comparison of candidate models with effect/noise reported.

## Methods

### single_run_vs_repeated_seeds

- Goal: Compare single-run model winner against repeated-seed evidence.
- Conditions: single_run_winner, repeated_seed_comparison
- Primary metric: `balanced_accuracy`
- Risks: seed_variance, single_split_overclaim

## Results And Verification

- Real run outputs: `runs/`
- Analyses: `analyses/`
- Full claim ledger: `claim_ledger.json`

## Lucky Loop Audit Trail

- Prediction log: `predictions/world_model_predictions.jsonl`
- Lab notebook: `notebook.jsonl`
- Generated code validation: `generated/static_validation.json`
- Next decision: `next_decision.json`

## Supported Claims

- None.

## Claims Not Supported By This Run

- **not supported `wrong_effect_direction`**: A single-run winner is not a robust best-model claim when the effect is smaller than seed noise.
  - Diagnostic: The measured effect was 0.0000, so the tested claim moved in the wrong direction.
  - Next action: Invert or revise the hypothesis, then run a confirmatory replication.
  - Allowed rewrite: Report that this experiment did not support the proposed direction.
- **not supported `protocol_warning`**: A single-run winner is not a robust best-model claim when the effect is smaller than seed noise.
  - Diagnostic: The generated experiment emitted protocol warnings: single-run winner is observation-only until repeated-seed effect exceeds seed noise
  - Next action: Inspect the warning, revise the protocol, and rerun before making a stronger claim.
  - Allowed rewrite: Report the result as warning-bounded evidence.
- **not supported `protocol_warning`**: A single-run winner is not a robust best-model claim when the effect is smaller than seed noise.
  - Diagnostic: The generated experiment emitted protocol warnings: single-run winner is observation-only until repeated-seed effect exceeds seed noise
  - Next action: Inspect the warning, revise the protocol, and rerun before making a stronger claim.
  - Allowed rewrite: Report the result as warning-bounded evidence.
- **not supported `protocol_warning`**: A single-run winner is not a robust best-model claim when the effect is smaller than seed noise.
  - Diagnostic: The generated experiment emitted protocol warnings: single-run winner is observation-only until repeated-seed effect exceeds seed noise
  - Next action: Inspect the warning, revise the protocol, and rerun before making a stronger claim.
  - Allowed rewrite: Report the result as warning-bounded evidence.

## Inconclusive / Observation-Only Findings

- None.

## Claim Ledger

- Full ledger: `claim_ledger.json`

## Limitations

- Literature review uses arXiv metadata, abstracts, and curated notes; this run does not parse full PDFs.
- Qwen-AgentWorld predictions are not evidence. They are pre-execution forecasts of lab observations.
- Generated experiment code is executed only after static validation and sandboxed local execution.

## Reproducibility Commands

- See `reproducibility.md`.
