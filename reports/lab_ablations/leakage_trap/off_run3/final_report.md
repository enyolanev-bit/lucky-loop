# Auto-Research Report

## Abstract

This report summarizes an autonomous ML research loop: literature review, dataset selection, generated protocol, real execution, claim verification, and follow-up decision.

## Research Question

Does feature scaling improve accuracy?

## Literature Review Summary

- Related work: `literature/related_work.md`
- Sources: `literature/sources.json`

## Hypotheses

### H1_leakage_invalidates_claim

- Candidate claim: A high score from leaky preprocessing is protocol-invalid and must not be reported as model generalization.
- Falsification condition: Proper and intentionally leaky preprocessing produce comparable performance.
- Minimum evidence: Controlled proper-vs-leaky protocol with a protocol warning attached to the leaky branch.

## Methods

### proper_vs_leaky_preprocessing

- Goal: Compare a proper train-fold preprocessing pipeline against an intentionally leaky preprocessing probe.
- Conditions: proper_pipeline, leaky_preprocessing
- Primary metric: `balanced_accuracy`
- Risks: data_leakage

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

- **not supported `protocol_invalid_leakage`**: The high score from the leaky branch is a valid generalization claim.
  - Diagnostic: The experiment included a leakage warning, so high scores from that branch cannot support generalization.
  - Next action: Remove leakage, rebuild the preprocessing pipeline inside train folds, and rerun.
  - Allowed rewrite: The leaky branch is an invalid protocol probe, not a valid model result.
- **not supported `protocol_invalid_leakage`**: The high score from the leaky branch is a valid generalization claim.
  - Diagnostic: The experiment included a leakage warning, so high scores from that branch cannot support generalization.
  - Next action: Remove leakage, rebuild the preprocessing pipeline inside train folds, and rerun.
  - Allowed rewrite: The leaky branch is an invalid protocol probe, not a valid model result.

## Inconclusive / Observation-Only Findings

- **observation_only**: A high score from leaky preprocessing is protocol-invalid and must not be reported as model generalization.. Reason: The lab recorded a real ML result, but no stronger verifier rule applied.
- **observation_only**: A high score from leaky preprocessing is protocol-invalid and must not be reported as model generalization.. Reason: The lab recorded a real ML result, but no stronger verifier rule applied.
- **observation_only**: A high score from leaky preprocessing is protocol-invalid and must not be reported as model generalization.. Reason: The lab recorded a real ML result, but no stronger verifier rule applied.

## Claim Ledger

- Full ledger: `claim_ledger.json`

## Limitations

- Literature review uses arXiv metadata, abstracts, and curated notes; this run does not parse full PDFs.
- Qwen-AgentWorld predictions are not evidence. They are pre-execution forecasts of lab observations.
- Generated experiment code is executed only after static validation and sandboxed local execution.

## Reproducibility Commands

- See `reproducibility.md`.
