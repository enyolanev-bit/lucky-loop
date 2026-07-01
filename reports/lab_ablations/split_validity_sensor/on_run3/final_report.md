# Auto-Research Report

## Abstract

This report summarizes an autonomous ML research loop: literature review, dataset selection, generated protocol, real execution, claim verification, and follow-up decision.

## Research Question

Does feature scaling improve accuracy?

## Literature Review Summary

- Related work: `literature/related_work.md`
- Sources: `literature/sources.json`

## Hypotheses

### H1_split_overstates_generalization

- Candidate claim: Random train/test splits overstate generalization on sequential sensor data compared with blocked splits.
- Falsification condition: Random and blocked split performance are similar within seed noise.
- Minimum evidence: Repeated-seed comparison where random split exceeds blocked split by more than seed noise.

## Methods

### random_vs_blocked_split

- Goal: Compare random train/test splitting against blocked sequential splitting on real sensor-style data.
- Conditions: random_split, blocked_split
- Primary metric: `balanced_accuracy`
- Risks: temporal_correlation, single_split_overclaim

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

- **supported**: Random train/test splits overstate generalization on sequential sensor data compared with blocked splits. Evidence: experiment_002. Reason: Random split exceeded blocked split by more than the estimated seed noise.
- **supported**: Random train/test splits overstate generalization on sequential sensor data compared with blocked splits. Evidence: experiment_004. Reason: Random split exceeded blocked split by more than the estimated seed noise.

## Claims Not Supported By This Run

- **not supported `split_effect_not_claimable`**: Random train/test splits overstate generalization on sequential sensor data compared with blocked splits.
  - Diagnostic: The blocked split did not underperform random split in a claimable way.
  - Next action: Run more seeds or use a stronger grouped/temporal holdout before claiming split overstatement.
  - Allowed rewrite: Observed split differences are not enough for a robust overstatement claim.
- **not supported `split_effect_not_claimable`**: Random train/test splits overstate generalization on sequential sensor data compared with blocked splits.
  - Diagnostic: The blocked split did not underperform random split in a claimable way.
  - Next action: Run more seeds or use a stronger grouped/temporal holdout before claiming split overstatement.
  - Allowed rewrite: Observed split differences are not enough for a robust overstatement claim.
- **not supported `split_effect_not_claimable`**: Random train/test splits overstate generalization on sequential sensor data compared with blocked splits.
  - Diagnostic: The blocked split did not underperform random split in a claimable way.
  - Next action: Run more seeds or use a stronger grouped/temporal holdout before claiming split overstatement.
  - Allowed rewrite: Observed split differences are not enough for a robust overstatement claim.

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
