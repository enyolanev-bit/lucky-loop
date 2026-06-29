# Auto-Research Report

## Abstract

This report summarizes an autonomous ML research loop: literature review, dataset selection, generated protocol, real execution, claim verification, and follow-up decision.

## Research Question

Does feature scaling improve logistic regression accuracy on breast_cancer?

## Domain Related Work

- Domain related work: `literature/domain_related_work.md`
- Domain sources: `literature/domain_sources.json`
- Domain gaps: `literature/domain_gaps.json`

## Method Safeguards

- Method related work: `literature/method_related_work.md`
- Method sources: `literature/method_sources.json`

## Research Agenda

- Agenda: `agenda/research_agenda.json`

## Hypotheses

### hypothesis_1

- Candidate claim: Feature scaling improves logistic regression accuracy on breast cancer classification.
- Falsification condition: No significant difference in accuracy with and without feature scaling across multiple seeds.
- Minimum evidence: Statistically significant improvement in accuracy across multiple data splits and seeds.

## Methods

### generated_ml_research_protocol

- Goal: Feature scaling improves logistic regression accuracy on breast cancer classification.
- Conditions: gradientboostingclassifier, logisticregression, randomforestclassifier, svc
- Primary metric: `balanced_accuracy`
- Risks: generated_protocol, dataset_selection_bias

## Dataset Selection

- Selection rationale: `datasets/selection_rationale.json`

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

- **supported**: Feature scaling improves logistic regression accuracy on breast cancer classification. Evidence: experiment_008. Reason: The measured effect exceeded seed noise by the support threshold.

## Claims Not Supported By This Run

- None.

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
