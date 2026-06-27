# Agent-Operated Autoresearch Report

Question: Can world-model-guided autoresearch produce more claimable ML evidence than classic autoresearch?

## Method

A coding agent operates inside the repository. Lucky Loop supplies the research protocol: literature context, safe experiment catalog, Qwen-AgentWorld predictions before compute, real sklearn execution, prediction-vs-reality comparison, deterministic verification, and claim ledger reporting.

## Literature-Derived Gaps

- Autonomous research agents are often evaluated by final score or report plausibility rather than claim calibration.
- Most ML-agent benchmarks do not measure whether the agent predicted experiment outcomes before spending compute.
- A strong single-run result can become an unsupported claim when seed variance or matched multi-seed checks are missing.
- World-model predictions are usually not logged as auditable prediction-vs-reality evidence in research-agent loops.

## Evidence Summary

- Related work: `literature/related_work.md`
- Experiment plan: `experiment_plan.json`
- Evidence manifest: `evidence_manifest.json`
- Backend ablation: `reports/ablations/world_model_ablation.md`

## Ablation Snapshot

| Task | Policy | Best single-run | Unsupported best-model claims | Claims blocked | Qwen predictions |
|---|---|---:|---:|---:|---:|
| breast_cancer_accuracy | classic_autoresearch | 0.9860 | 1 | 0 | 0 |
| breast_cancer_accuracy | classic_verified | 0.9860 | 0 | 1 | 0 |
| breast_cancer_accuracy | lucky_loop_full | 0.9860 | 0 | 1 | 84 |
| wine_accuracy | classic_autoresearch | 1.0000 | 1 | 0 | 0 |
| wine_accuracy | classic_verified | 1.0000 | 0 | 1 | 0 |
| wine_accuracy | lucky_loop_full | 1.0000 | 0 | 2 | 83 |
| digits_accuracy | classic_autoresearch | 0.9778 | 1 | 0 | 0 |
| digits_accuracy | classic_verified | 0.9778 | 0 | 1 | 0 |
| digits_accuracy | lucky_loop_full | 0.9800 | 0 | 1 | 84 |

## Claim Discipline

Classic autoresearch can find good single-run scores, but those are not robust claims. Lucky Loop full adds auditable pre-compute predictions and keeps unsupported best-model claims at zero in the generated ablation artifacts.

## Manifest

Workspace: `reports/autoresearch/can-world-model-guided-autoresearch-produce-more-claimable-ml-evidence-t`
