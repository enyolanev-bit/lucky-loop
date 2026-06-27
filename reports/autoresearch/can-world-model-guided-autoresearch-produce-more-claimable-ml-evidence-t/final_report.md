# Agent-Operated Autoresearch Report

Question: Can world-model-guided autoresearch produce more claimable ML evidence than classic autoresearch?

## Method

A coding agent operates inside the repository. Lucky Loop supplies the research protocol: literature context, safe experiment catalog, Qwen-AgentWorld predictions before compute, real sklearn execution, prediction-vs-reality comparison, deterministic verification, and claim ledger reporting.

## Literature-Derived Gaps

- Autonomous research agents can automate idea-to-report workflows, but final reports still need explicit claim calibration. Sources: [arxiv_2407_18367_2024], [arxiv_2408_06292_2024], [arxiv_2501_04227_2025], [arxiv_2506_01372_2025], [arxiv_2603_24647_2026]. Implication: Lucky Loop must separate observations from claims and force report claims through a verifier.
- ML-agent benchmarks and search policies emphasize final performance more than prospective prediction before compute. Sources: [arxiv_2410_07095_2024], [arxiv_2507_02554_2025], [arxiv_2509_02722_2025], [arxiv_2605_08956_2026], [arxiv_2606_24597_2026]. Implication: Lucky Loop should log state, candidate action, predicted observation, real observation, and comparison.
- Single-run leaderboard wins can overstate robustness when top models are close across seeds. Sources: [arxiv_2407_18367_2024], [arxiv_2410_07095_2024], [arxiv_2506_01372_2025], [arxiv_2507_02554_2025], [arxiv_2603_24647_2026]. Implication: A robust best-model claim requires matched multi-seed top-model verification.
- End-to-end autoresearch needs literature context, execution, analysis, and report generation tied to auditable evidence. Sources: [arxiv_2408_06292_2024], [arxiv_2501_04227_2025], [arxiv_2506_01372_2025], [arxiv_2605_08956_2026]. Implication: The workspace must preserve sources, experiment plans, commands, traces, claim ledgers, and final report links.

## Evidence Summary

- Related work: `literature/related_work.md`
- Sources: `literature/sources.json` and `literature/sources.bib`
- Experiment plan: `experiment_plan.json`
- Evidence manifest: `evidence_manifest.json`
- Backend ablation: `reports/ablations/world_model_ablation.md`
- Counterfactual evaluation: `reports/counterfactuals/counterfactual_evaluation.md`

## Ablation Snapshot

| Task | Policy | Best single-run | Best verified mean | Best claimable | Unsupported best-model claims | Claims blocked | Qwen predictions |
|---|---|---:|---:|---:|---:|---:|---:|
| breast_cancer_accuracy | classic_autoresearch | 0.9860 |  |  | 1 | 0 | 0 |
| breast_cancer_accuracy | classic_verified | 0.9860 | 0.9706 |  | 0 | 1 | 0 |
| breast_cancer_accuracy | lucky_loop_full | 0.9860 | 0.9706 |  | 0 | 1 | 84 |
| wine_accuracy | classic_autoresearch | 1.0000 |  |  | 1 | 0 | 0 |
| wine_accuracy | classic_verified | 1.0000 | 0.9956 |  | 0 | 1 | 0 |
| wine_accuracy | lucky_loop_full | 1.0000 | 0.9956 |  | 0 | 2 | 83 |
| digits_accuracy | classic_autoresearch | 0.9778 |  |  | 1 | 0 | 0 |
| digits_accuracy | classic_verified | 0.9778 | 0.9764 |  | 0 | 1 | 0 |
| digits_accuracy | lucky_loop_full | 0.9800 | 0.9764 |  | 0 | 1 | 84 |

## Counterfactual Result

- Cases: 3
- Lucky wins: 3
- Qwen choice usefulness: 100.00%

## Claim Discipline

Classic autoresearch can find good single-run scores, but those are not robust claims. Lucky Loop full adds auditable pre-compute predictions and keeps unsupported best-model claims at zero in the generated ablation artifacts.

Strict result: on these tasks, the verifier did not allow a robust best-model claim. That is the point of the evidence gate: Lucky Loop surfaces verified means and blocked overclaims instead of converting close single-run wins into publication claims.

## Source Mapping

| Gap | Sources | Metric | Experiment |
|---|---|---|---|
| gap_claim_calibration | [arxiv_2407_18367_2024], [arxiv_2408_06292_2024], [arxiv_2501_04227_2025], [arxiv_2506_01372_2025], [arxiv_2603_24647_2026] | `unsupported_best_model_claims` | Compare classic_autoresearch against lucky_loop_full claim ledger outcomes. |
| gap_prediction_before_compute | [arxiv_2410_07095_2024], [arxiv_2507_02554_2025], [arxiv_2509_02722_2025], [arxiv_2605_08956_2026], [arxiv_2606_24597_2026] | `prediction_interval_coverage` | Measure Qwen-AgentWorld prediction-vs-reality for every Lucky Loop candidate decision. |
| gap_single_run_overclaim | [arxiv_2407_18367_2024], [arxiv_2410_07095_2024], [arxiv_2506_01372_2025], [arxiv_2507_02554_2025], [arxiv_2603_24647_2026] | `best_claimable_score` | Detect top observed models, rerun matched seeds, and compare effect size against seed noise. |
| gap_end_to_end_auditability | [arxiv_2408_06292_2024], [arxiv_2501_04227_2025], [arxiv_2506_01372_2025], [arxiv_2605_08956_2026] | `evidence_manifest_completeness` | Generate reports/autoresearch/<slug>/ with sources.json, research_context.json, experiment_plan.json, and evidence_manifest.json. |

## Manifest

Workspace: `reports/autoresearch/can-world-model-guided-autoresearch-produce-more-claimable-ml-evidence-t`
