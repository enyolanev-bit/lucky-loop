# Backend Pitch Summary

Lucky Loop is a world-model-guided autoresearch backend. The agent proposes catalog actions, Qwen-AgentWorld predicts outcomes before compute, real sklearn experiments run, prediction-vs-reality is logged, and a deterministic verifier gates claims.

## What the ablation proves

- Classic autoresearch can find good single-run scores, but has no pre-compute prediction trace.
- Classic verified shows the trust gate alone can reduce overclaims, but still lacks prospective simulation.
- Lucky Loop full adds the missing world-model layer: every candidate is predicted before compute and every miss is logged.
- Claimable-score metrics are strict: inconclusive verifier outcomes block robust claims rather than becoming claimable wins.

## Artifacts

- `reports/ablations/world_model_ablation.md`
- `reports/ablations/classic_vs_lucky_loop.md`
- `reports/ablations/world_model_ablation.json`
- `reports/counterfactuals/counterfactual_evaluation.md`
- `runs/ablations/*/*/run_*.json`
- `runs/counterfactuals/*/*/*.json`
