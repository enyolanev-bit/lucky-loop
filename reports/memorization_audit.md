# Memorization audit — the mentor's cut-off warning, quantified

**World model:** `Qwen/Qwen-AgentWorld-35B-A3B` (vLLM, Team Pegasus / MI300X). 24 predictions.

## Question

Is the world-model's calibration on famous sklearn datasets (`breast_cancer`, `wine`, `digits`, `iris`) genuine prediction, or recall of memorized benchmark numbers (pre-cutoff contamination)?

## Design (confound-controlled)

Predict held-out accuracy under three conditions:

| Group | What the model sees | Memorizable? |
|---|---|---|
| `canonical_named` | famous dataset **with its name** | yes (name → recall) |
| `canonical_anon` | **same data, name hidden** (only `n, d, k` stats) | no name to recall |
| `synthetic_anon` | `make_classification` (random seeds), stats only | not at all |

If `error(named) << error(anon) ≈ error(synthetic)` → recall via name. If error tracks *familiarity of the data regime* instead → regime-bound calibration.

## Result

| Group | MAE | IC90 coverage | n |
|---|---|---|---|
| `canonical_named` | **0.023** | 50% | 8 |
| `canonical_anon` | **0.028** | 50% | 8 |
| `synthetic_anon` | **0.099** | 25% | 8 |

**Recall gap (MAE_anon − MAE_named) = +0.005** → negligible.

## Finding (honest)

1. **It is NOT name recall.** Hiding the dataset name barely changes accuracy (gap +0.005). The world-model does not rely on reading "breast_cancer" to predict well.

2. **Calibration is regime-bound, not structural.** On novel synthetic data the world-model is **4× less accurate** (MAE 0.099 vs 0.023) and its 90% intervals collapse from 50% to **25% coverage**. It anchors unknown tabular tasks at ~0.75 and systematically under-predicts (actual 0.84–0.91, especially random forest), with intervals too narrow to cover the truth.

3. **This is the cut-off warning made precise:** the agent is most overconfident exactly where research needs it most — on genuinely novel experiments. Its apparent skill on canonical benchmarks reflects familiarity with the *data regime*, and does not transfer to new distributions.

## Why this strengthens Lucky Loop

This is the empirical justification for the whole architecture:

- It is **why the deterministic verifier is necessary** — the world-model cannot be trusted on novel tasks, so claims must be gated on real measured effect-vs-noise, not on the model's confidence.
- It is **why selective guidance matters** — and an honest limitation: on novel tasks the model's confidence signal is itself miscalibrated (25% coverage), so confidence-gating is hardest exactly where it is most needed.
- It bounds the contribution honestly: Lucky Loop measures *when* the world-model is reliable rather than assuming it always is.

Reproduce: `python3 experiments/memorization_audit.py` (needs the vLLM endpoint).
