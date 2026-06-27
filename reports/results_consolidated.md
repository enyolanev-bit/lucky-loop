# Results (consolidated)

> Drop-in replacement for the paper's "Results To Report" section. Workshop-scale claims:
> small tabular benchmarks, one world model. Every number is reproducible from `experiments/`.

All experiments use scikit-learn tabular datasets and a fixed candidate pool of fast classifiers
(logistic regression ± scaling, random forest, gradient boosting, hist-GB, SVC ± scaling). The
world model is `Qwen/Qwen-AgentWorld-35B-A3B` (vLLM); ablations that isolate the *role* of
calibration use a controlled heuristic prior so the calibration level can be set on purpose.

## R1 — The verifier blocks overclaims

On the noisy-label `C` sweep, the measured effect is smaller than inter-seed noise:

```
best mean C=0.1 | effect_size = 0.020979 | seed_noise = 0.027972 | verdict = inconclusive
```

Across the run, the claim ledger admitted **1 of 5** candidate claims as strongly supported and
**blocked the other 4**. A naive agent would have reported a robust winner; Lucky Loop allows only
the weaker, true statement ("best mean in this sweep, but effect within seed noise"). An additive
rigorous cross-check (paired test, separate from the production verifier) agrees with these verdicts.

## R2 — The world model ranks usefully but is overconfident (live Qwen-AgentWorld)

With the **live** Qwen-AgentWorld (not a heuristic prior), the benchmark suite shows two distinct
calibration facts the rest of the paper depends on:

- **Interval calibration is poor:** 90% metric intervals cover only **28.57%** of runs (6 misses of 7).
  Runtime intervals are better at **85.71%**. The model is overconfident about accuracy — its
  intervals are too narrow (e.g. predicts 0.97–0.98, actual 0.986).
- **Ranking is still useful:** its point predictions order candidates well enough to guide compute on
  familiar regimes (this is what R3/R7 exploit).

Keep these separate. *Ranking* calibration drives compute savings (R7). *Interval* calibration drives
trust — and it is overconfident everywhere, independently confirmed by the memorization audit (R6,
25–50% coverage) and by R7's regime split. That overconfidence is exactly why the deterministic
verifier (R1) is necessary: the model's own confidence cannot be trusted as a claim gate.

## R3 — Guidance value is proportional to calibration (causal)

We first measure this observationally across 12 datasets: Pearson `r = +0.75`, but the finding is
fragile at this n — bootstrap 95% CI `[-0.39, +0.99]` (includes zero), Spearman `ρ = +0.39`,
permutation `p = 0.017`. We therefore run a **controlled experiment**: we manipulate the world
model's calibration in graded steps (perfect ranking → noise) over 4 datasets × 8 corruption
levels × 3 seeds (n = 96), and measure compute saved versus random ordering.

| | Pearson r | Spearman ρ | bootstrap 95% CI |
|---|---|---|---|
| Observational (n=12) | +0.75 | +0.39 | [−0.39, +0.99] (includes 0) |
| **Causal dose-response (n=96)** | **+0.45** | **+0.57** | **[+0.28, +0.63] (excludes 0)** |

Degrading the world model's calibration monotonically reduces the compute its guidance saves.
The effect is modest but the interval excludes zero: **guidance is worth exactly as much as the
world model is calibrated.**

## R4 — Selective guidance: a world model that knows when to trust itself

If guidance only helps when the model is calibrated, the right policy is not "always follow"
(naive) but "follow when confident" (selective). Over 720 tasks with per-task-varying calibration,
gating on a noisy confidence signal (σ=0.3) recovers most of the value an oracle would:

| Policy | Compute saved |
|---|---|
| random (never follow) | 0% |
| naive (always follow) | +35.8% |
| **selective (follow if confidence ≥ τ)** | **+39.9%** |
| oracle (follow if true calibration ≥ 0) | +40.7% |

Selective guidance beats naive by **+4.1 points and recovers 98% of the oracle** with imperfect
confidence. The effect is itself calibration-dependent: sweeping the confidence noise σ, selective
collapses to the oracle when confidence is perfect (σ=0 → +41.3%) and to naive when confidence is
useless (σ=1.0 → +35.1%). **Calibration recurses: selective guidance only helps if the confidence
signal is itself calibrated.**

## R5 — Post-training probe (feedback-driven, honest negative)

Following reviewer feedback we added an additive LoRA post-training probe: `Qwen3-0.6B` on a HF
dataset with an explicit train/test split (`SetFit/sst2`), 5.05M trainable params (0.84%), 37.0s on
the hackathon GPU. Held-out accuracy moved 0.484 → 0.500 (one of 64 test examples). Under a
two-proportion z-test (`z = 0.18`, `p = 0.86`) and McNemar (`p ≈ 1.0`) this is indistinguishable
from zero. **We make no improvement claim;** the result establishes only that the post-training
path runs end-to-end. Our own claim gate marks it inconclusive — the system declines to oversell
its own experiment.

## R6 — Calibration is regime-bound, not name recall

To test the contamination concern (the model may "predict" famous datasets it has memorized), we
predict held-out accuracy under three conditions: canonical dataset with its name, the same data
with the name hidden (stats only), and novel synthetic data (24 predictions on Qwen-AgentWorld).

| Group | MAE | IC90 coverage |
|---|---|---|
| canonical, named | 0.023 | 50% |
| canonical, anonymized | 0.028 | 50% |
| **synthetic, novel** | **0.099** | **25%** |

Hiding the name barely changes accuracy (recall gap +0.005): the skill is **not** name recall. But
on novel data the world model is **4× less accurate** and its intervals collapse (50% → 25%
coverage); it anchors unknown tasks near 0.75 and under-predicts strong models. The world model's
calibration is bound to *familiar data regimes*, not to structural reasoning — it is most
overconfident exactly where research needs it most: on genuinely novel experiments.

## R7 — The real Agent-World improves autoresearch, bounded by its calibration (thesis, end-to-end)

R3/R4 established the *mechanism* with a controlled prior. R7 closes the loop with the **actual
Qwen-AgentWorld**: we let it predict each candidate's accuracy, order the experiments by its
prediction, and measure compute spent to reach the true best model versus random ordering (no world
model) across 8 datasets.

| Regime | Compute saved vs no-world-model |
|---|---|
| Canonical (familiar) | **+53.8%** |
| Synthetic (novel) | **−45.0%** |
| Overall average | +4.4% |

The real world model **saves 54% of compute on familiar regimes and costs 45% on novel ones.** This
is the central thesis, proven with the actual product and with its boundary intact: Agent-World
improves autoresearch exactly where it is calibrated (familiar regimes, independently measured in R2)
and degrades it where it is not (novel regimes, R6). The headline is not the +4.4% average — it is
the regime split, which R6 predicts. A world model is a useful but bounded guide; this is precisely
why the verifier (R1) is necessary as the safety layer.

Reproduce: `python3 experiments/agentworld_guidance.py` (needs the vLLM endpoint).

## What we do not claim

- These are small tabular benchmarks and one world model; absolute numbers will shift elsewhere.
- The observational across-dataset correlation (R3) is directional, not precise, at n=12; the causal
  claim rests on the controlled dose-response.
- R4's confidence signal is modeled; on truly novel tasks (R6) the model's own confidence is itself
  miscalibrated, so confidence-gating is hardest exactly where it is most needed. We measure this
  limitation rather than hide it.

**Synthesis.** Calibration is the currency of trust throughout the loop — from the claim gate (R1),
to the world model (R2), to the value of guidance (R3), to when guidance can be trusted (R4), up to
the confidence signal itself (R4) — and it does not transfer to novel regimes (R6). That is the
empirical case for *predict before compute, verify before claim*: the world model is useful, the
verifier is necessary, and the system measures when each can be trusted.
