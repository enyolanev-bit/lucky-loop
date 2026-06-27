# Paper reframe — center everything on the thesis

> The project is research-first. The product is the **world model**. The single claim the paper must
> make and defend: **a world model (Qwen-AgentWorld) improves autoresearch — bounded by its
> calibration.** Everything else is evidence for, against, or around that claim.

## Drop-in abstract (replaces the current one)

> Autonomous research agents decide which experiments to run, often blindly. We ask whether a
> language **world model** — Qwen-AgentWorld — improves that decision by predicting experiment
> outcomes before compute is spent. We find that it does, **in proportion to its calibration, and
> only within familiar data regimes.** Using the real Agent-World to order candidate ML experiments
> saves **54% of compute on familiar tabular benchmarks** but **costs 45% on novel synthetic tasks**,
> where we independently show its predictions are miscalibrated (interval coverage collapses from 50%
> to 25%, prediction error grows 4×). A controlled dose-response confirms the mechanism causally:
> degrading a world model's calibration monotonically reduces the compute its guidance saves
> (r = 0.45, 95% CI [0.28, 0.63]). Because the world model is a useful but bounded guide, we pair it
> with a deterministic verifier and claim ledger that gate findings on measured effect-versus-noise,
> blocking overclaims (4 of 5 candidate claims) while admitting real effects (a LoRA post-training run
> that genuinely moves a model 48% → 93%). The contribution is not a world model that is always right;
> it is a measured account of **when a world model helps autoresearch and when it must not be
> trusted.**

## Central claim, one line (for the title/intro/pitch)

**"A world model improves autoresearch to the exact extent that it is calibrated — and we measure that
extent."**

## How each result maps to the thesis (use this as the Results spine)

| Result | Role in the thesis |
|---|---|
| R2 | The world model IS a predictor — 80% interval coverage on familiar regimes. |
| R6 | ...but its calibration is regime-bound: 4× worse + coverage 50%→25% on novel; not name recall. |
| R3 | Mechanism (causal): calibration → compute saved (dose-response, r=0.45, CI excludes 0). |
| R4 | Corollary: gating on the world model's confidence saves more — *if* confidence is calibrated. |
| **R7** | **Thesis, end-to-end with the REAL Agent-World: +54% compute saved on familiar, −45% on novel.** |
| R1 | Why it is safe to use a bounded guide: the verifier blocks overclaims regardless of the model. |
| R5 | The loop reasons about real experiments (post-training works, 48→93%) under the same gate. |

## Narrative arc for intro + conclusion

1. Autoresearch agents pick experiments blindly; a world model could guide them.
2. Does the real Agent-World actually help? Yes — but only where it is calibrated (R7, +54% familiar).
3. Where it is not calibrated, it actively hurts (−45% novel), and we can predict where that is (R6).
4. So the value of a world model for autoresearch is not "always on" — it is **calibration-bounded**,
   and that boundary is measurable (R2/R3/R6).
5. Therefore the safe architecture pairs the world model with a verifier (R1) that does not trust it.

## What to fix in paper.md

- Retitle around the thesis (e.g. "When Does a World Model Help Autonomous Research? A
  Calibration-Bounded Account of Qwen-AgentWorld").
- Lead Section 5 (Experiments) with R7, then explain it with R2/R6 (why) and R3/R4 (mechanism).
- Move claim-calibration (R1) to "safety layer," subordinate to the world-model thesis, not co-equal.
- State the honest limitation up front: the win is regime-bounded; the +4.4% average is the wrong
  summary, the regime split is the finding.
