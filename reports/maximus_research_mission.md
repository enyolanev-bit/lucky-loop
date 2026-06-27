# Maximus research mission — deep positioning for Lucky Loop (fire POST-submission)

> Status: DRAFT, queued. Do NOT dispatch during hackathon crunch. Fire after the Sunday submission.
> Lane: non-code, web research (Maximus). No repo writes. Output = one markdown report.

## Objective

Turn the quick landscape survey into a rigorous, citable positioning that proves the Lucky Loop
thesis is a genuine open lane, and surface the exact numbers needed to compare against the strongest
baselines later.

## Thesis to position against

"A world model (Qwen-AgentWorld) improves autoresearch in proportion to its calibration, and we
measure that boundary; claims are gated by a deterministic verifier, not the model's self-judgment."

## Questions to answer (specific, evidence-required)

1. **Confirm the white space.** Does any public system (paper or repo, 2024-2026) combine BOTH
   (a) a learned/predictive world model that selects experiments before compute AND (b) a calibrated
   claim gate based on measured effect-vs-noise (not LLM self-review)? Cite each near-miss and say
   exactly which half it has.
2. **Claim-by-claim diff** vs the 4 strongest: SakanaAI/AI-Scientist-v2, microsoft/RD-Agent,
   Just-Curieous/Curie, kyle8581/WMA-Agents. For each: how it selects the next experiment, how (if at
   all) it predicts outcomes, how it verifies claims. One row each.
3. **Baseline numbers** (for a future head-to-head): AI-Scientist-v2 cost/paper and acceptance
   evidence; RD-Agent's MLE-bench result (medal rate / rank); RE-Bench human-vs-agent score under
   equal budget. Cite source + number, mark anything you cannot verify.
4. **World-model calibration prior art.** Has anyone measured that an LLM world model is overconfident
   / regime-bound the way we found (R6: 25-50% interval coverage, 4× error on novel)? Find the closest
   calibration/overconfidence-of-LLM-predictions references.
5. **Best benchmark to prove the thesis at scale.** Between MLE-bench, RE-Bench, MLAgentBench: which is
   the cleanest harness to measure "compute saved + score delta with vs without world-model guidance"?
   Justify with what each measures and its baselines.

## Deliverable

One markdown report: (1) white-space verdict (yes/no + evidence), (2) claim-by-claim diff table,
(3) baseline numbers table with sources, (4) calibration prior-art list, (5) recommended benchmark +
why. Every claim cited with a real URL. Mark unverified items explicitly. No fabrication.

## What we do after Maximus's research

- **Tighten the paper's Related Work + Contributions** with the confirmed white-space statement and the
  claim-by-claim diff.
- **Pick ONE baseline + benchmark** and run Lucky Loop's world-model guidance on it for a head-to-head
  number ("X% compute saved / +Y score vs baseline"). This is the move that turns the hackathon project
  into an arXiv-worthy result.
- **ENYOLAB/HESTIA tie-in:** the predict-before-compute + verify-before-claim pattern, calibration-
  bounded, is a research thread that aligns with HESTIA (the model of the house predicting bench
  outcomes before spending compute). Decide whether to spin it into an ENYOLAB direction.
