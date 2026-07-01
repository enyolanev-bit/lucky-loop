# When Does a Language World Model Help Autonomous Research?

### A controlled negative result on tabular ML — and a pipeline that caught its own false positives

**Team Pegasus** — Nevil Enyola, Hicham, and collaborators
*Paris Research Hackathon (TUM.ai × Iterate), Track 3 — Autonomous AI Research Agent · June 2026*

---

## Abstract

We built **Lucky Loop**, an autonomous research agent that runs the full loop — literature review, dataset selection, code generation, real execution, and claim verification — and adds two ideas on top of the standard agent-scientist recipe: (1) **predict-before-compute**, a pretrained *language world model* (Qwen-AgentWorld-35B-A3B, served on an AMD MI300X) forecasts an experiment's outcome before it is run; and (2) **verify-before-claim**, a deterministic effect-vs-noise gate plus a machine-enforced *claim ledger* that admits a finding only when the measured effect exceeds estimated seed noise.

The system works end-to-end: in a real run it autonomously selected a dataset, reviewed arXiv literature, generated and executed code across multiple model families, and the verifier confirmed a single claim (effect/noise ratio 10.7). We then asked the scientific question we actually cared about: **does the language world model improve the research loop?** In two controlled ablations (a decision ablation over three studies, and a 16-candidate × 6-dataset experiment-ordering ablation), the answer on tabular ML is **no measurable benefit over trivial heuristics**. An apparent +98.7% compute saving evaporated under verification: it did not reproduce across the model's stochastic predictions, and a one-line "cheapest-first" baseline beat the world model on 5 of 6 datasets. We report this honestly. The most useful artifact of the project is that **our own verify-before-claim pipeline caught three false positives that were in our favor** — the clearest possible demonstration of the thesis it was built to test.

---

## 1. Introduction

Autonomous research agents (Sakana AI Scientist, RD-Agent, Agent Laboratory) can now propose ideas, write code, and draft papers. The recognized bottleneck is **not idea generation but verification**: "AI Scientists Fail Without Strong Implementation Capability" [2506.01372]. Two responses exist in isolation. One line *predicts before computing* — surrogate models (Bayesian optimization; LLM surrogates such as LLAMBO [2402.03921]) or outcome forecasters (Wen et al. [2506.00794], FOREAGENT [2601.05930]) — to spend compute well. A second line *verifies after computing* — deterministic, ground-truth gates (POPPER [2502.09858]; offline benchmarks like MLE-bench [2410.07095], CORE-Bench [2409.11363]).

**Lucky Loop fuses the two inside one live loop**: a language world model forecasts the outcome, but the forecast is never the decision — it is gated by a verifier that executes the real code and admits a claim only on effect-vs-noise grounds. No public system combines a pre-compute language-world-model forecast with an in-loop effect-vs-noise claim gate.

Building the system is one contribution. The other is **measuring whether the world model actually earns its place** — and reporting the result whichever way it falls.

## 2. Related Work

**Autonomous research agents (what we extend).** Verification here is post-hoc: LLM-as-reviewer of prose (Sakana v1 [2408.06292] / v2 [2504.08066], Agent Laboratory [2501.04227]); execution feedback without a claim gate (RD-Agent [2505.14738], Curie [2502.16069]); offline ground-truth benchmarks that score *after the loop* (PaperBench [2504.01848], RE-Bench [2411.15114]). The closest, **POPPER** [2502.09858], adds deterministic Type-I error control but has no pre-compute prediction step.

**Experiment selection / LLM surrogates (predict-before-compute).** Classic surrogate-then-select: BO [1206.2944], Hyperband [1603.06560], BOHB [1807.01774]; cost-aware variants FABOLAS [1605.07079], CAGES [2405.07760]. LLM-as-surrogate: LLAMBO [2402.03921], OPRO [2309.03409]. We replace the task-fitted surrogate with a *zero-shot pretrained language world model*, removing cold-start but losing calibrated uncertainty — hence the need for a verifier.

**Predicting experiment outcomes (direct prior art).** Wen et al. [2506.00794] predict which idea wins; FOREAGENT [2601.05930] forecasts ML-agent outcomes to skip executions. Both treat the forecast *as the decision*. We make it a speculative pre-filter **gated by ground-truth execution**.

**Verify-before-claim (the gate's foundation).** Reproducibility and effect-vs-noise rigor: "Are GANs Created Equal?" [1711.10337], "Deep RL That Matters" [1709.06560], Reimers-Gurevych [1707.09861], Bouthillier [2103.03098]; claim-verification structure [2408.14317]. We move this human, post-hoc discipline *into the loop* as a deterministic gate.

**Why confidence is not the gate (calibration).** Deep nets are systematically overconfident [1706.04599], arbitrarily confident far from training data [1812.05720], and RLHF degrades calibration [2303.08774]; memorization inflates self-knowledge [2506.18998]. Overconfidence concentrates exactly on the unfamiliar inputs that matter most → **we verify, we do not trust.** This principle ended up applying to our own results (§6).

## 3. System

Lucky Loop runs a staged loop over a research question:

1. **Literature review** — arXiv metadata/abstracts + curated notes feed a domain/method related-work pass and a gap list.
2. **Dataset selection** — discovery over Hugging Face / OpenML with a recorded rationale.
3. **Protocol generation** — a falsifiable protocol (hypothesis, falsification condition, controlled variables, seeds, primary metric).
4. **Code generation → static validation → sandboxed dry-run → real execution.**
5. **Verification** — every candidate claim enters a **claim ledger**: `claim → evidence_ids → metrics(effect_size, seed_noise, effect_to_noise_ratio) → verdict`. A claim is `supported` only if the measured effect exceeds estimated seed noise by a threshold; otherwise `blocked`.

**The world model (predict-before-compute).** Before each lab action, Qwen-AgentWorld is queried for a structured forecast `(recommendation, compute_waste_risk, value_of_information)`. When the model or sim is unavailable, a deterministic heuristic stub is used. The forecast informs action selection but **never overrides the verifier**: scientific evidence comes only from executed code.

**Serving.** Qwen-AgentWorld-35B-A3B runs under vLLM on an AMD MI300X (ROCm), OpenAI-compatible endpoint. All world-model calls in this paper hit the real model unless stated.

## 4. Experimental setup

We isolate one question: **does the world model change what the loop does, or do better than a trivial heuristic?**

- **M1 (does the loop run for real?).** One open-ended question, real Qwen required (`require_qwen=true`), budget 6.
- **A1 (decision ablation).** Three fixed-dataset studies, deterministic planner, no scientist agent (removes dataset-choice and LLM-text confounds). The *only* variable is the world model: **ON** = real Qwen (`wm_source=qwen_agentworld`) vs **OFF** = heuristic fallback (`wm_source=plumbing_not_called`). n=3 per arm.
- **A2 (experiment-ordering ablation, large candidate space).** 16 model/hyperparameter candidates (cost and accuracy both varied) × 6 datasets (breast_cancer, wine, digits, iris, two synthetic). The world model predicts each candidate's accuracy; we order candidates by the prediction and measure **compute-to-best** (runtime until the best-accuracy model is reached) against baselines.

## 5. Results

### 5.1 The loop runs end-to-end (M1)

Question: *"Does feature scaling improve logistic regression accuracy on breast_cancer?"* The agent selected the dataset, ran the literature pass, generated and executed a protocol over four model families (seeds 42/52/62/72/82), and the verifier returned **one `supported` claim**: effect_size **0.121**, seed_noise **0.011**, **effect/noise ratio 10.7**. All **8/8** world-model predictions came from the real Qwen (`qwen_agentworld`, 0 fallback). The full Track-3 cycle — review → code → analysis → verified report — is demonstrated on real hardware.

**First signal of the null:** all 8 real-Qwen predictions returned the identical structured `recommendation = run`. Even running for real, the world model's *decision-relevant* signal never diverged from the heuristic.

### 5.2 Decision ablation: NULL (A1)

| Study | exp_runs ON/OFF | supported ON/OFF | blocked ON/OFF | compute saved by WM |
|---|---|---|---|---|
| leakage_trap | 4 / 4 | 0 / 0 | 2 / 2 | **0.0** |
| seed_variance_claim | 3 / 3 | 0 / 0 | 4 / 4 | **0.0** |
| split_validity_sensor | 4 / 4 | 2 / 2 | 3 / 3 | **0.0** |

Every decision and every claim verdict is **identical** with the real world model vs the heuristic. The only non-zero difference is runtime jitter (±0.01–0.10 s, inconsistent sign). The real Qwen's structured signal collapses to the same `recommendation=run, compute_waste_risk=0.0, value_of_information=0.5` as the stub on these tasks. **No measurable effect.**

### 5.3 Experiment-ordering ablation: an apparent win that did not survive verification (A2)

A first run suggested ordering by Qwen's predicted accuracy saved **+98.7%** compute vs random on real datasets. Under scrutiny it collapsed:

| dataset | vs random | vs cheapest-first | cost-aware vs cheapest-first |
|---|---|---|---|
| breast_cancer | −135% (run 1: +98.7%) | −8727% | +0.0% |
| wine | +96% | −168% | +0.0% |
| digits | +99% | +57% | +0.0% |
| iris | +83% | −8% | +0.0% |
| synth_easy | −85% | −3969% | +0.0% |
| synth_hard | −100% | −2288% | +0.0% |

1. **Not reproducible.** breast_cancer flipped from +98.7% to −135% across two runs — Qwen's predictions vary between calls (temp 0.2), so the "win" was sampling noise.
2. **Beaten by one line of code.** A trivial *cheapest-first* heuristic (run the lowest-cost candidate first) beat the world-model ordering on 5 of 6 datasets.
3. **The cost term does all the work.** Making the ranking cost-aware (predicted-accuracy-per-unit-cost) yields **+0.0%** over cheapest-first everywhere — it degenerates *into* cheapest-first. The world model contributes nothing the cost heuristic doesn't.

**Conclusion.** On tabular ML, the language world model does not improve experiment selection over trivial heuristics — robust across both ablations.

## 6. The result behind the result: verify-before-claim, applied to ourselves

The project's thesis is that models are overconfident on what *looks* good and must be verified, not trusted. Over one working day, our own pipeline caught **three false positives, all in our favor**:

1. **Confounded pilot** — the autonomous agent chose *different datasets* per arm for the same question; the world-model effect was inseparable from the dataset change → fixed with the fixed-dataset controlled path.
2. **Self-deceiving analyzer** — our ablation analyzer reported "EFFECT MEASURED" on runtime jitter while every decision metric was identical → flagged and corrected.
3. **The +98.7% mirage** — an exciting headline number that was non-reproducible and beaten by a one-line baseline → killed before publication.

Had we shipped the +98.7%, we would have published exactly the overconfident, noise-as-signal claim the system exists to prevent. **The verifier stopped its own builders.** That is the contribution we trust most.

## 7. Limitations

- **Tabular sklearn tasks only.** The null may not transfer to open-ended, high-variance, or expensive research tasks where structured forecasts could genuinely diverge.
- **Baselines.** A1's OFF arm is a heuristic stub; A2's strongest baseline (cheapest-first) is itself trivial — the honest reading is "the world model does not beat trivial heuristics here," not "world models are useless."
- **Scale / variance.** n=3 (A1), single split per dataset (A2), low-temperature stochastic predictions. Reproducibility commands are emitted per run.
- **Forecast use.** We tested decision-gating and accuracy-ranking; other uses (failure prediction, early-stopping, multi-step planning) are untested.

## 8. Conclusion

We built a working autonomous research agent that fuses pre-compute language-world-model forecasting with an in-loop effect-vs-noise verifier, and we measured the world model honestly. On tabular ML it adds no decision value over trivial heuristics — a controlled negative result. More importantly, the verification machinery caught three false positives produced by its own authors, including a headline number we wanted to be true. **When does a language world model help autonomous research? On these tasks, not yet — and the discipline that let us say so is the real system.**

---

### Artifacts (reproducible)
- M1 workspace: `reports/lab/open-does-feature-scaling-improve-logistic-regression-accuracy-on-breast-canc/` (notebook, predictions, claim ledger, final report).
- A1: `reports/lab_ablations/<study>/{on,off}_run*/` + per-study `analysis.json`.
- A2: `experiments/wm_guidance_bigspace.py`, `reports/wm_guidance_bigspace.json`.
- World model: Qwen-AgentWorld-35B-A3B, vLLM on AMD MI300X (ROCm).

### References
Verified arXiv identifiers, by section. Agents: AI Scientist v1 [2408.06292] / v2 [2504.08066], Agent Laboratory [2501.04227], RD-Agent [2505.14738], Curie [2502.16069], MLE-bench [2410.07095], PaperBench [2504.01848], RE-Bench [2411.15114], CORE-Bench [2409.11363], POPPER [2502.09858], implementation-capability critique [2506.01372]. Selection/surrogates: BO [1206.2944], Hyperband [1603.06560], BOHB [1807.01774], FABOLAS [1605.07079], CAGES [2405.07760], LLAMBO [2402.03921], OPRO [2309.03409]. Outcome prediction: Wen et al. [2506.00794], FOREAGENT [2601.05930]. World models: Qwen-AgentWorld [2606.24597]. Reproducibility: [1711.10337], [1709.06560], [1707.09861], [2103.03098], claim verification [2408.14317]. Calibration: [1706.04599], [1812.05720], [2303.08774], [2506.18998].
