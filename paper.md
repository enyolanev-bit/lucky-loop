# Calibrated Autoresearch: An Autonomous Research Agent That Refuses to Fabricate Findings

**Team Pegasus** — Paris Research Hackathon (TUM.ai × Iterate), Track 3 "Lucky Loop", June 2026.

> 📝 Squelette de paper. Pré-rempli avec ce qui est déjà prouvé ; `[TODO]` = à compléter par l'équipe. Garder court (4-6 pages style workshop). Anglais.

---

## Abstract
Autonomous research agents promise to automate the full loop from literature review to experimentation and reporting. A critical failure mode, however, is **fabrication**: agents report findings their own data does not support. We present **Calibrated Autoresearch**, a multi-agent system running an end-to-end research loop with **two trust layers**: (1) a *world-model* that predicts each experiment's outcome before running it and checks prediction-vs-reality, and (2) a deterministic **Verifier** that gates every "best method" claim on the criterion *effect > inter-seed noise*. On a method-comparison benchmark across increasing label noise, a naive agent claims a winner at **all 4** noise levels, while our Verifier confirms only **1** and abstains on **3** — including on clean data, where the methods are statistically tied (gap 0.0035 vs noise 0.028). Our contribution is not a stronger model but a **trust layer** that makes autonomous research reproducible and honest.

## 1. Introduction
- Research is slow and painful: literature triage, hyperparameter search, compute scheduling. *(cf. challenge framing.)*
- Autonomous "AI scientist" agents aim to automate this. **But the #1 risk is hallucinated findings** — agents that claim results their data doesn't support.
- **Our claim**: the bottleneck isn't generating experiments, it's *trusting* them. We add a deterministic verification layer.
- **Contributions**:
  1. An end-to-end multi-agent research loop that executes **real** experiments (not simulated).
  2. A **Verifier** that gates findings on *effect > inter-seed noise* — deterministic, model-free, un-foolable by a persuasive LLM.
  3. An honest report format that separates *what is proven* from *what could not be confirmed*.

## 2. Related Work
- Autonomous research / AI Scientist agents *[TODO P3: cite 2-3, arXiv crawl]*.
- Data-efficiency under fixed data: Konwoo et al., *Pre-training under infinite compute* (Stanford, arXiv:2509.14786) — regularization + ensembling + distillation. *(domaine d'expérience possible pour l'agent.)*
- Generalization theory: Wilson, *Deep learning is not so mysterious* (inductive biases as the lever). *[TODO: positionner notre gating effet/bruit.]*
- Gap: existing agents optimize for output, not **calibration/honesty**.

## 3. Method
**Architecture** (5 agents, orchestrated):
1. **Literature** — retrieves relevant prior work (arXiv). *[TODO: real crawl]*
2. **Planner** — turns the question into an executable experiment plan (which hyperparameter, values, seeds).
3. **Experimenter** — runs **real** code in a sandbox; returns measured numbers only.
4. **Verifier** ⭐ — *the contribution*. A deterministic statistical test on the real results:
   - rank methods by mean accuracy; take **best vs 2nd**;
   - compute **per-seed paired differences** `acc_best,i − acc_2nd,i`;
   - a **95% Student-t confidence interval** on the mean paired difference;
   - a finding is **trustworthy iff the CI lower bound > 0**; else flagged "inconclusive (within noise)".
5. **Writer** — produces a report containing only verified findings + an explicit "not confirmed" section.

**Why deterministic verification?** An LLM verifier can be talked into agreeing. A paired-CI significance gate cannot — the truth comes from the logged numbers, not the model. We deliberately use a *paired* test (same seeds/splits for both methods) and gate on the CI lower bound to control for runner-up variance and the winner's-curse of selecting the best post-hoc.

## 4. Experimental Setup
- **Sandbox**: small MLP (1 hidden layer) on `sklearn digits` (10 classes), trained on the team's AMD MI300X. Fast, reproducible (fixed seeds). *[TODO P2: extend to more tasks/hyperparams + matplotlib figure.]*
- **Search space**: `weight_decay`, `dropout`, `lr`, `hidden`.
- **LLM backend**: [TODO: OpenAI gpt-4.1-mini / self-hosted vLLM on MI300X].

## 5. Results
**Headline experiment (proven, `experiments/noise_sweep.py`):** 4 methods (logreg-scaled, random-forest,
SVC-rbf, hist-GB) on `breast_cancer`, across label-noise levels {0, 0.1, 0.2, 0.4}, 4 seeds.
Verifier verdict per level (best−2nd paired difference, 95% Student-t CI):

| Noise | best − 2nd | 95% CI margin | CI lower bound | Verifier verdict |
|---|---|---|---|---|
| 0.0 | 0.0035 | 0.0232 | −0.0197 | ❌ inconclusive (CI crosses 0) |
| 0.1 | 0.0174 | 0.0264 | −0.0090 | ❌ inconclusive |
| 0.2 | 0.0244 | 0.0485 | −0.0240 | ❌ inconclusive |
| 0.4 | 0.1066 | 0.0858 | **+0.0208** | ✅ svc_rbf is a significant winner |

**Naive agent: 4 winners claimed. Verifier (paired CI): 1 confirmed. → 3 unsupported winner claims avoided.**

The counter-intuitive finding: even on **clean** labels, the four methods are statistically tied
(paired CI [−0.0197, +0.0267] crosses 0) — a naive agent that "picks the best" reports noise as signal.
The Verifier abstains, and only at heavy noise (0.4) does the CI clear zero (a genuine winner).
Result holds under the rigorous gate (independently reviewed, see `CODEX-REVIEW.md`).
*(Figure: `reports/noise_sweep.png`.)*

## 6. Limitations
- **Scope of the claim**: our contribution is a *calibration mechanism*, not a general empirical proof that autoresearch agents are unreliable. Evidence is one small `sklearn` benchmark, 4 seeds, accuracy only.
- "Unsupported winner claims avoided" ≠ proven false positives; the gate can still be fooled by biased protocols, non-independent seeds, post-hoc benchmark selection, or config p-hacking.
- The paired-CI gate handles best-vs-2nd; multiple-comparison correction across all methods is future work.
- Literature agent coverage is shallow in 48h.

## 7. Conclusion
Autonomous research is only useful if you can trust it. By gating findings on real effect-vs-noise, Calibrated Autoresearch turns an LLM research loop into a reproducible, honest one. The contribution is a **trust layer**, portable to any autoresearch system.

## References
*[TODO P3: Konwoo et al. 2509.14786; Wilson; AI Scientist papers; LeRobot/diffusion if used.]*
