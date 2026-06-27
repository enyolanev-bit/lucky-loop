# Calibrated Autoresearch: An Autonomous Research Agent That Refuses to Fabricate Findings

**Team Pegasus** — Paris Research Hackathon (TUM.ai × Iterate), Track 3 "Lucky Loop", June 2026.

> 📝 Squelette de paper. Pré-rempli avec ce qui est déjà prouvé ; `[TODO]` = à compléter par l'équipe. Garder court (4-6 pages style workshop). Anglais.

---

## Abstract
*[TODO P3 — 120 mots. Draft :]* Autonomous research agents promise to automate the full loop from literature review to experimentation and reporting. A critical failure mode, however, is **fabrication**: agents report findings that are not supported by their own data. We present **Calibrated Autoresearch**, a multi-agent system that runs an end-to-end research loop (literature → plan → real code execution → verification → report) with an explicit **Verifier** that gates every claimed finding on a simple, deterministic criterion: the measured effect must exceed inter-seed noise. On a hyperparameter-search benchmark, our system [TODO: headline result — e.g. "abstains from N% of unsupported claims while preserving M% of true findings"]. Our contribution is not a stronger model but a **trust layer** that makes autonomous research reproducible and honest.

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
4. **Verifier** ⭐ — *the contribution*. Deterministic checks on the real results:
   - `effect_size = max(acc) − min(acc)` across hyperparameter values;
   - `seed_noise = max−min` across seeds for the best config;
   - a finding is **trustworthy iff effect_size > seed_noise**; else flagged "inconclusive (within noise)".
5. **Writer** — produces a report containing only verified findings + an explicit "not confirmed" section.

**Why deterministic verification?** An LLM verifier can be talked into agreeing. A numeric effect-vs-noise gate cannot. The truth comes from the logged numbers, not the model.

## 4. Experimental Setup
- **Sandbox**: small MLP (1 hidden layer) on `sklearn digits` (10 classes), trained on the team's AMD MI300X. Fast, reproducible (fixed seeds). *[TODO P2: extend to more tasks/hyperparams + matplotlib figure.]*
- **Search space**: `weight_decay`, `dropout`, `lr`, `hidden`.
- **LLM backend**: [TODO: OpenAI gpt-4.1-mini / self-hosted vLLM on MI300X].

## 5. Results
**Preliminary (proven):** weight_decay sweep `{0, 1e-4, 1e-3, 1e-2}`, 2 seeds:
- best = `weight_decay=1e-3`, acc = **0.9731**
- **effect_size = 0.0064**, **seed_noise = 0.0055** → Verifier verdict: *trustworthy, but barely (effect ≈ noise)*.
- → The system reports this honestly instead of overselling "1e-3 is best". *(This is the headline behavior.)*

*[TODO: figure acc vs hyperparameter with seed error bars; a case where the Verifier correctly flags "inconclusive"; quantify fabrication avoided.]*

## 6. Limitations
- Sandbox is a toy task; the calibration principle generalizes but absolute numbers don't. *[honnêteté assumée]*
- Verifier uses a simple effect-vs-noise gate; richer statistics (CIs, significance tests) are future work.
- Literature agent coverage is shallow in 48h.

## 7. Conclusion
Autonomous research is only useful if you can trust it. By gating findings on real effect-vs-noise, Calibrated Autoresearch turns an LLM research loop into a reproducible, honest one. The contribution is a **trust layer**, portable to any autoresearch system.

## References
*[TODO P3: Konwoo et al. 2509.14786; Wilson; AI Scientist papers; LeRobot/diffusion if used.]*
