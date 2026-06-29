# Ablation world-model ON/OFF — analyse honnête

**Verdict : NULL → Option B (no measurable world-model effect). Experiment runs identical (compute_runs_saved=0) and claim verdicts identical (supported & blocked deltas=0). Runtime differences are timing jitter, not an effect; Qwen's free text differs from the fallback but its structured recommendation stays 'run' (same decisions). Present system + M1 with this as a stated limitation — do NOT claim a world-model compute win.**

- ON arm: 3 runs · source `['qwen_agentworld']`
- OFF arm: 3 runs · source `['plumbing_not_called']`

| Métrique (mean ± std) | ON (world-model) | OFF (fallback) | Δ | compte pour le verdict ? |
|---|---|---|---|---|
| **Runs d'expérience réels** | 3.0 ± 0.0 (n=3) | 3.0 ± 0.0 (n=3) | 0.0 (OFF−ON) | ✅ OUI |
| **Claims supported** | 0.0 ± 0.0 (n=3) | 0.0 ± 0.0 (n=3) | 0 (ON−OFF) | ✅ OUI |
| **Claims blocked** | 4.0 ± 0.0 (n=3) | 4.0 ± 0.0 (n=3) | 0.0 (ON−OFF) | ✅ OUI |
| Runtime total (s) | 1.7523 ± 0.0332 (n=3) | 1.8524 ± 0.1253 (n=3) | 0.1001 (jitter, hors verdict) — within noise | ❌ non (jitter) |
| Runs jusqu'au 1er supported | n/a | n/a | — | indicatif |

**Texte/champs Qwen ≠ fallback ?** oui — _informatif uniquement, NE compte PAS comme effet_ (la `recommendation` reste `run` → même décision).

## Intégrité
- Verdict driven ONLY by behavioural deltas: experiment runs + claim verdicts. Runtime jitter and Qwen free-text differences NEVER trigger an effect.
- Metrics are ACTUALS computed identically for both arms; counterfactual 'saved_remaining_runs' is NOT used.
- wm_source proves the arm: qwen_agentworld (ON) vs plumbing_not_called (OFF).
- qwen_text_differs is informational: the world model can emit different prose while the structured recommendation stays 'run' (same decision) -> that is NOT an effect.

## Détail par run
| arm | workspace | source | exp_runs | runtime_s | supported | blocked | non-trivial |
|---|---|---|---:|---:|---:|---:|---|
| ON | `reports/lab_ablations/seed_variance_claim/on_run1` | qwen_agentworld | 3 | 1.7327 | 0 | 4 | False |
| ON | `reports/lab_ablations/seed_variance_claim/on_run2` | qwen_agentworld | 3 | 1.799 | 0 | 4 | False |
| ON | `reports/lab_ablations/seed_variance_claim/on_run3` | qwen_agentworld | 3 | 1.7252 | 0 | 4 | False |
| OFF | `reports/lab_ablations/seed_variance_claim/off_run1` | plumbing_not_called | 3 | 2.0291 | 0 | 4 | False |
| OFF | `reports/lab_ablations/seed_variance_claim/off_run2` | plumbing_not_called | 3 | 1.7528 | 0 | 4 | False |
| OFF | `reports/lab_ablations/seed_variance_claim/off_run3` | plumbing_not_called | 3 | 1.7753 | 0 | 4 | False |
