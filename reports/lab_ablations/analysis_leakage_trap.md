# Ablation world-model ON/OFF — analyse honnête

**Verdict : NULL → Option B (no measurable world-model effect). Experiment runs identical (compute_runs_saved=0) and claim verdicts identical (supported & blocked deltas=0). Runtime differences are timing jitter, not an effect; Qwen's free text differs from the fallback but its structured recommendation stays 'run' (same decisions). Present system + M1 with this as a stated limitation — do NOT claim a world-model compute win.**

- ON arm: 3 runs · source `['qwen_agentworld']`
- OFF arm: 3 runs · source `['plumbing_not_called']`

| Métrique (mean ± std) | ON (world-model) | OFF (fallback) | Δ | compte pour le verdict ? |
|---|---|---|---|---|
| **Runs d'expérience réels** | 4.0 ± 0.0 (n=3) | 4.0 ± 0.0 (n=3) | 0.0 (OFF−ON) | ✅ OUI |
| **Claims supported** | 0.0 ± 0.0 (n=3) | 0.0 ± 0.0 (n=3) | 0 (ON−OFF) | ✅ OUI |
| **Claims blocked** | 2.0 ± 0.0 (n=3) | 2.0 ± 0.0 (n=3) | 0.0 (ON−OFF) | ✅ OUI |
| Runtime total (s) | 0.1924 ± 0.0043 (n=3) | 0.1843 ± 0.0022 (n=3) | -0.0081 (jitter, hors verdict) — within noise | ❌ non (jitter) |
| Runs jusqu'au 1er supported | n/a | n/a | — | indicatif |

**Texte/champs Qwen ≠ fallback ?** oui — _informatif uniquement, NE compte PAS comme effet_ (la `recommendation` reste `run` → même décision).

## Intégrité
- Verdict driven ONLY by behavioural deltas: experiment runs + claim verdicts. Runtime jitter and Qwen free-text differences NEVER trigger an effect.
- Metrics are ACTUALS computed identically for both arms; counterfactual 'saved_remaining_runs' is NOT used; claims are de-duplicated before counting.
- Each arm must be repeated >= 3x; a delta from fewer runs is flagged UNDER-POWERED (could be noise).
- Arm sources are validated: ON must be qwen_agentworld, OFF plumbing_not_called — otherwise the verdict is prefixed with a mislabel warning.
- qwen_text_differs is informational: the world model can emit different prose while the structured recommendation stays 'run' (same decision) -> that is NOT an effect.

## Détail par run
| arm | workspace | source | exp_runs | runtime_s | supported | blocked | non-trivial |
|---|---|---|---:|---:|---:|---:|---|
| ON | `reports/lab_ablations/leakage_trap/on_run1` | qwen_agentworld | 4 | 0.1875 | 0 | 2 | False |
| ON | `reports/lab_ablations/leakage_trap/on_run2` | qwen_agentworld | 4 | 0.1917 | 0 | 2 | False |
| ON | `reports/lab_ablations/leakage_trap/on_run3` | qwen_agentworld | 4 | 0.198 | 0 | 2 | False |
| OFF | `reports/lab_ablations/leakage_trap/off_run1` | plumbing_not_called | 4 | 0.185 | 0 | 2 | False |
| OFF | `reports/lab_ablations/leakage_trap/off_run2` | plumbing_not_called | 4 | 0.1814 | 0 | 2 | False |
| OFF | `reports/lab_ablations/leakage_trap/off_run3` | plumbing_not_called | 4 | 0.1866 | 0 | 2 | False |
