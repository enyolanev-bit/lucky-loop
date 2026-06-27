# Verifier cross-check: paired IC95 best-vs-2e

Résumé: 2/5 verdicts concordent.

| claim | verdict trust-ladder | verdict IC95 | accord oui/non | note |
|---|---|---|---|---|
| run_004_claim_001: C=0.1 is robustly better than C=10.0. | blocked | FAIL | oui | C=0.1 vs C=1.0; n=4; mean diff=0.008741; IC95=[-0.010260, 0.027742] |
| run_005_claim_001: candidate beat baseline by 0.1000 accuracy; effect/noise ratio=6.67, status=strongly supported. | strongly_supported | PASS | oui | candidate vs baseline; n=4; mean diff=0.100000; IC95=[0.100000, 0.100000] |
| run_006_claim_001: candidate is robustly better than baseline. | blocked | PASS | non | candidate vs baseline; n=4; mean diff=0.012000; IC95=[0.012000, 0.012000] |
| run_007_claim_001: leaky_protocol is a valid scientific winner. | blocked | PASS | non | leaky_protocol vs proper_protocol; n=4; mean diff=0.045250; IC95=[0.038453, 0.052047]; protocol warning in trust ladder: suspiciously high score; label-derived feature was included before the split |
| run_008_claim_001: balanced_objective is a valid scientific winner. | blocked | PASS | non | balanced_objective vs accuracy_only_baseline; n=4; mean diff=0.210000; IC95=[0.210000, 0.210000]; protocol warning in trust ladder: accuracy is misleading on an imbalanced dataset; balanced_accuracy is the verifier metric |

## Méthode

- Source claims: `reports/claim_ledger.json`.
- Source métriques: artifacts `runs/<run_id>.json` référencés par `evidence_run_ids`.
- Unité appariée: seed explicite quand disponible, sinon fallback par ordre intra-config.
- Comparaison: meilleur candidat vs deuxième meilleur candidat par moyenne de la métrique du sweep.
- Différence appariée: score(best) - score(second) sur les seeds communes.
- IC95: intervalle t bilatéral sur les différences appariées, self-contained via table t critique.
- Verdict IC95: PASS si la borne basse de l'IC95 est strictement > 0, sinon FAIL. INCONCLUSIVE si données insuffisantes.
- Accord: les statuts trust-ladder `allowed`, `weakly_supported`, `supported`, `strongly_supported` sont traités comme supportifs; `blocked`/`inconclusive` comme non-supportifs.

## Limitations

- `run_006_claim_001` diverge: l'IC95 apparié détecte un effet strictement positif, tandis que le trust ladder reste plus conservateur car il compare l'effet à la noise inter-seed.
- `run_007_claim_001` diverge: l'IC95 teste seulement la significativité métrique; le trust ladder bloque aussi les warnings de protocole.
- `run_008_claim_001` diverge: l'IC95 teste seulement la significativité métrique; le trust ladder bloque aussi les warnings de protocole.
