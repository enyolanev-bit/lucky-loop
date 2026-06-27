# Lucky Loop — cadrage final

## One-liner

Lucky Loop is an autonomous research loop that predicts before it computes and verifies before it claims.

## Positionnement

On ne vend pas juste un agent qui lance des expériences. On vend un agent de recherche qui combine deux comportements complémentaires :

1. Imagination avant l'expérience : un language world model, Qwen-AgentWorld, prédit les métriques, runtime, risques et décisions probables avant de dépenser du compute.
2. Scepticisme après l'expérience : un Verifier déterministe vérifie que les claims sont supportés par les vrais chiffres, notamment effet mesuré vs bruit inter-seed.

## Problème

Les agents de recherche autonomes savent produire des plans et des rapports convaincants, mais un failure mode critique reste les claims non supportés : l'agent raconte une découverte alors que les logs ne la prouvent pas vraiment.

## Solution

Lucky Loop structure chaque itération comme une expérience auditable :

question
-> Literature / context
-> Planner
-> World-model prediction with Qwen-AgentWorld
-> Experimenter runs real code
-> Comparator prediction vs actual
-> Verifier effect vs noise
-> Honest Writer
-> report + JSON evidence

## Deux trust layers

### 1. Predict before compute

Avant d'exécuter, Qwen-AgentWorld reçoit l'état courant et l'action proposée. Il prédit :
- metric range
- runtime
- risks
- recommendation: run / skip / modify
- rationale

Cette prédiction sert à mieux choisir la prochaine expérience et à rendre le raisonnement traçable.

### 2. Verify before claim

Après exécution, un Verifier déterministe décide ce qui peut être affirmé :
- effect_size = écart entre configurations
- seed_noise = variance inter-seed pour la meilleure config
- trustworthy = effect_size > seed_noise

Si l'effet est dans le bruit, le rapport doit dire "inconclusive" au lieu de survendre.

## Démo cible

### Acte 1 — world-model guided loop

Dataset sklearn breast_cancer.

Qwen-AgentWorld prédit qu'une logistic regression sans scaling peut être limitée. La boucle choisit ensuite le scaling. Résultat réel : scaled logistic regression atteint 0.9860 accuracy.

Preuve : traces JSON + final_report.md.

### Acte 2 — verifier trust layer

On lance un mini sweep multi-seed sur un MLP/digits ou une variante sklearn avec perturbation contrôlée.

Le système montre un cas où une amélioration apparente est proche du bruit inter-seed. Le Verifier refuse de transformer ça en claim fort.

Phrase démo :

The model is creative before the experiment. The verifier is skeptical after it.

## Ce qu'on doit construire maintenant

1. Fusionner le framing dans README/paper.
2. Ajouter un Verifier effect-vs-noise dans le module LuckyWorld/Lucky Loop.
3. Ajouter un scénario multi-seed ou noisy_labels pour obtenir un exemple "inconclusive".
4. Générer un rapport qui sépare :
   - supported claims
   - weak/inconclusive findings
   - prediction misses
5. Mettre à jour Streamlit pour afficher : prediction, actual, verifier verdict.
6. Poster les updates dans Notion.

## Demo script final

1. Montrer la question de recherche.
2. Montrer la prédiction Qwen-AgentWorld avant run.
3. Montrer l'exécution réelle.
4. Montrer l'écart prediction vs actual.
5. Montrer le Verifier qui accepte ou refuse un claim.
6. Montrer le rapport final honnête.

## Phrase finale

Lucky Loop does not just automate research. It makes autonomous research auditable: every claim must survive both prediction and verification.
