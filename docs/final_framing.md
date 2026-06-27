# Lucky Loop: cadrage final

## Framing

**Lucky Loop: World-Model-Guided Autonomous Research with Claim-Calibrated Reporting**

Tagline:

**Predict before compute. Verify before claim.**

One-liner:

Lucky Loop is a world-model-guided autonomous research loop: an autoresearch agent proposes experiments, Qwen-AgentWorld predicts their likely outcomes before compute, real code tests reality, and a deterministic verifier decides which claims survive the evidence.

## Positionnement

On ne vend pas "un agent qui lance des expériences ML". On vend un agent de recherche qui a deux comportements que les autoresearch agents classiques n'ont pas assez:

1. **Foresight before compute**: Qwen-AgentWorld sert de language world model. Il prédit métriques, runtime, risques et failure modes à partir de l'état courant et d'une action candidate.
2. **Discipline before claim**: après l'expérience réelle, un verifier déterministe bloque les claims non supportés et force le rapport à rester honnête.

Le world model est le cœur wow. Le verifier est la deuxième couche de confiance.

## Rôles

```text
Autoresearch agent / planner
    propose des actions candidates et décide quoi lancer

Qwen-AgentWorld
    prédit ce qui devrait arriver si une action est lancée

Executor
    lance la vraie expérience sklearn et logge les métriques

Comparator
    compare prédiction et réalité

Verifier
    gate déterministe des claims scientifiques

Claim ledger / reporter / UI
    expose les preuves et n'écrit que les claims autorisés
```

Pendant le build hackathon, Codex peut jouer le rôle de l'agent autoresearch. Le repo doit quand même produire des traces dans le format final: état, candidates, prédictions, décision, exécution, comparaison, verifier, claims.

## Boucle cible

```text
research question
-> explicit state s_t
-> candidate actions a_t
-> Qwen-AgentWorld predicts observations o_hat_t+1
-> selector chooses action using world-model signal
-> executor returns real observation o_t+1
-> comparator measures prediction-vs-reality
-> verifier gates claims
-> claim ledger
-> honest report and demo UI
-> updated state s_t+1
```

## Ce qu'il faut marteler

- Qwen-AgentWorld n'est pas seulement un backend LLM.
- Il joue le rôle de simulateur de l'environnement expérimental.
- Le système prédit avant de dépenser du compute.
- La prédiction est comparée au résultat réel.
- Les prédictions ratées restent visibles.
- Le verifier empêche de transformer un résultat fragile en claim.
- Le claim ledger rend le rapport auditable.

## Démo cible

### Acte 1 - Predict before compute

Montrer un état courant et plusieurs candidates. Qwen-AgentWorld prédit pour chaque candidate:

- metric range
- runtime
- risks
- recommendation
- rationale

### Acte 2 - World-model-guided decision

Montrer que le planner choisit une action à cause d'un signal du world model:

> Qwen predicted seed variance risk, so the agent selected a multi-seed verifier sweep before allowing a strong claim.

### Acte 3 - Reality check

L'expérience réelle tourne. Le comparator affiche hit/miss.

### Acte 4 - Verify before claim

Montrer le moment honnête:

```text
Apparent winner: C=0.1
effect_size = 0.020979
seed_noise = 0.027972
verdict = inconclusive
blocked claim: C=0.1 is robustly better
allowed claim: C=0.1 had the best mean, but the effect was smaller than seed noise
```

## Phrase finale

Most AI scientists hallucinate after the experiment. Lucky Loop makes a prediction before the experiment, runs the real code, compares prediction with reality, and only claims what survives verification.
