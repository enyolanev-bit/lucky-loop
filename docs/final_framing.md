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
Autoresearch agent / planner API
    propose une hypothèse, une shortlist d'actions candidates et l'action préférée

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

## Mode API-first

```text
Product path:
    Autoresearch agent = planner API OpenAI-compatible
    World model = Qwen-AgentWorld
    Safety selector / Executor / Comparator / Verifier = code Python dans le repo

Development path:
    Autoresearch agent = replay mode au même schema, uniquement pour tester sans clé API
    World model = Qwen-AgentWorld
    Executor / Comparator / Verifier = mêmes modules Python
```

Le point important: l'agent autoresearch est distinct de Qwen-AgentWorld. L'agent propose et décide; Qwen-AgentWorld prédit. Le replay mode n'est qu'un test double pour valider les traces et les rapports avant d'avoir une clé API planner.

## Boucle cible

```text
research question
-> explicit state s_t
-> candidate actions a_t
-> planner API returns AgentDecision
-> Qwen-AgentWorld predicts observations o_hat_t+1
-> safety selector validates the agent decision using world-model and evidence-risk signals
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
