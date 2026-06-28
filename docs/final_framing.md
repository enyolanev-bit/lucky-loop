# Lucky Loop: cadrage final

## Framing

**Lucky Loop: Predictive Research Lab OS with Claim-Calibrated Reporting**

Tagline:

**Predict before compute. Verify before claim.**

One-liner:

Lucky Loop is a predictive research lab backend: an autoresearch agent proposes research actions, Qwen-AgentWorld predicts their likely outcome, cost, protocol risk, and claim impact before compute, real code tests reality, and a deterministic verifier decides which claims survive the evidence.

## Positionnement

On ne vend pas "un agent qui lance des expériences ML". On vend un research lab OS prédictif qui a trois comportements que les autoresearch agents classiques n'ont pas assez:

1. **Foresight before compute**: Qwen-AgentWorld sert de language world model. Il prédit métriques, runtime, risques et failure modes à partir de l'état courant et d'une action candidate.
2. **Protocol awareness**: le système sait si une action va changer le rapport final ou juste ajouter un score non claimable.
3. **Discipline before claim**: après l'expérience réelle, un verifier déterministe bloque les claims non supportés et force le rapport à rester honnête.

Le world model est le cœur wow: il contrôle la boucle avant compute. Le verifier est la deuxième couche de confiance.

## Rôles

```text
Autoresearch agent / planner API
    propose une hypothèse, une shortlist d'actions candidates et l'action préférée

Qwen-AgentWorld
    prédit ce qui devrait arriver si une action est lancée, combien elle coûte, quels risques de protocole elle porte, et ce qu'elle change dans les claims

Executor
    lance la vraie expérience sklearn et logge les métriques

Comparator
    compare prédiction et réalité

Verifier
    gate déterministe des claims scientifiques

Claim ledger / reporter / UI
    expose les preuves et n'écrit que les claims autorisés
```

## Modes backend

```text
Product path:
    Autoresearch agent = planner API OpenAI-compatible
    World model = Qwen-AgentWorld
    Safety selector / Executor / Comparator / Verifier = code Python dans le repo

Agent-in-repo path:
    Autoresearch agent = Codex / Claude Code / OpenClaw / Hermes opérant depuis program.md
    Backend trace = planner_mode=operator_driven, agent_backend=<agent_name>_operator
    World model = Qwen-AgentWorld

Development smoke path:
    Autoresearch agent = replay mode au même schema, uniquement pour tester sans clé API ni agent
    World model = Qwen-AgentWorld
    Executor / Comparator / Verifier = mêmes modules Python
```

Le point important: l'agent autoresearch est distinct de Qwen-AgentWorld. L'agent propose et décide; Qwen-AgentWorld prédit. Le mode `operator_driven` garde le chemin agent-in-repo propre pour le hackathon; le mode API peut ensuite remplacer la décision agent sans changer executor, comparator, verifier, claim ledger ou reports.

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

## Ablation attendue

Pour prouver l'apport réel du world model, le backend doit comparer trois politiques:

1. `classic_autoresearch`: agent classique qui explore et peut reporter le meilleur score single-run.
2. `classic_verified`: même agent classique, mais avec verification déterministe des top models.
3. `lucky_loop_full`: agent + Qwen-AgentWorld avant compute + comparator + verifier + claim ledger.

La comparaison ne doit pas seulement regarder le best score. Elle doit mesurer:

- nombre de prédictions Qwen pré-compute
- prediction misses loggées
- top-model verification lancée ou non
- claims unsupported restants
- claims bloqués / supportés
- preuves JSON disponibles par run

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
