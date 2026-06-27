# Lucky Loop — Autoresearch you can trust 🔬

> Track 3, Paris Research Hackathon (TUM.ai × Iterate). Équipe **Pegasus**. Juge : Noah (Tzafon).
> Soumission : dim 28/06 **12:30** via ehl.gg + Entire.

## Le pitch en 1 phrase
Un agent de recherche autonome qui automatise la **ruée vers l'hyperparamètre** — il lit la littérature, conçoit et **lance de vraies expériences**, les analyse, et écrit un rapport. **Différenciateur : un agent Vérifieur qui ne fabrique rien** — il distingue l'effet réel du bruit et flag ce qui n'est pas prouvé.

## Pourquoi on gagne
La plupart des agents autoresearch **hallucinent des résultats**. Le nôtre **vérifie avant d'affirmer** (effet vs bruit inter-seed). Démontré : sur un sweep weight_decay, effet=0.0064 vs bruit=0.0055 → l'agent dit "à peine significatif" au lieu de survendre. C'est la vertu n°1 d'un chercheur : l'honnêteté reproductible.

## Architecture (5 agents + sandbox réel)
```
question → Literature → Planner → Experimenter(RUNS CODE) → Verifier(skeptic) → Writer → rapport.md
```
- `llm.py` — client LLM unique (OpenAI $50 crédits, ou vLLM self-hosted MI300X via LLM_BASE_URL)
- `sandbox.py` — vraie expé : MLP sur digits, sweep d'1 hyperparamètre. Rapide, reproductible. ✅ prouvé
- `orchestrator.py` — le loop + les 5 agents. Verifier = déterministe (la vérité = les chiffres). ✅ verifier prouvé

## Lancer
```bash
# 1. clé LLM
export OPENAI_API_KEY=sk-...            # $50 crédits hacka (mail)
#   OU self-hosted MI300X :  export LLM_BASE_URL=http://<MI300X-IP>:8000/v1 ; export LLM_MODEL=<id>

# 2. env
uv venv .venv && uv pip install --python .venv/bin/python torch scikit-learn

# 3. le loop complet
.venv/bin/python orchestrator.py --question "La régularisation améliore-t-elle la généralisation d'un petit MLP ?"
# -> reports/report-*.md + data-*.json

# test sandbox seul (sans LLM) :
.venv/bin/python sandbox.py --hp weight_decay --values 0 1e-4 1e-3 1e-2 --seeds 0 1 2
```

## Division de labo (équipe Pegasus, 4)
- **Nevil** : orchestrateur + **Verifier** (la signature) + démo/pitch
- **P2** : Experimenter + sandbox (élargir : plus d'hyperparamètres, figure matplotlib, run sur MI300X)
- **P3** : Literature (vrai crawl arXiv) + Writer (rapport + figure)
- **P4** : infra GPU (vLLM sur MI300X) + Entire + interface "fun" (voice/Telegram)
- **CC HQ** : scaffold (fait) + box GPU + debug on-call

## État (Day 1)
- ✅ Sandbox prouvé (vraies expés, 3.5s CPU)
- ✅ Verifier prouvé (effet vs bruit)
- ✅ GPU MI300X provisionné (<MI300X-IP>), image `rocm/vllm` en cours de pull
- ⏳ Clé OpenAI ($50) → tester le loop complet
- ⏳ Brancher experimenter sur le MI300X + figure + arXiv crawl + interface fun

## Scope (set up the exams)
**MVP (ce soir)** : le loop tourne end-to-end UNE fois → un rapport réel + le verdict du Verifier.
**Démo (dimanche)** : l'agent répond à une vraie question, montre une figure, et le Verifier flag honnêtement. Bonus : self-hosted MI300X + voice.
