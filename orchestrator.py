"""Lucky Loop — agent de recherche autonome end-to-end ("autoresearch you can trust").

Loop : Literature -> Planner -> Experimenter (RUNS REAL CODE) -> Verifier (skeptic) -> Writer.
Le différenciateur = le Verifier : il re-checke les findings contre les vrais chiffres loggés
et flag ce qui n'est pas prouvé. Il ne fabrique rien.

Usage :
    export OPENAI_API_KEY=sk-...        # ou LLM_BASE_URL=http://<box>:8000/v1 (vLLM MI300X)
    python orchestrator.py --question "La régularisation améliore-t-elle la généralisation d'un petit MLP ?"
"""
from __future__ import annotations
import argparse, json
from datetime import datetime
from pathlib import Path
import llm, sandbox

OUT = Path(__file__).parent / "reports"


# ── 1. LITERATURE ────────────────────────────────────────────────────────────
def literature(question: str) -> str:
    """(MVP : résumé from-knowledge. TODO P3 : brancher un vrai crawl arXiv.)"""
    return llm.chat(
        "Tu es un agent de revue de littérature ML. Concis, factuel, cite les idées clés.",
        f"Question de recherche : {question}\nRésume en 5 puces l'état de l'art pertinent "
        f"(régularisation, généralisation, data-efficiency). Mentionne Konwoo et al. "
        f"'Pre-training under infinite compute' si pertinent.")


# ── 2. PLANNER ───────────────────────────────────────────────────────────────
def planner(question: str, lit: str) -> dict:
    """Transforme la question en plan d'expérience EXÉCUTABLE par le sandbox."""
    return llm.chat_json(
        "Tu conçois une expérience ML minimale et exécutable. Le sandbox entraîne un petit MLP "
        "sur 'digits' et peut varier UN hyperparamètre parmi: weight_decay, dropout, lr, hidden.",
        f"Question: {question}\nLittérature:\n{lit}\n\n"
        'Donne un plan JSON: {"hypothesis": str, "hp": one of [weight_decay,dropout,lr,hidden], '
        '"values": [list of 4-6 floats], "seeds": [list of ints], "success_criterion": str}')


# ── 3. EXPERIMENTER (RUNS REAL CODE) ─────────────────────────────────────────
def experimenter(plan: dict) -> dict:
    """Lance les VRAIES expériences. Renvoie les chiffres réels (rien d'inventé)."""
    return sandbox.sweep(plan["hp"], plan["values"], plan.get("seeds", [0, 1]))


# ── 4. VERIFIER (le différenciateur — skeptic) ───────────────────────────────
def verifier(plan: dict, results: dict) -> dict:
    """Re-checke : les claims du plan tiennent-ils face aux VRAIS chiffres ? Flag le non-prouvé."""
    # checks déterministes (pas de LLM) — la vérité vient des chiffres, pas du modèle
    rows = results["results"]
    accs = [r["acc_mean"] for r in rows]
    spread = round(max(accs) - min(accs), 4)
    best = results["best"]
    # variance inter-seed : un "best" est-il significatif ou dans le bruit ?
    best_seed_spread = round(max(best["acc_per_seed"]) - min(best["acc_per_seed"]), 4)
    verdict = {
        "best_value": best["value"], "best_acc": best["acc_mean"],
        "effect_size": spread,            # écart max-min entre hyperparamètres
        "seed_noise": best_seed_spread,   # bruit inter-seed sur le best
        "trustworthy": spread > best_seed_spread,  # effet > bruit ?
        "flags": [],
    }
    if spread <= best_seed_spread:
        verdict["flags"].append("EFFET DANS LE BRUIT : l'écart entre hyperparamètres ≤ bruit inter-seed → non concluant.")
    if len(results["seeds"]) < 2:
        verdict["flags"].append("1 seed seulement : pas d'estimation de variance.")
    return verdict


# ── 5. WRITER ────────────────────────────────────────────────────────────────
def writer(question: str, lit: str, plan: dict, results: dict, verdict: dict) -> str:
    return llm.chat(
        "Tu écris un mini-rapport de recherche honnête. Inclus SEULEMENT ce qui est prouvé par "
        "les chiffres. Si le Verifier a flaggé un doute, dis-le explicitement. Pas de survente.",
        f"Question: {question}\n\nLittérature:\n{lit}\n\nPlan:\n{json.dumps(plan)}\n\n"
        f"Résultats réels:\n{json.dumps(results['results'])}\n\nVerdict Verifier:\n{json.dumps(verdict)}\n\n"
        "Écris un rapport markdown: ## Question / ## Méthode / ## Résultats / "
        "## Ce qui est prouvé / ## Ce qu'on n'a PAS pu confirmer / ## Conclusion.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    a = ap.parse_args()
    print("① Literature…");  lit = literature(a.question)
    print("② Planner…");     plan = planner(a.question, lit); print("   plan:", json.dumps(plan))
    print("③ Experimenter (vraies expés)…"); results = experimenter(plan)
    print(f"   best={results['best']}  ({results['elapsed_s']}s sur {results['device']})")
    print("④ Verifier (skeptic)…"); verdict = verifier(plan, results); print("   verdict:", json.dumps(verdict))
    print("⑤ Writer…");      report = writer(a.question, lit, plan, results, verdict)
    OUT.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (OUT / f"report-{stamp}.md").write_text(report)
    (OUT / f"data-{stamp}.json").write_text(json.dumps({"plan": plan, "results": results, "verdict": verdict}, indent=2))
    print(f"\n✅ Rapport: reports/report-{stamp}.md")
    print(report[:600])


if __name__ == "__main__":
    main()
