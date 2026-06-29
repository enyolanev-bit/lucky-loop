# Brancher le frontend sur run.json (statique, pas de backend live)

Deux changements, rien d'autre. Le reste de l'UI est inchangé.

## 1. Poser le fichier
Copie `public/run.json` (dans ce bundle) à la racine `public/` de
`lucky-loop-frontend`. Next le sert alors à l'URL `/run.json`.

## 2. Patcher getRun() dans lib/oracle-data.ts

REMPLACER :

    export async function getRun(): Promise<Run> {
      return RUN
    }

PAR :

    export async function getRun(): Promise<Run> {
      try {
        const res = await fetch("/run.json", { cache: "no-store" })
        if (!res.ok) throw new Error(`run.json ${res.status}`)
        return (await res.json()) as Run
      } catch {
        return RUN   // fallback offline = fixture embarqué
      }
    }

C'est tout. `RUN` reste comme fallback si /run.json manque.

## Régénérer run.json plus tard
Depuis le repo backend, après un run :
    PYTHONPATH=src .venv/bin/python scripts/export_run.py --out reports/run_export/run.json
puis recopier dans public/run.json.

## Deux narratifs prêts (les deux 100% réels, sklearn)
- `public/run.json`         → verdict CONFIRMED (le scaling bat le bruit) — actif par défaut.
- `public/run_blocked.json` → verdict BLOCKED  (svc vs random_forest: écart dans le bruit de seed).
Pour basculer la démo sur le verdict bloqué :
    cp public/run_blocked.json public/run.json
getRun() fetch toujours /run.json — il suffit d'écraser ce fichier.
