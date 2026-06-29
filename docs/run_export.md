# Static run export — backend → frontend `Run`

Option A: the backend projects a **finished** run into a single `run.json` that
matches the frontend `Run` type field-for-field. The frontend serves the file
and `getRun()` fetches it. **No Python runs on Vercel. No live execution.**

```
reports/lab/<slug>/        run artifacts (a finished run)
        │  scripts/export_run.py  (luckyloop.run_export)
        ▼
reports/run_export/run.json   ← strictly conformant Run
        │  copy
        ▼
lucky-loop-frontend/public/run.json   ← served statically
        │  getRun() = fetch("/run.json")
        ▼
UI (unchanged)
```

## Generate run.json

```bash
# from a real finished workspace (auto-picks the latest reports/lab/*)
PYTHONPATH=src .venv/bin/python scripts/export_run.py --out reports/run_export/run.json

# or point at a specific workspace
PYTHONPATH=src .venv/bin/python scripts/export_run.py \
  --workspace reports/lab/<slug> --out reports/run_export/run.json
```

The exporter prints a one-line summary and any anti-fabrication warnings
(e.g. `findings` empty when no benchmark artifact is present).

## Wire the frontend (one seam, nothing else)

1. Copy the file the frontend will serve:
   ```bash
   cp reports/run_export/run.json ../lucky-loop-frontend/public/run.json
   ```
2. Switch `getRun()` in `lib/oracle-data.ts` (this is the only frontend edit):
   ```ts
   export async function getRun(): Promise<Run> {
     const res = await fetch("/run.json", { cache: "no-store" })
     return (await res.json()) as Run
   }
   ```
   The embedded `RUN` fixture can stay as an offline fallback inside a `try/catch`.

`run.json` carries no `_warnings` key — that is stripped before writing and only
printed to the operator.

## Mapping (every number from a real artifact)

| `Run` field | Source artifact |
|---|---|
| `currentState` | `study_result.json` state id · `dataset_audit.json` · literature source count |
| `papers` | `literature/study_inference.json` `included_sources` (exact arXiv titles), authors resolved from the real curated catalogue, `ref`/`url` from the arXiv id |
| `predictions` | `top_model_summary.json` `top_models` — real mean ± std → `accLow/accHigh`; `probability` = softmax (T=0.20) over the **real** ranked metrics |
| `diffs` | `analyses/analysis_*.json` `condition_means` progression (`delta = after − before`) |
| `verdict` | `claim_ledger.json` verdict + `analysis.effect_size` / `seed_noise` |
| `traces` | literature sources + `dataset_audit` dims (`Loaded 569×30`) |
| `pipeline[].log` | real values above (paper refs, dataset dims, diffs, effect/noise, verdict) |
| `findings` | **only** a real `findings.json` benchmark artifact; absent → `[]` |

### Anti-fabrication rules baked in

- No measurement is invented. Presentation-only fields (`probability` bar,
  gothic `verdict.title/reason`, `pipeline[].speak`) are **derived** from real
  numbers; they never introduce a new metric.
- `findings` is empty unless a real `findings.json` exists. The four head-line
  stats in the frontend fixture (`+53.8%/−45%`, `48.3→92.6`, `r=0.45`, `28.6%`)
  are **not** computed by this repo (`simulator.py` returns a constant `0.45`),
  so the exporter refuses to emit them. Provide a real benchmark artifact to
  populate `findings`.

## Demo workspace (real metrics, no LLM)

No completed run ships in the repo, and the full loop can't run offline (no
`openai` dep, no Qwen-AgentWorld simulator, no network). To produce a workspace
whose numbers are **really computed** by sklearn (LLM orchestration bypassed):

```bash
PYTHONPATH=src .venv/bin/python scripts/make_demo_workspace.py
```

It fits logistic regression (raw / scaled / C-tuned), random forest and SVC on
`breast_cancer` across seeds `[0..4]`, then writes real `condition_means`,
`effect_size`, `seed_noise`, a verdict, and the four verified arXiv references.
Provenance is stamped `offline_real_metrics_no_llm` in `_provenance.json`. A
true end-to-end run (LLM scientist + Qwen world model) still needs the full
backend environment; point `export_run.py` at that workspace instead.

## Test

```bash
PYTHONPATH=src python -m pytest tests/test_run_export.py -q
```

Asserts the exact `Run` shape, that metrics trace to source artifacts (not
placeholders), and that `findings` stays empty without a real artifact.
