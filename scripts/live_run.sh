#!/usr/bin/env bash
# Live end-to-end run -> run.json -> frontend. One command.
#
# Runs Icham's full Lucky Loop backend (scientist LLM + Qwen-AgentWorld world
# model + real sklearn), projects the resulting workspace into run.json via the
# exporter, and (optionally) drops it into the frontend's public/ folder.
#
# Usage:
#   ./scripts/live_run.sh                                  # defaults
#   QUESTION="..." BUDGET=6 FRONTEND=/path/to/lucky-loop-frontend ./scripts/live_run.sh
#
# Prereqs (on the machine that runs this):
#   - Python deps installed:  .venv/bin/pip install -r requirements.txt
#   - .env filled (agent + Qwen simulator) — already done in this repo
#   - The Qwen-AgentWorld server actually running at LUCKYWORLD_SIMULATOR_BASE_URL
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

QUESTION="${QUESTION:-Does feature scaling improve logistic regression on breast_cancer beyond seed noise?}"
BUDGET="${BUDGET:-6}"
PY="${PY:-.venv/bin/python}"
FRONTEND="${FRONTEND:-}"   # path to lucky-loop-frontend repo root (optional)

say() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
die() { printf '\n\033[31mABORT:\033[0m %s\n' "$1" >&2; exit 1; }

say "0. Load .env"
[ -f .env ] || die ".env not found. Fill it (agent + Qwen simulator) first."
set -a; source .env; set +a

say "1. Preflight checks"
[ -x "$PY" ] || die "No python at $PY. Create the venv and: $PY -m pip install -r requirements.txt"
"$PY" -c "import openai, sklearn" 2>/dev/null || die "Missing deps. Run: $PY -m pip install -r requirements.txt"
for v in LUCKYLOOP_AGENT_BASE_URL LUCKYLOOP_AGENT_MODEL LUCKYLOOP_AGENT_API_KEY \
         LUCKYWORLD_SIMULATOR_BASE_URL LUCKYWORLD_SIMULATOR_MODEL LUCKYWORLD_SIMULATOR_API_KEY; do
  [ -n "$(printenv "$v")" ] || die "$v is empty in .env"
done
echo "  agent     -> $LUCKYLOOP_AGENT_BASE_URL ($LUCKYLOOP_AGENT_MODEL)"
echo "  simulator -> $LUCKYWORLD_SIMULATOR_BASE_URL ($LUCKYWORLD_SIMULATOR_MODEL)"

say "2. Is the Qwen-AgentWorld server up?"
if curl -fsS --max-time 8 "$LUCKYWORLD_SIMULATOR_BASE_URL/models" \
      -H "Authorization: Bearer $LUCKYWORLD_SIMULATOR_API_KEY" >/tmp/_qwen_models.json 2>/dev/null; then
  if grep -q "$LUCKYWORLD_SIMULATOR_MODEL" /tmp/_qwen_models.json; then
    echo "  reachable, serving $LUCKYWORLD_SIMULATOR_MODEL ✓"
  else
    echo "  reachable, but did NOT list $LUCKYWORLD_SIMULATOR_MODEL — check --served-model-name."
    die "Simulator model name mismatch. It must equal LUCKYWORLD_SIMULATOR_MODEL exactly."
  fi
else
  die "Qwen simulator not reachable at $LUCKYWORLD_SIMULATOR_BASE_URL. Start it first (vLLM), then re-run."
fi

say "3. Run the full loop (this calls the LLMs + sklearn — takes a few minutes)"
echo "  question: $QUESTION"
echo "  budget:   $BUDGET"
PYTHONPATH=src "$PY" -m luckyloop.lab --question "$QUESTION" --budget "$BUDGET"

say "4. Export the fresh workspace -> run.json"
PYTHONPATH=src "$PY" scripts/export_run.py --out reports/run_export/run.json

say "5. Wire into the frontend"
if [ -n "$FRONTEND" ] && [ -d "$FRONTEND/public" ]; then
  cp reports/run_export/run.json "$FRONTEND/public/run.json"
  echo "  copied -> $FRONTEND/public/run.json ✓  (refresh the page)"
else
  echo "  run.json ready at: reports/run_export/run.json"
  echo "  copy it yourself:  cp reports/run_export/run.json <lucky-loop-frontend>/public/run.json"
fi

say "DONE — live end-to-end data is now in run.json"
