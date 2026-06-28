#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import urllib.parse
from http.server import SimpleHTTPRequestHandler, HTTPServer
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports" / "lab"
PYTHON = ROOT / ".venv" / "bin" / "python"

active_processes = {}

def _env_from_dotenv():
    env = {}
    dotenv = ROOT / ".env"
    if not dotenv.exists():
        return env
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env

def detected_models():
    dotenv = _env_from_dotenv()
    get = lambda key: os.getenv(key) or dotenv.get(key) or ""
    return {
        "world_model": get("LUCKYWORLD_SIMULATOR_MODEL") or get("LUCKYLOOP_WORLD_MODEL") or "not configured",
        "agent_model": get("LUCKYLOOP_AGENT_MODEL") or "not configured",
    }

def run_env():
    return {**os.environ, **_env_from_dotenv(), "PYTHONPATH": str(ROOT / "src")}

def safe_workspace(slug):
    if not slug or "/" in slug or "\\" in slug or slug.startswith("."):
        return None
    workspace = (REPORTS_DIR / slug).resolve()
    if REPORTS_DIR.resolve() not in workspace.parents:
        return None
    return workspace

def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default

def read_jsonl(path):
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    except Exception:
        return []

def build_complete_report(slug):
    workspace = safe_workspace(slug)
    if not workspace or not workspace.exists():
        return None

    models = detected_models()
    result = read_json(workspace / "study_result.json", {})
    events = read_jsonl(workspace / "events.jsonl")
    notebook = read_jsonl(workspace / "notebook.jsonl")
    predictions = read_jsonl(workspace / "predictions" / "world_model_predictions.jsonl")
    claims = read_json(workspace / "claim_ledger.json", [])
    final_report = (workspace / "final_report.md").read_text(encoding="utf-8") if (workspace / "final_report.md").exists() else ""

    papers = []
    for lit_type in ["domain", "method"]:
      sources = read_json(workspace / "literature" / lit_type / "sources.json", {}).get("sources", [])
      for source in sources:
          papers.append((lit_type, source))

    question = result.get("lab_question", {}).get("question") or slug
    lines = [
        f"# Lucky Loop Complete Run Report",
        "",
        f"- Run: `{slug}`",
        f"- Question: {question}",
        f"- World model: `{models['world_model']}`",
        f"- Agent model: `{models['agent_model']}`",
        "",
        "## Pipeline Events",
    ]

    for event in events:
        lines.append(f"- `{event.get('ts', '')}` **{event.get('event', '')}**: {event.get('message', '')}")
    if not events:
        lines.append("- No event trace found.")

    lines.extend(["", "## Lab Notebook"])
    for item in notebook:
        lines.append(f"- Step {item.get('step', '?')}: {item.get('event', item.get('message', item.get('kind', 'notebook entry')))}")
    if not notebook:
        lines.append("- No notebook entries found.")

    lines.extend(["", "## Papers And Sources"])
    for lit_type, paper in papers:
        authors = ", ".join(paper.get("authors", [])[:4])
        lines.append(f"- **[{lit_type}] {paper.get('title', 'Untitled')}** ({paper.get('year', 'n/a')})")
        if authors:
            lines.append(f"  - Authors: {authors}")
        if paper.get("url"):
            lines.append(f"  - URL: {paper['url']}")
        if paper.get("abstract"):
            lines.append(f"  - Abstract: {paper['abstract']}")
    if not papers:
        lines.append("- No literature sources found.")

    lines.extend(["", "## World Model Predictions"])
    for item in predictions:
        action = item.get("action", {})
        pred = item.get("prediction", {})
        lines.extend([
            f"### Step {item.get('step', '?')} - {action.get('kind', action.get('action_id', 'action'))}",
            f"- Action ID: `{action.get('action_id', '')}`",
            f"- Scientific goal: {action.get('scientific_goal', '')}",
            f"- Recommendation: `{pred.get('recommendation', '')}`",
            f"- Claim support probability: `{pred.get('claim_support_probability', '')}`",
            f"- Compute waste risk: `{pred.get('compute_waste_risk', '')}`",
            f"- Expected claim delta: `{pred.get('expected_claim_delta', '')}`",
            f"- Predicted observation: {pred.get('predicted_terminal_observation', '')}",
            f"- Rationale: {pred.get('rationale', '')}",
            "",
        ])
    if not predictions:
        lines.append("- No world-model predictions found.")

    lines.extend(["", "## Claim Ledger"])
    claim_items = claims if isinstance(claims, list) else claims.get("claims", [])
    for claim in claim_items:
        lines.append(f"- **{claim.get('claim', claim.get('text', 'Claim'))}**")
        lines.append(f"  - Verdict: `{claim.get('verdict', 'n/a')}`")
        if claim.get("reason"):
            lines.append(f"  - Reason: {claim['reason']}")
    if not claim_items:
        lines.append("- No claim ledger entries found.")

    if final_report:
        lines.extend(["", "## Final Report", "", final_report])

    return "\n".join(lines).strip() + "\n"

class LuckyLoopHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        # Default behavior: serve from web/ directory
        web_dir = ROOT / "web"
        parsed = urllib.parse.urlparse(path)
        rel_path = parsed.path.lstrip("/")
        
        # Route reports or runs requests to the workspace root
        if rel_path.startswith("reports/") or rel_path.startswith("runs/"):
            return str(ROOT / rel_path)
            
        return str(web_dir / rel_path)

    def do_POST(self):
        if self.path == "/api/run":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                params = json.loads(post_data)
                question = params.get("question", "")
                budget = int(params.get("budget", 8))
            except Exception:
                question = ""
                budget = 8

            if not question:
                self.send_error(400, "Missing question parameter")
                return
            budget = max(1, min(32, budget))

            # Compute slug to know where output goes
            slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:72].strip("-")
            slug = f"open-{slug or 'research'}"
            workspace = REPORTS_DIR / slug

            # Start backend run in a background process.
            proc = subprocess.Popen(
                [str(PYTHON if PYTHON.exists() else Path(sys.executable)), "-m", "luckyloop.lab", "--question", question, "--budget", str(budget)],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=run_env(),
            )
            active_processes[slug] = {"proc": proc, "error": ""}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "started", "slug": slug}).encode('utf-8'))
            return

        self.send_error(404)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            parsed_url = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            slug = query_params.get("slug", [None])[0]

            if not slug:
                self.send_error(400, "Missing slug parameter")
                return

            workspace = REPORTS_DIR / slug
            status_data = {
                "slug": slug,
                "running": False,
                "models": detected_models(),
                "error": "",
                "events": [],
                "notebook": [],
                "predictions": [],
                "claims": [],
                "papers": [],
                "result": None
            }

            # Check if background process is still running
            active = active_processes.get(slug)
            if active:
                proc = active["proc"]
                poll = proc.poll()
                if poll is None:
                    status_data["running"] = True
                else:
                    stderr = proc.stderr.read() if proc.stderr else ""
                    stdout = proc.stdout.read() if proc.stdout else ""
                    if poll != 0:
                        active["error"] = (stderr or stdout or f"Process exited with code {poll}")[-4000:]
                    status_data["error"] = active.get("error", "")
                    del active_processes[slug]

            # Read events.jsonl
            events_file = workspace / "events.jsonl"
            if events_file.exists():
                try:
                    with open(events_file, "r", encoding="utf-8") as f:
                        status_data["events"] = [json.loads(line) for line in f if line.strip()]
                except Exception as e:
                    print(f"Error reading events: {e}")

            # Read notebook.jsonl
            notebook_file = workspace / "notebook.jsonl"
            if notebook_file.exists():
                try:
                    with open(notebook_file, "r", encoding="utf-8") as f:
                        status_data["notebook"] = [json.loads(line) for line in f if line.strip()]
                except Exception as e:
                    print(f"Error reading notebook: {e}")

            # Read world_model_predictions.jsonl
            pred_file = workspace / "predictions" / "world_model_predictions.jsonl"
            if pred_file.exists():
                try:
                    with open(pred_file, "r", encoding="utf-8") as f:
                        status_data["predictions"] = [json.loads(line) for line in f if line.strip()]
                except Exception as e:
                    print(f"Error reading predictions: {e}")

            # Read claim_ledger.json
            claims_file = workspace / "claim_ledger.json"
            if claims_file.exists():
                try:
                    status_data["claims"] = json.loads(claims_file.read_text(encoding="utf-8"))
                except Exception as e:
                    print(f"Error reading claims: {e}")

            # Read study_result.json
            result_file = workspace / "study_result.json"
            if result_file.exists():
                try:
                    status_data["result"] = json.loads(result_file.read_text(encoding="utf-8"))
                except Exception as e:
                    print(f"Error reading result: {e}")

            # Read papers from literature sources.json
            papers_list = []
            for lit_type in ["domain", "method"]:
                sources_file = workspace / "literature" / lit_type / "sources.json"
                if sources_file.exists():
                    try:
                        sources_data = json.loads(sources_file.read_text(encoding="utf-8"))
                        sources = sources_data.get("sources", [])
                        for s in sources:
                            papers_list.append({
                                "title": s.get("title", ""),
                                "authors": s.get("authors", []),
                                "year": s.get("year", ""),
                                "url": s.get("url", ""),
                                "abstract": s.get("abstract", ""),
                                "citation_id": s.get("citation_id", ""),
                                "source": s.get("source", ""),
                                "arxiv_id": s.get("arxiv_id", "")
                            })
                    except Exception as e:
                        print(f"Error reading papers from {lit_type}: {e}")
            status_data["papers"] = papers_list

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status_data).encode('utf-8'))
            return

        elif self.path.startswith("/api/report"):
            parsed_url = urllib.parse.urlparse(self.path)
            slug = urllib.parse.parse_qs(parsed_url.query).get("slug", [None])[0]
            report = build_complete_report(slug)
            if report is None:
                self.send_error(404, "Run report not found")
                return
            filename = f"{slug}-complete-report.md"
            self.send_response(200)
            self.send_header("Content-Type", "text/markdown; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(report.encode("utf-8"))
            return

        elif self.path == "/api/config":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"models": detected_models()}).encode("utf-8"))
            return

        elif self.path == "/api/previous-runs":
            runs = []
            if REPORTS_DIR.exists():
                for item in REPORTS_DIR.iterdir():
                    if item.is_dir() and (item / "study_result.json").exists():
                        try:
                            res = json.loads((item / "study_result.json").read_text(encoding="utf-8"))
                            question = res.get("lab_question", {}).get("question", item.name)
                            runs.append({
                                "slug": item.name,
                                "question": question,
                                "status": "Archived",
                                "report_path": res.get("final_report", "")
                            })
                        except Exception as e:
                            print(f"Error parsing previous run {item.name}: {e}")
                            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(runs).encode('utf-8'))
            return

        # Serve static file normally
        super().do_GET()

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, LuckyLoopHandler)
    print(f"Lucky Loop UI Server running on http://localhost:{port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
