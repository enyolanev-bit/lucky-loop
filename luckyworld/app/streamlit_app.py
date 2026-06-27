from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="LuckyWorld", layout="wide")
st.title("LuckyWorld")
st.caption("Predict before you compute: world-model-guided autonomous research loops")
run_files = sorted((ROOT / "runs").glob("run_*.json"))
if not run_files:
    st.warning("No runs yet. Run: python -m src.luckyworld.loop")
    st.stop()
traces = [json.loads(p.read_text()) for p in run_files]
rows=[]
for t in traces:
    rows.append({
        "run": t["run_id"],
        "hypothesis": t["hypothesis"],
        "model": t["proposed_action"]["model"],
        "predicted": t["world_model_prediction"]["expected_metric"],
        "actual_accuracy": t["actual_result"].get("accuracy"),
        "runtime_s": t["actual_result"].get("runtime_seconds"),
        "metric_match": t["comparison"].get("metric_match"),
        "verifier": (t.get("verification") or {}).get("status"),
        "effect_size": (t.get("verification") or {}).get("effect_size"),
        "seed_noise": (t.get("verification") or {}).get("seed_noise"),
        "trustworthy": (t.get("verification") or {}).get("trustworthy"),
        "decision": t["next_decision"],
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True)
for t in traces:
    with st.expander(f"{t['run_id']} - {t['proposed_action']['model']}"):
        st.json(t)
report = ROOT / "reports" / "final_report.md"
if report.exists():
    st.markdown(report.read_text())
