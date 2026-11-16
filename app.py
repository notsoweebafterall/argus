"""Argus ML — Streamlit product surface (v1 + upload).

Run: streamlit run app.py
"""
from __future__ import annotations

import io
import json
import os
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.orchestrator import run_task
from agents.llm import MockLLMClient

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Argus ML", layout="wide", page_icon="🧠")
st.title("Argus — Autonomous ML Engineering")
st.caption("Upload a dataset, review its profile, and run an end-to-end ML workflow.")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    source = st.radio("Data source", ["BAF variant", "Upload file"], horizontal=True)

    variant = None
    uploaded = None
    if source == "BAF variant":
        variant = st.selectbox("Dataset variant", ["Base", "I", "II", "III", "IV", "V"])
    else:
        uploaded = st.file_uploader(
            "Upload dataset",
            type=["csv", "xlsx", "xls", "json"],
            help="CSV, Excel, or JSON. Generic ingestion is being wired (v2); files are validated and previewed now.",
        )

    use_live_llm = st.checkbox("Use live LLM (Groq)", value=False)
    run_btn = st.button("Run ML Workflow", type="primary", use_container_width=True)
    st.divider()
    st.caption("Past runs: `mlflow ui`.")


def _read_uploaded(uploaded) -> pd.DataFrame | None:
    """Read an uploaded file into a DataFrame, or return None on failure."""
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded)
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded)
        if name.endswith(".json"):
            return pd.read_json(uploaded)
    except Exception as exc:
        st.error(f"Could not parse {uploaded.name}: {exc}")
        return None
    st.error(f"Unsupported file type: {uploaded.name}")
    return None


# ---------------------------------------------------------------------------
# Upload path: validate + preview + save (honest about v2)
# ---------------------------------------------------------------------------
if source == "Upload file" and uploaded is not None:
    st.subheader("Dataset Upload")
    df = _read_uploaded(uploaded)
    if df is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing cells", int(df.isna().sum().sum()))

        st.write("**Preview**")
        st.dataframe(df.head(20), use_container_width=True)

        st.write("**Column profile**")
        profile = pd.DataFrame({
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "nulls": df.isna().sum().values,
            "null_pct": (df.isna().mean() * 100).round(1).values,
            "unique": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(profile, use_container_width=True)

        target = st.selectbox("Confirm target column", df.columns, index=len(df.columns) - 1)
        st.info(
            "Generic ingestion (any CSV → full pipeline) is the v2 build: it needs a "
            "configurable split column and target column in the DataAgent/feature layer. "
            "For now, this validates and profiles your file; run a BAF variant to execute the pipeline."
        )

        if st.button("Save upload for v2 ingestion"):
            dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{uploaded.name}"
            dest.write_bytes(uploaded.getbuffer())
            st.success(f"Saved to {dest}. Target confirmed: `{target}`.")

# ---------------------------------------------------------------------------
# Run path: BAF variant only (until v2 generic ingestion lands)
# ---------------------------------------------------------------------------
if run_btn:
    if source == "Upload file":
        st.warning(
            "Generic CSV execution is v2. Pick a **BAF variant** in the sidebar to run the "
            "pipeline now, or save your upload and come back after v2."
        )
    else:
        with st.status("Running Argus pipeline...", expanded=True) as status:
            st.write("Data validation → Feature engineering → Model training → Critic review → MLOps")
            state = run_task(variant, llm_client=None)
            status.update(
                label=f"Pipeline {state.status}",
                state="complete" if state.status == "completed" else "error",
            )
        st.session_state["state"] = state

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
state = st.session_state.get("state")
if state is None:
    st.info("Pick a BAF variant (or upload to preview) and click **Run ML Workflow**.")
    st.stop()

st.subheader("Execution Summary")
st.metric("Status", state.status.upper())

for rec in state.stage_history:
    icon = {"success": "✅", "failure": "❌", "needs_retry": "⚠️"}.get(rec.output.status, "•")
    with st.expander(f"{icon} {rec.stage} — {rec.output.status}"):
        st.write(rec.output.rationale)
        if rec.critic_verdict:
            st.write(f"**Critic verdict:** `{rec.critic_verdict}`")
        if rec.output.errors:
            st.error("\n".join(rec.output.errors))

critic_rec = next((r for r in state.stage_history if r.stage == "critic_review"), None)
if critic_rec and critic_rec.output.artifact:
    st.subheader("Critic Validation")
    st.write(f"**Verdict:** `{critic_rec.output.artifact.get('verdict')}`")
    failed = critic_rec.output.artifact.get("failed_checks", [])
    if failed:
        st.dataframe(failed, use_container_width=True)
    retrieved = critic_rec.output.artifact.get("retrieved_similar_failures", [])
    if retrieved:
        st.subheader("Similar Past Failures (RAG)")
        for m in retrieved:
            st.write(f"Run `{str(m.get('run_id',''))[:8]}` — similarity {m.get('score',0):.2f}")
            st.json(m.get("failed_checks", []))

report = state.final_report or {}
mlops = report.get("mlops_registration") or {}
model_training = report.get("model_training") or {}
feature_eng = report.get("feature_engineering") or {}

st.subheader("Model & Artifacts")
c1, c2, c3 = st.columns(3)
c1.metric("Model ID", mlops.get("model_id", "—"))
c2.metric("Registry version", mlops.get("version", "—"))
c3.metric("Stage", mlops.get("stage", "—"))

def _dl(label, path, mime, key):
    if path and Path(path).exists():
        st.download_button(label, Path(path).read_bytes(), file_name=Path(path).name,
                           mime=mime, key=key, use_container_width=True)
    else:
        st.caption(f"{label}: not produced")

_dl("Trained model (.joblib)", model_training.get("model_path"), "application/octet-stream", "dl_model")
_dl("Evaluation data (.csv)", feature_eng.get("eval_set_path"), "text/csv", "dl_eval")
st.download_button("Evaluation report (.json)",
                   json.dumps(report, indent=2, default=str).encode(),
                   file_name=f"argus_report_{state.task_id}.json",
                   mime="application/json", key="dl_report", use_container_width=True)