"""Argus ML — Streamlit product surface (v1 + upload + HTML Reports)."""
from __future__ import annotations
import json
import os
import uuid
import base64
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from pipeline.orchestrator import run_task
from agents.llm import MockLLMClient
import app_handlers

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Argus ML", layout="wide", page_icon="🧠")
st.title("Argus — Autonomous ML Engineering")


# ---------------------------------------------------------------------------
# HTML Report Generation Helper
# ---------------------------------------------------------------------------
def generate_html_report(state, report, mlops, model_training, feature_eng, critic_rec):
    eval_metrics = model_training.get("eval_metrics", {})
    pr_auc = eval_metrics.get("pr_auc", 0)
    rec = eval_metrics.get("recall_at_fixed_fpr", 0)

    # Extract plot data from evaluate_model
    cm = eval_metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    roc_data = eval_metrics.get("roc_curve", {"fpr": [], "tpr": []})

    # Generate ROC Chart
    roc_base64 = ""
    if roc_data["fpr"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        roc_auc = auc(roc_data["fpr"], roc_data["tpr"])
        ax.plot(roc_data["fpr"], roc_data["tpr"], color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0]);
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate');
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic');
        ax.legend(loc="lower right")
        buf = BytesIO();
        fig.savefig(buf, format="png", bbox_inches="tight");
        plt.close(fig)
        roc_base64 = base64.b64encode(buf.getvalue()).decode()

    # Generate Confusion Matrix Chart
    cm_base64 = ""
    if cm:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted');
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        buf = BytesIO();
        fig.savefig(buf, format="png", bbox_inches="tight");
        plt.close(fig)
        cm_base64 = base64.b64encode(buf.getvalue()).decode()

    # Build HTML
    html = f"""
    <html>
    <head>
        <title>Argus ML Report - {state.task_id}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; color: #333; }}
            h1, h2 {{ color: #1f77b4; }}
            .card {{ background: #f9f9f9; padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #eee; }}
            .metric {{ font-size: 24px; font-weight: bold; color: #2ca02c; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            .flex {{ display: flex; gap: 40px; }}
        </style>
    </head>
    <body>
        <h1>Argus ML Validation Report</h1>
        <p><strong>Task ID:</strong> {state.task_id} | <strong>Status:</strong> {state.status.upper()}</p>

        <div class="card">
            <h2>Executive Summary</h2>
            <div class="flex">
                <div>
                    <p><strong>Winning Model:</strong> {mlops.get('model_id', 'N/A')} ({model_training.get('model_type', 'N/A')})</p>
                    <p><strong>Critic Verdict:</strong> {critic_rec.output.artifact.get('verdict', 'N/A').upper()}</p>
                    <p><strong>Eval PR-AUC:</strong> <span class="metric">{pr_auc:.4f}</span></p>
                    <p><strong>Recall @ 5% FPR:</strong> <span class="metric">{rec:.4f}</span></p>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Performance Visuals</h2>
            <div class="flex">
                <div><img src="data:image/png;base64,{roc_base64}" style="max-width: 500px;"></div>
                <div><img src="data:image/png;base64,{cm_base64}" style="max-width: 400px;"></div>
            </div>
        </div>

        <div class="card">
            <h2>Critic Validation & RAG</h2>
            <p>The deterministic Critic system issued a <strong>{critic_rec.output.artifact.get('verdict').upper()}</strong> verdict.</p>
    """

    # Add RAG citations
    retrieved = critic_rec.output.artifact.get("retrieved_similar_failures", [])
    if retrieved:
        html += "<h3>Similar Past Failures Consulted (RAG):</h3><ul>"
        for m in retrieved:
            html += f"<li>Run <code>{m['run_id'][:8]}</code> (Similarity: {m['score']:.2f})</li>"
        html += "</ul>"

    html += """
        </div>
    </body>
    </html>
    """
    return html.encode("utf-8")


# ---------------------------------------------------------------------------
# Sidebar & Upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    source = st.radio("Data source", ["BAF variant", "Upload file"], horizontal=True)
    variant = None
    uploaded = None
    if source == "BAF variant":
        variant = st.selectbox("Dataset variant", ["Base", "I", "II", "III", "IV", "V"])
    else:
        uploaded = st.file_uploader("Upload dataset", type=["csv"])

    run_btn = st.button("Run ML Workflow", type="primary", use_container_width=True)

if source == "Upload file" and uploaded is not None:
    st.subheader("Dataset Upload")
    df = pd.read_csv(uploaded)
    st.dataframe(df.head(10), use_container_width=True)

    has_baf_schema = {"fraud_bool", "month"}.issubset(set(df.columns))
    if has_baf_schema:
        st.success("BAF schema detected — this upload can run the full pipeline.")
        if st.button("Run pipeline on this upload", type="primary"):
            dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{uploaded.name}"
            dest.write_bytes(uploaded.getbuffer())
            app_handlers.reset()
            app_handlers._state["upload_path"] = str(dest.resolve())

            with st.status("Running Argus pipeline...", expanded=True) as status:
                state = run_task("uploaded", llm_client=MockLLMClient(handler=app_handlers.demo_pipeline_handler))
                status.update(label=f"Pipeline {state.status}", state="complete")
            st.session_state["state"] = state
            st.rerun()
    else:
        st.warning("Missing 'fraud_bool' and 'month'. Generic ingestion is coming in v2.")

# ---------------------------------------------------------------------------
# Run Standard BAF Variant
# ---------------------------------------------------------------------------
if run_btn and source == "BAF variant":
    app_handlers.reset()
    app_handlers._state["variant"] = variant
    with st.status("Running Argus pipeline...", expanded=True) as status:
        state = run_task(variant, llm_client=MockLLMClient(handler=app_handlers.demo_pipeline_handler))
        status.update(label=f"Pipeline {state.status}", state="complete")
    st.session_state["state"] = state

# ---------------------------------------------------------------------------
# Results & Download
# ---------------------------------------------------------------------------
state = st.session_state.get("state")
if state is None:
    st.stop()

critic_rec = next((r for r in state.stage_history if r.stage == "critic_review"), None)
report = state.final_report or {}
mlops = report.get("mlops_registration") or {}
model_training = report.get("model_training") or {}
feature_eng = report.get("feature_engineering") or {}

st.subheader("Results")
col1, col2, col3 = st.columns(3)
col1.metric("Model ID", mlops.get("model_id", "—"))
col2.metric("PR-AUC", model_training.get("eval_metrics", {}).get("pr_auc", "—"))
col3.metric("Critic Verdict", critic_rec.output.artifact.get("verdict", "—") if critic_rec else "—")

st.subheader("Downloads")
st.write("Generate a comprehensive HTML report containing charts, metrics, and validation results.")
html_report = generate_html_report(state, report, mlops, model_training, feature_eng, critic_rec)
st.download_button(
    label="📄 Download Full HTML Report",
    data=html_report,
    file_name=f"Argus_Report_{state.task_id}.html",
    mime="text/html",
    type="primary",
    use_container_width=True
)

# Raw artifacts
st.write("Raw Artifacts:")
if model_training.get("model_path") and Path(model_training["model_path"]).exists():
    st.download_button("Trained Model (.joblib)", Path(model_training["model_path"]).read_bytes(),
                       file_name="model.joblib")