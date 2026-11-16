"""Deterministic pipeline handler for the Streamlit demo."""
from __future__ import annotations

import json
import os
import pandas as pd
from typing import Any

from agents.llm import LLMMessage, LLMResponse, ToolCallRequest

_state: dict[str, Any] = {}


def reset() -> None:
    _state.clear()


def _variant_key(raw: str) -> str:
    return raw.replace("Variant", "").strip().lower() or "base"


def _ensure_dummy_data() -> str:
    """Creates a tiny fake BAF CSV for Streamlit Cloud if the real one is missing."""
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "dummy_baf.csv")
    if not os.path.exists(path):
        # Create a tiny 100-row dataset that looks like BAF
        df = pd.DataFrame({
            "fraud_bool": [0] * 90 + [1] * 10,
            "month": [0] * 50 + [7] * 50,
            "income": [50000.0] * 100,
            "customer_age": [30] * 50 + [60] * 50,
            "feature_1": [0.5] * 100
        })
        df.to_csv(path, index=False)
    return path


def demo_pipeline_handler(messages: list[LLMMessage], tools: list[dict[str, Any]] | None) -> LLMResponse:
    sys = (messages[0].content or "") if messages else ""
    last = messages[-1] if messages else None

    # ---- Data Agent ----
    if "You are the Argus Data Agent" in sys:
        if last.role == "user":
            # If the real data folder is missing (like on Cloud), use the dummy data
            try:
                from tools.data_tools import load_dataset
                real_path = load_dataset(variant=_variant_key(_state.get("variant", "Base")))["path"]
            except Exception:
                real_path = _ensure_dummy_data()

            _state["dataset_path"] = real_path
            return LLMResponse(tool_calls=[
                ToolCallRequest(id="v", name="validate_dataset", arguments={"dataset_path": _state["dataset_path"]})])

        if last.role == "tool" and last.name == "validate_dataset":
            return LLMResponse(tool_calls=[
                ToolCallRequest(id="e", name="run_eda", arguments={"dataset_path": _state["dataset_path"]})])

        if last.role == "tool" and last.name == "run_eda":
            return LLMResponse(content=json.dumps({
                "status": "success", "rationale": "Dataset loaded, validated, EDA complete.",
                "artifact": {"dataset_path": _state["dataset_path"], "row_count": 100, "column_count": 5,
                             "validation": {"severity": "ok", "total_rows": 100, "class_balance": {"0": 90, "1": 10},
                                            "sentinel_missing_counts": {}},
                             "eda": {"correlation_with_target": {}, "fraud_rate_by_month": {},
                                     "high_correlation_pairs": []}, "issues_found": []},
            }))

    # ---- Feature Agent ----
    elif "You are the Argus Feature Agent" in sys:
        if last.role == "user":
            return LLMResponse(tool_calls=[
                ToolCallRequest(id="f", name="create_features", arguments={"dataset_path": _state["dataset_path"]})])
        if last.role == "tool" and last.name == "create_features":
            d = json.loads(last.content or "{}")
            _state["train"] = d["train_set_path"]
            _state["eval"] = d["eval_set_path"]
            return LLMResponse(content=json.dumps({
                "status": "success", "rationale": "Temporal split months 0-5 train / 6-7 eval.",
                "artifact": {"feature_set_path": d["feature_set_path"], "train_set_path": d["train_set_path"],
                             "eval_set_path": d["eval_set_path"], "manifest": d["manifest"],
                             "feature_count": d["manifest"]["feature_count"]},
            }))

    # ---- ML Agent (Bypassed for UI Demo) ----
    elif "You are the Argus ML Agent" in sys:
        if last.role == "user":
            scores = {"precision": 0.45, "recall": 0.60, "pr_auc": 0.85, "recall_at_fixed_fpr": 0.55}
            roc_curve = {"fpr": [0.0, 0.05, 0.2, 0.5, 1.0], "tpr": [0.0, 0.6, 0.8, 0.9, 1.0]}
            confusion_matrix = [[950, 50], [40, 60]]
            reason = f"Selected random_forest with eval PR-AUC of {scores['pr_auc']:.4f} and recall@5%FPR of {scores['recall_at_fixed_fpr']:.4f} on the temporal eval split."

            return LLMResponse(content=json.dumps({
                "status": "success", "rationale": "Trained and evaluated candidate.",
                "artifact": {
                    "model_id": "model_demo_123", "model_path": "dummy_path.joblib", "run_id": "run_demo_123",
                    "model_type": "random_forest",
                    "train_metrics": {"pr_auc": 0.88},
                    "eval_metrics": {**scores, "roc_curve": roc_curve, "confusion_matrix": confusion_matrix},
                    "candidates_considered": [{"model_type": "random_forest", "eval_metrics": scores},
                                              {"model_type": "logistic_regression", "eval_metrics": scores}],
                    "selection_reason": reason,
                },
            }))

    return LLMResponse(content=json.dumps({"status": "success", "rationale": "done", "artifact": {}}))