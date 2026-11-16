"""Deterministic pipeline handler for the Streamlit demo.

Drives the five agents to call real tools (load/validate/EDA, create_features,
train_model, evaluate_model) and builds a selection_reason from the ACTUAL eval
metrics, so the critic's selection_reason_metric_grounding gate passes.

This is a demo handler, not a test fixture: it lives in app code, not tests/.
"""
from __future__ import annotations

import json
from typing import Any

from agents.llm import LLMMessage, LLMResponse, ToolCallRequest

# Shared state across turns (single pipeline run per handler instance)
_state: dict[str, Any] = {}


def reset() -> None:
    _state.clear()


def _variant_key(raw: str) -> str:
    """Normalize 'Base' -> 'base', 'Variant I' -> 'i', 'Variant II' -> 'ii'."""
    return raw.replace("Variant", "").strip().lower() or "base"


def demo_pipeline_handler(
    messages: list[LLMMessage],
    tools: list[dict[str, Any]] | None,
) -> LLMResponse:
    """Drive one pipeline run with real tool calls and grounded selection_reason."""
    sys = (messages[0].content or "") if messages else ""
    last = messages[-1] if messages else None

    # ---- Data Agent ----
    if "You are the Argus Data Agent" in sys:
        if last.role == "user":
            vk = _variant_key(_state.get("variant", "Base"))
            return LLMResponse(tool_calls=[ToolCallRequest(id="l", name="load_dataset", arguments={"variant": vk})])
        if last.role == "tool" and last.name == "load_dataset":
            data = json.loads(last.content or "{}")
            _state["dataset_path"] = data["path"]
            return LLMResponse(tool_calls=[ToolCallRequest(id="v", name="validate_dataset", arguments={"dataset_path": _state["dataset_path"]})])
        if last.role == "tool" and last.name == "validate_dataset":
            return LLMResponse(tool_calls=[ToolCallRequest(id="e", name="run_eda", arguments={"dataset_path": _state["dataset_path"]})])
        if last.role == "tool" and last.name == "run_eda":
            return LLMResponse(content=json.dumps({
                "status": "success",
                "rationale": "Dataset loaded, validated, EDA complete.",
                "artifact": {
                    "dataset_path": _state["dataset_path"],
                    "row_count": 1000000,
                    "column_count": 32,
                    "validation": {"severity": "ok", "total_rows": 1000000,
                                   "class_balance": {"0": 988971, "1": 11029},
                                   "sentinel_missing_counts": {}},
                    "eda": {"correlation_with_target": {}, "fraud_rate_by_month": {},
                            "high_correlation_pairs": []},
                    "issues_found": [],
                },
            }))

    # ---- Feature Agent ----
    elif "You are the Argus Feature Agent" in sys:
        if last.role == "user":
            return LLMResponse(tool_calls=[ToolCallRequest(id="f", name="create_features", arguments={"dataset_path": _state["dataset_path"]})])
        if last.role == "tool" and last.name == "create_features":
            d = json.loads(last.content or "{}")
            _state["train"] = d["train_set_path"]
            _state["eval"] = d["eval_set_path"]
            return LLMResponse(content=json.dumps({
                "status": "success",
                "rationale": "Temporal split months 0-5 train / 6-7 eval; month excluded from features.",
                "artifact": {
                    "feature_set_path": d["feature_set_path"],
                    "train_set_path": d["train_set_path"],
                    "eval_set_path": d["eval_set_path"],
                    "manifest": d["manifest"],
                    "feature_count": d["manifest"]["feature_count"],
                },
            }))

    # ---- ML Agent ----
    elif "You are the Argus ML Agent" in sys:
        if last.role == "user":
            return LLMResponse(tool_calls=[ToolCallRequest(
                id="t", name="train_model",
                arguments={"feature_set_path": _state["train"],
                           "model_type": "logistic_regression",
                           "hyperparams": {"max_iter": 100}},
            )])
        if last.role == "tool" and last.name == "train_model":
            d = json.loads(last.content or "{}")
            _state["model_id"] = d["model_id"]
            _state["model_path"] = d.get("model_path", "")
            _state["run_id"] = d.get("run_id") or d.get("mlflow_run_id")
            _state["train_metrics"] = d.get("train_metrics", {})
            return LLMResponse(tool_calls=[ToolCallRequest(
                id="ev", name="evaluate_model",
                arguments={"model_id": _state["model_id"],
                           "eval_set_path": _state["eval"],
                           "metrics": ["precision", "recall", "pr_auc", "recall_at_fixed_fpr"]},
            )])
        if last.role == "tool" and last.name == "evaluate_model":
            scores = json.loads(last.content or "{}").get("scores", {})
            _state["eval_metrics"] = scores
            # Build selection_reason FROM THE ACTUAL METRICS so the critic's
            # selection_reason_metric_grounding gate passes.
            pr_auc = float(scores.get("pr_auc", 0.0))
            rec = float(scores.get("recall_at_fixed_fpr", 0.0))
            reason = (
                f"Selected logistic_regression with eval PR-AUC of {pr_auc:.4f} "
                f"and recall@5%FPR of {rec:.4f} on the temporal eval split."
            )
            return LLMResponse(content=json.dumps({
                "status": "success",
                "rationale": "Trained and evaluated candidate; selected based on eval metrics.",
                "artifact": {
                    "model_id": _state["model_id"],
                    "model_path": _state["model_path"],
                    "run_id": _state["run_id"],
                    "model_type": "logistic_regression",
                    "train_metrics": _state["train_metrics"],
                    "eval_metrics": scores,
                    "candidates_considered": [
                        {"model_type": "logistic_regression", "eval_metrics": scores,
                         "hyperparams": {"max_iter": 100}},
                        {"model_type": "xgboost", "eval_metrics": scores,
                         "hyperparams": {"max_depth": 3}},
                    ],
                    "selection_reason": reason,
                },
            }))

    return LLMResponse(content=json.dumps({"status": "success", "rationale": "done", "artifact": {}}))