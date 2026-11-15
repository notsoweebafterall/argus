"""Unit tests for Argus ML Agent."""

import json
import os
import re
from typing import Any
import pytest

from agents.contracts import AgentInput
from agents.llm import LLMMessage, LLMResponse, MockLLMClient, ToolCallRequest
from agents.ml_agent import MLAgent
from tools.data_tools import load_dataset
from tools.ml_tools import create_features


@pytest.fixture(scope="module")
def feature_paths():
    """Fixture providing created train and eval feature set paths."""
    load_res = load_dataset("base")
    feat_res = create_features(dataset_path=load_res["path"])
    return feat_res


def test_ml_agent_standalone_multi_candidate_selection(feature_paths):
    """Verify MLAgent executes multi-candidate comparison, search, and justified model selection."""
    saved_state: dict[str, Any] = {
        "models": {},
        "evals": {},
    }

    def dynamic_handler(messages: list[LLMMessage], tools: list[dict[str, Any]] | None) -> LLMResponse:
        last_msg = messages[-1]

        if last_msg.role == "user":
            # Turn 1: Train baseline Logistic Regression
            return LLMResponse(
                tool_calls=[
                    ToolCallRequest(
                        id="call_train_lr",
                        name="train_model",
                        arguments={
                            "feature_set_path": feature_paths["train_set_path"],
                            "model_type": "logistic_regression",
                            "hyperparams": {"max_iter": 50},
                        },
                    )
                ]
            )

        if last_msg.role == "tool":
            if last_msg.name == "train_model":
                data = json.loads(last_msg.content or "{}")
                saved_state["models"]["lr"] = data
                # Turn 2: Evaluate Logistic Regression
                return LLMResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="call_eval_lr",
                            name="evaluate_model",
                            arguments={
                                "model_id": data["model_id"],
                                "eval_set_path": feature_paths["eval_set_path"],
                            },
                        )
                    ]
                )

            elif last_msg.name == "evaluate_model" and "lr" in saved_state["models"] and "lr" not in saved_state["evals"]:
                data = json.loads(last_msg.content or "{}")
                saved_state["evals"]["lr"] = data.get("scores", {})
                # Turn 3: Run hyperparameter search on XGBoost candidate
                return LLMResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="call_search_xgb",
                            name="search_hyperparams",
                            arguments={
                                "feature_set_path": feature_paths["train_set_path"],
                                "model_type": "xgboost",
                                "param_grid": {"n_estimators": [10], "max_depth": [3, 5]},
                            },
                        )
                    ]
                )

            elif last_msg.name == "search_hyperparams":
                data = json.loads(last_msg.content or "{}")
                saved_state["models"]["xgb"] = data
                # Turn 4: Evaluate winning searched XGBoost model on eval set
                return LLMResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="call_eval_xgb",
                            name="evaluate_model",
                            arguments={
                                "model_id": data["model_id"],
                                "eval_set_path": feature_paths["eval_set_path"],
                            },
                        )
                    ]
                )

            elif last_msg.name == "evaluate_model" and "xgb" in saved_state["models"]:
                data = json.loads(last_msg.content or "{}")
                saved_state["evals"]["xgb"] = data.get("scores", {})

                lr_scores = saved_state["evals"]["lr"]
                xgb_scores = saved_state["evals"]["xgb"]
                xgb_train = saved_state["models"]["xgb"]

                candidates = [
                    {
                        "model_type": "logistic_regression",
                        "eval_metrics": lr_scores,
                        "hyperparams": {"max_iter": 50},
                    },
                    {
                        "model_type": "xgboost",
                        "eval_metrics": xgb_scores,
                        "hyperparams": xgb_train.get("best_hyperparams", {}),
                    },
                ]

                selection_reason = (
                    f"XGBoost selected with PR-AUC of {xgb_scores.get('pr_auc', 0.0):.4f} "
                    f"and recall@5%FPR of {xgb_scores.get('recall_at_fixed_fpr', 0.0):.4f}, "
                    f"outperforming Logistic Regression PR-AUC of {lr_scores.get('pr_auc', 0.0):.4f} "
                    f"and recall@5%FPR of {lr_scores.get('recall_at_fixed_fpr', 0.0):.4f}."
                )

                return LLMResponse(
                    content=json.dumps({
                        "status": "success",
                        "rationale": (
                            f"Evaluated multiple candidates: Logistic Regression and XGBoost. "
                            f"XGBoost demonstrated superior operating trade-offs for fraud detection. "
                            f"{selection_reason}"
                        ),
                        "artifact": {
                            "model_id": xgb_train["model_id"],
                            "model_path": xgb_train["model_path"],
                            "model_type": "xgboost",
                            "train_metrics": xgb_train["train_metrics"],
                            "eval_metrics": xgb_scores,
                            "candidates_considered": candidates,
                            "selection_reason": selection_reason,
                        },
                    })
                )

        return LLMResponse(content="ML Agent fallback")

    mock_llm = MockLLMClient(handler=dynamic_handler)
    agent = MLAgent(llm_client=mock_llm)

    output = agent.run_stage(task_id="test-ml-multi-candidate", feature_artifact=feature_paths)

    assert output.status == "success"
    assert len(output.tool_calls) == 4
    assert output.artifact is not None

    artifact = output.artifact
    assert artifact["model_type"] == "xgboost"
    assert os.path.exists(artifact["model_path"])

    # 1. Assert candidates_considered has more than one entry
    candidates = artifact.get("candidates_considered", [])
    assert len(candidates) > 1, f"Expected >1 candidates_considered, got {len(candidates)}"
    candidate_types = [c["model_type"] for c in candidates]
    assert "logistic_regression" in candidate_types
    assert "xgboost" in candidate_types

    # 2. Assert selection_reason is present and grounded in actual metric numbers
    selection_reason = artifact.get("selection_reason", "")
    assert len(selection_reason) > 0, "selection_reason must not be empty"

    # Verify selection_reason contains numeric decimal values matching metrics
    numbers_in_reason = re.findall(r"\d+\.\d+", selection_reason)
    assert len(numbers_in_reason) >= 1, (
        f"selection_reason must contain formatted metric numbers, got: '{selection_reason}'"
    )

    # Verify at least one actual metric value from artifact eval_metrics is referenced
    xgb_pr_auc_str = f"{artifact['eval_metrics']['pr_auc']:.4f}"
    xgb_recall_fpr_str = f"{artifact['eval_metrics']['recall_at_fixed_fpr']:.4f}"
    assert (
        xgb_pr_auc_str in selection_reason
        or xgb_recall_fpr_str in selection_reason
        or str(artifact["eval_metrics"]["pr_auc"]) in selection_reason
        or str(artifact["eval_metrics"]["recall_at_fixed_fpr"]) in selection_reason
    ), f"selection_reason must cite actual eval metrics from artifact, got: '{selection_reason}'"
