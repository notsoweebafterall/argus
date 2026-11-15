"""Unit tests for Argus Feature Agent."""

import json
import os
from typing import Any
import pytest

from agents.contracts import AgentInput
from agents.feature_agent import FeatureAgent
from agents.llm import LLMMessage, LLMResponse, MockLLMClient, ToolCallRequest
from tools.data_tools import load_dataset


@pytest.fixture(scope="module")
def base_dataset_path():
    res = load_dataset("base")
    return res["path"]


def test_feature_agent_standalone_success(base_dataset_path):
    """Verify FeatureAgent standalone execution with temporal split artifact production."""
    data_artifact = {
        "dataset_path": base_dataset_path,
        "row_count": 1000000,
        "column_count": 32,
        "issues_found": ["Severe class imbalance"],
        "eda": {
            "high_correlation_pairs": [
                {"feature_a": "velocity_6h", "feature_b": "velocity_24h", "correlation": 0.95}
            ],
            "correlation_with_target": {},
            "distribution_summary": {},
            "fraud_rate_by_month": {},
        },
    }

    def dynamic_handler(messages: list[LLMMessage], tools: list[dict[str, Any]] | None) -> LLMResponse:
        last_msg = messages[-1]

        if last_msg.role == "user":
            return LLMResponse(
                tool_calls=[
                    ToolCallRequest(
                        id="call_create_feat",
                        name="create_features",
                        arguments={
                            "dataset_path": base_dataset_path,
                            "feature_spec": {
                                "drop_correlated_pairs": [
                                    {"feature_a": "velocity_6h", "feature_b": "velocity_24h"}
                                ]
                            },
                        },
                    )
                ]
            )

        if last_msg.role == "tool" and last_msg.name == "create_features":
            data = json.loads(last_msg.content or "{}")
            manifest = data["manifest"]
            return LLMResponse(
                content=json.dumps({
                    "status": "success",
                    "rationale": (
                        "Temporal split used: months 0-5 train, months 6-7 eval. "
                        "velocity_24h dropped for multicollinearity with velocity_6h (|corr|=0.95). "
                        "'month' confirmed absent from feature set."
                    ),
                    "artifact": {
                        "feature_set_path": data["feature_set_path"],
                        "train_set_path": data["train_set_path"],
                        "eval_set_path": data["eval_set_path"],
                        "manifest": manifest,
                        "feature_count": manifest["feature_count"],
                    },
                })
            )

        return LLMResponse(content="Feature agent fallback")

    mock_llm = MockLLMClient(handler=dynamic_handler)
    agent = FeatureAgent(llm_client=mock_llm)

    output = agent.run_stage(task_id="test-feat-task", data_artifact=data_artifact)

    assert output.status == "success"
    assert len(output.tool_calls) == 1
    assert output.artifact is not None
    assert os.path.exists(output.artifact["train_set_path"])
    assert os.path.exists(output.artifact["eval_set_path"])
    assert output.artifact["feature_count"] > 0

    # Manifest must reflect temporal split
    manifest = output.artifact["manifest"]
    assert manifest["split_method"] == "temporal"
    assert manifest["train_months"] == [0, 1, 2, 3, 4, 5]
    assert manifest["eval_months"] == [6, 7]

    # 'month' must not be in feature columns
    assert "month" not in manifest["feature_columns"], (
        "'month' must not appear as a model input feature"
    )

    # Rationale must mention temporal split
    assert "temporal" in output.rationale.lower() or "month" in output.rationale.lower()


def test_feature_agent_month_in_features_causes_failure(base_dataset_path):
    """Verify FeatureAgent self-check returns failure if 'month' leaks into feature_columns.

    Simulates a scenario where the manifest (somehow) lists 'month' as a feature.
    The _enforce_month_absence post-processing must catch this and flip to failure.
    """
    def handler_with_month_leak(messages: list[LLMMessage], tools: list[dict[str, Any]] | None) -> LLMResponse:
        last_msg = messages[-1]
        if last_msg.role == "user":
            return LLMResponse(
                tool_calls=[
                    ToolCallRequest(
                        id="call_create",
                        name="create_features",
                        arguments={"dataset_path": base_dataset_path},
                    )
                ]
            )
        if last_msg.role == "tool" and last_msg.name == "create_features":
            data = json.loads(last_msg.content or "{}")
            # Tamper: inject 'month' into feature_columns to simulate the leakage scenario
            manifest = dict(data["manifest"])
            manifest["feature_columns"] = manifest["feature_columns"] + ["month"]
            return LLMResponse(
                content=json.dumps({
                    "status": "success",
                    "rationale": "Done.",
                    "artifact": {
                        "feature_set_path": data["feature_set_path"],
                        "train_set_path": data["train_set_path"],
                        "eval_set_path": data["eval_set_path"],
                        "manifest": manifest,
                        "feature_count": manifest["feature_count"],
                    },
                })
            )
        return LLMResponse(content="fallback")

    mock_llm = MockLLMClient(handler=handler_with_month_leak)
    agent = FeatureAgent(llm_client=mock_llm)
    output = agent.run_stage(task_id="test-month-leak", data_artifact={"dataset_path": base_dataset_path})

    assert output.status == "failure", (
        "FeatureAgent must return failure when 'month' is found in feature_columns"
    )
    assert len(output.errors) >= 1
    assert any("leakage_detected" in e for e in output.errors)


def test_feature_agent_high_correlation_pairs_surfaced_in_context(base_dataset_path):
    """Verify high_correlation_pairs from data_artifact['eda'] are surfaced in stage context."""
    hcp = [{"feature_a": "velocity_6h", "feature_b": "velocity_24h", "correlation": 0.95}]
    data_artifact = {
        "dataset_path": base_dataset_path,
        "eda": {"high_correlation_pairs": hcp},
    }

    context_captured: dict[str, Any] = {}

    def capture_handler(messages: list[LLMMessage], tools: list[dict[str, Any]] | None) -> LLMResponse:
        # Capture the system message content (which contains the serialized Execution Context)
        if messages and messages[0].role == "system":
            context_captured["content"] = messages[0].content or ""
        # Return a minimal success to end the run
        last_msg = messages[-1]
        if last_msg.role == "user":
            return LLMResponse(
                tool_calls=[
                    ToolCallRequest(
                        id="c1", name="create_features",
                        arguments={"dataset_path": base_dataset_path}
                    )
                ]
            )
        if last_msg.role == "tool":
            data = json.loads(last_msg.content or "{}")
            manifest = data["manifest"]
            return LLMResponse(
                content=json.dumps({
                    "status": "success",
                    "rationale": "temporal split, month absent",
                    "artifact": {
                        "feature_set_path": data["feature_set_path"],
                        "train_set_path": data["train_set_path"],
                        "eval_set_path": data["eval_set_path"],
                        "manifest": manifest,
                        "feature_count": manifest["feature_count"],
                    },
                })
            )
        return LLMResponse(content="fallback")

    mock_llm = MockLLMClient(handler=capture_handler)
    agent = FeatureAgent(llm_client=mock_llm)
    agent.run_stage(task_id="test-hcp-ctx", data_artifact=data_artifact)

    # high_correlation_pairs must appear somewhere in the serialized context passed to LLM
    assert "high_correlation_pairs" in context_captured.get("content", ""), (
        "high_correlation_pairs must be surfaced in the context passed to the LLM"
    )
