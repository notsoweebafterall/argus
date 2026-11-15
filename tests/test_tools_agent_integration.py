"""Integration tests running full Agent workflows with real tools on real BAF data."""

import json
from typing import Any
import pytest

from agents.base import Agent
from agents.contracts import AgentInput
from agents.llm import LLMMessage, LLMResponse, MockLLMClient, ToolCallRequest
from tools import ALL_TOOLS


def test_agent_workflow_prompt_1_class_balance():
    """Manual Prompt 1: 'Load the BAF base variant and tell me the class balance.'

    Workflow:
    Turn 1: Agent calls load_dataset(variant='base')
    Turn 2: Agent calls inspect_dataset(dataset_path=...)
    Turn 3: Agent completes with class balance analysis in rationale and artifact.
    """
    turn1_response = LLMResponse(
        tool_calls=[
            ToolCallRequest(
                id="call_load_base",
                name="load_dataset",
                arguments={"variant": "base"},
            )
        ]
    )

    def dynamic_llm_handler(messages: list[LLMMessage], tools: list[dict[str, Any]] | None) -> LLMResponse:
        last_msg = messages[-1]
        if last_msg.role == "tool" and last_msg.name == "load_dataset":
            data = json.loads(last_msg.content or "{}")
            path = data.get("path")
            return LLMResponse(
                tool_calls=[
                    ToolCallRequest(
                        id="call_inspect_base",
                        name="inspect_dataset",
                        arguments={"dataset_path": path},
                    )
                ]
            )
        elif last_msg.role == "tool" and last_msg.name == "inspect_dataset":
            data = json.loads(last_msg.content or "{}")
            cb = data.get("class_balance", {})
            return LLMResponse(
                content=json.dumps({
                    "status": "success",
                    "artifact": {"class_balance": cb, "total_rows": data.get("row_count")},
                    "rationale": f"Loaded BAF base variant and extracted class balance: {cb}",
                })
            )
        return turn1_response

    mock_llm = MockLLMClient(handler=dynamic_llm_handler)
    agent = Agent(llm_client=mock_llm)

    agent_input = AgentInput(
        task_id="task-prompt-1",
        stage="data_inspection",
        system_prompt="You are an autonomous data agent for BAF analysis.",
        available_tools=ALL_TOOLS,
        context={"prompt": "Load the BAF base variant and tell me the class balance."},
    )

    output = agent.run(agent_input)

    assert output.status == "success"
    assert "class balance" in output.rationale.lower()
    assert len(output.tool_calls) == 2
    assert output.tool_calls[0].tool_name == "load_dataset"
    assert output.tool_calls[0].output["rows"] > 0
    assert output.tool_calls[1].tool_name == "inspect_dataset"
    assert "0" in output.tool_calls[1].output["class_balance"]["counts"]
    assert "1" in output.tool_calls[1].output["class_balance"]["counts"]
    assert output.artifact is not None
    assert "class_balance" in output.artifact


def test_agent_workflow_prompt_2_train_and_evaluate():
    """Manual Prompt 2:

    'Load the base variant, create a simple feature set using all numeric columns,
    train a logistic regression, and report precision, recall, and PR-AUC.'

    Workflow:
    Turn 1: load_dataset(variant='base')
    Turn 2: create_features(dataset_path=...)
    Turn 3: train_model(feature_set_path=train_set_path, model_type='logistic_regression')
    Turn 4: evaluate_model(model_id=..., eval_set_path=eval_set_path, metrics=['precision', 'recall', 'pr_auc'])
    Turn 5: Output final evaluation metrics.
    """
    saved_state: dict[str, Any] = {}

    def dynamic_llm_handler(messages: list[LLMMessage], tools: list[dict[str, Any]] | None) -> LLMResponse:
        last_msg = messages[-1]

        if last_msg.role == "user":
            return LLMResponse(
                tool_calls=[
                    ToolCallRequest(
                        id="step_1_load",
                        name="load_dataset",
                        arguments={"variant": "base"},
                    )
                ]
            )

        if last_msg.role == "tool":
            if last_msg.name == "load_dataset":
                data = json.loads(last_msg.content or "{}")
                saved_state["dataset_path"] = data["path"]
                return LLMResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="step_2_features",
                            name="create_features",
                            arguments={"dataset_path": saved_state["dataset_path"]},
                        )
                    ]
                )
            elif last_msg.name == "create_features":
                data = json.loads(last_msg.content or "{}")
                saved_state["train_set_path"] = data.get("train_set_path", data.get("feature_set_path"))
                saved_state["eval_set_path"] = data.get("eval_set_path", data.get("feature_set_path"))
                return LLMResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="step_3_train",
                            name="train_model",
                            arguments={
                                "feature_set_path": saved_state["train_set_path"],
                                "model_type": "logistic_regression",
                                "hyperparams": {"max_iter": 50},
                            },
                        )
                    ]
                )
            elif last_msg.name == "train_model":
                data = json.loads(last_msg.content or "{}")
                saved_state["model_id"] = data["model_id"]
                return LLMResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="step_4_eval",
                            name="evaluate_model",
                            arguments={
                                "model_id": saved_state["model_id"],
                                "eval_set_path": saved_state["eval_set_path"],
                                "metrics": ["precision", "recall", "pr_auc"],
                            },
                        )
                    ]
                )
            elif last_msg.name == "evaluate_model":
                data = json.loads(last_msg.content or "{}")
                scores = data.get("scores", {})
                return LLMResponse(
                    content=json.dumps({
                        "status": "success",
                        "artifact": {
                            "model_id": saved_state.get("model_id"),
                            "metrics": scores,
                        },
                        "rationale": (
                            f"Model trained on 80% train split and evaluated on 20% eval split of BAF base variant. "
                            f"Metrics achieved: Precision={scores.get('precision')}, "
                            f"Recall={scores.get('recall')}, PR-AUC={scores.get('pr_auc')}."
                        ),
                    })
                )

        return LLMResponse(content="Unexpected turn state")

    mock_llm = MockLLMClient(handler=dynamic_llm_handler)
    agent = Agent(llm_client=mock_llm)

    agent_input = AgentInput(
        task_id="task-prompt-2",
        stage="model_training_and_eval",
        system_prompt="You are an autonomous ML engineer agent.",
        available_tools=ALL_TOOLS,
        context={
            "prompt": "Load the base variant, create a simple feature set using all numeric columns, train a logistic regression, and report precision, recall, and PR-AUC."
        },
    )

    output = agent.run(agent_input)

    assert output.status == "success"
    assert "precision=" in output.rationale.lower() or "pr-auc=" in output.rationale.lower()
    assert len(output.tool_calls) == 4

    # Verify tool call sequence
    tool_sequence = [tc.tool_name for tc in output.tool_calls]
    assert tool_sequence == ["load_dataset", "create_features", "train_model", "evaluate_model"]

    for tc in output.tool_calls:
        assert tc.error is None
        assert tc.duration_ms >= 0.0

    assert output.artifact is not None
    assert "metrics" in output.artifact
    assert "pr_auc" in output.artifact["metrics"]
