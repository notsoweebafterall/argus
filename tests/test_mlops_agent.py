"""Unit tests for Argus MLOps Agent."""

from unittest.mock import patch
import pytest
from agents.contracts import AgentInput
from agents.mlops_agent import MLOpsAgent
from tools.data_tools import load_dataset
from tools.ml_tools import create_features, train_model


@pytest.fixture(scope="module")
def base_dataset_path():
    res = load_dataset("base")
    return res["path"]


def test_mlops_agent_happy_path(base_dataset_path):
    """Verify MLOpsAgent registers approved model and returns structured registration metadata."""
    feat_res = create_features(dataset_path=base_dataset_path)
    train_res = train_model(
        feature_set_path=feat_res["train_set_path"],
        model_type="logistic_regression",
        hyperparams={"max_iter": 50},
    )

    mlops = MLOpsAgent()
    agent_input = AgentInput(
        task_id="test-mlops-happy",
        stage="mlops_registration",
        system_prompt=mlops.system_prompt,
        context={
            "model_training_artifact": {
                "model_id": train_res["model_id"],
                "run_id": train_res["run_id"],
            },
            "critic_verdict": "approve",
            "registry_name": "argus_fraud_detector",
        },
    )

    output = mlops.run(agent_input)

    assert output.status == "success"
    assert output.errors == []
    assert output.artifact is not None
    assert output.artifact["action"] == "registered"
    assert output.artifact["model_id"] == train_res["model_id"]
    assert output.artifact["registry_name"] == "argus_fraud_detector"
    assert isinstance(output.artifact["version"], int)
    assert output.artifact["version"] >= 1
    assert output.artifact["stage"] == "Staging"


def test_mlops_agent_defensive_critic_unapproved_rejection():
    """Verify MLOpsAgent defensively fails when Critic verdict is not 'approve'."""
    mlops = MLOpsAgent()

    # Case 1: verdict is 'reject'
    agent_input_reject = AgentInput(
        task_id="test-mlops-defensive-reject",
        stage="mlops_registration",
        system_prompt=mlops.system_prompt,
        context={
            "model_training_artifact": {"model_id": "m1", "run_id": "r1"},
            "critic_verdict": "reject",
        },
    )
    output_reject = mlops.run(agent_input_reject)
    assert output_reject.status == "failure"
    assert "unapproved_model" in output_reject.errors[0]
    assert output_reject.artifact is None

    # Case 2: verdict missing entirely
    agent_input_missing = AgentInput(
        task_id="test-mlops-defensive-missing",
        stage="mlops_registration",
        system_prompt=mlops.system_prompt,
        context={
            "model_training_artifact": {"model_id": "m1", "run_id": "r1"},
        },
    )
    output_missing = mlops.run(agent_input_missing)
    assert output_missing.status == "failure"
    assert "unapproved_model" in output_missing.errors[0]


def test_mlops_agent_missing_model_inputs():
    """Verify MLOpsAgent fails when model_id or run_id is missing from context."""
    mlops = MLOpsAgent()
    agent_input = AgentInput(
        task_id="test-mlops-missing-inputs",
        stage="mlops_registration",
        system_prompt=mlops.system_prompt,
        context={
            "critic_verdict": "approve",
            "model_training_artifact": {},
        },
    )
    output = mlops.run(agent_input)
    assert output.status == "failure"
    assert "missing_registration_inputs" in output.errors[0]

