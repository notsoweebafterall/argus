"""Direct unit tests for ML MCP Server and MLflow Experiment Tracking."""

import asyncio
import json
import os
import mlflow
import pytest
from mcp_servers.ml_mcp.server import EXPERIMENT_NAME, TRACKING_URI, ml_mcp_server
from tools.data_tools import load_dataset


def _run_async(coro):
    return asyncio.run(coro)


@pytest.fixture(scope="module")
def base_dataset_path():
    res = load_dataset("base")
    return res["path"]


def test_ml_mcp_create_features(base_dataset_path):
    """Verify create_features executes through ML MCP Server."""
    result = _run_async(
        ml_mcp_server.call_tool(
            "create_features",
            {"dataset_path": base_dataset_path},
        )
    )
    assert not result.is_error
    data = json.loads(result.content[0].text)
    assert os.path.exists(data["train_set_path"])
    assert os.path.exists(data["eval_set_path"])
    assert data["manifest"]["target_column"] == "fraud_bool"


def test_ml_mcp_train_and_evaluate_with_mlflow_tracking(base_dataset_path):
    """Verify train_model and evaluate_model through ML MCP log parameters, metrics, and run_id to MLflow."""
    # 1. Create features
    feat_res = _run_async(
        ml_mcp_server.call_tool(
            "create_features",
            {"dataset_path": base_dataset_path},
        )
    )
    feat_data = json.loads(feat_res.content[0].text)

    # 2. Train model via MCP
    train_res = _run_async(
        ml_mcp_server.call_tool(
            "train_model",
            {
                "feature_set_path": feat_data["train_set_path"],
                "model_type": "logistic_regression",
                "hyperparams": {"max_iter": 100},
            },
        )
    )
    assert not train_res.is_error
    train_data = json.loads(train_res.content[0].text)
    assert "model_id" in train_data
    assert "mlflow_run_id" in train_data
    model_id = train_data["model_id"]
    run_id = train_data["mlflow_run_id"]

    # 3. Evaluate model via MCP
    eval_res = _run_async(
        ml_mcp_server.call_tool(
            "evaluate_model",
            {
                "model_id": model_id,
                "eval_set_path": feat_data["eval_set_path"],
                "metrics": ["precision", "recall", "pr_auc"],
            },
        )
    )
    assert not eval_res.is_error
    eval_data = json.loads(eval_res.content[0].text)
    assert "pr_auc" in eval_data["scores"]

    # 4. Assert MLflow tracking actually logged the run
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    assert experiment is not None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.model_id = '{model_id}'",
    )
    assert not runs.empty
    logged_run = runs.iloc[0]
    assert logged_run["run_id"] == run_id
    assert logged_run["params.model_type"] == "logistic_regression"
    assert "metrics.train_accuracy" in logged_run
    assert "metrics.eval_pr_auc" in logged_run


def test_ml_mcp_search_hyperparams(base_dataset_path):
    """Verify search_hyperparams executes via ML MCP and logs search best score to MLflow."""
    feat_res = _run_async(
        ml_mcp_server.call_tool(
            "create_features",
            {"dataset_path": base_dataset_path},
        )
    )
    feat_data = json.loads(feat_res.content[0].text)

    search_res = _run_async(
        ml_mcp_server.call_tool(
            "search_hyperparams",
            {
                "feature_set_path": feat_data["train_set_path"],
                "model_type": "random_forest",
                "param_grid": {"n_estimators": [10], "max_depth": [3, 5]},
            },
        )
    )
    assert not search_res.is_error
    search_data = json.loads(search_res.content[0].text)
    assert "model_id" in search_data
    assert "best_score" in search_data
    assert "mlflow_run_id" in search_data
