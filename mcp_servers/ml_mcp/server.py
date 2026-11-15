"""ML MCP Server for Argus ML with MLflow Experiment Tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import mlflow
from mcp.server.mcpserver import MCPServer
from pipeline.config import settings
from tools.ml_tools import (
    check_fairness,
    create_features,
    evaluate_model,
    load_model,
    register_model,
    save_model,
    search_hyperparams,
    train_model,
)

ml_mcp_server = MCPServer("ml_mcp")

# Configure MLflow tracking
DB_PATH = settings.workspace_root / "experiments" / "mlflow.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
TRACKING_URI = f"sqlite:///{DB_PATH.as_posix()}"
EXPERIMENT_NAME = "argus_fraud_detection"

# Registry mapping model_id to mlflow run_id
_MODEL_RUN_MAP: dict[str, str] = {}


def _init_mlflow() -> None:
    """Initialize MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


@ml_mcp_server.tool(
    name="create_features",
    description="Creates a tabular feature set from dataset using a fixed temporal train/eval split (months 0-5 train, 6-7 eval).",
)
def mcp_create_features(
    dataset_path: str,
    feature_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """MCP tool wrapper for create_features."""
    return create_features(
        dataset_path=dataset_path,
        feature_spec=feature_spec,
    )


@ml_mcp_server.tool(
    name="train_model",
    description="Trains a machine learning model and logs parameters and metrics to MLflow.",
)
def mcp_train_model(
    feature_set_path: str,
    model_type: str,
    hyperparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """MCP tool wrapper for train_model with automated MLflow experiment tracking."""
    _init_mlflow()
    result = train_model(
        feature_set_path=feature_set_path,
        model_type=model_type,
        hyperparams=hyperparams,
    )

    model_id = result["model_id"]
    train_metrics = result.get("train_metrics", {})
    params = hyperparams or {}

    try:
        with mlflow.start_run(run_name=f"train_{model_id}") as run:
            run_id = run.info.run_id
            _MODEL_RUN_MAP[model_id] = run_id

            # Log parameters
            mlflow.log_param("model_id", model_id)
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("feature_set_path", feature_set_path)
            for k, v in params.items():
                mlflow.log_param(f"param_{k}", v)

            # Log metrics
            for metric_name, val in train_metrics.items():
                mlflow.log_metric(metric_name, float(val))

            # Set tags
            mlflow.set_tag("pipeline_stage", "model_training")
            mlflow.set_tag("model_id", model_id)

            result["mlflow_run_id"] = run_id
    except Exception:
        # Fallback without failing tool execution if MLflow server has an issue
        pass

    return result


@ml_mcp_server.tool(
    name="search_hyperparams",
    description="Searches hyperparameters for tree-based models ('random_forest', 'xgboost') and logs winning parameters and scores to MLflow.",
)
def mcp_search_hyperparams(
    feature_set_path: str,
    model_type: str,
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """MCP tool wrapper for search_hyperparams with MLflow experiment tracking."""
    _init_mlflow()
    result = search_hyperparams(
        feature_set_path=feature_set_path,
        model_type=model_type,
        param_grid=param_grid,
    )

    model_id = result["model_id"]
    train_metrics = result.get("train_metrics", {})
    best_params = result.get("best_hyperparams", {})

    try:
        with mlflow.start_run(run_name=f"search_{model_id}") as run:
            run_id = run.info.run_id
            _MODEL_RUN_MAP[model_id] = run_id

            mlflow.log_param("model_id", model_id)
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("feature_set_path", feature_set_path)
            mlflow.log_param("search_mode", "hyperparam_search")
            for k, v in best_params.items():
                mlflow.log_param(f"best_param_{k}", v)

            mlflow.log_metric("search_best_pr_auc", float(result.get("best_score", 0.0)))
            for metric_name, val in train_metrics.items():
                mlflow.log_metric(metric_name, float(val))

            mlflow.set_tag("pipeline_stage", "hyperparameter_search")
            mlflow.set_tag("model_id", model_id)

            result["mlflow_run_id"] = run_id
    except Exception:
        pass

    return result


@ml_mcp_server.tool(
    name="evaluate_model",
    description="Evaluates a trained model on evaluation features and logs scores to MLflow.",
)
def mcp_evaluate_model(
    model_id: str,
    eval_set_path: str,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """MCP tool wrapper for evaluate_model with MLflow evaluation tracking."""
    _init_mlflow()
    result = evaluate_model(
        model_id=model_id,
        eval_set_path=eval_set_path,
        metrics=metrics,
    )

    scores = result.get("scores", {})
    run_id = _MODEL_RUN_MAP.get(model_id)

    if not run_id:
        # Search MLflow for run by model_id tag if not in local map
        try:
            experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"tags.model_id = '{model_id}'",
                )
                if not runs.empty:
                    run_id = runs.iloc[0]["run_id"]
        except Exception:
            pass

    if run_id:
        try:
            with mlflow.start_run(run_id=run_id):
                for metric_name, val in scores.items():
                    mlflow.log_metric(f"eval_{metric_name}", float(val))
            result["mlflow_run_id"] = run_id
        except Exception:
            pass

    return result


@ml_mcp_server.tool(
    name="check_fairness",
    description="Evaluates algorithmic fairness across customer age groups and logs predictive equality metrics to MLflow.",
)
def mcp_check_fairness(
    model_id: str,
    eval_set_path: str,
) -> dict[str, Any]:
    """MCP tool wrapper for check_fairness with MLflow evaluation tracking."""
    _init_mlflow()
    result = check_fairness(
        model_id=model_id,
        eval_set_path=eval_set_path,
    )

    ratio = result.get("predictive_equality_ratio")
    fpr_y = result.get("fpr_younger")
    fpr_o = result.get("fpr_older")
    run_id = _MODEL_RUN_MAP.get(model_id)

    if run_id and ratio is not None:
        try:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("eval_predictive_equality_ratio", float(ratio))
                if fpr_y is not None:
                    mlflow.log_metric("eval_fpr_younger", float(fpr_y))
                if fpr_o is not None:
                    mlflow.log_metric("eval_fpr_older", float(fpr_o))
        except Exception:
            pass

    return result


@ml_mcp_server.tool(
    name="save_model",
    description="Saves a trained model to a designated destination file path.",
)
def mcp_save_model(model_id: str, destination: str) -> dict[str, Any]:
    """MCP tool wrapper for save_model."""
    return save_model(model_id=model_id, destination=destination)


@ml_mcp_server.tool(
    name="load_model",
    description="Retrieves a model artifact path and its metadata by model ID.",
)
def mcp_load_model(model_id: str) -> dict[str, Any]:
    """MCP tool wrapper for load_model."""
    return load_model(model_id=model_id)


@ml_mcp_server.tool(
    name="register_model",
    description="Registers an approved model version in MLflow Model Registry in stage 'Staging'.",
)
def mcp_register_model(
    model_id: str,
    run_id: str,
    registry_name: str = "argus_fraud_detector",
) -> dict[str, Any]:
    """MCP tool wrapper for register_model."""
    return register_model(
        model_id=model_id,
        run_id=run_id,
        registry_name=registry_name,
    )


if __name__ == "__main__":
    ml_mcp_server.run(transport="stdio")
