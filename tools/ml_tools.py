"""Machine Learning tools for Argus ML."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ImportError:
    xgb = None
    XGBClassifier = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

from agents.contracts import ToolSpec
from pipeline.config import settings

import subprocess
import mlflow

# Experiment directories
EXPERIMENTS_DIR = settings.workspace_root / "experiments"
MODELS_DIR = EXPERIMENTS_DIR / "models"
FEATURES_DIR = settings.workspace_root / "data" / "features"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# MLflow tracking configuration
DB_PATH = EXPERIMENTS_DIR / "mlflow.db"
TRACKING_URI = f"sqlite:///{DB_PATH.as_posix()}"
EXPERIMENT_NAME = "argus_fraud_detection"


def _init_mlflow() -> None:
    """Initialize MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def _get_git_commit() -> str:
    """Retrieve current git HEAD commit hash or 'unknown'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        commit = res.stdout.strip()
        return commit if commit else "unknown"
    except Exception:
        return "unknown"


# Fixed temporal split boundaries for BAF (months 0-7).
# Train: months 0-5 (inclusive). Eval: months 6-7.
# These are non-negotiable constants — not parameters the agent picks per run.
TRAIN_MONTHS = range(0, 6)   # months 0, 1, 2, 3, 4, 5
EVAL_MONTHS = range(6, 8)    # months 6, 7


if torch is not None and nn is not None:
    class PyTorchTabularMLP(nn.Module):
        """Small feedforward neural network for tabular fraud classification."""

        def __init__(self, input_dim: int, hidden_dims: list[int] | None = None, dropout: float = 0.1) -> None:
            super().__init__()
            h_dims = hidden_dims or [64, 32]
            layers: list[nn.Module] = []
            prev_dim = input_dim
            for h_dim in h_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x).squeeze(-1)

        def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
            """Predict class probabilities returning shape (N, 2)."""
            self.eval()
            if isinstance(X, pd.DataFrame):
                X_arr = X.values
            else:
                X_arr = np.asarray(X)
            with torch.no_grad():
                tensor_x = torch.tensor(X_arr, dtype=torch.float32)
                logits = self.forward(tensor_x)
                probs_pos = torch.sigmoid(logits).cpu().numpy()
                probs_neg = 1.0 - probs_pos
                return np.column_stack([probs_neg, probs_pos])

        def predict(self, X: np.ndarray | pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
            """Predict binary labels returning shape (N,)."""
            proba = self.predict_proba(X)[:, 1]
            return (proba >= threshold).astype(int)
else:
    PyTorchTabularMLP = None


def create_features(
    dataset_path: str,
    feature_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a tabular feature set from dataset_path using a fixed temporal train/eval split.

    Split rule (non-configurable):
      Train: rows where month in TRAIN_MONTHS (0-5)
      Eval:  rows where month in EVAL_MONTHS  (6-7)

    Leakage checks:
      - 'fraud_bool' in features_requested → raises ValueError("leakage_detected: ...")
      - 'month' in features_requested → raises ValueError("leakage_detected: ...")
        ('month' is the split key and must not be used as a model input feature)

    Multicollinearity reduction:
      If feature_spec contains 'drop_correlated_pairs' (a list of
      {feature_a, feature_b} dicts, e.g. from run_eda's high_correlation_pairs),
      for each pair the second column (feature_b) is dropped from the feature set.
      Dropped columns and the reason are recorded in the manifest.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"invalid_spec: Dataset path '{dataset_path}' does not exist.")

    spec = feature_spec or {}

    # --- Leakage checks (before reading data) ---
    features_requested = spec.get("features") or spec.get("columns") or spec.get("selected_features")
    if features_requested and isinstance(features_requested, list):
        if "fraud_bool" in features_requested:
            raise ValueError("leakage_detected: 'fraud_bool' cannot be used as an input feature.")
        if "month" in features_requested:
            raise ValueError(
                "leakage_detected: 'month' is the temporal split key and cannot be used "
                "as a model input feature."
            )
    elif isinstance(features_requested, str):
        if features_requested == "fraud_bool":
            raise ValueError("leakage_detected: 'fraud_bool' cannot be used as an input feature.")
        if features_requested == "month":
            raise ValueError(
                "leakage_detected: 'month' is the temporal split key and cannot be used "
                "as a model input feature."
            )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"invalid_spec: Failed to read dataset CSV: {exc}") from exc

    if "fraud_bool" not in df.columns:
        raise ValueError("invalid_spec: Dataset missing required target column 'fraud_bool'.")
    if "month" not in df.columns:
        raise ValueError("invalid_spec: Dataset missing required split column 'month'.")

    # --- Feature column selection ---
    if features_requested and isinstance(features_requested, list):
        missing = set(features_requested) - set(df.columns)
        if missing:
            raise ValueError(f"invalid_spec: Requested features do not exist in dataset: {missing}")
        feature_cols = [c for c in features_requested if c not in ("fraud_bool", "month")]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in ("fraud_bool", "month")]

    if not feature_cols:
        raise ValueError("invalid_spec: No valid feature columns identified after filtering.")

    # --- Multicollinearity reduction ---
    drop_pairs: list[dict[str, str]] = spec.get("drop_correlated_pairs") or []
    dropped_columns: list[dict[str, str]] = []
    for pair in drop_pairs:
        col_to_drop = pair.get("feature_b") or pair.get("drop")
        col_kept = pair.get("feature_a") or pair.get("keep")
        if col_to_drop and col_to_drop in feature_cols:
            feature_cols = [c for c in feature_cols if c != col_to_drop]
            dropped_columns.append({
                "column": col_to_drop,
                "reason": f"multicollinearity: |corr| > 0.9 with '{col_kept}'",
            })

    # --- Fill NaN with median for baseline features ---
    feature_df = df[feature_cols].copy()
    for col in feature_cols:
        if feature_df[col].isnull().any():
            median_val = feature_df[col].median()
            feature_df[col] = feature_df[col].fillna(median_val if not pd.isna(median_val) else 0)

    # Assemble output with target and month (month used only for splitting, not as a feature)
    output_df = feature_df.copy()
    output_df["fraud_bool"] = df["fraud_bool"]
    output_df["month"] = df["month"]
    if "customer_age" in df.columns and "customer_age" not in output_df.columns:
        output_df["customer_age"] = df["customer_age"]

    # --- Temporal split (fixed, non-negotiable) ---
    train_month_vals = list(TRAIN_MONTHS)
    eval_month_vals = list(EVAL_MONTHS)
    train_df = output_df[output_df["month"].isin(train_month_vals)].copy()
    eval_df = output_df[output_df["month"].isin(eval_month_vals)].copy()

    if train_df.empty:
        raise ValueError(
            f"invalid_spec: Temporal split produced empty training set. "
            f"Dataset has no rows with month in {train_month_vals}."
        )
    if eval_df.empty:
        raise ValueError(
            f"invalid_spec: Temporal split produced empty eval set. "
            f"Dataset has no rows with month in {eval_month_vals}."
        )

    # Compute training distribution summary and baseline fraud rate for monitoring
    distribution_summary: dict[str, Any] = {}
    for col in feature_cols:
        s = train_df[col]
        quantiles = [round(float(q), 6) for q in s.quantile(np.linspace(0.1, 0.9, 9)).tolist()]
        distribution_summary[col] = {
            "mean": round(float(s.mean()), 6),
            "std": round(float(s.std()), 6),
            "min": round(float(s.min()), 6),
            "max": round(float(s.max()), 6),
            "q25": round(float(s.quantile(0.25)), 6),
            "q50": round(float(s.quantile(0.50)), 6),
            "q75": round(float(s.quantile(0.75)), 6),
            "deciles": quantiles,
        }

    training_fraud_rate = (
        round(float(train_df["fraud_bool"].mean()), 6)
        if "fraud_bool" in train_df.columns
        else 0.011
    )

    feature_id = f"feat_{uuid.uuid4().hex[:8]}"
    train_path = FEATURES_DIR / f"train_{feature_id}.csv"
    eval_path = FEATURES_DIR / f"eval_{feature_id}.csv"

    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    manifest = {
        "feature_id": feature_id,
        "feature_set_path": str(train_path.resolve()),
        "train_set_path": str(train_path.resolve()),
        "eval_set_path": str(eval_path.resolve()),
        "feature_columns": feature_cols,
        "target_column": "fraud_bool",
        "split_method": "temporal",
        "train_months": train_month_vals,
        "eval_months": eval_month_vals,
        "total_rows": len(output_df),
        "train_rows": len(train_df),
        "train_row_count": len(train_df),
        "eval_rows": len(eval_df),
        "eval_row_count": len(eval_df),
        "feature_count": len(feature_cols),
        "dropped_columns": dropped_columns,
        "training_fraud_rate": training_fraud_rate,
        "distribution_summary": distribution_summary,
    }

    manifest_path = FEATURES_DIR / f"{feature_id}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    dist_path = FEATURES_DIR / f"{feature_id}_distribution.json"
    with open(dist_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_id": feature_id,
                "training_fraud_rate": training_fraud_rate,
                "distribution_summary": distribution_summary,
            },
            f,
            indent=2,
        )

    return {
        "feature_set_path": str(train_path.resolve()),
        "train_set_path": str(train_path.resolve()),
        "eval_set_path": str(eval_path.resolve()),
        "manifest": manifest,
    }


def train_model(
    feature_set_path: str,
    model_type: str,
    hyperparams: dict[str, Any] | None = None,
    dataset_variant: str | None = None,
) -> dict[str, Any]:
    """Train a machine learning model on the provided feature set and save unified model+scaler bundle.

    Supported model_type values:
      - 'logistic_regression' (with StandardScaler and max_iter)
      - 'random_forest' (RandomForestClassifier)
      - 'xgboost' (XGBClassifier with scale_pos_weight for class imbalance)
      - 'pytorch_baseline' (Feedforward neural network with BCEWithLogitsLoss pos_weight)
    """
    path = Path(feature_set_path)
    if not path.exists():
        raise FileNotFoundError(f"training_failed: Feature set '{feature_set_path}' not found.")

    m_type = model_type.strip().lower()
    valid_types = {
        "logistic_regression": "logistic_regression",
        "logisticregression": "logistic_regression",
        "lr": "logistic_regression",
        "random_forest": "random_forest",
        "randomforest": "random_forest",
        "rf": "random_forest",
        "xgboost": "xgboost",
        "xgb": "xgboost",
        "pytorch_baseline": "pytorch_baseline",
        "pytorch": "pytorch_baseline",
        "mlp": "pytorch_baseline",
        "torch": "pytorch_baseline",
    }
    if m_type not in valid_types:
        raise ValueError(
            f"training_failed: Unsupported model_type '{model_type}'. "
            f"Supported types: ['logistic_regression', 'random_forest', 'xgboost', 'pytorch_baseline']."
        )

    canonical_type = valid_types[m_type]
    params = dict(hyperparams or {})

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"training_failed: Could not read feature set: {exc}") from exc

    if "fraud_bool" not in df.columns:
        raise ValueError("training_failed: Target column 'fraud_bool' missing from feature set.")

    feature_cols = [c for c in df.columns if c not in ("fraud_bool", "month")]
    X = df[feature_cols].fillna(0)
    y = df["fraud_bool"]

    neg_count = float((y == 0).sum())
    pos_count = float((y == 1).sum())
    class_ratio = (neg_count / pos_count) if pos_count > 0 else 1.0

    scaler: StandardScaler | None = None
    is_torch = False

    try:
        if canonical_type == "logistic_regression":
            if "max_iter" not in params:
                params["max_iter"] = 1000
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = LogisticRegression(**params)
            model.fit(X_scaled, y)
            X_eval_for_train_metrics = X_scaled

        elif canonical_type == "random_forest":
            if "random_state" not in params:
                params["random_state"] = 42
            if "n_estimators" not in params:
                params["n_estimators"] = 100
            model = RandomForestClassifier(**params)
            model.fit(X, y)
            X_eval_for_train_metrics = X

        elif canonical_type == "xgboost":
            if XGBClassifier is None:
                raise RuntimeError("training_failed: xgboost package is not installed.")
            if "scale_pos_weight" not in params:
                params["scale_pos_weight"] = float(round(class_ratio, 2))
            if "random_state" not in params:
                params["random_state"] = 42
            if "eval_metric" not in params:
                params["eval_metric"] = "logloss"
            if "n_estimators" not in params:
                params["n_estimators"] = 100
            model = XGBClassifier(**params)
            model.fit(X, y)
            X_eval_for_train_metrics = X

        elif canonical_type == "pytorch_baseline":
            if torch is None or nn is None or PyTorchTabularMLP is None:
                raise RuntimeError("training_failed: torch package is not installed.")
            is_torch = True
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            epochs = int(params.pop("epochs", 5))
            lr = float(params.pop("lr", 0.001))
            batch_size = int(params.pop("batch_size", 256))
            hidden_dims = params.pop("hidden_dims", [64, 32])
            dropout = float(params.pop("dropout", 0.1))

            model = PyTorchTabularMLP(input_dim=X.shape[1], hidden_dims=hidden_dims, dropout=dropout)
            pos_weight_tensor = torch.tensor([class_ratio], dtype=torch.float32)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            tensor_X = torch.tensor(X_scaled, dtype=torch.float32)
            tensor_y = torch.tensor(y.values, dtype=torch.float32)
            dataset = TensorDataset(tensor_X, tensor_y)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            model.train()
            for _ in range(epochs):
                for bx, by in loader:
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
            model.eval()
            X_eval_for_train_metrics = X_scaled

    except TypeError as exc:
        raise ValueError(f"invalid_hyperparams: Invalid hyperparameters for {model_type}: {exc}") from exc
    except Exception as exc:
        if "invalid_hyperparams" in str(exc):
            raise
        raise RuntimeError(f"training_failed: Model training error for {model_type}: {exc}") from exc

    model_id = f"model_{uuid.uuid4().hex[:8]}"
    model_path = MODELS_DIR / f"{model_id}.joblib"
    meta_path = MODELS_DIR / f"{model_id}_meta.json"

    metadata = {
        "model_id": model_id,
        "model_type": canonical_type,
        "hyperparams": params,
        "feature_columns": feature_cols,
        "feature_set_path": str(path.resolve()),
        "train_rows": len(df),
        "is_torch": is_torch,
    }

    # Save model and scaler together in a single unified bundle dictionary
    bundle = {
        "model": model,
        "scaler": scaler,
        "metadata": metadata,
        "is_torch": is_torch,
    }
    joblib.dump(bundle, model_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Quick train metrics
    y_pred = model.predict(X_eval_for_train_metrics)
    try:
        if hasattr(model, "predict_proba"):
            y_train_proba = model.predict_proba(X_eval_for_train_metrics)[:, 1]
        else:
            y_train_proba = y_pred
        train_prec_c, train_rec_c, _ = precision_recall_curve(y, y_train_proba)
        train_pr_auc_val = float(round(auc(train_rec_c, train_prec_c), 4))
    except Exception:
        train_pr_auc_val = float(round(accuracy_score(y, y_pred), 4))

    train_metrics = {
        "train_rows": len(df),
        "train_accuracy": float(round(accuracy_score(y, y_pred), 4)),
        "pr_auc": train_pr_auc_val,
    }

    # Load manifest details if available
    split_method = "temporal"
    train_months = list(TRAIN_MONTHS)
    eval_months = list(EVAL_MONTHS)

    manifest_file = path.parent / f"{path.stem.replace('train_', '').replace('eval_', '')}_manifest.json"
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                m_data = json.load(f)
                split_method = m_data.get("split_method", split_method)
                train_months = m_data.get("train_months", train_months)
                eval_months = m_data.get("eval_months", eval_months)
        except Exception:
            pass

    # Log to MLflow
    _init_mlflow()
    run_id = ""
    code_ver = _get_git_commit()
    try:
        with mlflow.start_run(run_name=f"train_{model_id}") as run:
            run_id = run.info.run_id

            # Parameters
            mlflow.log_param("model_id", model_id)
            mlflow.log_param("model_type", canonical_type)
            mlflow.log_param("feature_set_path", str(path.resolve()))
            mlflow.log_param("dataset_variant", dataset_variant or "base")
            mlflow.log_param("split_method", split_method)
            mlflow.log_param("train_months", str(train_months))
            mlflow.log_param("eval_months", str(eval_months))
            mlflow.log_param("code_version", code_ver)
            for k, v in params.items():
                mlflow.log_param(f"param_{k}", v)

            # Metrics
            for metric_name, val in train_metrics.items():
                mlflow.log_metric(metric_name, float(val))

            # Tags
            mlflow.set_tag("pipeline_stage", "model_training")
            mlflow.set_tag("model_id", model_id)
            mlflow.set_tag("dataset_variant", dataset_variant or "base")
            mlflow.set_tag("split_method", split_method)
            mlflow.set_tag("train_months", str(train_months))
            mlflow.set_tag("eval_months", str(eval_months))
            mlflow.set_tag("model_type", canonical_type)
            mlflow.set_tag("code_version", code_ver)

            # Log model with MLflow flavor loggers to support Model Registry
            if canonical_type in ("logistic_regression", "random_forest"):
                mlflow.sklearn.log_model(model, artifact_path="model")
            elif canonical_type == "xgboost":
                try:
                    mlflow.xgboost.log_model(model, artifact_path="model")
                except Exception:
                    mlflow.sklearn.log_model(model, artifact_path="model")
            elif canonical_type == "pytorch_baseline":
                try:
                    mlflow.pytorch.log_model(model, artifact_path="model")
                except Exception:
                    mlflow.sklearn.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")
            mlflow.log_artifact(str(model_path), artifact_path="bundle")
    except Exception:
        if not run_id:
            run_id = f"run_{uuid.uuid4().hex[:12]}"

    metadata["run_id"] = run_id
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "model_id": model_id,
        "model_path": str(model_path.resolve()),
        "train_metrics": train_metrics,
        "run_id": run_id,
        "mlflow_run_id": run_id,
    }


def search_hyperparams(
    feature_set_path: str,
    model_type: str,
    param_grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Perform hyperparameter search for tree-based models (random_forest, xgboost).

    Search is strictly evaluated on a held-out slice of the training months (months 0-5),
    never touching the eval set (months 6-7).
    Scores candidates using PR-AUC on the training holdout slice.
    Trains the final winning model on the full training set.
    """
    path = Path(feature_set_path)
    if not path.exists():
        raise FileNotFoundError(f"search_failed: Feature set '{feature_set_path}' not found.")

    m_type = model_type.strip().lower()
    if m_type in ("random_forest", "randomforest", "rf"):
        canonical_type = "random_forest"
    elif m_type in ("xgboost", "xgb"):
        canonical_type = "xgboost"
    else:
        raise ValueError(
            f"search_failed: Model type '{model_type}' is not supported for hyperparameter search. "
            f"Supported types: ['random_forest', 'xgboost']."
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"search_failed: Could not read feature set: {exc}") from exc

    if "fraud_bool" not in df.columns:
        raise ValueError("search_failed: Target column 'fraud_bool' missing from feature set.")

    feature_cols = [c for c in df.columns if c not in ("fraud_bool", "month")]

    # Split training data into train slice and holdout validation slice strictly within training months
    if "month" in df.columns:
        # Default: months 0-4 for train slice, month 5 for search validation
        train_slice_df = df[df["month"] < 5].copy()
        val_slice_df = df[df["month"] == 5].copy()
        if train_slice_df.empty or val_slice_df.empty:
            split_idx = int(len(df) * 0.8)
            train_slice_df = df.iloc[:split_idx].copy()
            val_slice_df = df.iloc[split_idx:].copy()
    else:
        split_idx = int(len(df) * 0.8)
        train_slice_df = df.iloc[:split_idx].copy()
        val_slice_df = df.iloc[split_idx:].copy()

    X_train_slice = train_slice_df[feature_cols].fillna(0)
    y_train_slice = train_slice_df["fraud_bool"]
    X_val_slice = val_slice_df[feature_cols].fillna(0)
    y_val_slice = val_slice_df["fraud_bool"]

    neg_count = float((y_train_slice == 0).sum())
    pos_count = float((y_train_slice == 1).sum())
    class_ratio = (neg_count / pos_count) if pos_count > 0 else 1.0

    # Default grids for fast iterative search
    if canonical_type == "random_forest":
        grid = param_grid or {
            "n_estimators": [25, 50],
            "max_depth": [5, 10],
        }
    else:
        if XGBClassifier is None:
            raise RuntimeError("search_failed: xgboost package is not installed.")
        grid = param_grid or {
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [25, 50],
        }

    search_results: list[dict[str, Any]] = []
    best_score = -1.0
    best_params: dict[str, Any] = {}

    grid_combinations = list(ParameterGrid(grid))
    for combo in grid_combinations:
        p = dict(combo)
        if canonical_type == "random_forest":
            if "random_state" not in p:
                p["random_state"] = 42
            cand_model = RandomForestClassifier(**p)
            cand_model.fit(X_train_slice, y_train_slice)
            y_val_proba = cand_model.predict_proba(X_val_slice)[:, 1]
        else:
            if "scale_pos_weight" not in p:
                p["scale_pos_weight"] = float(round(class_ratio, 2))
            if "random_state" not in p:
                p["random_state"] = 42
            if "eval_metric" not in p:
                p["eval_metric"] = "logloss"
            cand_model = XGBClassifier(**p)
            cand_model.fit(X_train_slice, y_train_slice)
            y_val_proba = cand_model.predict_proba(X_val_slice)[:, 1]

        prec, rec, _ = precision_recall_curve(y_val_slice, y_val_proba)
        cand_pr_auc = float(round(auc(rec, prec), 4))

        search_results.append({
            "hyperparams": combo,
            "pr_auc": cand_pr_auc,
        })

        if cand_pr_auc > best_score:
            best_score = cand_pr_auc
            best_params = combo

    # Train final winning model on the FULL training set
    final_res = train_model(
        feature_set_path=feature_set_path,
        model_type=canonical_type,
        hyperparams=best_params,
    )

    return {
        "model_id": final_res["model_id"],
        "model_path": final_res["model_path"],
        "model_type": canonical_type,
        "best_hyperparams": best_params,
        "best_score": float(round(best_score, 4)),
        "metric_used": "pr_auc",
        "search_results": search_results,
        "train_metrics": final_res["train_metrics"],
        "run_id": final_res.get("run_id", ""),
        "mlflow_run_id": final_res.get("mlflow_run_id", ""),
    }


def evaluate_model(
        model_id: str,
        eval_set_path: str,
        metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a trained model and return scores, including CM and ROC for reporting."""
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    import numpy as np

    model_path = MODELS_DIR / f"{model_id}.joblib"
    meta_path = MODELS_DIR / f"{model_id}_meta.json"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"model_not_found: Model ID '{model_id}' was not found in model registry.")

    eval_path = Path(eval_set_path)
    if not eval_path.exists():
        raise FileNotFoundError(f"model_not_found: Evaluation dataset '{eval_set_path}' not found.")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    bundle_data = joblib.load(model_path)
    if isinstance(bundle_data, dict) and "model" in bundle_data:
        model = bundle_data["model"]
        scaler = bundle_data.get("scaler")
    else:
        model = bundle_data
        scaler = None

    eval_df = pd.read_csv(eval_path)
    feature_cols = metadata.get("feature_columns", [])
    if not feature_cols:
        feature_cols = [c for c in eval_df.columns if c not in ("fraud_bool", "month", "customer_age")]

    X_eval = eval_df[feature_cols].fillna(0)
    y_eval = eval_df["fraud_bool"]

    if scaler is not None:
        X_eval_transformed = scaler.transform(X_eval)
    else:
        X_eval_transformed = X_eval

    y_pred = model.predict(X_eval_transformed)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_eval_transformed)[:, 1]
    else:
        y_proba = y_pred

    requested_metrics = [m.lower() for m in (metrics or ["precision", "recall", "pr_auc", "recall_at_fixed_fpr"])]
    scores: dict[str, Any] = {}  # Changed from float to Any to hold arrays

    if "precision" in requested_metrics:
        scores["precision"] = float(round(precision_score(y_eval, y_pred, zero_division=0), 4))
    if "recall" in requested_metrics:
        scores["recall"] = float(round(recall_score(y_eval, y_pred, zero_division=0), 4))

    # PR-AUC
    prec_curve, rec_curve, _ = precision_recall_curve(y_eval, y_proba)
    scores["pr_auc"] = float(round(auc(rec_curve, prec_curve), 4))

    # Recall @ 5% FPR
    fpr_arr, tpr_arr, _ = roc_curve(y_eval, y_proba)
    valid_indices = np.where(fpr_arr <= 0.05)[0]
    scores["recall_at_fixed_fpr"] = float(round(tpr_arr[valid_indices[-1]], 4)) if len(valid_indices) > 0 else 0.0

    # --- NEW: Add Chart Data for the Report ---
    cm = confusion_matrix(y_eval, y_pred)
    scores["confusion_matrix"] = cm.tolist()

    # Subsample ROC curve so JSON doesn't explode (max 100 points)
    if len(fpr_arr) > 100:
        idx = np.linspace(0, len(fpr_arr) - 1, 100).astype(int)
        scores["roc_curve"] = {"fpr": fpr_arr[idx].tolist(), "tpr": tpr_arr[idx].tolist()}
    else:
        scores["roc_curve"] = {"fpr": fpr_arr.tolist(), "tpr": tpr_arr.tolist()}

    # Log to MLflow
    _init_mlflow()
    run_id = metadata.get("run_id")
    if run_id:
        try:
            with mlflow.start_run(run_id=run_id):
                # Log only scalar metrics to MLflow
                mlflow.log_metric("eval_precision", scores.get("precision", 0))
                mlflow.log_metric("eval_recall", scores.get("recall", 0))
                mlflow.log_metric("eval_pr_auc", scores["pr_auc"])
                mlflow.log_metric("eval_recall_at_5pct_fpr", scores["recall_at_fixed_fpr"])
        except Exception:
            pass

    return {"scores": scores, "run_id": run_id}


def save_model(model_id: str, destination: str) -> dict[str, Any]:
    """Save/copy a unified model+scaler bundle and its metadata to a destination path."""
    model_path = MODELS_DIR / f"{model_id}.joblib"
    meta_path = MODELS_DIR / f"{model_id}_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"save_failed: Model ID '{model_id}' does not exist.")

    dest_path = Path(destination)
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, dest_path)
        if meta_path.exists():
            dest_meta = dest_path.parent / f"{dest_path.stem}_meta.json"
            shutil.copy2(meta_path, dest_meta)
        return {"saved_path": str(dest_path.resolve())}
    except Exception as exc:
        raise RuntimeError(f"save_failed: Could not save model to '{destination}': {exc}") from exc


def load_model(model_id: str) -> dict[str, Any]:
    """Load model metadata and verify existence."""
    model_path = MODELS_DIR / f"{model_id}.joblib"
    meta_path = MODELS_DIR / f"{model_id}_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"model_not_found: Model ID '{model_id}' does not exist.")

    metadata: dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return {
        "model_path": str(model_path.resolve()),
        "metadata": metadata,
    }


def check_fairness(
    model_id: str,
    eval_set_path: str,
) -> dict[str, Any]:
    """Evaluate predictive equality and false positive rate disparities across age groups.

    Group definition (BAF benchmark):
      - younger: customer_age < 50
      - older:   customer_age >= 50

    Metrics:
      - fpr_younger: False Positive Rate for younger group
      - fpr_older: False Positive Rate for older group
      - predictive_equality_ratio: min(fpr_younger, fpr_older) / max(fpr_younger, fpr_older)
      - group_sizes: {"younger": int, "older": int}

    Edge cases:
      - If both FPRs are 0: predictive_equality_ratio = 1.0 (no false positives in either group)
      - If only one FPR is 0: predictive_equality_ratio = 0.0 (maximal disparity)
    """
    model_path = MODELS_DIR / f"{model_id}.joblib"
    meta_path = MODELS_DIR / f"{model_id}_meta.json"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"model_not_found: Model ID '{model_id}' was not found in model registry.")

    eval_path = Path(eval_set_path)
    if not eval_path.exists():
        raise FileNotFoundError(f"model_not_found: Evaluation dataset '{eval_set_path}' not found.")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Load unified bundle
    bundle_data = joblib.load(model_path)
    if isinstance(bundle_data, dict) and "model" in bundle_data:
        model = bundle_data["model"]
        scaler = bundle_data.get("scaler")
    else:
        model = bundle_data
        scaler = None

    try:
        eval_df = pd.read_csv(eval_path)
    except Exception as exc:
        raise RuntimeError(f"model_not_found: Failed to read evaluation CSV: {exc}") from exc

    if "customer_age" not in eval_df.columns:
        raise ValueError("age_column_missing: 'customer_age' column is missing from evaluation dataset.")

    if "fraud_bool" not in eval_df.columns:
        raise ValueError("model_not_found: Target column 'fraud_bool' missing in evaluation set.")

    feature_cols = metadata.get("feature_columns", [])
    if not feature_cols:
        feature_cols = [c for c in eval_df.columns if c not in ("fraud_bool", "month", "customer_age")]

    X_eval = eval_df[feature_cols].fillna(0)
    if scaler is not None:
        X_eval_transformed = scaler.transform(X_eval)
    else:
        X_eval_transformed = X_eval

    y_pred = model.predict(X_eval_transformed)
    eval_df_res = eval_df.copy()
    eval_df_res["y_pred"] = y_pred

    younger_mask = eval_df_res["customer_age"] < 50
    older_mask = eval_df_res["customer_age"] >= 50

    younger_df = eval_df_res[younger_mask]
    older_df = eval_df_res[older_mask]

    younger_size = int(len(younger_df))
    older_size = int(len(older_df))

    # Calculate False Positive Rate (FPR) = FP / (FP + TN) where true label is negative (fraud_bool == 0)
    younger_negs = (younger_df["fraud_bool"] == 0)
    if younger_negs.sum() > 0:
        younger_fps = ((younger_df["fraud_bool"] == 0) & (younger_df["y_pred"] == 1)).sum()
        raw_fpr_younger = younger_fps / younger_negs.sum()
        fpr_younger = float(round(raw_fpr_younger, 4))
    else:
        raw_fpr_younger = 0.0
        fpr_younger = 0.0

    older_negs = (older_df["fraud_bool"] == 0)
    if older_negs.sum() > 0:
        older_fps = ((older_df["fraud_bool"] == 0) & (older_df["y_pred"] == 1)).sum()
        raw_fpr_older = older_fps / older_negs.sum()
        fpr_older = float(round(raw_fpr_older, 4))
    else:
        raw_fpr_older = 0.0
        fpr_older = 0.0

    # Predictive equality ratio with edge-case handling
    if (fpr_younger == 0.0 and fpr_older == 0.0) or (raw_fpr_younger < 0.001 and raw_fpr_older < 0.001):
        ratio = 1.0
    elif fpr_younger == 0.0 or fpr_older == 0.0:
        ratio = 0.0
    else:
        ratio = float(round(min(fpr_younger, fpr_older) / max(fpr_younger, fpr_older), 4))

    return {
        "fpr_younger": fpr_younger,
        "fpr_older": fpr_older,
        "predictive_equality_ratio": ratio,
        "group_sizes": {
            "younger": younger_size,
            "older": older_size,
        },
    }


# ML Tool Specs
CREATE_FEATURES_SPEC = ToolSpec(
    name="create_features",
    description="Creates a tabular feature set from dataset using a fixed temporal train/eval split (train: months 0-5, eval: months 6-7). Rejects fraud_bool or month as input features. Supports multicollinearity dropping via drop_correlated_pairs in feature_spec.",
    parameters={
        "type": "object",
        "properties": {
            "dataset_path": {"type": "string", "description": "Path to the raw CSV dataset."},
            "feature_spec": {"type": "object", "description": "Specification: optional 'features' list, optional 'drop_correlated_pairs' list of {feature_a, feature_b} dicts."},
        },
        "required": ["dataset_path"],
    },
    handler=create_features,
)

TRAIN_MODEL_SPEC = ToolSpec(
    name="train_model",
    description="Trains a machine learning model ('logistic_regression', 'random_forest', 'xgboost', 'pytorch_baseline') on a feature set.",
    parameters={
        "type": "object",
        "properties": {
            "feature_set_path": {"type": "string", "description": "Path to the feature set CSV file."},
            "model_type": {"type": "string", "description": "Model algorithm type: 'logistic_regression', 'random_forest', 'xgboost', 'pytorch_baseline'."},
            "hyperparams": {"type": "object", "description": "Hyperparameters for model training."},
        },
        "required": ["feature_set_path", "model_type"],
    },
    handler=train_model,
)

SEARCH_HYPERPARAMS_SPEC = ToolSpec(
    name="search_hyperparams",
    description="Searches hyperparameters for tree-based models ('random_forest', 'xgboost') using a holdout slice of training months only, scored by PR-AUC.",
    parameters={
        "type": "object",
        "properties": {
            "feature_set_path": {"type": "string", "description": "Path to training feature set CSV file."},
            "model_type": {"type": "string", "description": "Model type to search: 'random_forest' or 'xgboost'."},
            "param_grid": {"type": "object", "description": "Optional custom parameter grid dictionary."},
        },
        "required": ["feature_set_path", "model_type"],
    },
    handler=search_hyperparams,
)

EVALUATE_MODEL_SPEC = ToolSpec(
    name="evaluate_model",
    description="Evaluates a trained model on an evaluation set, computing precision, recall, PR-AUC, and recall_at_fixed_fpr (5% FPR).",
    parameters={
        "type": "object",
        "properties": {
            "model_id": {"type": "string", "description": "ID of the trained model."},
            "eval_set_path": {"type": "string", "description": "Path to evaluation dataset or feature set."},
            "metrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of evaluation metrics (e.g. ['precision', 'recall', 'pr_auc', 'recall_at_fixed_fpr']).",
            },
        },
        "required": ["model_id", "eval_set_path"],
    },
    handler=evaluate_model,
)

CHECK_FAIRNESS_SPEC = ToolSpec(
    name="check_fairness",
    description="Evaluates algorithmic fairness across customer age demographic groups (younger < 50, older >= 50) and computes predictive equality ratio.",
    parameters={
        "type": "object",
        "properties": {
            "model_id": {"type": "string", "description": "ID of the model to evaluate for fairness."},
            "eval_set_path": {"type": "string", "description": "Path to the evaluation dataset CSV containing customer_age."},
        },
        "required": ["model_id", "eval_set_path"],
    },
    handler=check_fairness,
)

def update_critic_verdict_tag(
    run_id: str,
    verdict: str,
    failed_checks: list[dict[str, Any]] | None = None,
    rationale: str | None = None,
) -> None:
    """Update existing MLflow run with critic_verdict tag and (optionally) failed_checks + rationale.

    Persisting failed_checks is what makes the critic-failure corpus buildable:
    retrieval_tools.build_critic_failure_corpus() reads tags.critic_failed_checks.
    """
    if not run_id:
        return
    _init_mlflow()
    try:
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tag("critic_verdict", verdict)
            if failed_checks is not None:
                mlflow.set_tag("critic_failed_checks", json.dumps(failed_checks))
            if rationale:
                mlflow.set_tag("critic_rationale", rationale[:5000])  # tag size guard
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            f"update_critic_verdict_tag FAILED for run_id={run_id!r}: "
            f"{type(exc).__name__}: {exc}"
        )


def register_model(
    model_id: str,
    run_id: str,
    registry_name: str = "argus_fraud_detector",
) -> dict[str, Any]:
    """Register an approved trained model in the MLflow Model Registry in stage 'Staging'.

    Input:
      - model_id: Unique model ID
      - run_id: MLflow run ID
      - registry_name: Target model registry name (default: 'argus_fraud_detector')

    Output:
      - version: int
      - registry_name: str
      - stage: "Staging"

    Errors:
      - ValueError("run_not_found: ...") if run_id does not exist
      - RuntimeError("registration_failed: ...") if registration fails
    """
    if not run_id:
        raise ValueError("run_not_found: run_id must not be empty.")

    _init_mlflow()
    client = mlflow.tracking.MlflowClient()

    try:
        run = client.get_run(run_id)
    except Exception as exc:
        raise ValueError(f"run_not_found: MLflow run '{run_id}' not found: {exc}") from exc

    if not run:
        raise ValueError(f"run_not_found: MLflow run '{run_id}' not found.")

    model_uri = f"runs:/{run_id}/model"
    try:
        try:
            client.create_registered_model(registry_name)
        except Exception:
            pass  # Already exists

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=registry_name,
        )
        version_num = int(model_version.version)

        client.transition_model_version_stage(
            name=registry_name,
            version=version_num,
            stage="Staging",
        )

        return {
            "version": version_num,
            "registry_name": registry_name,
            "stage": "Staging",
        }
    except Exception as exc:
        if "run_not_found" in str(exc):
            raise
        raise RuntimeError(f"registration_failed: Failed to register model '{model_id}' ({run_id}): {exc}") from exc


REGISTER_MODEL_SPEC = ToolSpec(
    name="register_model",
    description="Registers an approved model version in MLflow Model Registry in stage 'Staging'.",
    parameters={
        "type": "object",
        "properties": {
            "model_id": {"type": "string", "description": "Unique identifier of the model."},
            "run_id": {"type": "string", "description": "MLflow run ID associated with model training."},
            "registry_name": {"type": "string", "description": "Target model registry name (default: 'argus_fraud_detector').", "default": "argus_fraud_detector"},
        },
        "required": ["model_id", "run_id"],
    },
    handler=register_model,
)

SAVE_MODEL_SPEC = ToolSpec(
    name="save_model",
    description="Saves a trained model to a designated destination file path.",
    parameters={
        "type": "object",
        "properties": {
            "model_id": {"type": "string", "description": "ID of the model to save."},
            "destination": {"type": "string", "description": "Target destination path for the model file."},
        },
        "required": ["model_id", "destination"],
    },
    handler=save_model,
)

LOAD_MODEL_SPEC = ToolSpec(
    name="load_model",
    description="Retrieves a model artifact path and its metadata by model ID.",
    parameters={
        "type": "object",
        "properties": {
            "model_id": {"type": "string", "description": "ID of the model to load."},
        },
        "required": ["model_id"],
    },
    handler=load_model,
)
