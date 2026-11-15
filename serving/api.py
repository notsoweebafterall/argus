"""FastAPI Serving Application for Argus Fraud Detector.

Loads the latest registered model in the configured stage (default: 'Staging')
from MLflow Model Registry and serves real-time fraud predictions.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

try:
    import torch
except ImportError:
    torch = None

from pipeline.config import settings
from tools.ml_tools import _init_mlflow, MODELS_DIR

logger = logging.getLogger("argus.serving")
logging.basicConfig(level=logging.INFO)

# Default configurations
DEFAULT_REGISTRY_NAME = "argus_fraud_detector"
DEFAULT_STAGE = "Staging"
DEFAULT_LOG_PATH = settings.workspace_root / "monitoring" / "prediction_log.jsonl"


class ModelContainer:
    """Holds the in-memory loaded model, preprocessor, and metadata."""

    def __init__(self) -> None:
        self.model: Any = None
        self.scaler: Any = None
        self.feature_columns: list[str] = []
        self.model_version: str | None = None
        self.model_type: str | None = None
        self.registered_at: str | None = None
        self.eval_metrics: dict[str, float] = {}
        self.model_loaded: bool = False
        self.is_torch: bool = False

    def reset(self) -> None:
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.model_version = None
        self.model_type = None
        self.registered_at = None
        self.eval_metrics = {}
        self.model_loaded = False
        self.is_torch = False


model_container = ModelContainer()


def get_prediction_log_path() -> Path:
    """Return configured path for prediction audit/latency log."""
    env_path = os.getenv("ARGUS_PREDICTION_LOG_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_LOG_PATH


def append_prediction_log(entry: dict[str, Any], log_path: Path | None = None) -> None:
    """Append a prediction event record to the JSONL log."""
    target_path = log_path or get_prediction_log_path()
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning("Failed to write to prediction log %s: %s", target_path, exc)


def load_model_from_registry(
    registry_name: str | None = None,
    stage: str | None = None,
) -> bool:
    """Load latest model version from MLflow Model Registry for given stage.

    Returns True if successfully loaded, False otherwise.
    """
    model_container.reset()
    reg_name = registry_name or os.getenv("ARGUS_MODEL_NAME", DEFAULT_REGISTRY_NAME)
    target_stage = stage or os.getenv("ARGUS_MODEL_STAGE", DEFAULT_STAGE)

    try:
        _init_mlflow()
        client = mlflow.tracking.MlflowClient()

        # Query model versions in the requested stage
        versions = []
        try:
            versions = client.get_latest_versions(name=reg_name, stages=[target_stage])
        except Exception as exc:
            logger.info("Could not fetch latest versions via get_latest_versions for %s: %s", reg_name, exc)

        if not versions:
            try:
                all_versions = client.search_model_versions(f"name='{reg_name}'")
                versions = [
                    v for v in all_versions
                    if str(v.current_stage).lower() == str(target_stage).lower()
                ]
            except Exception as exc:
                logger.info("Could not search model versions for %s: %s", reg_name, exc)

        if not versions:
            logger.warning(
                "No model found in registry '%s' with stage '%s'. App starting in unloaded state.",
                reg_name,
                target_stage,
            )
            return False

        # Select highest version
        latest_ver = max(versions, key=lambda v: int(v.version))
        version_str = str(latest_ver.version)
        run_id = latest_ver.run_id

        # Registration timestamp
        if hasattr(latest_ver, "creation_timestamp") and latest_ver.creation_timestamp:
            ts_sec = latest_ver.creation_timestamp / 1000.0
            registered_at = datetime.datetime.fromtimestamp(ts_sec, tz=datetime.timezone.utc).isoformat()
        else:
            registered_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Fetch MLflow Run for metadata & metrics
        run = client.get_run(run_id)
        eval_metrics: dict[str, float] = {}
        for k, v in run.data.metrics.items():
            clean_k = k[5:] if k.startswith("eval_") else k
            eval_metrics[clean_k] = float(v)
            if k.startswith("eval_"):
                eval_metrics[k] = float(v)

        model_id = run.data.tags.get("model_id") or run.data.params.get("model_id")
        model_type = run.data.tags.get("model_type") or run.data.params.get("model_type") or "unknown"

        # Attempt to load the unified model+scaler bundle
        bundle_loaded = False
        loaded_model = None
        loaded_scaler = None
        feature_cols: list[str] = []
        is_torch = False

        # Strategy 1: Check local MODELS_DIR if model_id is known
        if model_id:
            local_bundle = MODELS_DIR / f"{model_id}.joblib"
            local_meta = MODELS_DIR / f"{model_id}_meta.json"
            if local_bundle.exists():
                try:
                    b_data = joblib.load(local_bundle)
                    if isinstance(b_data, dict) and "model" in b_data:
                        loaded_model = b_data["model"]
                        loaded_scaler = b_data.get("scaler")
                        is_torch = bool(b_data.get("is_torch", False))
                        if local_meta.exists():
                            with open(local_meta, "r", encoding="utf-8") as f:
                                meta_json = json.load(f)
                                feature_cols = meta_json.get("feature_columns", [])
                                model_type = meta_json.get("model_type", model_type)
                        elif "metadata" in b_data:
                            feature_cols = b_data["metadata"].get("feature_columns", [])
                            model_type = b_data["metadata"].get("model_type", model_type)
                        bundle_loaded = True
                except Exception as exc:
                    logger.warning("Failed loading local bundle %s: %s", local_bundle, exc)

        # Strategy 2: Download bundle artifact from MLflow Run
        if not bundle_loaded and run_id:
            try:
                bundle_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="bundle")
                bundle_path = Path(bundle_dir)
                if bundle_path.is_dir():
                    candidates = list(bundle_path.glob("*.joblib"))
                    if candidates:
                        bundle_path = candidates[0]
                if bundle_path.exists() and bundle_path.is_file():
                    b_data = joblib.load(bundle_path)
                    if isinstance(b_data, dict) and "model" in b_data:
                        loaded_model = b_data["model"]
                        loaded_scaler = b_data.get("scaler")
                        is_torch = bool(b_data.get("is_torch", False))
                        if "metadata" in b_data:
                            feature_cols = b_data["metadata"].get("feature_columns", [])
                            model_type = b_data["metadata"].get("model_type", model_type)
                        bundle_loaded = True
            except Exception as exc:
                logger.info("Could not download bundle artifact for run %s: %s", run_id, exc)

        # Strategy 3: Flavor-specific fallback loader from MLflow model uri
        if not bundle_loaded:
            model_uri = f"runs:/{run_id}/model"
            try:
                if model_type in ("logistic_regression", "random_forest"):
                    loaded_model = mlflow.sklearn.load_model(model_uri)
                elif model_type == "xgboost":
                    try:
                        loaded_model = mlflow.xgboost.load_model(model_uri)
                    except Exception:
                        loaded_model = mlflow.sklearn.load_model(model_uri)
                elif model_type == "pytorch_baseline":
                    try:
                        loaded_model = mlflow.pytorch.load_model(model_uri)
                        is_torch = True
                    except Exception:
                        loaded_model = mlflow.sklearn.load_model(model_uri)
                else:
                    loaded_model = mlflow.pyfunc.load_model(model_uri)
                bundle_loaded = True
            except Exception as exc:
                logger.error("Failed loading model from URI %s: %s", model_uri, exc)

        if not bundle_loaded or loaded_model is None:
            logger.error("Model could not be reconstructed for version %s", version_str)
            return False

        model_container.model = loaded_model
        model_container.scaler = loaded_scaler
        model_container.feature_columns = feature_cols
        model_container.model_version = version_str
        model_container.model_type = model_type
        model_container.registered_at = registered_at
        model_container.eval_metrics = eval_metrics
        model_container.model_loaded = True
        model_container.is_torch = is_torch

        logger.info(
            "Successfully loaded model '%s' version %s (type=%s, features=%d) from stage '%s'.",
            reg_name,
            version_str,
            model_type,
            len(feature_cols),
            target_stage,
        )
        return True

    except Exception as exc:
        logger.warning("Unexpected error during model registry loading: %s", exc)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load the model on startup gracefully."""
    load_model_from_registry()
    yield


app = FastAPI(
    title="Argus Fraud Detector API",
    description="Real-time fraud prediction API powered by Argus ML Registry",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Feature dictionary mapping feature names to numerical values.",
    )


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None


class ModelInfoResponse(BaseModel):
    model_version: str
    model_type: str
    registered_at: str
    eval_metrics: dict[str, float]


@app.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Health check endpoint indicating whether a model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=model_container.model_loaded,
        model_version=model_container.model_version,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def get_model_info() -> ModelInfoResponse:
    """Return metadata and evaluation metrics for the currently loaded model."""
    if not model_container.model_loaded or model_container.model_version is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model is currently loaded.",
        )

    return ModelInfoResponse(
        model_version=model_container.model_version,
        model_type=model_container.model_type or "unknown",
        registered_at=model_container.registered_at or "",
        eval_metrics=model_container.eval_metrics,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Serve real-time fraud prediction and log request inference latency."""
    if not model_container.model_loaded or model_container.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model is currently loaded to serve predictions.",
        )

    expected_cols = model_container.feature_columns
    req_cols = set(req.features.keys())
    expected_set = set(expected_cols)

    # Feature key validation: exact match required
    if expected_cols and req_cols != expected_set:
        missing_keys = sorted(list(expected_set - req_cols))
        unexpected_keys = sorted(list(req_cols - expected_set))
        error_msg = f"Feature validation failed. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}"
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "feature_validation_failed",
                "message": error_msg,
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
            },
        )

    # Prepare feature vector
    if expected_cols:
        row_values = [req.features[col] for col in expected_cols]
    else:
        row_values = list(req.features.values())

    X = np.array([row_values], dtype=float)

    # Apply fitted training scaler
    if model_container.scaler is not None:
        try:
            X_scaled = model_container.scaler.transform(X)
        except Exception as exc:
            logger.error("Scaler transformation failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Preprocessing error: {exc}",
            ) from exc
    else:
        X_scaled = X

    # Measure wall-clock inference time
    pred_val = 0
    prob_val = 0.0
    t0 = time.perf_counter()
    try:
        model = model_container.model
        if model_container.is_torch or (torch is not None and isinstance(model, torch.nn.Module)):
            with torch.no_grad():
                tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
                out = model(tensor_x)
                prob_val = float(torch.sigmoid(out).squeeze().item())
                pred_val = int(prob_val >= 0.5)
        else:
            pred_arr = model.predict(X_scaled)
            pred_val = int(pred_arr[0])
            if hasattr(model, "predict_proba"):
                prob_val = float(model.predict_proba(X_scaled)[0, 1])
            else:
                prob_val = float(pred_val)
        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000.0, 4)

        # Log successful prediction event
        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "latency_ms": latency_ms,
            "prediction": pred_val,
            "probability": round(prob_val, 6),
            "model_version": model_container.model_version,
            "features": req.features,
            "status": "success",
            "error": None,
        }
        append_prediction_log(log_entry)

        return PredictResponse(
            prediction=pred_val,
            probability=round(prob_val, 6),
            model_version=model_container.model_version or "unknown",
            latency_ms=latency_ms,
        )

    except Exception as exc:
        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000.0, 4)
        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "latency_ms": latency_ms,
            "prediction": None,
            "probability": None,
            "model_version": model_container.model_version,
            "features": req.features,
            "status": "error",
            "error": str(exc),
        }
        append_prediction_log(log_entry)
        logger.error("Prediction inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        ) from exc
