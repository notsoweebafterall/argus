"""Unit and integration tests for Argus FastAPI serving application."""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from serving.api import (
    app,
    model_container,
    load_model_from_registry,
    append_prediction_log,
)
from tools.data_tools import load_dataset
from tools.ml_tools import (
    create_features,
    train_model,
    evaluate_model,
    register_model,
)


@pytest.fixture(scope="module")
def base_dataset_path():
    res = load_dataset("base")
    return res["path"]


@pytest.fixture(scope="module")
def registered_test_model(base_dataset_path):
    """Train, evaluate, and register a real logistic regression model in 'Staging'."""
    feat_res = create_features(dataset_path=base_dataset_path)
    train_res = train_model(
        feature_set_path=feat_res["train_set_path"],
        model_type="logistic_regression",
        hyperparams={"max_iter": 50},
    )
    eval_res = evaluate_model(
        model_id=train_res["model_id"],
        eval_set_path=feat_res["eval_set_path"],
    )
    reg_res = register_model(
        model_id=train_res["model_id"],
        run_id=train_res["run_id"],
        registry_name="argus_fraud_detector",
    )

    manifest_path = Path(feat_res["train_set_path"]).parent / f"{Path(feat_res['train_set_path']).stem.replace('train_', '')}_manifest.json"
    feature_cols = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f).get("feature_columns", [])

    return {
        "model_id": train_res["model_id"],
        "run_id": train_res["run_id"],
        "version": str(reg_res["version"]),
        "feature_columns": feature_cols,
        "eval_scores": eval_res["scores"],
    }


def test_health_no_model_loaded():
    """Verify /health reports model_loaded=False when container is empty."""
    model_container.reset()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is False
    assert data["model_version"] is None


def test_predict_no_model_loaded():
    """Verify /predict returns 503 error if no model is loaded."""
    model_container.reset()
    client = TestClient(app)
    response = client.post("/predict", json={"features": {"col1": 1.0}})
    assert response.status_code == 503
    assert "No model is currently loaded" in response.json()["detail"]


def test_model_info_no_model_loaded():
    """Verify /model-info returns 503 error if no model is loaded."""
    model_container.reset()
    client = TestClient(app)
    response = client.get("/model-info")
    assert response.status_code == 503
    assert "No model is currently loaded" in response.json()["detail"]


def test_model_loading_and_health_loaded(registered_test_model):
    """Verify loading from registry populates container and /health reflects it."""
    loaded = load_model_from_registry(stage="Staging")
    assert loaded is True
    assert model_container.model_loaded is True
    assert model_container.model_version is not None

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_version"] == model_container.model_version


def test_model_info_endpoint(registered_test_model):
    """Verify /model-info returns model version, type, registration time, and real eval metrics."""
    load_model_from_registry(stage="Staging")
    client = TestClient(app)
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == model_container.model_version
    assert data["model_type"] == "logistic_regression"
    assert data["registered_at"] != ""
    assert isinstance(data["eval_metrics"], dict)
    assert "pr_auc" in data["eval_metrics"] or "eval_pr_auc" in data["eval_metrics"]


def test_predict_happy_path(registered_test_model, tmp_path, monkeypatch):
    """Verify /predict returns valid prediction, probability, version, and latency."""
    log_file = tmp_path / "prediction_log.jsonl"
    monkeypatch.setenv("ARGUS_PREDICTION_LOG_PATH", str(log_file))

    load_model_from_registry(stage="Staging")
    client = TestClient(app)

    # Construct valid features dictionary with all expected columns
    feature_cols = model_container.feature_columns
    valid_features = {col: 1.0 for col in feature_cols}

    response = client.post("/predict", json={"features": valid_features})
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in (0, 1)
    assert isinstance(data["probability"], float)
    assert 0.0 <= data["probability"] <= 1.0
    assert data["model_version"] == model_container.model_version
    assert isinstance(data["latency_ms"], float)
    assert data["latency_ms"] >= 0.0


def test_predict_validation_missing_keys(registered_test_model):
    """Verify /predict rejects request with 422 when expected feature keys are missing."""
    load_model_from_registry(stage="Staging")
    client = TestClient(app)

    # Omit some required keys
    feature_cols = model_container.feature_columns
    if len(feature_cols) >= 2:
        incomplete_features = {col: 1.0 for col in feature_cols[:-2]}
        missing_expected = sorted(feature_cols[-2:])
    else:
        incomplete_features = {}
        missing_expected = sorted(feature_cols)

    response = client.post("/predict", json={"features": incomplete_features})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "feature_validation_failed"
    assert detail["missing_keys"] == missing_expected


def test_predict_validation_unexpected_keys(registered_test_model):
    """Verify /predict rejects request with 422 when extra/unexpected feature keys are present."""
    load_model_from_registry(stage="Staging")
    client = TestClient(app)

    feature_cols = model_container.feature_columns
    extra_features = {col: 1.0 for col in feature_cols}
    extra_features["bogus_feature_x"] = 99.0
    extra_features["bogus_feature_y"] = 100.0

    response = client.post("/predict", json={"features": extra_features})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "feature_validation_failed"
    assert detail["unexpected_keys"] == ["bogus_feature_x", "bogus_feature_y"]


def test_predict_validation_missing_and_unexpected_keys(registered_test_model):
    """Verify /predict returns 422 naming both missing and unexpected keys when both occur."""
    load_model_from_registry(stage="Staging")
    client = TestClient(app)

    feature_cols = model_container.feature_columns
    bad_features = {col: 1.0 for col in feature_cols[:-1]}
    bad_features["unexpected_alpha"] = 42.0

    response = client.post("/predict", json={"features": bad_features})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "feature_validation_failed"
    assert feature_cols[-1] in detail["missing_keys"]
    assert "unexpected_alpha" in detail["unexpected_keys"]


def test_prediction_logging_audit_file(registered_test_model, tmp_path, monkeypatch):
    """Verify prediction events are appended to prediction_log.jsonl with all required fields."""
    log_file = tmp_path / "custom_prediction_log.jsonl"
    monkeypatch.setenv("ARGUS_PREDICTION_LOG_PATH", str(log_file))

    load_model_from_registry(stage="Staging")
    client = TestClient(app)

    feature_cols = model_container.feature_columns
    valid_features = {col: 0.5 for col in feature_cols}

    response = client.post("/predict", json={"features": valid_features})
    assert response.status_code == 200

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1

    last_record = json.loads(lines[-1])
    assert "timestamp" in last_record
    assert "latency_ms" in last_record
    assert isinstance(last_record["latency_ms"], float)
    assert last_record["prediction"] in (0, 1)
    assert isinstance(last_record["probability"], float)
    assert last_record["model_version"] == model_container.model_version
    assert "features" in last_record
    assert last_record["features"] == valid_features
    assert last_record["status"] == "success"
    assert last_record["error"] is None
