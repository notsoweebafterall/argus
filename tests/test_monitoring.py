"""Unit and integration tests for Argus ML Monitoring."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from monitoring.monitor import (
    calculate_psi,
    run_monitoring_pass,
    generate_monitoring_report,
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
def registered_monitoring_model(base_dataset_path):
    """Train, evaluate, and register a model in 'Staging' for monitoring integration tests."""
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
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    return {
        "model_id": train_res["model_id"],
        "run_id": train_res["run_id"],
        "version": str(reg_res["version"]),
        "train_set_path": feat_res["train_set_path"],
        "eval_set_path": feat_res["eval_set_path"],
        "feature_columns": manifest.get("feature_columns", []),
        "eval_pr_auc": eval_res["scores"]["pr_auc"],
        "training_fraud_rate": manifest.get("training_fraud_rate", 0.011),
    }


# =========================================================================
# PSI Function & Epsilon Guard Tests
# =========================================================================

def test_psi_identical_distributions():
    """Verify PSI is near 0 for identical expected and actual distributions."""
    np.random.seed(42)
    expected = np.random.normal(loc=10.0, scale=2.0, size=1000)
    actual = np.random.normal(loc=10.0, scale=2.0, size=1000)

    psi = calculate_psi(expected, actual)
    assert isinstance(psi, float)
    assert 0.0 <= psi < 0.05


def test_psi_shifted_distribution():
    """Verify PSI exceeds 0.20 when actual distribution undergoes significant shift."""
    np.random.seed(42)
    expected = np.random.normal(loc=10.0, scale=2.0, size=1000)
    actual = np.random.normal(loc=25.0, scale=4.0, size=1000)

    psi = calculate_psi(expected, actual)
    assert psi > 0.20


def test_psi_epsilon_guard_empty_buckets():
    """Verify PSI handles empty buckets without ZeroDivisionError or -inf/inf values."""
    expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    # Actual has values in only one bucket, leaving other buckets with 0 counts
    actual = [1.0, 1.0, 1.0, 1.0, 1.0]

    psi = calculate_psi(expected, actual, epsilon=1e-4)
    assert not np.isnan(psi)
    assert not np.isinf(psi)
    assert psi > 0.0


def test_psi_constant_features():
    """Verify PSI returns 0.0 for completely constant features with zero variance."""
    expected = [5.0] * 100
    actual = [5.0] * 100
    psi = calculate_psi(expected, actual)
    assert psi == 0.0


# =========================================================================
# Signal 1: Latency Tests (Flagged vs Not Flagged)
# =========================================================================

def test_signal_latency_not_flagged(tmp_path):
    """Verify p95 latency <= 500ms produces no latency issue."""
    log_file = tmp_path / "pred_log_normal_lat.jsonl"
    entries = [
        {"latency_ms": 25.0 + i, "prediction": 0, "status": "success"}
        for i in range(100)
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    report = run_monitoring_pass(log_path=log_file, baseline_pr_auc=0.8)
    assert report["latency_p95_ms"] < 500.0
    assert not any("high_latency" in issue for issue in report["issues"])


def test_signal_latency_flagged(tmp_path):
    """Verify p95 latency > 500ms triggers high_latency issue."""
    log_file = tmp_path / "pred_log_high_lat.jsonl"
    entries = [
        {"latency_ms": 600.0 + i, "prediction": 0, "status": "success"}
        for i in range(100)
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    report = run_monitoring_pass(log_path=log_file, baseline_pr_auc=0.8)
    assert report["latency_p95_ms"] > 500.0
    assert any("high_latency" in issue for issue in report["issues"])


# =========================================================================
# Signal 2: Prediction Distribution Tests (Flagged vs Not Flagged)
# =========================================================================

def test_signal_prediction_distribution_not_flagged(tmp_path):
    """Verify predicted fraud rate within 3x bounds produces no distribution issue."""
    log_file = tmp_path / "pred_log_normal_fraud.jsonl"
    # 1 fraud out of 100 = 1.0% (within 3x of 1.1% baseline: [0.37%, 3.3%])
    entries = [
        {"prediction": 1 if i == 0 else 0, "status": "success", "latency_ms": 10.0}
        for i in range(100)
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    report = run_monitoring_pass(
        log_path=log_file,
        baseline_fraud_rate=0.011,
        baseline_pr_auc=0.8,
    )
    assert report["prediction_fraud_rate"] == 0.01
    assert not any("prediction_distribution_drift" in issue for issue in report["issues"])


def test_signal_prediction_distribution_flagged(tmp_path):
    """Verify predicted fraud rate deviating > 3x triggers distribution drift issue."""
    log_file = tmp_path / "pred_log_high_fraud.jsonl"
    # 15 frauds out of 100 = 15.0% (> 3.3% upper bound for 1.1% baseline)
    entries = [
        {"prediction": 1 if i < 15 else 0, "status": "success", "latency_ms": 10.0}
        for i in range(100)
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    report = run_monitoring_pass(
        log_path=log_file,
        baseline_fraud_rate=0.011,
        baseline_pr_auc=0.8,
    )
    assert report["prediction_fraud_rate"] == 0.15
    assert any("prediction_distribution_drift" in issue for issue in report["issues"])


# =========================================================================
# Signal 3: Input Drift (PSI) Tests (Flagged vs Not Flagged)
# =========================================================================

def test_signal_input_drift_not_flagged(tmp_path, registered_monitoring_model):
    """Verify features matching training distribution have PSI < 0.10 and are not flagged."""
    log_file = tmp_path / "pred_log_no_drift.jsonl"
    feature_cols = registered_monitoring_model["feature_columns"]

    # Read real representative sample values from training set
    train_df = pd.read_csv(registered_monitoring_model["train_set_path"])
    sample_rows = train_df[feature_cols].sample(n=200, random_state=42).to_dict(orient="records")

    entries = [
        {
            "prediction": 0,
            "status": "success",
            "latency_ms": 10.0,
            "features": row,
        }
        for row in sample_rows
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    report = run_monitoring_pass(
        log_path=log_file,
        train_set_path=registered_monitoring_model["train_set_path"],
        baseline_pr_auc=0.8,
    )
    for feat_name, drift_info in report["input_drift"].items():
        assert drift_info["flagged"] is False
    assert not any("input_drift" in issue for issue in report["issues"])


def test_signal_input_drift_flagged(tmp_path, registered_monitoring_model):
    """Verify shifted feature values produce PSI > 0.20 and trigger input_drift issue."""
    log_file = tmp_path / "pred_log_drifted.jsonl"
    feature_cols = registered_monitoring_model["feature_columns"]

    train_df = pd.read_csv(registered_monitoring_model["train_set_path"])
    sample_rows = train_df[feature_cols].head(50).to_dict(orient="records")

    # Massively shift first feature
    drifted_col = feature_cols[0]
    for row in sample_rows:
        row[drifted_col] = float(row[drifted_col]) + 10000.0

    entries = [
        {
            "prediction": 0,
            "status": "success",
            "latency_ms": 10.0,
            "features": row,
        }
        for row in sample_rows
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    report = run_monitoring_pass(
        log_path=log_file,
        train_set_path=registered_monitoring_model["train_set_path"],
        baseline_pr_auc=0.8,
    )
    assert report["input_drift"][drifted_col]["flagged"] is True
    assert report["input_drift"][drifted_col]["psi"] > 0.20
    assert any(f"input_drift: feature '{drifted_col}'" in issue for issue in report["issues"])


# =========================================================================
# Signal 4: Delayed-Label Performance Tests (Flagged vs Not Flagged)
# =========================================================================

def test_signal_delayed_label_performance_not_degraded(registered_monitoring_model):
    """Verify small or zero drop in PR-AUC sets degraded=False with no issue."""
    baseline_pr_auc = registered_monitoring_model["eval_pr_auc"]
    report = run_monitoring_pass(
        model_id=registered_monitoring_model["model_id"],
        eval_set_path=registered_monitoring_model["eval_set_path"],
        baseline_pr_auc=baseline_pr_auc,
    )
    delayed_perf = report["delayed_label_performance"]
    assert "pr_auc" in delayed_perf
    assert delayed_perf["degraded"] is False
    assert not any("performance_degradation" in issue for issue in report["issues"])


def test_signal_delayed_label_performance_degraded(registered_monitoring_model):
    """Verify drop > 0.10 in PR-AUC sets degraded=True and adds performance_degradation issue."""
    # Artificially set an unrealistically high baseline to trigger > 0.10 drop
    synthetic_high_baseline = 0.999
    report = run_monitoring_pass(
        model_id=registered_monitoring_model["model_id"],
        eval_set_path=registered_monitoring_model["eval_set_path"],
        baseline_pr_auc=synthetic_high_baseline,
    )
    delayed_perf = report["delayed_label_performance"]
    assert delayed_perf["degraded"] is True
    assert any("performance_degradation" in issue for issue in report["issues"])


# =========================================================================
# Signal 5: Error Rate Tests (Flagged vs Not Flagged)
# =========================================================================

def test_signal_error_rate_not_flagged(tmp_path):
    """Verify error rate <= 1.0% produces no error rate issue."""
    log_file = tmp_path / "pred_log_low_errors.jsonl"
    entries = [{"status": "success", "prediction": 0, "latency_ms": 10.0} for _ in range(100)]
    with open(log_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    report = run_monitoring_pass(log_path=log_file, baseline_pr_auc=0.8)
    assert report["error_rate"] == 0.0
    assert not any("high_error_rate" in issue for issue in report["issues"])


def test_signal_error_rate_flagged(tmp_path):
    """Verify error rate > 1.0% triggers high_error_rate issue."""
    log_file = tmp_path / "pred_log_high_errors.jsonl"
    entries = [
        {"status": "error", "error": "test_err", "latency_ms": 5.0} if i < 5
        else {"status": "success", "prediction": 0, "latency_ms": 10.0}
        for i in range(100)
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    report = run_monitoring_pass(log_path=log_file, baseline_pr_auc=0.8)
    assert report["error_rate"] == 0.05
    assert any("high_error_rate" in issue for issue in report["issues"])


# =========================================================================
# End-to-End Report Shape & Integration Test
# =========================================================================

def test_full_monitoring_report_shape_and_types(tmp_path, registered_monitoring_model):
    """Verify full monitoring report matches Stage 7 spec shape and field types exactly."""
    log_file = tmp_path / "pred_log_e2e.jsonl"
    feature_cols = registered_monitoring_model["feature_columns"]

    train_df = pd.read_csv(registered_monitoring_model["train_set_path"])
    sample_rows = train_df[feature_cols].head(20).to_dict(orient="records")

    entries = [
        {
            "timestamp": "2026-09-17T01:00:00+00:00",
            "latency_ms": 15.5 + i,
            "prediction": 0,
            "probability": 0.02,
            "model_version": registered_monitoring_model["version"],
            "features": row,
            "status": "success",
            "error": None,
        }
        for i, row in enumerate(sample_rows)
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    report = generate_monitoring_report(
        log_path=log_file,
        model_id=registered_monitoring_model["model_id"],
        train_set_path=registered_monitoring_model["train_set_path"],
        eval_set_path=registered_monitoring_model["eval_set_path"],
        baseline_pr_auc=registered_monitoring_model["eval_pr_auc"],
        baseline_fraud_rate=registered_monitoring_model["training_fraud_rate"],
    )

    # Validate exact top-level keys
    expected_keys = {
        "timestamp",
        "latency_p95_ms",
        "prediction_fraud_rate",
        "input_drift",
        "delayed_label_performance",
        "error_rate",
        "issues",
    }
    assert set(report.keys()) == expected_keys

    # Validate types
    assert isinstance(report["timestamp"], str)
    assert isinstance(report["latency_p95_ms"], float)
    assert isinstance(report["prediction_fraud_rate"], float)
    assert isinstance(report["input_drift"], dict)
    assert isinstance(report["delayed_label_performance"], dict)
    assert "pr_auc" in report["delayed_label_performance"]
    assert "degraded" in report["delayed_label_performance"]
    assert isinstance(report["delayed_label_performance"]["pr_auc"], float)
    assert isinstance(report["delayed_label_performance"]["degraded"], bool)
    assert isinstance(report["error_rate"], float)
    assert isinstance(report["issues"], list)
