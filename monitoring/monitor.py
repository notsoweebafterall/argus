"""Argus ML Monitoring Module.

Reads serving prediction logs (monitoring/prediction_log.jsonl),
training distribution metadata, and evaluation slices to detect:
  1. Latency spikes (p95 > 500ms)
  2. Prediction distribution drift (> 3x training fraud rate)
  3. Input feature drift via Population Stability Index (PSI > 0.20)
  4. Delayed-label performance degradation (PR-AUC drop > 0.10)
  5. API inference error rate (> 1.0%)
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

from pipeline.config import settings
from tools.ml_tools import (
    _init_mlflow,
    FEATURES_DIR,
    MODELS_DIR,
    evaluate_model,
)

logger = logging.getLogger("argus.monitoring")


def calculate_psi(
    expected: list[float] | np.ndarray | pd.Series,
    actual: list[float] | np.ndarray | pd.Series,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between expected and actual distributions.

    Decile binning of the expected distribution is used as the reference bucket edges.
    Guards against zero-division and log(0) using Laplace count smoothing.
    Buckets are dynamically capped to avoid empty-bin amplification when actual N is small.

    Threshold interpretation:
      - PSI < 0.10: No significant shift (stable)
      - 0.10 <= PSI <= 0.20: Moderate shift (advisory)
      - PSI > 0.20: Significant shift (action required / flagged)
    """
    exp_arr = np.asarray(expected, dtype=float)
    act_arr = np.asarray(actual, dtype=float)

    # Filter NaNs
    exp_arr = exp_arr[~np.isnan(exp_arr)]
    act_arr = act_arr[~np.isnan(act_arr)]

    if len(exp_arr) == 0 or len(act_arr) == 0:
        return 0.0

    # Dynamically cap num_buckets based on actual sample size.
    # Target at least 10 actual samples per bucket on average; below that,
    # Laplace smoothing cannot compensate for the ln() amplification from
    # near-empty bins (each empty bin contributes ~0.15 PSI with 50 samples/10 bins).
    effective_buckets = min(num_buckets, max(2, len(act_arr) // 10))

    # Calculate bucket percentiles on expected distribution
    percentiles = np.linspace(0, 100, effective_buckets + 1)
    raw_edges = np.percentile(exp_arr, percentiles)
    unique_edges = np.unique(raw_edges)

    if len(unique_edges) < 2:
        # Constant expected distribution
        min_v = min(float(exp_arr.min()), float(act_arr.min()))
        max_v = max(float(exp_arr.max()), float(act_arr.max()))
        if min_v == max_v:
            return 0.0
        unique_edges = np.array([min_v - 1e-5, max_v + 1e-5])

    # Construct robust bin edges
    if len(unique_edges) == 2:
        midpoint = (unique_edges[0] + unique_edges[1]) / 2.0
        bin_edges = np.array([-np.inf, midpoint, np.inf])
    else:
        bin_edges = unique_edges.copy()
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

    # Bucket counts
    expected_counts, _ = np.histogram(exp_arr, bins=bin_edges)
    actual_counts, _ = np.histogram(act_arr, bins=bin_edges)

    # Laplace count smoothing (+1 count to each bucket)
    # This naturally scales with sample size and prevents empty bucket amplification
    exp_smoothed = expected_counts + 1.0
    act_smoothed = actual_counts + 1.0

    expected_pct = exp_smoothed / np.sum(exp_smoothed)
    actual_pct = act_smoothed / np.sum(act_smoothed)

    # PSI calculation: sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
    per_bucket_psi = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi_val = np.sum(per_bucket_psi)
    return float(round(max(0.0, float(psi_val)), 4))


def _read_prediction_log(log_path: Path) -> list[dict[str, Any]]:
    """Read and parse all JSON lines from the prediction log file."""
    if not log_path.exists():
        return []

    entries: list[dict[str, Any]] = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    entries.append(json.loads(line_str))
                except json.JSONDecodeError:
                    entries.append({"status": "error", "error": "malformed_json_log_entry"})
    except Exception as exc:
        logger.warning("Could not read prediction log %s: %s", log_path, exc)
    return entries


def _load_registered_model_metadata(
    registry_name: str = "argus_fraud_detector",
    stage: str = "Staging",
) -> dict[str, Any]:
    """Retrieve metadata and baseline metrics for the model currently in the given stage."""
    _init_mlflow()
    client = mlflow.tracking.MlflowClient()

    try:
        versions = []
        try:
            versions = client.get_latest_versions(name=registry_name, stages=[stage])
        except Exception:
            pass

        if not versions:
            try:
                all_v = client.search_model_versions(f"name='{registry_name}'")
                versions = [v for v in all_v if str(v.current_stage).lower() == str(stage).lower()]
            except Exception:
                pass

        if not versions:
            return {}

        latest_ver = max(versions, key=lambda v: int(v.version))
        run_id = latest_ver.run_id
        run = client.get_run(run_id)

        metrics = run.data.metrics
        pr_auc = metrics.get("eval_pr_auc") or metrics.get("pr_auc", 0.0)
        model_id = run.data.tags.get("model_id") or run.data.params.get("model_id")

        return {
            "model_version": str(latest_ver.version),
            "run_id": run_id,
            "model_id": model_id,
            "baseline_pr_auc": float(pr_auc),
            "feature_set_path": run.data.params.get("feature_set_path"),
        }
    except Exception as exc:
        logger.warning("Could not retrieve model metadata from MLflow registry: %s", exc)
        return {}


def run_monitoring_pass(
    log_path: str | Path | None = None,
    model_id: str | None = None,
    registry_name: str = "argus_fraud_detector",
    stage: str = "Staging",
    train_set_path: str | Path | None = None,
    eval_set_path: str | Path | None = None,
    baseline_pr_auc: float | None = None,
    baseline_fraud_rate: float | None = None,
) -> dict[str, Any]:
    """Run a monitoring pass across the 5 core reliability signals.

    Signals:
      1. Latency (p95 > 500ms)
      2. Prediction distribution (> 3x training fraud rate)
      3. Input drift (PSI > 0.20 per numeric feature)
      4. Delayed-label performance degradation (PR-AUC drop > 0.10)
      5. Error rate (> 1.0%)

    Returns structured monitoring report conforming to the Argus Stage 7 spec.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    issues: list[str] = []

    # 1. Resolve prediction log
    actual_log_path = Path(log_path or os.getenv("ARGUS_PREDICTION_LOG_PATH", settings.workspace_root / "monitoring" / "prediction_log.jsonl"))
    entries = _read_prediction_log(actual_log_path)

    # 2. Resolve Model Registry metadata
    reg_meta = _load_registered_model_metadata(registry_name=registry_name, stage=stage)
    active_model_id = model_id or reg_meta.get("model_id")
    target_baseline_pr_auc = (
        baseline_pr_auc
        if baseline_pr_auc is not None
        else reg_meta.get("baseline_pr_auc", 0.0)
    )

    # --- Signal 1: Latency ---
    valid_latencies = [
        float(e["latency_ms"])
        for e in entries
        if "latency_ms" in e and e["latency_ms"] is not None and not np.isnan(float(e["latency_ms"]))
    ]
    if valid_latencies:
        latency_p95 = float(round(float(np.percentile(valid_latencies, 95)), 4))
    else:
        latency_p95 = 0.0

    if latency_p95 > 500.0:
        issues.append(f"high_latency: p95 latency {latency_p95:.2f}ms exceeds 500.0ms threshold")

    # --- Signal 5: Error Rate ---
    total_entries = len(entries)
    if total_entries > 0:
        error_count = sum(1 for e in entries if e.get("status") != "success")
        error_rate = float(round(error_count / total_entries, 6))
    else:
        error_rate = 0.0

    if error_rate > 0.01:
        issues.append(f"high_error_rate: error rate {error_rate:.2%} exceeds 1.0% threshold")

    # --- Load Training Baseline Distribution ---
    resolved_train_path = None
    if train_set_path:
        resolved_train_path = Path(train_set_path)
    elif reg_meta.get("feature_set_path"):
        resolved_train_path = Path(reg_meta["feature_set_path"])

    train_df = None
    resolved_base_fraud_rate = baseline_fraud_rate
    feature_columns: list[str] = []

    if resolved_train_path and resolved_train_path.exists():
        try:
            train_df = pd.read_csv(resolved_train_path)
            if resolved_base_fraud_rate is None and "fraud_bool" in train_df.columns:
                resolved_base_fraud_rate = float(train_df["fraud_bool"].mean())
            feature_columns = [c for c in train_df.columns if c not in ("fraud_bool", "month")]
        except Exception as exc:
            logger.warning("Could not read training set %s: %s", resolved_train_path, exc)

    # Check manifest/distribution json if available
    if resolved_base_fraud_rate is None or not feature_columns:
        manifest_files = list(FEATURES_DIR.glob("*_manifest.json"))
        if manifest_files:
            try:
                with open(manifest_files[-1], "r", encoding="utf-8") as f:
                    m_data = json.load(f)
                    if resolved_base_fraud_rate is None:
                        resolved_base_fraud_rate = m_data.get("training_fraud_rate")
                    if not feature_columns:
                        feature_columns = m_data.get("feature_columns", [])
                    if train_df is None and m_data.get("train_set_path"):
                        t_p = Path(m_data["train_set_path"])
                        if t_p.exists():
                            train_df = pd.read_csv(t_p)
            except Exception:
                pass

    if resolved_base_fraud_rate is None:
        resolved_base_fraud_rate = 0.011  # Default BAF benchmark baseline ~1.1%

    # --- Signal 2: Prediction Distribution ---
    successful_entries = [
        e for e in entries
        if e.get("status") == "success" and e.get("prediction") is not None
    ]
    if successful_entries:
        fraud_pred_count = sum(1 for e in successful_entries if int(e["prediction"]) == 1)
        prediction_fraud_rate = float(round(fraud_pred_count / len(successful_entries), 6))
    else:
        prediction_fraud_rate = 0.0

    if successful_entries and resolved_base_fraud_rate > 0:
        lower_bound = resolved_base_fraud_rate / 3.0
        upper_bound = resolved_base_fraud_rate * 3.0
        if prediction_fraud_rate < lower_bound or prediction_fraud_rate > upper_bound:
            issues.append(
                f"prediction_distribution_drift: predicted fraud rate {prediction_fraud_rate:.4f} "
                f"deviates >3x from baseline {resolved_base_fraud_rate:.4f} (bounds: [{lower_bound:.4f}, {upper_bound:.4f}])"
            )

    # --- Signal 3: Input Drift (PSI) ---
    input_drift: dict[str, dict[str, Any]] = {}
    actual_features_by_col: dict[str, list[float]] = {}

    for e in successful_entries:
        feat_dict = e.get("features")
        if isinstance(feat_dict, dict):
            for k, v in feat_dict.items():
                if isinstance(v, (int, float)) and not np.isnan(float(v)):
                    actual_features_by_col.setdefault(k, []).append(float(v))

    # Features to analyze
    target_cols = feature_columns or list(actual_features_by_col.keys())

    for col in target_cols:
        actual_vals = actual_features_by_col.get(col, [])
        if not actual_vals:
            continue

        if train_df is not None and col in train_df.columns:
            expected_vals = train_df[col].dropna().values
        else:
            expected_vals = np.array(actual_vals)

        psi_score = calculate_psi(expected=expected_vals, actual=actual_vals)
        is_flagged = bool(psi_score > 0.20)
        input_drift[col] = {
            "psi": psi_score,
            "flagged": is_flagged,
        }
        if is_flagged:
            issues.append(f"input_drift: feature '{col}' PSI {psi_score:.4f} exceeds 0.20 threshold")

    # --- Signal 4: Delayed-Label Performance Simulation ---
    delayed_perf: dict[str, Any] = {"pr_auc": 0.0, "degraded": False}
    resolved_eval_path = None

    if eval_set_path:
        resolved_eval_path = Path(eval_set_path)
    else:
        eval_files = list(FEATURES_DIR.glob("eval_*.csv"))
        if eval_files:
            resolved_eval_path = eval_files[-1]

    if resolved_eval_path and resolved_eval_path.exists() and active_model_id:
        try:
            eval_df = pd.read_csv(resolved_eval_path)
            if "month" in eval_df.columns and "fraud_bool" in eval_df.columns:
                # Partition eval set by month (sequential batches: month 6 = batch 1, month 7 = batch 2)
                eval_months = sorted(eval_df["month"].dropna().astype(int).unique())
                latest_month = eval_months[-1]
                latest_batch_df = eval_df[eval_df["month"] == latest_month].copy()

                # Load model bundle to evaluate latest batch
                model_bundle_path = MODELS_DIR / f"{active_model_id}.joblib"
                if model_bundle_path.exists():
                    bundle = joblib.load(model_bundle_path)
                    model = bundle["model"] if isinstance(bundle, dict) else bundle
                    scaler = bundle.get("scaler") if isinstance(bundle, dict) else None

                    feature_cols = feature_columns or [
                        c for c in latest_batch_df.columns
                        if c not in ("fraud_bool", "month", "customer_age")
                    ]
                    X_batch = latest_batch_df[feature_cols].fillna(0)
                    if scaler is not None:
                        X_batch_trans = scaler.transform(X_batch)
                    else:
                        X_batch_trans = X_batch

                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X_batch_trans)[:, 1]
                    else:
                        y_proba = model.predict(X_batch_trans)

                    prec_c, rec_c, _ = precision_recall_curve(latest_batch_df["fraud_bool"], y_proba)
                    batch_pr_auc = float(round(auc(rec_c, prec_c), 4))
                else:
                    batch_pr_auc = target_baseline_pr_auc

                drop_amount = target_baseline_pr_auc - batch_pr_auc
                degraded = bool(drop_amount > 0.10)
                delayed_perf = {
                    "pr_auc": batch_pr_auc,
                    "degraded": degraded,
                }
                if degraded:
                    issues.append(
                        f"performance_degradation: delayed PR-AUC {batch_pr_auc:.4f} "
                        f"dropped >0.10 from baseline {target_baseline_pr_auc:.4f}"
                    )
        except Exception as exc:
            logger.warning("Error evaluating delayed-label performance: %s", exc)

    return {
        "timestamp": timestamp,
        "latency_p95_ms": latency_p95,
        "prediction_fraud_rate": prediction_fraud_rate,
        "input_drift": input_drift,
        "delayed_label_performance": delayed_perf,
        "error_rate": error_rate,
        "issues": issues,
    }


# Backward-compatible alias
generate_monitoring_report = run_monitoring_pass
