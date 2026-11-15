"""Unit tests for Argus ML Tools."""

import os
from unittest.mock import patch
import pandas as pd
import pytest
from tools.data_tools import load_dataset
from tools.ml_tools import (
    EVAL_MONTHS,
    TRAIN_MONTHS,
    check_fairness,
    create_features,
    evaluate_model,
    load_model,
    register_model,
    save_model,
    search_hyperparams,
    train_model,
    update_critic_verdict_tag,
)


@pytest.fixture(scope="module")
def base_dataset_path():
    """Fixture providing loaded base dataset path."""
    res = load_dataset("base")
    return res["path"]


# ---------------------------------------------------------------------------
# create_features — temporal split
# ---------------------------------------------------------------------------

def test_create_features_success(base_dataset_path):
    """Verify temporal split: manifest has correct keys, both paths exist, no fraud_bool in features."""
    res = create_features(dataset_path=base_dataset_path)
    assert os.path.exists(res["train_set_path"])
    assert os.path.exists(res["eval_set_path"])
    m = res["manifest"]
    assert m["feature_count"] > 0
    assert m["target_column"] == "fraud_bool"
    assert "fraud_bool" not in m["feature_columns"]
    assert "month" not in m["feature_columns"], "'month' must not appear as a feature column"
    assert m["train_rows"] > 0
    assert m["eval_rows"] > 0
    assert m["split_method"] == "temporal"
    assert m["train_months"] == list(TRAIN_MONTHS)
    assert m["eval_months"] == list(EVAL_MONTHS)
    assert m["train_row_count"] == m["train_rows"]
    assert m["eval_row_count"] == m["eval_rows"]
    assert "dropped_columns" in m


def test_temporal_split_boundary_train(base_dataset_path):
    """Pinned: no row in train_set_path has month >= 6."""
    res = create_features(dataset_path=base_dataset_path)
    train_df = pd.read_csv(res["train_set_path"])
    assert "month" in train_df.columns, "train CSV must carry month column for verification"
    out_of_bound = train_df[train_df["month"] >= 6]
    assert len(out_of_bound) == 0, (
        f"Train set must contain only months 0-5. Found {len(out_of_bound)} rows with month >= 6."
    )


def test_temporal_split_boundary_eval(base_dataset_path):
    """Pinned: no row in eval_set_path has month < 6."""
    res = create_features(dataset_path=base_dataset_path)
    eval_df = pd.read_csv(res["eval_set_path"])
    assert "month" in eval_df.columns, "eval CSV must carry month column for verification"
    out_of_bound = eval_df[eval_df["month"] < 6]
    assert len(out_of_bound) == 0, (
        f"Eval set must contain only months 6-7. Found {len(out_of_bound)} rows with month < 6."
    )


def test_month_not_in_feature_columns(base_dataset_path):
    """Pinned: 'month' never appears as a column in the final feature set manifest."""
    res = create_features(dataset_path=base_dataset_path)
    assert "month" not in res["manifest"]["feature_columns"]


def test_create_features_month_as_feature_rejected(base_dataset_path):
    """Pinned: feature_spec explicitly listing 'month' is rejected with leakage_detected."""
    with pytest.raises(ValueError) as exc_info:
        create_features(
            dataset_path=base_dataset_path,
            feature_spec={"features": ["income", "month", "credit_risk_score"]},
        )
    assert "leakage_detected" in str(exc_info.value)
    assert "month" in str(exc_info.value)


def test_create_features_leakage_detected(base_dataset_path):
    """Verify fraud_bool target leakage detection raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        create_features(
            dataset_path=base_dataset_path,
            feature_spec={"features": ["income", "fraud_bool", "credit_risk_score"]},
        )
    assert "leakage_detected" in str(exc_info.value)


def test_create_features_multicollinearity_drop(base_dataset_path):
    """Verify drop_correlated_pairs drops feature_b from the feature set and records it in manifest."""
    res = create_features(
        dataset_path=base_dataset_path,
        feature_spec={
            "drop_correlated_pairs": [
                {"feature_a": "velocity_6h", "feature_b": "velocity_24h"}
            ]
        },
    )
    m = res["manifest"]
    assert "velocity_24h" not in m["feature_columns"], (
        "velocity_24h should have been dropped due to multicollinearity with velocity_6h"
    )
    assert any(d["column"] == "velocity_24h" for d in m["dropped_columns"]), (
        "dropped_columns must record the dropped column"
    )
    assert any("multicollinearity" in d["reason"] for d in m["dropped_columns"])


# ---------------------------------------------------------------------------
# train_model — all supported candidate types
# ---------------------------------------------------------------------------

def test_train_model_logistic_regression_success(base_dataset_path):
    """Verify training logistic regression model on temporally-split feature set."""
    feat_res = create_features(dataset_path=base_dataset_path)
    train_res = train_model(
        feature_set_path=feat_res["train_set_path"],
        model_type="logistic_regression",
        hyperparams={"max_iter": 50},
    )
    assert train_res["model_id"].startswith("model_")
    assert os.path.exists(train_res["model_path"])
    assert "train_accuracy" in train_res["train_metrics"]


def test_train_model_random_forest_success(base_dataset_path):
    """Verify training random forest model on temporally-split feature set."""
    feat_res = create_features(dataset_path=base_dataset_path)
    train_res = train_model(
        feature_set_path=feat_res["train_set_path"],
        model_type="random_forest",
        hyperparams={"n_estimators": 10, "max_depth": 5, "random_state": 42},
    )
    assert train_res["model_id"].startswith("model_")
    assert os.path.exists(train_res["model_path"])
    assert "train_accuracy" in train_res["train_metrics"]

    eval_res = evaluate_model(
        model_id=train_res["model_id"],
        eval_set_path=feat_res["eval_set_path"],
    )
    assert "pr_auc" in eval_res["scores"]
    assert "recall_at_fixed_fpr" in eval_res["scores"]


def test_train_model_xgboost_success(base_dataset_path):
    """Verify training xgboost model with automated class imbalance handling."""
    feat_res = create_features(dataset_path=base_dataset_path)
    train_res = train_model(
        feature_set_path=feat_res["train_set_path"],
        model_type="xgboost",
        hyperparams={"n_estimators": 10, "max_depth": 3, "random_state": 42},
    )
    assert train_res["model_id"].startswith("model_")
    assert os.path.exists(train_res["model_path"])
    assert "train_accuracy" in train_res["train_metrics"]

    eval_res = evaluate_model(
        model_id=train_res["model_id"],
        eval_set_path=feat_res["eval_set_path"],
    )
    assert "pr_auc" in eval_res["scores"]
    assert "recall_at_fixed_fpr" in eval_res["scores"]


def test_train_model_pytorch_baseline_success(base_dataset_path):
    """Verify training small feedforward neural network in PyTorch with class weighting."""
    feat_res = create_features(dataset_path=base_dataset_path)
    train_res = train_model(
        feature_set_path=feat_res["train_set_path"],
        model_type="pytorch_baseline",
        hyperparams={"epochs": 2, "batch_size": 512, "lr": 0.005},
    )
    assert train_res["model_id"].startswith("model_")
    assert os.path.exists(train_res["model_path"])
    assert "train_accuracy" in train_res["train_metrics"]

    eval_res = evaluate_model(
        model_id=train_res["model_id"],
        eval_set_path=feat_res["eval_set_path"],
    )
    assert "pr_auc" in eval_res["scores"]
    assert "recall_at_fixed_fpr" in eval_res["scores"]
    assert 0.0 <= eval_res["scores"]["pr_auc"] <= 1.0


def test_train_model_invalid_type(base_dataset_path):
    """Verify unsupported model type raises ValueError."""
    feat_res = create_features(dataset_path=base_dataset_path)
    with pytest.raises(ValueError) as exc_info:
        train_model(
            feature_set_path=feat_res["train_set_path"],
            model_type="unsupported_deep_forest",
        )
    assert "training_failed" in str(exc_info.value)


def test_train_model_invalid_hyperparams(base_dataset_path):
    """Verify invalid hyperparameters raise ValueError with invalid_hyperparams."""
    feat_res = create_features(dataset_path=base_dataset_path)
    with pytest.raises(ValueError) as exc_info:
        train_model(
            feature_set_path=feat_res["train_set_path"],
            model_type="logistic_regression",
            hyperparams={"invalid_hyperparam_xyz": 999},
        )
    assert "invalid_hyperparams" in str(exc_info.value)


# ---------------------------------------------------------------------------
# search_hyperparams — training-months holdout & leakage prevention
# ---------------------------------------------------------------------------

def test_search_hyperparams_never_touches_eval_set(base_dataset_path):
    """Provable assertion: search_hyperparams strictly operates on train_set_path and never accesses eval_set_path."""
    feat_res = create_features(dataset_path=base_dataset_path)
    train_set_path = feat_res["train_set_path"]
    eval_set_path = feat_res["eval_set_path"]

    # 1. Verify train_set_path itself only contains training months 0-5
    train_df = pd.read_csv(train_set_path)
    assert set(train_df["month"].unique()).issubset(set(range(0, 6))), (
        "train_set_path must only contain months 0-5"
    )

    # 2. Track all file paths read via pd.read_csv during hyperparameter search
    original_read_csv = pd.read_csv
    files_read: list[str] = []

    def tracking_read_csv(filepath_or_buffer, *args, **kwargs):
        path_str = str(filepath_or_buffer)
        files_read.append(path_str)
        return original_read_csv(filepath_or_buffer, *args, **kwargs)

    with patch("pandas.read_csv", side_effect=tracking_read_csv):
        search_res = search_hyperparams(
            feature_set_path=train_set_path,
            model_type="random_forest",
            param_grid={"n_estimators": [10], "max_depth": [3, 5]},
        )

    # 3. Assert eval_set_path was never touched during search
    eval_normalized = str(os.path.abspath(eval_set_path))
    for f in files_read:
        assert os.path.abspath(f) != eval_normalized, (
            f"Hyperparameter search must NEVER read eval_set_path ({eval_set_path})"
        )

    # 4. Assert search outputs are structured and auditable
    assert "model_id" in search_res
    assert search_res["model_type"] == "random_forest"
    assert "best_hyperparams" in search_res
    assert "best_score" in search_res
    assert search_res["metric_used"] == "pr_auc"
    assert len(search_res["search_results"]) == 2
    assert all("pr_auc" in r and "hyperparams" in r for r in search_res["search_results"])


# ---------------------------------------------------------------------------
# evaluate_model — metrics & recall_at_fixed_fpr enforcement
# ---------------------------------------------------------------------------

def test_evaluate_model_success(base_dataset_path):
    """Verify model evaluation on eval_set_path returns precision, recall, PR-AUC, and recall_at_fixed_fpr."""
    feat_res = create_features(dataset_path=base_dataset_path)
    train_res = train_model(
        feature_set_path=feat_res["train_set_path"],
        model_type="logistic_regression",
        hyperparams={"max_iter": 50},
    )

    eval_res = evaluate_model(
        model_id=train_res["model_id"],
        eval_set_path=feat_res["eval_set_path"],
        metrics=["precision", "recall", "pr_auc", "recall_at_fixed_fpr"],
    )

    scores = eval_res["scores"]
    assert "precision" in scores
    assert "recall" in scores
    assert "pr_auc" in scores
    assert "recall_at_fixed_fpr" in scores
    assert 0.0 <= scores["pr_auc"] <= 1.0
    assert 0.0 <= scores["recall_at_fixed_fpr"] <= 1.0


def test_evaluate_model_always_enforces_pr_auc_and_recall_at_fixed_fpr(base_dataset_path):
    """Verify requesting only accuracy still enforces both PR-AUC and recall_at_fixed_fpr inclusion."""
    feat_res = create_features(dataset_path=base_dataset_path)
    train_res = train_model(
        feature_set_path=feat_res["train_set_path"],
        model_type="logistic_regression",
        hyperparams={"max_iter": 50},
    )

    eval_res = evaluate_model(
        model_id=train_res["model_id"],
        eval_set_path=feat_res["eval_set_path"],
        metrics=["accuracy"],
    )

    scores = eval_res["scores"]
    assert "accuracy" in scores
    assert "pr_auc" in scores, "PR-AUC must always be included for fraud detection"
    assert "recall_at_fixed_fpr" in scores, "recall_at_fixed_fpr must always be included for fraud detection"


def test_evaluate_model_unsupported_metric(base_dataset_path):
    """Verify unsupported metric raises ValueError with metric_not_supported."""
    feat_res = create_features(dataset_path=base_dataset_path)
    train_res = train_model(
        feature_set_path=feat_res["train_set_path"],
        model_type="logistic_regression",
        hyperparams={"max_iter": 50},
    )

    with pytest.raises(ValueError) as exc_info:
        evaluate_model(
            model_id=train_res["model_id"],
            eval_set_path=feat_res["eval_set_path"],
            metrics=["non_existing_metric"],
        )
    assert "metric_not_supported" in str(exc_info.value)


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------

def test_save_and_load_model(base_dataset_path, tmp_path):
    """Verify saving model to destination and loading metadata."""
    feat_res = create_features(dataset_path=base_dataset_path)
    train_res = train_model(
        feature_set_path=feat_res["train_set_path"],
        model_type="logistic_regression",
        hyperparams={"max_iter": 50},
    )

    dest = tmp_path / "custom_export" / "model.joblib"
    save_res = save_model(model_id=train_res["model_id"], destination=str(dest))
    assert os.path.exists(save_res["saved_path"])

    load_res = load_model(model_id=train_res["model_id"])
    assert os.path.exists(load_res["model_path"])
    assert load_res["metadata"]["model_type"] == "logistic_regression"


# ---------------------------------------------------------------------------
# check_fairness
# ---------------------------------------------------------------------------

def test_check_fairness_success(base_dataset_path):
    """Verify check_fairness computes younger/older group sizes, FPRs, and predictive_equality_ratio."""
    feat_res = create_features(dataset_path=base_dataset_path)
    train_res = train_model(
        feature_set_path=feat_res["train_set_path"],
        model_type="logistic_regression",
        hyperparams={"max_iter": 50},
    )

    fairness_res = check_fairness(
        model_id=train_res["model_id"],
        eval_set_path=feat_res["eval_set_path"],
    )

    assert "fpr_younger" in fairness_res
    assert "fpr_older" in fairness_res
    assert "predictive_equality_ratio" in fairness_res
    assert "group_sizes" in fairness_res

    sizes = fairness_res["group_sizes"]
    assert sizes["younger"] > 0
    assert sizes["older"] > 0
    assert 0.0 <= fairness_res["fpr_younger"] <= 1.0
    assert 0.0 <= fairness_res["fpr_older"] <= 1.0
    assert 0.0 <= fairness_res["predictive_equality_ratio"] <= 1.0


def test_check_fairness_missing_age_column(base_dataset_path, tmp_path):
    """Verify check_fairness raises ValueError with age_column_missing if customer_age is missing."""
    feat_res = create_features(dataset_path=base_dataset_path)
    train_res = train_model(
        feature_set_path=feat_res["train_set_path"],
        model_type="logistic_regression",
        hyperparams={"max_iter": 50},
    )

    # Create an eval CSV without customer_age
    eval_df = pd.read_csv(feat_res["eval_set_path"])
    if "customer_age" in eval_df.columns:
        eval_df = eval_df.drop(columns=["customer_age"])
    bad_eval_path = tmp_path / "eval_no_age.csv"
    eval_df.to_csv(bad_eval_path, index=False)

    with pytest.raises(ValueError) as exc_info:
        check_fairness(
            model_id=train_res["model_id"],
            eval_set_path=str(bad_eval_path),
        )
    assert "age_column_missing" in str(exc_info.value)


def test_check_fairness_zero_fpr_edge_cases(base_dataset_path, tmp_path):
    """Verify predictive_equality_ratio edge cases: both FPRs zero -> 1.0; one FPR zero -> 0.0."""
    feat_res = create_features(dataset_path=base_dataset_path)
    train_res = train_model(
        feature_set_path=feat_res["train_set_path"],
        model_type="logistic_regression",
        hyperparams={"max_iter": 50},
    )

    eval_df = pd.read_csv(feat_res["eval_set_path"]).head(200).copy()

    # Case 1: Both groups have zero false positives
    # All true labels 0, model predicts 0 everywhere
    custom_eval_both_zero = tmp_path / "eval_both_zero.csv"
    eval_df.to_csv(custom_eval_both_zero, index=False)

    # Use patch on model predict to return all 0s (no false positives)
    with patch("sklearn.linear_model.LogisticRegression.predict", return_value=pd.Series([0] * len(eval_df)).values):
        res_both_zero = check_fairness(
            model_id=train_res["model_id"],
            eval_set_path=str(custom_eval_both_zero),
        )
        assert res_both_zero["fpr_younger"] == 0.0
        assert res_both_zero["fpr_older"] == 0.0
        assert res_both_zero["predictive_equality_ratio"] == 1.0, (
            "When both groups have 0 FPR, predictive_equality_ratio must be 1.0"
        )

    # Case 2: One group has zero false positives, other group has non-zero false positives
    # Younger predicts 0 (FPR=0), older predicts 1 (FPR > 0)
    preds = [1 if row["customer_age"] >= 50 else 0 for _, row in eval_df.iterrows()]
    with patch("sklearn.linear_model.LogisticRegression.predict", return_value=pd.Series(preds).values):
        res_one_zero = check_fairness(
            model_id=train_res["model_id"],
            eval_set_path=str(custom_eval_both_zero),
        )
        assert res_one_zero["fpr_younger"] == 0.0
        assert res_one_zero["fpr_older"] > 0.0
        assert res_one_zero["predictive_equality_ratio"] == 0.0, (
            "When only one group has 0 FPR, predictive_equality_ratio must be 0.0 (maximal disparity)"
        )


# ---------------------------------------------------------------------------
# Batch 7a: Tracking Tags, Critic Tagging & Model Registry
# ---------------------------------------------------------------------------

def test_train_model_and_evaluate_mlflow_tracking_tags(base_dataset_path):
    """Verify train_model logs required tracking tags and returns run_id."""
    import mlflow

    feat_res = create_features(dataset_path=base_dataset_path)
    train_res = train_model(
        feature_set_path=feat_res["train_set_path"],
        model_type="logistic_regression",
        hyperparams={"max_iter": 50},
        dataset_variant="base",
    )

    assert "run_id" in train_res
    run_id = train_res["run_id"]
    assert run_id != ""

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    tags = run.data.tags
    assert "dataset_variant" in tags
    assert "split_method" in tags
    assert "train_months" in tags
    assert "eval_months" in tags
    assert "model_type" in tags
    assert "code_version" in tags
    assert tags["model_type"] == "logistic_regression"

    # Evaluate model and confirm metrics are appended to the same run
    eval_res = evaluate_model(
        model_id=train_res["model_id"],
        eval_set_path=feat_res["eval_set_path"],
    )
    assert eval_res["run_id"] == run_id

    run_after_eval = client.get_run(run_id)
    metrics = run_after_eval.data.metrics
    assert "eval_pr_auc" in metrics


def test_critic_verdict_tag_updates_existing_run(base_dataset_path):
    """Verify update_critic_verdict_tag sets critic_verdict on existing run without creating new run."""
    import mlflow

    feat_res = create_features(dataset_path=base_dataset_path)
    train_res = train_model(
        feature_set_path=feat_res["train_set_path"],
        model_type="logistic_regression",
        hyperparams={"max_iter": 50},
    )
    run_id = train_res["run_id"]

    client = mlflow.tracking.MlflowClient()
    run_before = client.get_run(run_id)
    assert "critic_verdict" not in run_before.data.tags

    update_critic_verdict_tag(run_id=run_id, verdict="approve")

    run_after = client.get_run(run_id)
    assert run_after.data.tags.get("critic_verdict") == "approve"
    assert run_after.info.run_id == run_id


def test_register_model_success_and_errors(base_dataset_path):
    """Verify register_model happy path and error modes (run_not_found, registration_failed)."""
    feat_res = create_features(dataset_path=base_dataset_path)
    train_res = train_model(
        feature_set_path=feat_res["train_set_path"],
        model_type="logistic_regression",
        hyperparams={"max_iter": 50},
    )
    model_id = train_res["model_id"]
    run_id = train_res["run_id"]

    # 1. Happy path
    reg_res = register_model(
        model_id=model_id,
        run_id=run_id,
        registry_name="argus_fraud_detector",
    )
    assert "version" in reg_res
    assert isinstance(reg_res["version"], int)
    assert reg_res["version"] >= 1
    assert reg_res["registry_name"] == "argus_fraud_detector"
    assert reg_res["stage"] == "Staging"

    # 2. Error case: run_not_found
    with pytest.raises(ValueError) as exc_info_run:
        register_model(
            model_id=model_id,
            run_id="non_existent_run_id_9999",
            registry_name="argus_fraud_detector",
        )
    assert "run_not_found" in str(exc_info_run.value)

    # 3. Error case: registration_failed
    with patch("mlflow.register_model", side_effect=Exception("Backend connection timeout")):
        with pytest.raises(RuntimeError) as exc_info_fail:
            register_model(
                model_id=model_id,
                run_id=run_id,
                registry_name="argus_fraud_detector",
            )
        assert "registration_failed" in str(exc_info_fail.value)

