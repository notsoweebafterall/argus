"""Comprehensive unit tests for the Argus Critic Agent."""

import json
from unittest.mock import patch
import pytest

from agents.contracts import AgentInput
from agents.critic_agent import CriticAgent
from tools.data_tools import load_dataset
from tools.ml_tools import EVAL_MONTHS, TRAIN_MONTHS, create_features, train_model


@pytest.fixture(scope="module")
def base_dataset_path():
    res = load_dataset("base")
    return res["path"]


@pytest.fixture(scope="module")
def clean_pipeline_artifacts(base_dataset_path):
    """Fixture providing clean, valid artifacts from Data, Feature, and ML stages."""
    feat_res = create_features(dataset_path=base_dataset_path)
    train_res = train_model(
        feature_set_path=feat_res["train_set_path"],
        model_type="logistic_regression",
        hyperparams={"max_iter": 50},
    )

    data_artifact = {
        "dataset_path": base_dataset_path,
        "row_count": 1000000,
        "column_count": 32,
        "validation": {
            "severity": "ok",
            "total_rows": 1000000,
            "class_balance": {"0": 989000, "1": 11000},
            "sentinel_missing_counts": {"prev_address_months_count": 50000},
        },
        "eda": {
            "correlation_with_target": {"income": 0.05, "credit_risk_score": 0.12},
            "fraud_rate_by_month": {"0": 0.011, "1": 0.011},
            "high_correlation_pairs": [],
        },
    }

    feature_artifact = {
        "feature_set_path": feat_res["feature_set_path"],
        "train_set_path": feat_res["train_set_path"],
        "eval_set_path": feat_res["eval_set_path"],
        "manifest": feat_res["manifest"],
    }

    eval_scores = {
        "precision": 0.08,
        "recall": 0.65,
        "pr_auc": 0.22,
        "recall_at_fixed_fpr": 0.45,
    }

    ml_artifact = {
        "model_id": train_res["model_id"],
        "model_path": train_res["model_path"],
        "model_type": "logistic_regression",
        "train_metrics": {"train_rows": 800000, "pr_auc": 0.25, "train_accuracy": 0.98},
        "eval_metrics": eval_scores,
        "candidates_considered": [
            {"model_type": "logistic_regression", "eval_metrics": eval_scores, "hyperparams": {"max_iter": 50}},
            {"model_type": "xgboost", "eval_metrics": eval_scores, "hyperparams": {"max_depth": 3}},
        ],
        "selection_reason": "Selected Logistic Regression with PR-AUC of 0.2200 and recall@5%FPR of 0.4500 over XGBoost.",
    }

    return {
        "data_validation_artifact": data_artifact,
        "feature_engineering_artifact": feature_artifact,
        "model_training_artifact": ml_artifact,
        "eval_set_path": feat_res["eval_set_path"],
    }


# ---------------------------------------------------------------------------
# Full Approve & Full Reject Multi-Failure Tests
# ---------------------------------------------------------------------------

def test_critic_full_approve_clean_pipeline_artifacts(clean_pipeline_artifacts):
    """Verify CriticAgent approves clean pipeline artifacts and returns status='success' with no errors."""
    critic = CriticAgent()
    agent_input = AgentInput(
        task_id="test-critic-clean",
        stage="critic_review",
        system_prompt=critic.system_prompt,
        context=clean_pipeline_artifacts,
    )

    output = critic.run(agent_input)

    assert output.status == "success"
    assert output.errors == []
    assert output.artifact is not None
    assert output.artifact["verdict"] == "approve"
    assert output.artifact["failed_checks"] == []
    assert len(output.artifact["advisory_notes"]) > 0


def test_critic_full_reject_multiple_simultaneous_failures(clean_pipeline_artifacts):
    """Verify CriticAgent captures ALL failed checks simultaneously across categories, not just the first one."""
    critic = CriticAgent()

    # Create mutated context with failures across all 4 categories:
    # 1. Data: critical severity + sentinel missing > 30%
    # 2. Modeling: month in feature_columns + candidates < 2 + massive gap
    # 3. Experiment: selection reason too short
    # 4. Fairness: predictive equality ratio < 0.70
    bad_context = {
        "data_validation_artifact": {
            "row_count": 1000,
            "validation": {
                "severity": "critical",
                "sentinel_missing_counts": {"prev_address_months_count": 450},  # 45% > 30%
                "class_balance": {"0": 990, "1": 10},
            },
            "eda": {
                "correlation_with_target": {"leaky_feat": 0.99},  # > 0.95
            },
        },
        "feature_engineering_artifact": {
            "manifest": {
                "split_method": "random",  # not temporal
                "train_months": [0, 1],
                "eval_months": [2, 3],
                "feature_columns": ["income", "month", "fraud_bool"],  # month + fraud_bool leakage
            },
        },
        "model_training_artifact": {
            "model_id": clean_pipeline_artifacts["model_training_artifact"]["model_id"],
            "train_metrics": {"pr_auc": 0.90},
            "eval_metrics": {"pr_auc": 0.02},  # gap 0.88 > 0.15, and eval_pr_auc <= 0.05
            "candidates_considered": [{"model_type": "lr"}],  # < 2
            "selection_reason": "too short",  # < 20 chars
        },
        "eval_set_path": clean_pipeline_artifacts["eval_set_path"],
    }

    with patch("agents.critic_agent.check_fairness", return_value={
        "predictive_equality_ratio": 0.45,  # < 0.70
        "fpr_younger": 0.20,
        "fpr_older": 0.09,
        "group_sizes": {"younger": 500, "older": 500},
    }):
        agent_input = AgentInput(
            task_id="test-critic-multi-reject",
            stage="critic_review",
            system_prompt=critic.system_prompt,
            context=bad_context,
        )
        output = critic.run(agent_input)

    assert output.status == "needs_retry"
    assert output.errors == []  # Rejection is a valid audit outcome, not an agent exception
    assert output.artifact["verdict"] == "reject"

    failed = output.artifact["failed_checks"]
    failed_names = {fc["check_name"] for fc in failed}
    categories = {fc["category"] for fc in failed}

    # Must contain failures from all 4 categories
    assert "data" in categories
    assert "modeling" in categories
    assert "experiment" in categories
    assert "fairness" in categories

    # Verify specific checks failed simultaneously
    assert "data_severity_not_critical" in failed_names
    assert "sentinel_missing_threshold" in failed_names
    assert "no_extreme_target_correlation" in failed_names
    assert "temporal_split_integrity" in failed_names
    assert "month_not_in_feature_columns" in failed_names
    assert "fraud_bool_not_in_feature_columns" in failed_names
    assert "candidates_considered_minimum" in failed_names
    assert "generalization_gap_threshold" in failed_names
    assert "eval_pr_auc_plausibility_lower" in failed_names
    assert "selection_reason_minimum_length" in failed_names
    assert "predictive_equality_ratio_threshold" in failed_names
    assert len(failed) >= 10


# ---------------------------------------------------------------------------
# 1. Data Checks Tests (Pass & Fail)
# ---------------------------------------------------------------------------

def test_critic_data_severity_check(clean_pipeline_artifacts):
    """Verify data_severity_not_critical flags critical severity."""
    critic = CriticAgent()
    ctx = json.loads(json.dumps(clean_pipeline_artifacts))
    ctx["data_validation_artifact"]["validation"]["severity"] = "critical"

    output = critic.run(AgentInput(task_id="t1", stage="critic_review", system_prompt="", context=ctx))
    assert output.artifact["verdict"] == "reject"
    assert any(fc["check_name"] == "data_severity_not_critical" for fc in output.artifact["failed_checks"])


def test_critic_sentinel_missing_threshold_check(clean_pipeline_artifacts):
    """Verify sentinel_missing_threshold fails when a column has >30% sentinel missing values."""
    critic = CriticAgent()
    ctx = json.loads(json.dumps(clean_pipeline_artifacts))
    ctx["data_validation_artifact"]["row_count"] = 1000
    ctx["data_validation_artifact"]["validation"]["sentinel_missing_counts"] = {"current_address_months_count": 350}

    output = critic.run(AgentInput(task_id="t2", stage="critic_review", system_prompt="", context=ctx))
    assert output.artifact["verdict"] == "reject"
    assert any(fc["check_name"] == "sentinel_missing_threshold" for fc in output.artifact["failed_checks"])


def test_critic_target_correlation_threshold_check(clean_pipeline_artifacts):
    """Verify no_extreme_target_correlation fails when a feature has >0.95 correlation with fraud_bool."""
    critic = CriticAgent()
    ctx = json.loads(json.dumps(clean_pipeline_artifacts))
    ctx["data_validation_artifact"]["eda"]["correlation_with_target"]["velocity_24h"] = 0.98

    output = critic.run(AgentInput(task_id="t3", stage="critic_review", system_prompt="", context=ctx))
    assert output.artifact["verdict"] == "reject"
    assert any(fc["check_name"] == "no_extreme_target_correlation" for fc in output.artifact["failed_checks"])


def test_critic_minority_class_ratio_check(clean_pipeline_artifacts):
    """Verify minority_class_ratio fails when ratio is outside [0.001, 0.5]."""
    critic = CriticAgent()
    ctx = json.loads(json.dumps(clean_pipeline_artifacts))
    ctx["data_validation_artifact"]["row_count"] = 1000000
    ctx["data_validation_artifact"]["validation"]["class_balance"] = {"0": 999950, "1": 50}  # ratio 0.00005 < 0.001

    output = critic.run(AgentInput(task_id="t4", stage="critic_review", system_prompt="", context=ctx))
    assert output.artifact["verdict"] == "reject"
    assert any(fc["check_name"] == "minority_class_ratio" for fc in output.artifact["failed_checks"])


# ---------------------------------------------------------------------------
# 2. Modeling Checks Tests (Pass & Fail)
# ---------------------------------------------------------------------------

def test_critic_temporal_split_integrity_check(clean_pipeline_artifacts):
    """Verify temporal_split_integrity fails when split_method or month ranges differ from constants."""
    critic = CriticAgent()
    ctx = json.loads(json.dumps(clean_pipeline_artifacts))
    ctx["feature_engineering_artifact"]["manifest"]["split_method"] = "random_80_20"

    output = critic.run(AgentInput(task_id="t5", stage="critic_review", system_prompt="", context=ctx))
    assert output.artifact["verdict"] == "reject"
    assert any(fc["check_name"] == "temporal_split_integrity" for fc in output.artifact["failed_checks"])


def test_critic_month_in_features_check(clean_pipeline_artifacts):
    """Verify month_not_in_feature_columns fails when 'month' appears in manifest.feature_columns."""
    critic = CriticAgent()
    ctx = json.loads(json.dumps(clean_pipeline_artifacts))
    ctx["feature_engineering_artifact"]["manifest"]["feature_columns"].append("month")

    output = critic.run(AgentInput(task_id="t6", stage="critic_review", system_prompt="", context=ctx))
    assert output.artifact["verdict"] == "reject"
    assert any(fc["check_name"] == "month_not_in_feature_columns" for fc in output.artifact["failed_checks"])


def test_critic_fraud_bool_in_features_check(clean_pipeline_artifacts):
    """Verify fraud_bool_not_in_feature_columns fails when 'fraud_bool' appears in manifest.feature_columns."""
    critic = CriticAgent()
    ctx = json.loads(json.dumps(clean_pipeline_artifacts))
    ctx["feature_engineering_artifact"]["manifest"]["feature_columns"].append("fraud_bool")

    output = critic.run(AgentInput(task_id="t7", stage="critic_review", system_prompt="", context=ctx))
    assert output.artifact["verdict"] == "reject"
    assert any(fc["check_name"] == "fraud_bool_not_in_feature_columns" for fc in output.artifact["failed_checks"])


def test_critic_candidates_considered_minimum_check(clean_pipeline_artifacts):
    """Verify candidates_considered_minimum fails when fewer than 2 candidates are tracked."""
    critic = CriticAgent()
    ctx = json.loads(json.dumps(clean_pipeline_artifacts))
    ctx["model_training_artifact"]["candidates_considered"] = [
        {"model_type": "logistic_regression", "eval_metrics": {"pr_auc": 0.22}}
    ]

    output = critic.run(AgentInput(task_id="t8", stage="critic_review", system_prompt="", context=ctx))
    assert output.artifact["verdict"] == "reject"
    assert any(fc["check_name"] == "candidates_considered_minimum" for fc in output.artifact["failed_checks"])


def test_critic_generalization_gap_check(clean_pipeline_artifacts):
    """Verify generalization_gap_threshold fails when |train_pr_auc - eval_pr_auc| > 0.15."""
    critic = CriticAgent()
    ctx = json.loads(json.dumps(clean_pipeline_artifacts))
    ctx["model_training_artifact"]["train_metrics"]["pr_auc"] = 0.50
    ctx["model_training_artifact"]["eval_metrics"]["pr_auc"] = 0.20  # gap = 0.30 > 0.15

    output = critic.run(AgentInput(task_id="t9", stage="critic_review", system_prompt="", context=ctx))
    assert output.artifact["verdict"] == "reject"
    assert any(fc["check_name"] == "generalization_gap_threshold" for fc in output.artifact["failed_checks"])


def test_critic_eval_pr_auc_plausibility_bounds(clean_pipeline_artifacts):
    """Verify eval_pr_auc fails when > 0.999 (leakage) or <= 0.05 (chance)."""
    critic = CriticAgent()

    # Case 1: > 0.999
    ctx_high = json.loads(json.dumps(clean_pipeline_artifacts))
    ctx_high["model_training_artifact"]["eval_metrics"]["pr_auc"] = 1.00
    ctx_high["model_training_artifact"]["train_metrics"]["pr_auc"] = 0.99
    out_high = critic.run(AgentInput(task_id="t10a", stage="critic_review", system_prompt="", context=ctx_high))
    assert any(fc["check_name"] == "eval_pr_auc_plausibility_upper" for fc in out_high.artifact["failed_checks"])

    # Case 2: <= 0.05
    ctx_low = json.loads(json.dumps(clean_pipeline_artifacts))
    ctx_low["model_training_artifact"]["eval_metrics"]["pr_auc"] = 0.03
    ctx_low["model_training_artifact"]["train_metrics"]["pr_auc"] = 0.04
    out_low = critic.run(AgentInput(task_id="t10b", stage="critic_review", system_prompt="", context=ctx_low))
    assert any(fc["check_name"] == "eval_pr_auc_plausibility_lower" for fc in out_low.artifact["failed_checks"])


# ---------------------------------------------------------------------------
# 3. Experiment Checks Tests (Pass & Fail)
# ---------------------------------------------------------------------------

def test_critic_selection_reason_metric_grounding_check(clean_pipeline_artifacts):
    """Verify selection_reason_metric_grounding fails when reason does not match any eval metric values."""
    critic = CriticAgent()
    ctx = json.loads(json.dumps(clean_pipeline_artifacts))
    ctx["model_training_artifact"]["selection_reason"] = (
        "This model is definitely the greatest model ever built because it looks fantastic and smells nice."
    )

    output = critic.run(AgentInput(task_id="t11", stage="critic_review", system_prompt="", context=ctx))
    assert output.artifact["verdict"] == "reject"
    assert any(fc["check_name"] == "selection_reason_metric_grounding" for fc in output.artifact["failed_checks"])


def test_critic_selection_reason_length_check(clean_pipeline_artifacts):
    """Verify selection_reason_minimum_length fails when reason is shorter than 20 characters."""
    critic = CriticAgent()
    ctx = json.loads(json.dumps(clean_pipeline_artifacts))
    ctx["model_training_artifact"]["selection_reason"] = "best 0.22"

    output = critic.run(AgentInput(task_id="t12", stage="critic_review", system_prompt="", context=ctx))
    assert output.artifact["verdict"] == "reject"
    assert any(fc["check_name"] == "selection_reason_minimum_length" for fc in output.artifact["failed_checks"])


# ---------------------------------------------------------------------------
# 4. Fairness Checks Tests (Pass & Fail)
# ---------------------------------------------------------------------------

def test_critic_predictive_equality_ratio_check(clean_pipeline_artifacts):
    """Verify predictive_equality_ratio_threshold fails when ratio < 0.70."""
    critic = CriticAgent()
    with patch("agents.critic_agent.check_fairness", return_value={
        "predictive_equality_ratio": 0.55,  # < 0.70
        "fpr_younger": 0.20,
        "fpr_older": 0.11,
        "group_sizes": {"younger": 500, "older": 500},
    }):
        output = critic.run(AgentInput(task_id="t13", stage="critic_review", system_prompt="", context=clean_pipeline_artifacts))

    assert output.artifact["verdict"] == "reject"
    assert any(fc["check_name"] == "predictive_equality_ratio_threshold" for fc in output.artifact["failed_checks"])


def test_critic_fairness_empty_group_check(clean_pipeline_artifacts):
    """Verify fairness_groups_non_empty fails when one demographic group is empty."""
    critic = CriticAgent()
    with patch("agents.critic_agent.check_fairness", return_value={
        "predictive_equality_ratio": 1.0,
        "fpr_younger": 0.05,
        "fpr_older": 0.05,
        "group_sizes": {"younger": 500, "older": 0},  # older group is empty
    }):
        output = critic.run(AgentInput(task_id="t14", stage="critic_review", system_prompt="", context=clean_pipeline_artifacts))

    assert output.artifact["verdict"] == "reject"
    assert any(fc["check_name"] == "fairness_groups_non_empty" for fc in output.artifact["failed_checks"])

def test_critic_cites_similar_past_failure_on_reject(clean_pipeline_artifacts, monkeypatch):
    """When retrieval returns a matching past failure, the critic cites it in rationale and artifact."""
    from tools import retrieval_tools

    monkeypatch.setattr(
        retrieval_tools,
        "retrieve_similar_failures",
        lambda query, k=3: {
            "matches": [
                {
                    "run_id": "run_abc123",
                    "score": 0.91,
                    "failed_checks": [
                        {"check_name": "sentinel_missing_threshold", "category": "data"}
                    ],
                }
            ]
        },
        raising=False,
    )

    critic = CriticAgent()
    bad_context = dict(clean_pipeline_artifacts)
    bad_context["data_validation_artifact"]["validation"]["severity"] = "critical"

    out = critic.run(AgentInput(task_id="t", stage="critic_review", system_prompt="", context=bad_context))

    assert out.artifact["verdict"] == "reject"
    assert out.artifact["retrieved_similar_failures"]  # non-empty
    assert "run_abc123" in out.rationale