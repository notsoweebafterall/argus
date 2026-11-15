"""Unit and integration tests for the Argus Orchestrator."""

import json
import os
from typing import Any
from unittest.mock import patch
import pytest

from agents.base import Agent
from agents.contracts import AgentInput, AgentOutput
from agents.data_agent import DataAgent
from agents.feature_agent import FeatureAgent
from agents.ml_agent import MLAgent
from agents.critic_agent import CriticAgent
from agents.mlops_agent import MLOpsAgent
from agents.llm import LLMMessage, LLMResponse, MockLLMClient, ToolCallRequest
from pipeline.orchestrator import Orchestrator, run_task
from pipeline.config import MAX_RETRIES_PER_STAGE


def test_orchestrator_e2e_run_task_base():
    """End-to-end integration test: Full 5-stage graph runs on Base variant."""
    shared_data: dict[str, Any] = {}

    def full_pipeline_handler(messages: list[LLMMessage], tools: list[dict[str, Any]] | None) -> LLMResponse:
        system_msg = messages[0].content or ""
        last_msg = messages[-1]

        # 1. Data Agent
        if "You are the Argus Data Agent" in system_msg:
            if last_msg.role == "user":
                return LLMResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="call_load_data",
                            name="load_dataset",
                            arguments={"variant": "base"},
                        )
                    ]
                )
            elif last_msg.role == "tool" and last_msg.name == "load_dataset":
                data = json.loads(last_msg.content or "{}")
                shared_data["dataset_path"] = data["path"]
                return LLMResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="call_inspect_data",
                            name="inspect_dataset",
                            arguments={"dataset_path": shared_data["dataset_path"]},
                        )
                    ]
                )
            elif last_msg.role == "tool" and last_msg.name == "inspect_dataset":
                return LLMResponse(
                    content=json.dumps({
                        "status": "success",
                        "rationale": "Data validation passed. Verified 1M rows across 32 columns.",
                        "artifact": {
                            "dataset_path": shared_data["dataset_path"],
                            "row_count": 1000000,
                            "column_count": 32,
                            "validation": {
                                "severity": "ok",
                                "total_rows": 1000000,
                                "class_balance": {"0": 988971, "1": 11029},
                                "sentinel_missing_counts": {},
                            },
                            "eda": {
                                "correlation_with_target": {},
                                "fraud_rate_by_month": {},
                                "high_correlation_pairs": [],
                            },
                            "null_summary": {},
                            "duplicate_row_count": 0,
                            "class_balance": {"0": 988971, "1": 11029},
                            "month_range": [0, 7],
                            "issues_found": ["Severe class imbalance"],
                        },
                    })
                )

        # 2. Feature Agent
        elif "You are the Argus Feature Agent" in system_msg:
            if last_msg.role == "user":
                dpath = shared_data.get("dataset_path")
                return LLMResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="call_create_features",
                            name="create_features",
                            arguments={"dataset_path": dpath},
                        )
                    ]
                )
            elif last_msg.role == "tool" and last_msg.name == "create_features":
                data = json.loads(last_msg.content or "{}")
                shared_data["train_set_path"] = data["train_set_path"]
                shared_data["eval_set_path"] = data["eval_set_path"]
                return LLMResponse(
                    content=json.dumps({
                        "status": "success",
                        "rationale": "Generated train and eval feature sets with temporal split. Excluded fraud_bool.",
                        "artifact": {
                            "feature_set_path": data["feature_set_path"],
                            "train_set_path": data["train_set_path"],
                            "eval_set_path": data["eval_set_path"],
                            "manifest": data["manifest"],
                            "feature_count": data["manifest"]["feature_count"],
                        },
                    })
                )

        # 3. ML Agent
        elif "You are the Argus ML Agent" in system_msg:
            if last_msg.role == "user":
                train_p = shared_data.get("train_set_path")
                return LLMResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="call_train_model",
                            name="train_model",
                            arguments={
                                "feature_set_path": train_p,
                                "model_type": "logistic_regression",
                                "hyperparams": {"max_iter": 50},
                            },
                        )
                    ]
                )
            elif last_msg.role == "tool" and last_msg.name == "train_model":
                data = json.loads(last_msg.content or "{}")
                shared_data["model_id"] = data["model_id"]
                shared_data["model_path"] = data["model_path"]
                shared_data["train_metrics"] = data["train_metrics"]
                shared_data["run_id"] = data.get("run_id") or data.get("mlflow_run_id")
                eval_p = shared_data.get("eval_set_path")
                return LLMResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="call_eval_model",
                            name="evaluate_model",
                            arguments={
                                "model_id": shared_data["model_id"],
                                "eval_set_path": eval_p,
                                "metrics": ["precision", "recall", "pr_auc", "recall_at_fixed_fpr"],
                            },
                        )
                    ]
                )
            elif last_msg.role == "tool" and last_msg.name == "evaluate_model":
                eval_data = json.loads(last_msg.content or "{}")
                scores = eval_data.get("scores", {})
                return LLMResponse(
                    content=json.dumps({
                        "status": "success",
                        "rationale": f"Trained baseline model. Achieved PR-AUC={scores.get('pr_auc')}.",
                        "artifact": {
                            "model_id": shared_data["model_id"],
                            "model_path": shared_data["model_path"],
                            "run_id": shared_data.get("run_id"),
                            "model_type": "logistic_regression",
                            "train_metrics": shared_data["train_metrics"],
                            "eval_metrics": scores,
                            "candidates_considered": [
                                {
                                    "model_type": "logistic_regression",
                                    "eval_metrics": scores,
                                    "hyperparams": {"max_iter": 50},
                                },
                                {
                                    "model_type": "xgboost",
                                    "eval_metrics": scores,
                                    "hyperparams": {"max_depth": 3},
                                },
                            ],
                            "selection_reason": f"Baseline logistic regression selected with PR-AUC of {scores.get('pr_auc')}.",
                        },
                    })
                )

        return LLMResponse(content="Fallback turn")

    mock_llm = MockLLMClient(handler=full_pipeline_handler)
    state = run_task(dataset_variant="base", llm_client=mock_llm)

    assert state.status == "completed"
    assert len(state.stage_history) == 5

    stages_recorded = [r.stage for r in state.stage_history]
    assert stages_recorded == [
        "data_validation",
        "feature_engineering",
        "model_training",
        "critic_review",
        "mlops_registration",
    ]

    # Check Critic record
    critic_record = state.stage_history[3]
    assert critic_record.stage == "critic_review"
    assert critic_record.agent == "CriticAgent"
    assert critic_record.critic_verdict == "approve"

    # Check MLOps record
    mlops_record = state.stage_history[4]
    assert mlops_record.stage == "mlops_registration"
    assert mlops_record.agent == "MLOpsAgent"
    assert mlops_record.output.status == "success"
    assert mlops_record.output.artifact["action"] == "registered"
    assert mlops_record.output.artifact["model_id"] == shared_data["model_id"]
    assert mlops_record.output.artifact["registry_name"] == "argus_fraud_detector"
    assert "version" in mlops_record.output.artifact
    assert mlops_record.output.artifact["stage"] == "Staging"

    # Final report
    assert state.final_report is not None
    assert state.final_report["status"] == "completed"
    assert "data_validation" in state.final_report
    assert "feature_engineering" in state.final_report
    assert "model_training" in state.final_report
    assert "critic_review" in state.final_report
    assert "mlops_registration" in state.final_report


def test_orchestrator_critic_rejection_routes_back_to_model_training():
    """Verify Orchestrator routes back to model_training when Critic rejects."""
    critic_call_count = 0

    class RejectingCriticAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            nonlocal critic_call_count
            critic_call_count += 1
            if critic_call_count == 1:
                return AgentOutput(
                    status="success",
                    artifact={
                        "verdict": "reject",
                        "reason": "PR-AUC metric below quality threshold.",
                    },
                    rationale="Quality threshold not met on first evaluation.",
                )
            return AgentOutput(
                status="success",
                artifact={"verdict": "approve", "reason": "Retrained model approved."},
                rationale="Approved on second review.",
            )

    ml_training_call_count = 0

    class CountingMLAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            nonlocal ml_training_call_count
            ml_training_call_count += 1
            if ml_training_call_count == 2:
                # Assert that critic rejection feedback was routed back to model_training
                assert agent_input.context.get("critic_verdict") == "reject"
                assert "critic_feedback" in agent_input.context
            return AgentOutput(
                status="success",
                artifact={
                    "model_id": f"model_v{ml_training_call_count}",
                    "run_id": f"run_v{ml_training_call_count}",
                },
                rationale="Trained candidate model.",
            )

    class FastDataAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            return AgentOutput(status="success", artifact={"dataset_path": "data/Base.csv"}, rationale="ok")

    class FastFeatureAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            return AgentOutput(status="success", artifact={"train_set_path": "data/train.csv"}, rationale="ok")

    orchestrator = Orchestrator(
        llm_client=MockLLMClient(),
        agent_registry={
            "data_validation": FastDataAgent(llm_client=MockLLMClient()),
            "feature_engineering": FastFeatureAgent(llm_client=MockLLMClient()),
            "model_training": CountingMLAgent(llm_client=MockLLMClient()),
            "critic_review": RejectingCriticAgent(llm_client=MockLLMClient()),
            "mlops_registration": MLOpsAgent(llm_client=MockLLMClient()),
        },
    )

    with patch("agents.mlops_agent.register_model", return_value={"version": 1, "registry_name": "argus_fraud_detector", "stage": "Staging"}):
        state = orchestrator.run(dataset_variant="base")

    assert state.status == "completed"
    assert ml_training_call_count == 2
    assert critic_call_count == 2

    # Expected sequence: data_val, feat_eng, model_train (1), critic (reject), model_train (2), critic (approve), mlops
    history_stages = [r.stage for r in state.stage_history]
    assert history_stages == [
        "data_validation",
        "feature_engineering",
        "model_training",
        "critic_review",
        "model_training",
        "critic_review",
        "mlops_registration",
    ]


def test_orchestrator_forced_failure_on_bad_variant():
    """Verify Orchestrator sets status='failed' and stops immediately on stage failure."""
    def fail_handler(messages: list[LLMMessage], tools: list[dict[str, Any]] | None) -> LLMResponse:
        return LLMResponse(
            content=json.dumps({
                "status": "failure",
                "rationale": "Variant does not exist.",
                "artifact": None,
                "errors": ["file_not_found: Non-existent variant"],
            })
        )

    mock_llm = MockLLMClient(handler=fail_handler)
    state = run_task(dataset_variant="invalid_variant_xyz", llm_client=mock_llm)

    assert state.status == "failed"
    assert len(state.stage_history) == 1
    assert state.stage_history[0].stage == "data_validation"
    assert state.stage_history[0].output.status == "failure"
    assert state.final_report is None


def test_orchestrator_retry_enrichment_and_recovery():
    """Verify Orchestrator enriches context with previous errors on needs_retry, and agent recovers."""
    attempt = 0

    class RetryingMockAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                return AgentOutput(
                    status="needs_retry",
                    artifact=None,
                    rationale="Encountered transient file lock during ingestion.",
                    tool_calls=[],
                    errors=["transient_lock_error: File busy"],
                )
            else:
                assert "previous_attempt_errors" in agent_input.context
                assert "transient_lock_error: File busy" in agent_input.context["previous_attempt_errors"]
                assert agent_input.context["retry_attempt"] == 1
                assert agent_input.retry_count == 1
                return AgentOutput(
                    status="success",
                    artifact={"recovered": True},
                    rationale="Recovered successfully on retry attempt 2.",
                    tool_calls=[],
                    errors=[],
                )

    retrying_agent = RetryingMockAgent(llm_client=MockLLMClient())
    orchestrator = Orchestrator(
        llm_client=MockLLMClient(),
        stages=["data_validation"],
        agent_registry={"data_validation": retrying_agent},
    )

    state = orchestrator.run(dataset_variant="base")

    assert state.status == "completed"
    assert len(state.stage_history) == 2
    assert state.stage_history[0].output.status == "needs_retry"
    assert state.stage_history[1].output.status == "success"
    assert state.retry_counts["data_validation"] == 1


def test_orchestrator_max_retries_exceeded():
    """Verify Orchestrator halts when retry budget is exceeded."""
    class EndlessRetryAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            return AgentOutput(
                status="needs_retry",
                artifact=None,
                rationale="Still failing transient condition.",
                tool_calls=[],
                errors=["persistent_failure"],
            )

    endless_agent = EndlessRetryAgent(llm_client=MockLLMClient())
    orchestrator = Orchestrator(
        llm_client=MockLLMClient(),
        stages=["data_validation"],
        agent_registry={"data_validation": endless_agent},
        max_retries_per_stage=MAX_RETRIES_PER_STAGE,
    )

    state = orchestrator.run(dataset_variant="base")

    assert state.status == "failed"
    assert len(state.stage_history) == MAX_RETRIES_PER_STAGE + 1
    assert state.retry_counts["data_validation"] == MAX_RETRIES_PER_STAGE + 1


# ---------------------------------------------------------------------------
# Batch 6b: Category-Aware Routing & Failed-Checks Context Enrichment Tests
# ---------------------------------------------------------------------------

def test_orchestrator_critic_rejection_data_category_routes_to_data_validation():
    """Verify data-category rejection routes back to data_validation and increments its counter."""
    data_call_count = 0
    critic_call_count = 0

    class CountingDataAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            nonlocal data_call_count
            data_call_count += 1
            return AgentOutput(status="success", artifact={"dataset_path": "data/Base.csv"}, rationale="data ok")

    class FastFeatureAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            return AgentOutput(status="success", artifact={"train_set_path": "data/train.csv"}, rationale="feat ok")

    class FastMLAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            return AgentOutput(status="success", artifact={"model_id": "m1", "run_id": "run_test_1"}, rationale="ml ok")

    class DataRejectingCriticAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            nonlocal critic_call_count
            critic_call_count += 1
            if critic_call_count == 1:
                return AgentOutput(
                    status="needs_retry",
                    artifact={
                        "verdict": "reject",
                        "failed_checks": [
                            {
                                "check_name": "data_severity_not_critical",
                                "category": "data",
                                "reason": "Data validation produced critical severity",
                            }
                        ],
                    },
                    rationale="Data severity critical on first run.",
                )
            return AgentOutput(
                status="success",
                artifact={"verdict": "approve", "failed_checks": []},
                rationale="Approved on rerun.",
            )

    orchestrator = Orchestrator(
        llm_client=MockLLMClient(),
        agent_registry={
            "data_validation": CountingDataAgent(llm_client=MockLLMClient()),
            "feature_engineering": FastFeatureAgent(llm_client=MockLLMClient()),
            "model_training": FastMLAgent(llm_client=MockLLMClient()),
            "critic_review": DataRejectingCriticAgent(llm_client=MockLLMClient()),
            "mlops_registration": MLOpsAgent(llm_client=MockLLMClient()),
        },
    )

    with patch("agents.mlops_agent.register_model", return_value={"version": 1, "registry_name": "argus_fraud_detector", "stage": "Staging"}):
        state = orchestrator.run(dataset_variant="base")

    assert state.status == "completed"
    assert data_call_count == 2
    assert state.retry_counts["data_validation"] == 1
    assert state.retry_counts["critic_review"] == 0
    assert state.retry_counts["model_training"] == 0

    history_stages = [r.stage for r in state.stage_history]
    assert history_stages == [
        "data_validation",
        "feature_engineering",
        "model_training",
        "critic_review",
        "data_validation",
        "feature_engineering",
        "model_training",
        "critic_review",
        "mlops_registration",
    ]


def test_orchestrator_critic_rejection_leakage_routes_to_feature_engineering():
    """Verify split-key and target leakage rejections route back to feature_engineering."""
    feat_call_count = 0
    critic_call_count = 0

    class FastDataAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            return AgentOutput(status="success", artifact={"dataset_path": "data/Base.csv"}, rationale="data ok")

    class CountingFeatureAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            nonlocal feat_call_count
            feat_call_count += 1
            return AgentOutput(status="success", artifact={"train_set_path": "data/train.csv"}, rationale="feat ok")

    class FastMLAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            return AgentOutput(status="success", artifact={"model_id": "m1", "run_id": "run_test_1"}, rationale="ml ok")

    class LeakageRejectingCriticAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            nonlocal critic_call_count
            critic_call_count += 1
            if critic_call_count == 1:
                return AgentOutput(
                    status="needs_retry",
                    artifact={
                        "verdict": "reject",
                        "failed_checks": [
                            {
                                "check_name": "month_not_in_feature_columns",
                                "category": "modeling",
                                "reason": "'month' is the temporal split key and must not be used as a model feature",
                            }
                        ],
                    },
                    rationale="Leakage detected in feature columns.",
                )
            return AgentOutput(
                status="success",
                artifact={"verdict": "approve", "failed_checks": []},
                rationale="Approved on rerun.",
            )

    orchestrator = Orchestrator(
        llm_client=MockLLMClient(),
        agent_registry={
            "data_validation": FastDataAgent(llm_client=MockLLMClient()),
            "feature_engineering": CountingFeatureAgent(llm_client=MockLLMClient()),
            "model_training": FastMLAgent(llm_client=MockLLMClient()),
            "critic_review": LeakageRejectingCriticAgent(llm_client=MockLLMClient()),
            "mlops_registration": MLOpsAgent(llm_client=MockLLMClient()),
        },
    )

    with patch("agents.mlops_agent.register_model", return_value={"version": 1, "registry_name": "argus_fraud_detector", "stage": "Staging"}):
        state = orchestrator.run(dataset_variant="base")

    assert state.status == "completed"
    assert feat_call_count == 2
    assert state.retry_counts["feature_engineering"] == 1
    assert state.retry_counts["data_validation"] == 0
    assert state.retry_counts["model_training"] == 0

    history_stages = [r.stage for r in state.stage_history]
    assert history_stages == [
        "data_validation",
        "feature_engineering",
        "model_training",
        "critic_review",
        "feature_engineering",
        "model_training",
        "critic_review",
        "mlops_registration",
    ]


def test_orchestrator_critic_rejection_modeling_fairness_routes_to_model_training():
    """Verify non-leakage modeling, experiment, and fairness rejections route back to model_training."""
    ml_call_count = 0
    critic_call_count = 0

    class FastDataAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            return AgentOutput(status="success", artifact={"dataset_path": "data/Base.csv"}, rationale="data ok")

    class FastFeatureAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            return AgentOutput(status="success", artifact={"train_set_path": "data/train.csv"}, rationale="feat ok")

    class CountingMLAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            nonlocal ml_call_count
            ml_call_count += 1
            return AgentOutput(status="success", artifact={"model_id": f"m{ml_call_count}", "run_id": f"run_test_{ml_call_count}"}, rationale="ml ok")

    class FairnessRejectingCriticAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            nonlocal critic_call_count
            critic_call_count += 1
            if critic_call_count == 1:
                return AgentOutput(
                    status="needs_retry",
                    artifact={
                        "verdict": "reject",
                        "failed_checks": [
                            {
                                "check_name": "predictive_equality_ratio_threshold",
                                "category": "fairness",
                                "reason": "Predictive equality ratio 0.45 below 0.70 threshold",
                            }
                        ],
                    },
                    rationale="Fairness disparity detected.",
                )
            return AgentOutput(
                status="success",
                artifact={"verdict": "approve", "failed_checks": []},
                rationale="Approved on rerun.",
            )

    orchestrator = Orchestrator(
        llm_client=MockLLMClient(),
        agent_registry={
            "data_validation": FastDataAgent(llm_client=MockLLMClient()),
            "feature_engineering": FastFeatureAgent(llm_client=MockLLMClient()),
            "model_training": CountingMLAgent(llm_client=MockLLMClient()),
            "critic_review": FairnessRejectingCriticAgent(llm_client=MockLLMClient()),
            "mlops_registration": MLOpsAgent(llm_client=MockLLMClient()),
        },
    )

    with patch("agents.mlops_agent.register_model", return_value={"version": 1, "registry_name": "argus_fraud_detector", "stage": "Staging"}):
        state = orchestrator.run(dataset_variant="base")

    assert state.status == "completed"
    assert ml_call_count == 2
    assert state.retry_counts["model_training"] == 1
    assert state.retry_counts["feature_engineering"] == 0
    assert state.retry_counts["data_validation"] == 0

    history_stages = [r.stage for r in state.stage_history]
    assert history_stages == [
        "data_validation",
        "feature_engineering",
        "model_training",
        "critic_review",
        "model_training",
        "critic_review",
        "mlops_registration",
    ]


def test_orchestrator_critic_rejection_priority_mixed_data_and_modeling_routes_to_data_validation():
    """Verify mixed data + modeling failures prioritize routing to data_validation first."""
    data_call_count = 0
    feat_call_count = 0
    ml_call_count = 0
    critic_call_count = 0

    class CountingDataAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            nonlocal data_call_count
            data_call_count += 1
            return AgentOutput(status="success", artifact={"dataset_path": "data/Base.csv"}, rationale="data ok")

    class CountingFeatureAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            nonlocal feat_call_count
            feat_call_count += 1
            return AgentOutput(status="success", artifact={"train_set_path": "data/train.csv"}, rationale="feat ok")

    class CountingMLAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            nonlocal ml_call_count
            ml_call_count += 1
            return AgentOutput(status="success", artifact={"model_id": "m1", "run_id": "run_test_1"}, rationale="ml ok")

    class MultiRejectingCriticAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            nonlocal critic_call_count
            critic_call_count += 1
            if critic_call_count == 1:
                return AgentOutput(
                    status="needs_retry",
                    artifact={
                        "verdict": "reject",
                        "failed_checks": [
                            {
                                "check_name": "month_not_in_feature_columns",
                                "category": "modeling",
                                "reason": "Month column found in feature_columns",
                            },
                            {
                                "check_name": "sentinel_missing_threshold",
                                "category": "data",
                                "reason": "Column prev_address_months_count has 35% sentinel missing",
                            },
                        ],
                    },
                    rationale="Multiple simultaneous failures.",
                )
            return AgentOutput(
                status="success",
                artifact={"verdict": "approve", "failed_checks": []},
                rationale="Approved on rerun.",
            )

    orchestrator = Orchestrator(
        llm_client=MockLLMClient(),
        agent_registry={
            "data_validation": CountingDataAgent(llm_client=MockLLMClient()),
            "feature_engineering": CountingFeatureAgent(llm_client=MockLLMClient()),
            "model_training": CountingMLAgent(llm_client=MockLLMClient()),
            "critic_review": MultiRejectingCriticAgent(llm_client=MockLLMClient()),
            "mlops_registration": MLOpsAgent(llm_client=MockLLMClient()),
        },
    )

    with patch("agents.mlops_agent.register_model", return_value={"version": 1, "registry_name": "argus_fraud_detector", "stage": "Staging"}):
        state = orchestrator.run(dataset_variant="base")

    assert state.status == "completed"
    assert data_call_count == 2
    assert state.retry_counts["data_validation"] == 1
    assert state.retry_counts["feature_engineering"] == 0
    assert state.retry_counts["model_training"] == 0


def test_orchestrator_critic_rejection_enriches_stage_context_with_failed_checks():
    """Verify target stage context receives detailed failed_checks with reasons on retry."""
    captured_retry_context: dict[str, Any] = {}
    data_call_count = 0
    critic_call_count = 0

    class ContextCapturingDataAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            nonlocal data_call_count, captured_retry_context
            data_call_count += 1
            if data_call_count == 2:
                captured_retry_context = dict(agent_input.context)
            return AgentOutput(status="success", artifact={"dataset_path": "data/Base.csv"}, rationale="data ok")

    class FastFeatureAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            return AgentOutput(status="success", artifact={"train_set_path": "data/train.csv"}, rationale="feat ok")

    class FastMLAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            return AgentOutput(status="success", artifact={"model_id": "m1", "run_id": "run_test_1"}, rationale="ml ok")

    class DetailRejectingCriticAgent(Agent):
        def run(self, agent_input: AgentInput) -> AgentOutput:
            nonlocal critic_call_count
            critic_call_count += 1
            if critic_call_count == 1:
                return AgentOutput(
                    status="needs_retry",
                    artifact={
                        "verdict": "reject",
                        "failed_checks": [
                            {
                                "check_name": "sentinel_missing_threshold",
                                "category": "data",
                                "expected": "sentinel_missing <= 30.0% of rows for all columns",
                                "actual": "column 'prev_address_months_count' has 350000 sentinel values (35.0%)",
                                "reason": "Column 'prev_address_months_count' has 350000/1000000 (35.0%) sentinel-missing values, exceeding the 30% threshold.",
                            }
                        ],
                    },
                    rationale="Sentinel missing threshold violated.",
                )
            return AgentOutput(
                status="success",
                artifact={"verdict": "approve", "failed_checks": []},
                rationale="Approved on rerun.",
            )

    orchestrator = Orchestrator(
        llm_client=MockLLMClient(),
        agent_registry={
            "data_validation": ContextCapturingDataAgent(llm_client=MockLLMClient()),
            "feature_engineering": FastFeatureAgent(llm_client=MockLLMClient()),
            "model_training": FastMLAgent(llm_client=MockLLMClient()),
            "critic_review": DetailRejectingCriticAgent(llm_client=MockLLMClient()),
            "mlops_registration": MLOpsAgent(llm_client=MockLLMClient()),
        },
    )

    with patch("agents.mlops_agent.register_model", return_value={"version": 1, "registry_name": "argus_fraud_detector", "stage": "Staging"}):
        state = orchestrator.run(dataset_variant="base")

    assert state.status == "completed"
    assert data_call_count == 2
    assert captured_retry_context.get("critic_verdict") == "reject"
    assert "failed_checks" in captured_retry_context
    assert len(captured_retry_context["failed_checks"]) == 1

    fc = captured_retry_context["failed_checks"][0]
    assert fc["check_name"] == "sentinel_missing_threshold"
    assert fc["category"] == "data"
    assert "35.0%" in fc["reason"]
    assert captured_retry_context.get("retry_attempt") == 1

