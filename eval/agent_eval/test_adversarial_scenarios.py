"""Adversarial evaluation scenarios for Argus ML multi-agent reliability and robustness.

Tests 10 adversarial and failure-mode scenarios scored against:
  - Rubric A: Model Performance (PR-AUC, Recall@5%FPR, Predictive Equality Ratio)
  - Rubric B: Agent-System Reliability (Terminal State Classification, Tool Retry Behavior,
              No Hallucinated Success, Reproducibility)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agents.base import Agent
from agents.contracts import AgentInput, AgentOutput, ToolCallLog, ToolSpec
from agents.critic_agent import CriticAgent
from agents.data_agent import DataAgent
from agents.llm import LLMClient, LLMMessage, LLMResponse, MockLLMClient, ToolCallRequest
from eval.agent_eval.scoring import ScenarioResult, eval_collector
from monitoring.monitor import run_monitoring_pass
from pipeline.config import MAX_RETRIES_PER_STAGE
from pipeline.orchestrator import Orchestrator
from serving.api import app, model_container
from tools.data_tools import inspect_dataset, load_dataset
from tools.ml_tools import (
    check_fairness,
    create_features,
    evaluate_model,
    register_model,
    train_model,
)


# ============================================================================
# Scenario 1: Missing dataset variant
# ============================================================================

def test_scenario_01_missing_dataset_variant():
    """Scenario 1: Request a nonexistent BAF variant.

    Expect: load_dataset returns file_not_found, Data Agent returns status: 'failure',
    Orchestrator ends status: 'failed' cleanly without crashing.
    """
    start = time.perf_counter()
    variant = "nonexistent_baf_variant_xyz999"

    # 1. Direct tool call check
    with pytest.raises(FileNotFoundError) as exc_info:
        load_dataset(variant)
    assert "file_not_found" in str(exc_info.value)

    # 2. Data Agent check with simulated tool execution and failure recovery
    def missing_variant_handler(messages: list[LLMMessage], tools: list[dict[str, Any]] | None) -> LLMResponse:
        last_msg = messages[-1]
        if last_msg.role == "user":
            return LLMResponse(
                tool_calls=[
                    ToolCallRequest(
                        id="call_load_s1",
                        name="load_dataset",
                        arguments={"variant": variant},
                    )
                ]
            )
        elif last_msg.role == "tool":
            return LLMResponse(
                content=json.dumps({
                    "status": "failure",
                    "rationale": f"Fatal: Dataset variant '{variant}' not found on disk.",
                    "artifact": None,
                    "errors": [f"file_not_found: Dataset variant '{variant}' not found"],
                })
            )
        return LLMResponse(content="Fatal error")

    mock_llm = MockLLMClient(handler=missing_variant_handler)
    data_agent = DataAgent(llm_client=mock_llm)
    agent_output = data_agent.run_stage(task_id="eval-s1-data", dataset_variant=variant)
    assert agent_output.status == "failure"
    assert "not found" in agent_output.rationale.lower() or len(agent_output.errors) > 0
    # Confirm no hallucinated success artifact
    if agent_output.artifact:
        assert agent_output.artifact.get("row_count") in (0, None)
        assert agent_output.artifact.get("dataset_path") in ("", None)

    # 3. Full Orchestrator execution check
    orchestrator = Orchestrator(llm_client=mock_llm)
    task_state = orchestrator.run(dataset_variant=variant, task_id="eval-s1-orch")
    assert task_state.status == "failed"
    assert task_state.current_stage == "data_validation"

    # 4. Reproducibility check: run a second time
    mock_llm_2 = MockLLMClient(handler=missing_variant_handler)
    orchestrator_2 = Orchestrator(llm_client=mock_llm_2)
    task_state_2 = orchestrator_2.run(dataset_variant=variant, task_id="eval-s1-orch-2")
    assert task_state_2.status == "failed"

    latency = time.perf_counter() - start
    result = ScenarioResult(
        scenario="Scenario 1: Missing dataset variant",
        expected_terminal_state="failed",
        actual_terminal_state=task_state.status,
        passed=bool(task_state.status == "failed" and agent_output.status == "failure"),
        notes="Clean file_not_found propagation; DataAgent and Orchestrator failed deterministically.",
        latency_s=latency,
        rubric_b={
            "terminal_state_matched": True,
            "no_hallucinated_success": True,
            "reproducible": True,
        },
    )
    eval_collector.record(result)
    assert result.passed is True


# ============================================================================
# Scenario 2: Corrupted/truncated CSV
# ============================================================================

def test_scenario_02_corrupted_truncated_csv(tmp_path: Path):
    """Scenario 2: Construct a malformed CSV and point dataset tools at it.

    Expect: A clear raised error (RuntimeError / ValueError), not a silent partial read.
    """
    start = time.perf_counter()

    # Create corrupted CSV with broken header and truncated rows
    bad_csv = tmp_path / "corrupted_dataset.csv"
    bad_csv.write_text("invalid_col1,invalid_col2\n1,2,3,4,extra\ncorrupted_line\x00\x01\n", encoding="utf-8")

    # 1. Direct tool inspect_dataset check (missing required fraud_bool column)
    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        inspect_dataset(str(bad_csv))
    assert any(err in str(exc_info.value) for err in ("missing", "dataset_not_loaded", "fraud_bool"))

    # 2. Reproducibility: run a second time
    with pytest.raises((ValueError, RuntimeError)):
        inspect_dataset(str(bad_csv))

    latency = time.perf_counter() - start
    result = ScenarioResult(
        scenario="Scenario 2: Corrupted/truncated CSV",
        expected_terminal_state="error_raised",
        actual_terminal_state="error_raised",
        passed=True,
        notes="Corrupted CSV rejected with explicit schema/target validation error; no silent partial read.",
        latency_s=latency,
        rubric_b={
            "terminal_state_matched": True,
            "no_silent_success": True,
            "reproducible": True,
        },
    )
    eval_collector.record(result)
    assert result.passed is True


# ============================================================================
# Scenario 3: Extreme target leakage injected
# ============================================================================

def test_scenario_03_extreme_target_leakage_injected():
    """Scenario 3: Inject synthetic feature with extreme correlation to fraud_bool.

    Expect: no_extreme_target_correlation check fires in Critic review and rejects.
    """
    start = time.perf_counter()

    critic = CriticAgent()
    leaked_data_artifact = {
        "dataset_path": "data/Base.csv",
        "row_count": 1000000,
        "validation": {"severity": "ok", "total_rows": 1000000, "class_balance": {"0": 989000, "1": 11000}},
        "eda": {
            "correlation_with_target": {
                "income": 0.05,
                "credit_risk_score": 0.12,
                "synthetic_target_clone": 0.998,  # Extreme target leakage (>0.95)
            },
            "fraud_rate_by_month": {"0": 0.011, "1": 0.011},
            "high_correlation_pairs": [],
        },
    }

    critic_input = AgentInput(
        task_id="eval-s3-leakage",
        stage="critic_review",
        system_prompt=critic.system_prompt,
        context={
            "data_validation_artifact": leaked_data_artifact,
            "feature_engineering_artifact": {"manifest": {"split_method": "temporal", "feature_columns": ["income", "synthetic_target_clone"]}},
            "model_training_artifact": {
                "model_id": "model_leakage_test",
                "train_metrics": {"pr_auc": 0.999, "train_rows": 800000},
                "eval_metrics": {"pr_auc": 0.998, "recall_at_fixed_fpr": 0.995},
                "candidates_considered": [{"model_type": "logistic_regression", "eval_metrics": {"pr_auc": 0.998}}],
                "selection_reason": "Selected model with PR-AUC of 0.9980 exceeding all thresholds.",
            },
        },
    )

    output = critic.run(critic_input)
    assert output.artifact is not None
    verdict = output.artifact.get("verdict")
    failed_checks = [fc.get("check_name") for fc in output.artifact.get("failed_checks", [])]

    assert verdict == "reject"
    assert "no_extreme_target_correlation" in failed_checks

    latency = time.perf_counter() - start
    result = ScenarioResult(
        scenario="Scenario 3: Extreme target leakage injected",
        expected_terminal_state="rejected",
        actual_terminal_state=verdict or "unknown",
        passed=bool(verdict == "reject" and "no_extreme_target_correlation" in failed_checks),
        notes="Critic detected 0.998 target correlation and correctly rejected stage with no_extreme_target_correlation.",
        latency_s=latency,
        rubric_a={
            "train_pr_auc": 0.999,
            "eval_pr_auc": 0.998,
        },
        rubric_b={
            "terminal_state_matched": True,
            "failed_check_verified": "no_extreme_target_correlation",
            "no_hallucinated_success": True,
        },
    )
    eval_collector.record(result)
    assert result.passed is True


# ============================================================================
# Scenario 4: Malformed tool call arguments
# ============================================================================

def test_scenario_04_malformed_tool_call_arguments():
    """Scenario 4: Call tool with invalid types/missing args; verify agent handles it gracefully.

    Expect: Direct call raises typed error; agent runtime catches tool error in ToolCallLog and continues.
    """
    start = time.perf_counter()

    # 1. Direct tool call with invalid argument types
    with pytest.raises(TypeError):
        # Missing required model_type argument
        train_model(feature_set_path="dummy.csv")  # type: ignore

    # 2. Agent runtime handling malformed arguments through LLM tool-calling loop
    class FaultyLLMClient(LLMClient):
        def __init__(self):
            self.call_count = 0

        def generate(self, messages: list[LLMMessage], tools=None) -> LLMResponse:
            self.call_count += 1
            if self.call_count == 1:
                # Issue malformed tool call (missing required argument)
                return LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCallRequest(
                            id="call_err_1",
                            name="load_dataset",
                            arguments={},  # missing required 'variant'
                        )
                    ],
                )
            else:
                # After tool error feedback, LLM returns graceful handled response
                return LLMResponse(
                    content=json.dumps({
                        "status": "failure",
                        "rationale": "Tool call failed due to missing required argument 'variant'. Correcting execution.",
                        "artifact": None,
                    })
                )

    dummy_tool = ToolSpec(
        name="load_dataset",
        description="Load dataset",
        parameters={"type": "object", "properties": {"variant": {"type": "string"}}, "required": ["variant"]},
        handler=load_dataset,
    )

    faulty_client = FaultyLLMClient()
    agent = Agent(llm_client=faulty_client, max_turns=3)
    agent_input = AgentInput(
        task_id="eval-s4-malformed",
        stage="data_validation",
        system_prompt="Test agent",
        available_tools=[dummy_tool],
    )

    output = agent.run(agent_input)

    assert output.status == "failure"
    assert len(output.tool_calls) == 1
    assert output.tool_calls[0].error is not None
    assert "missing" in output.tool_calls[0].error.lower() or "typeerror" in output.tool_calls[0].error.lower()

    # 3. Reproducibility
    faulty_client_2 = FaultyLLMClient()
    agent_2 = Agent(llm_client=faulty_client_2, max_turns=3)
    output_2 = agent_2.run(agent_input)
    assert len(output_2.tool_calls) == 1
    assert output_2.tool_calls[0].error is not None

    latency = time.perf_counter() - start
    result = ScenarioResult(
        scenario="Scenario 4: Malformed tool call arguments",
        expected_terminal_state="handled_gracefully",
        actual_terminal_state="handled_gracefully",
        passed=bool(len(output.tool_calls) > 0 and output.tool_calls[0].error is not None),
        notes="Tool argument errors captured in ToolCallLog; agent did not crash unhandled.",
        latency_s=latency,
        rubric_b={
            "terminal_state_matched": True,
            "tool_error_logged": True,
            "reproducible": True,
        },
    )
    eval_collector.record(result)
    assert result.passed is True


# ============================================================================
# Scenario 5: Model training failure
# ============================================================================

def test_scenario_05_model_training_failure(tmp_path: Path):
    """Scenario 5: Call train_model with unsupported model_type or invalid hyperparams.

    Expect: ValueError with unsupported_model_type or invalid_hyperparams; no silent fallback.
    """
    start = time.perf_counter()

    # Create small valid feature CSV
    feat_csv = tmp_path / "train_feat_eval_s5.csv"
    pd.DataFrame({
        "income": [0.1, 0.2, 0.3, 0.4],
        "fraud_bool": [0, 0, 0, 1],
    }).to_csv(feat_csv, index=False)

    # 1. Unsupported model type
    with pytest.raises(ValueError) as exc_info:
        train_model(feature_set_path=str(feat_csv), model_type="deep_quantum_network")
    assert "training_failed" in str(exc_info.value)
    assert "Unsupported model_type" in str(exc_info.value)

    # 2. Invalid hyperparameters (e.g. unsupported hyperparameter keyword)
    with pytest.raises(ValueError) as exc_info_hp:
        train_model(
            feature_set_path=str(feat_csv),
            model_type="logistic_regression",
            hyperparams={"invalid_hyperparam_name": 123},
        )
    assert "invalid_hyperparams" in str(exc_info_hp.value)

    # 3. Reproducibility
    with pytest.raises(ValueError) as exc_info_2:
        train_model(feature_set_path=str(feat_csv), model_type="deep_quantum_network")
    assert "training_failed" in str(exc_info_2.value)

    latency = time.perf_counter() - start
    result = ScenarioResult(
        scenario="Scenario 5: Model training failure",
        expected_terminal_state="error_raised",
        actual_terminal_state="error_raised",
        passed=True,
        notes="Unsupported model_type and invalid hyperparams rejected with explicit errors; no silent fallback.",
        latency_s=latency,
        rubric_b={
            "terminal_state_matched": True,
            "unsupported_model_type_blocked": True,
            "reproducible": True,
        },
    )
    eval_collector.record(result)
    assert result.passed is True


# ============================================================================
# Scenario 6: Critic retry-loop exhaustion
# ============================================================================

def test_scenario_06_critic_retry_loop_exhaustion():
    """Scenario 6: Critic rejects every attempt continuously.

    Expect: Orchestrator halts after MAX_RETRIES_PER_STAGE attempts, status: 'failed', no infinite loop.
    """
    start = time.perf_counter()

    class AlwaysRejectCritic(Agent):
        def __init__(self):
            super().__init__(llm_client=MockLLMClient(), max_turns=1)
            self.system_prompt = "Always reject"
            self.tools = []

        def run(self, agent_input: AgentInput) -> AgentOutput:
            return AgentOutput(
                status="needs_retry",
                artifact={
                    "verdict": "reject",
                    "failed_checks": [
                        {"category": "modeling", "check_name": "generalization_gap_threshold", "reason": "Persistent gap"}
                    ],
                },
                rationale="Always rejecting to test retry budget exhaustion.",
            )

    class FastSuccessAgent(Agent):
        def __init__(self):
            super().__init__(llm_client=MockLLMClient(), max_turns=1)
            self.tools = []

        def run(self, agent_input: AgentInput) -> AgentOutput:
            return AgentOutput(
                status="success",
                artifact={"dataset_path": "data/Base.csv", "model_id": "m_test", "run_id": "r_test"},
                rationale="Fast stub success.",
            )

    custom_registry = {
        "data_validation": FastSuccessAgent(),
        "feature_engineering": FastSuccessAgent(),
        "model_training": FastSuccessAgent(),
        "critic_review": AlwaysRejectCritic(),
        "mlops_registration": FastSuccessAgent(),
    }

    orchestrator = Orchestrator(
        llm_client=MockLLMClient(),
        agent_registry=custom_registry,
        max_retries_per_stage=MAX_RETRIES_PER_STAGE,
    )

    state = orchestrator.run(dataset_variant="base", task_id="eval-s6-loop")

    assert state.status == "failed"
    # model_training should have exhausted its retry limit (MAX_RETRIES_PER_STAGE + 1 attempts total)
    assert state.retry_counts.get("model_training", 0) > MAX_RETRIES_PER_STAGE
    # Confirm pipeline bounded
    assert len(state.stage_history) <= (MAX_RETRIES_PER_STAGE + 2) * 3

    latency = time.perf_counter() - start
    result = ScenarioResult(
        scenario="Scenario 6: Critic retry-loop exhaustion",
        expected_terminal_state="failed",
        actual_terminal_state=state.status,
        passed=bool(state.status == "failed" and state.retry_counts.get("model_training", 0) > MAX_RETRIES_PER_STAGE),
        notes=f"Orchestrator cleanly halted at retry limit {MAX_RETRIES_PER_STAGE} without infinite looping.",
        latency_s=latency,
        rubric_b={
            "terminal_state_matched": True,
            "retry_budget_enforced": True,
            "max_retries": MAX_RETRIES_PER_STAGE,
            "no_infinite_loop": True,
        },
    )
    eval_collector.record(result)
    assert result.passed is True


# ============================================================================
# Scenario 7: Fairness violation on a skewed variant
# ============================================================================

def test_scenario_07_fairness_violation_on_skewed_variant():
    """Scenario 7: Evaluate algorithmic fairness on real skewed BAF dataset (Variant I or IV).

    Expect: check_fairness calculates non-empty group sizes, valid predictive equality ratio.
    """
    start = time.perf_counter()

    # Load real Variant I
    v1_data = load_dataset("i")
    v1_path = v1_data["path"]

    # Generate features & train fast model on Variant I
    feat_res = create_features(dataset_path=v1_path)
    train_res = train_model(
        feature_set_path=feat_res["train_set_path"],
        model_type="logistic_regression",
        hyperparams={"max_iter": 30},
        dataset_variant="Variant I",
    )

    fairness_res = check_fairness(
        model_id=train_res["model_id"],
        eval_set_path=feat_res["eval_set_path"],
    )

    group_sizes = fairness_res["group_sizes"]
    ratio = fairness_res["predictive_equality_ratio"]

    assert group_sizes["younger"] > 0
    assert group_sizes["older"] > 0
    assert 0.0 <= ratio <= 1.0

    eval_res = evaluate_model(
        model_id=train_res["model_id"],
        eval_set_path=feat_res["eval_set_path"],
    )
    scores = eval_res["scores"]

    latency = time.perf_counter() - start
    result = ScenarioResult(
        scenario="Scenario 7: Fairness violation on skewed variant",
        expected_terminal_state="evaluated",
        actual_terminal_state="evaluated",
        passed=bool(group_sizes["younger"] > 0 and group_sizes["older"] > 0 and 0.0 <= ratio <= 1.0),
        notes=f"Fairness evaluated on Variant I: younger={group_sizes['younger']}, older={group_sizes['older']}, ratio={ratio:.4f}",
        latency_s=latency,
        rubric_a={
            "pr_auc": scores.get("pr_auc"),
            "recall_at_fixed_fpr": scores.get("recall_at_fixed_fpr"),
            "predictive_equality_ratio": ratio,
            "fpr_younger": fairness_res["fpr_younger"],
            "fpr_older": fairness_res["fpr_older"],
            "group_sizes": group_sizes,
        },
        rubric_b={
            "terminal_state_matched": True,
            "demographic_sanity_check_passed": True,
        },
    )
    eval_collector.record(result)
    assert result.passed is True


# ============================================================================
# Scenario 8: Prediction request with unavailable model
# ============================================================================

def test_scenario_08_prediction_request_with_unavailable_model():
    """Scenario 8: Call /predict and /model-info on FastAPI serving when no model is loaded.

    Expect: Graceful HTTP 503 response ('No model is currently loaded'), not an unhandled 500 crash.
    """
    start = time.perf_counter()

    model_container.reset()
    client = TestClient(app)

    # 1. /predict endpoint
    resp_predict = client.post("/predict", json={"features": {"income": 0.5, "customer_age": 35}})
    assert resp_predict.status_code == 503
    assert "No model is currently loaded" in resp_predict.json()["detail"]

    # 2. /model-info endpoint
    resp_info = client.get("/model-info")
    assert resp_info.status_code == 503
    assert "No model is currently loaded" in resp_info.json()["detail"]

    # 3. Reproducibility
    resp_predict_2 = client.post("/predict", json={"features": {"income": 0.5}})
    assert resp_predict_2.status_code == 503

    latency = time.perf_counter() - start
    result = ScenarioResult(
        scenario="Scenario 8: Prediction request with unavailable model",
        expected_terminal_state="503_unavailable",
        actual_terminal_state="503_unavailable",
        passed=bool(resp_predict.status_code == 503 and resp_info.status_code == 503),
        notes="FastAPI serving returned 503 with informative detail message; no unhandled 500 error.",
        latency_s=latency,
        rubric_b={
            "terminal_state_matched": True,
            "graceful_error_code": 503,
            "reproducible": True,
        },
    )
    eval_collector.record(result)
    assert result.passed is True


# ============================================================================
# Scenario 9: Drifted input at serving time
# ============================================================================

def test_scenario_09_drifted_input_at_serving_time(tmp_path: Path):
    """Scenario 9: Feed drifted feature inputs into prediction log and run monitoring pass.

    Expect: Population Stability Index (PSI) signal correctly flags drifted features (>0.20 threshold).
    """
    start = time.perf_counter()

    # 1. Train and register baseline model to create reference feature set
    base_res = load_dataset("base")
    feat_res = create_features(dataset_path=base_res["path"])
    train_res = train_model(
        feature_set_path=feat_res["train_set_path"],
        model_type="logistic_regression",
        hyperparams={"max_iter": 30},
    )
    register_model(model_id=train_res["model_id"], run_id=train_res["run_id"])

    # 2. Construct synthetic drifted serving logs in tmp_path
    drift_log_path = tmp_path / "drifted_prediction_log.jsonl"
    np.random.seed(42)

    with open(drift_log_path, "w", encoding="utf-8") as f:
        # Generate 150 requests with severely drifted 'income' and 'credit_risk_score'
        for _ in range(150):
            record = {
                "request_id": f"req_{_}",
                "timestamp": "2026-09-17T12:00:00Z",
                "model_version": "1",
                "features": {
                    "income": float(np.random.normal(loc=999.0, scale=50.0)),  # Heavy shift
                    "customer_age": int(np.random.randint(20, 70)),
                    "credit_risk_score": float(np.random.normal(loc=850.0, scale=10.0)),  # Heavy shift
                },
                "prediction": 1,
                "probability": 0.85,
                "latency_ms": 12.5,
                "status": "success",
            }
            f.write(json.dumps(record) + "\n")

    # 3. Run monitoring pass against the drifted log
    report = run_monitoring_pass(
        log_path=drift_log_path,
        train_set_path=feat_res["train_set_path"],
    )

    input_drift = report.get("input_drift", {})

    income_drift = input_drift.get("income", {})
    income_flagged = income_drift.get("flagged", False)
    income_psi = income_drift.get("psi", 0.0)

    assert income_flagged is True
    assert income_psi > 0.20
    assert any("input_drift: feature 'income'" in issue for issue in report.get("issues", []))

    latency = time.perf_counter() - start
    result = ScenarioResult(
        scenario="Scenario 9: Drifted input at serving time",
        expected_terminal_state="drift_detected",
        actual_terminal_state="drift_detected" if income_flagged else "normal",
        passed=bool(income_flagged and income_psi > 0.20),
        notes=f"PSI successfully flagged drifted feature 'income' with PSI={income_psi:.4f} > 0.20.",
        latency_s=latency,
        rubric_b={
            "terminal_state_matched": True,
            "drift_signal_flagged": True,
            "psi_score": round(income_psi, 4),
        },
    )
    eval_collector.record(result)
    assert result.passed is True


# ============================================================================
# Scenario 10: Duplicate model registration
# ============================================================================

def test_scenario_10_duplicate_model_registration():
    """Scenario 10: Register the same model run_id twice in MLflow model registry.

    Expect: Handled cleanly and idempotently, returning valid version info without corrupted state.
    """
    start = time.perf_counter()

    base_res = load_dataset("base")
    feat_res = create_features(dataset_path=base_res["path"])
    train_res = train_model(
        feature_set_path=feat_res["train_set_path"],
        model_type="logistic_regression",
        hyperparams={"max_iter": 30},
    )

    # First registration
    res1 = register_model(
        model_id=train_res["model_id"],
        run_id=train_res["run_id"],
        registry_name="argus_fraud_detector",
    )
    assert "version" in res1
    assert res1["stage"] == "Staging"

    # Duplicate registration of the same run_id
    res2 = register_model(
        model_id=train_res["model_id"],
        run_id=train_res["run_id"],
        registry_name="argus_fraud_detector",
    )
    assert "version" in res2
    assert res2["stage"] == "Staging"

    # Both version numbers should be valid positive integers
    assert int(res1["version"]) >= 1
    assert int(res2["version"]) >= 1

    latency = time.perf_counter() - start
    result = ScenarioResult(
        scenario="Scenario 10: Duplicate model registration",
        expected_terminal_state="handled_gracefully",
        actual_terminal_state="handled_gracefully",
        passed=bool("version" in res1 and "version" in res2),
        notes=f"Duplicate model registration handled cleanly: version {res1['version']} -> version {res2['version']}.",
        latency_s=latency,
        rubric_b={
            "terminal_state_matched": True,
            "idempotent_or_versioned": True,
            "reproducible": True,
        },
    )
    eval_collector.record(result)
    assert result.passed is True


# ============================================================================
# Summary & Aggregation Verification Test
# ============================================================================

def test_summary_report_aggregation():
    """Verify that report aggregation correctly renders summary table with all scenario results."""
    table = eval_collector.format_summary_table()
    assert isinstance(table, str)
    assert "Argus Adversarial Evaluation Harness" in table
    assert "Summary:" in table
