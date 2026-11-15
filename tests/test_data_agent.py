"""Unit tests for Argus Data Agent."""

import json
from typing import Any
import pytest

from agents.contracts import AgentInput
from agents.data_agent import DataAgent
from agents.llm import LLMMessage, LLMResponse, MockLLMClient, ToolCallRequest


# Shared validation and EDA outputs used across tests
MOCK_VALIDATION = {
    "missingness": {
        "true_nulls": {"income": 0, "credit_risk_score": 0},
        "sentinel_missing": {"prev_address_months_count": 15200},  # > 10% of 100000 -> triggers issue
    },
    "dtype_issues": [],
    "duplicate_row_count": 0,
    "class_balance": {"counts": {"0": 988971, "1": 11029}, "ratio": 0.011029},
    "temporal_integrity": {
        "month_min": 0,
        "month_max": 7,
        "expected_range": [0, 7],
        "gaps": [],
        "out_of_range_count": 0,
    },
    "severity": "warning",  # ratio above 0.01 but sentinel_missing > 10%
}

MOCK_EDA = {
    "distribution_summary": {"income": {"mean": 0.5, "std": 0.1, "min": 0.0, "max": 1.0, "q25": 0.4, "q50": 0.5, "q75": 0.6}},
    "correlation_with_target": {
        "credit_risk_score": 0.75,   # above 0.7 → triggers leakage_signal issue
        "income": 0.12,
    },
    "high_correlation_pairs": [
        {"feature_a": "velocity_6h", "feature_b": "velocity_24h", "correlation": 0.95}
    ],
    "fraud_rate_by_month": {"0": 0.011, "1": 0.012},
}


def test_data_agent_standalone_success():
    """Verify DataAgent standalone execution produces expected artifact shape with new fields."""
    saved_path: dict[str, str] = {}

    def dynamic_handler(messages: list[LLMMessage], tools: list[dict[str, Any]] | None) -> LLMResponse:
        last_msg = messages[-1]

        if last_msg.role == "user":
            return LLMResponse(
                tool_calls=[
                    ToolCallRequest(
                        id="call_load",
                        name="load_dataset",
                        arguments={"variant": "base"},
                    )
                ]
            )

        if last_msg.role == "tool":
            if last_msg.name == "load_dataset":
                data = json.loads(last_msg.content or "{}")
                saved_path["path"] = data["path"]
                return LLMResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="call_validate",
                            name="validate_dataset",
                            arguments={"dataset_path": saved_path["path"]},
                        )
                    ]
                )
            elif last_msg.name == "validate_dataset":
                return LLMResponse(
                    tool_calls=[
                        ToolCallRequest(
                            id="call_eda",
                            name="run_eda",
                            arguments={"dataset_path": saved_path["path"]},
                        )
                    ]
                )
            elif last_msg.name == "run_eda":
                # Agent produces final artifact including both validation and eda outputs
                issues = [
                    # Deterministic minimums from MOCK_VALIDATION/EDA:
                    "sentinel_missing: prev_address_months_count has 15200 sentinel -1 values (>10% of rows)",
                    "severity: validate_dataset returned severity=warning",
                    "leakage_signal: credit_risk_score has correlation 0.75 with fraud_bool (above 0.7 — review for leakage)",
                    "high_correlation: velocity_6h and velocity_24h are correlated at 0.95 (potential multicollinearity)",
                ]
                return LLMResponse(
                    content=json.dumps({
                        "status": "success",
                        "rationale": "Dataset loaded, validated, and EDA complete. Issues catalogued.",
                        "artifact": {
                            "dataset_path": saved_path["path"],
                            "row_count": 1000000,
                            "column_count": 32,
                            "validation": MOCK_VALIDATION,
                            "eda": MOCK_EDA,
                            "issues_found": issues,
                        },
                    })
                )

        return LLMResponse(content="Fallback turn")

    mock_llm = MockLLMClient(handler=dynamic_handler)
    agent = DataAgent(llm_client=mock_llm)

    output = agent.run_stage(task_id="test-data-task", dataset_variant="base")

    assert output.status == "success"
    assert output.artifact is not None

    # New artifact shape
    assert "validation" in output.artifact
    assert "eda" in output.artifact
    assert "issues_found" in output.artifact

    # Deterministic minimum issues must be present
    issues_text = " ".join(output.artifact["issues_found"])
    assert "sentinel_missing" in issues_text, "Expected sentinel_missing issue"
    assert "leakage_signal" in issues_text, "Expected leakage_signal issue for credit_risk_score"
    assert "high_correlation" in issues_text, "Expected high_correlation multicollinearity issue"
    assert "severity" in issues_text, "Expected severity issue"

    # Validation sub-fields
    validation = output.artifact["validation"]
    assert "missingness" in validation
    assert "temporal_integrity" in validation
    assert "severity" in validation

    # EDA sub-fields
    eda = output.artifact["eda"]
    assert "distribution_summary" in eda
    assert "correlation_with_target" in eda
    assert "high_correlation_pairs" in eda
    assert "fraud_rate_by_month" in eda


def test_data_agent_critical_severity_causes_failure():
    """Verify DataAgent returns failure when validate_dataset returns severity=critical."""
    saved_path: dict[str, str] = {}

    critical_validation = {
        **MOCK_VALIDATION,
        "temporal_integrity": {
            **MOCK_VALIDATION["temporal_integrity"],
            "out_of_range_count": 500,  # triggers critical
        },
        "severity": "critical",
    }

    def handler(messages: list[LLMMessage], tools: list[dict[str, Any]] | None) -> LLMResponse:
        last_msg = messages[-1]
        if last_msg.role == "user":
            return LLMResponse(
                tool_calls=[ToolCallRequest(id="call_load", name="load_dataset", arguments={"variant": "base"})]
            )
        if last_msg.role == "tool":
            if last_msg.name == "load_dataset":
                data = json.loads(last_msg.content or "{}")
                saved_path["path"] = data["path"]
                return LLMResponse(
                    tool_calls=[ToolCallRequest(id="call_val", name="validate_dataset", arguments={"dataset_path": saved_path["path"]})]
                )
            elif last_msg.name == "validate_dataset":
                return LLMResponse(
                    tool_calls=[ToolCallRequest(id="call_eda", name="run_eda", arguments={"dataset_path": saved_path["path"]})]
                )
            elif last_msg.name == "run_eda":
                return LLMResponse(
                    content=json.dumps({
                        "status": "failure",
                        "rationale": "Critical: temporal integrity violation detected.",
                        "artifact": None,
                        "errors": ["severity: validate_dataset returned severity=critical"],
                    })
                )
        return LLMResponse(content="Fallback")

    mock_llm = MockLLMClient(handler=handler)
    agent = DataAgent(llm_client=mock_llm)
    output = agent.run_stage(task_id="test-critical-task", dataset_variant="base")

    assert output.status == "failure"
    assert len(output.errors) >= 1


def test_data_agent_fatal_failure_on_bad_variant():
    """Verify DataAgent handles non-existent variant by returning failure status."""
    def error_handler(messages: list[LLMMessage], tools: list[dict[str, Any]] | None) -> LLMResponse:
        last_msg = messages[-1]
        if last_msg.role == "user":
            return LLMResponse(
                tool_calls=[
                    ToolCallRequest(
                        id="call_load_fail",
                        name="load_dataset",
                        arguments={"variant": "invalid_variant_999"},
                    )
                ]
            )
        elif last_msg.role == "tool":
            return LLMResponse(
                content=json.dumps({
                    "status": "failure",
                    "rationale": "Fatal: Dataset file not found.",
                    "artifact": None,
                    "errors": ["file_not_found: Dataset variant 'invalid_variant_999' not found"],
                })
            )
        return LLMResponse(content="Fatal error")

    mock_llm = MockLLMClient(handler=error_handler)
    agent = DataAgent(llm_client=mock_llm)

    output = agent.run_stage(task_id="test-fail-task", dataset_variant="invalid_variant_999")

    assert output.status == "failure"
    assert len(output.errors) >= 1


def test_data_agent_deterministic_issues_enforced_regardless_of_llm():
    """Critical: verify _enforce_deterministic_issues appends the 4 required issues
    even when the LLM emits issues_found=[] (prompt non-compliance simulation).

    This is the proof that issue population is guaranteed by code, not by prompt.
    The mock LLM deliberately returns an empty issues_found list.
    After run_stage returns, the 4 minimums must be present in the artifact.
    """
    saved_path: dict[str, str] = {}

    # Validation with sentinel >10% and severity=warning
    noncompliant_validation = {
        "missingness": {
            "true_nulls": {"income": 0},
            "sentinel_missing": {"prev_address_months_count": 120000},  # 12% of 1000000
        },
        "dtype_issues": [],
        "duplicate_row_count": 0,
        "class_balance": {"counts": {"0": 988971, "1": 11029}, "ratio": 0.011029},
        "temporal_integrity": {
            "month_min": 0, "month_max": 7,
            "expected_range": [0, 7], "gaps": [], "out_of_range_count": 0,
        },
        "severity": "warning",
    }

    # EDA with a leakage signal and a high-corr pair
    noncompliant_eda = {
        "distribution_summary": {},
        "correlation_with_target": {"credit_risk_score": 0.82},  # > 0.7
        "high_correlation_pairs": [
            {"feature_a": "velocity_6h", "feature_b": "velocity_24h", "correlation": 0.95}
        ],
        "fraud_rate_by_month": {},
    }

    def llm_ignores_issues(messages: list[LLMMessage], tools: list[dict[str, Any]] | None) -> LLMResponse:
        last_msg = messages[-1]
        if last_msg.role == "user":
            return LLMResponse(
                tool_calls=[ToolCallRequest(id="c1", name="load_dataset", arguments={"variant": "base"})]
            )
        if last_msg.role == "tool":
            if last_msg.name == "load_dataset":
                saved_path["path"] = json.loads(last_msg.content or "{}")["path"]
                return LLMResponse(
                    tool_calls=[ToolCallRequest(id="c2", name="validate_dataset",
                                               arguments={"dataset_path": saved_path["path"]})]
                )
            elif last_msg.name == "validate_dataset":
                return LLMResponse(
                    tool_calls=[ToolCallRequest(id="c3", name="run_eda",
                                               arguments={"dataset_path": saved_path["path"]})]
                )
            elif last_msg.name == "run_eda":
                # LLM deliberately returns issues_found=[] — simulates prompt non-compliance
                return LLMResponse(
                    content=json.dumps({
                        "status": "success",
                        "rationale": "Done.",
                        "artifact": {
                            "dataset_path": saved_path["path"],
                            "row_count": 1000000,
                            "column_count": 32,
                            "validation": noncompliant_validation,
                            "eda": noncompliant_eda,
                            "issues_found": [],  # LLM deliberately empty
                        },
                    })
                )
        return LLMResponse(content="fallback")

    mock_llm = MockLLMClient(handler=llm_ignores_issues)
    agent = DataAgent(llm_client=mock_llm)
    output = agent.run_stage(task_id="test-deterministic", dataset_variant="base")

    assert output.status == "success"
    assert output.artifact is not None
    issues_text = " ".join(output.artifact["issues_found"])

    # All 4 deterministic minimums must be present regardless of LLM output
    assert "sentinel_missing: prev_address_months_count" in issues_text, \
        "sentinel_missing rule must be enforced in code"
    assert "high_correlation: velocity_6h and velocity_24h" in issues_text, \
        "high_correlation rule must be enforced in code"
    assert "leakage_signal: credit_risk_score" in issues_text, \
        "leakage_signal rule must be enforced in code"
    assert "severity: validate_dataset returned severity=warning" in issues_text, \
        "severity rule must be enforced in code"
