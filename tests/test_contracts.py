"""Tests for Argus ML data contracts."""

import pytest
from pydantic import ValidationError

from agents.contracts import (
    AgentInput,
    AgentOutput,
    StageRecord,
    TaskState,
    ToolCallLog,
    ToolSpec,
)
from pipeline.config import MAX_RETRIES_PER_STAGE


def test_max_retries_per_stage_constant():
    """Verify MAX_RETRIES_PER_STAGE constant is defined and equal to 3."""
    assert MAX_RETRIES_PER_STAGE == 3


def test_tool_call_log_contract():
    """Verify ToolCallLog fields, defaults, and serialization."""
    log = ToolCallLog(
        tool_name="test_tool",
        input={"param": "value"},
        output={"result": 123},
        error=None,
        duration_ms=45.2,
    )
    assert log.tool_name == "test_tool"
    assert log.input == {"param": "value"}
    assert log.output == {"result": 123}
    assert log.error is None
    assert log.duration_ms == 45.2

    serialized = log.model_dump()
    assert serialized["tool_name"] == "test_tool"
    assert serialized["duration_ms"] == 45.2


def test_agent_input_contract():
    """Verify AgentInput fields, explicit default factories, and immutability of defaults."""
    input1 = AgentInput(
        task_id="task-001",
        stage="data_analysis",
        system_prompt="You are a data agent.",
    )
    input2 = AgentInput(
        task_id="task-002",
        stage="feature_engineering",
        system_prompt="You are a feature agent.",
    )

    # Verify default factories create independent instances
    assert input1.available_tools == []
    assert input1.context == {}
    assert input1.retry_count == 0

    input1.context["key"] = "value"
    assert "key" not in input2.context


def test_agent_output_contract():
    """Verify AgentOutput requires rationale and validates status."""
    output = AgentOutput(
        status="success",
        artifact={"model_path": "models/baseline.pkl"},
        rationale="Model trained successfully with ROC-AUC 0.89.",
        tool_calls=[],
        errors=[],
    )
    assert output.status == "success"
    assert output.rationale == "Model trained successfully with ROC-AUC 0.89."
    assert output.artifact == {"model_path": "models/baseline.pkl"}

    # Validation failure when status is invalid
    with pytest.raises(ValidationError):
        AgentOutput(
            status="unknown_status",  # type: ignore
            rationale="Some rationale",
        )


def test_stage_record_contract():
    """Verify StageRecord fields and automatic ISO timestamp generation."""
    agent_output = AgentOutput(
        status="success",
        artifact=None,
        rationale="Completed stage data_prep.",
    )
    record = StageRecord(
        stage="data_prep",
        agent="DataAgent",
        output=agent_output,
        critic_verdict="approve",
    )
    assert record.stage == "data_prep"
    assert record.agent == "DataAgent"
    assert record.critic_verdict == "approve"
    assert record.timestamp is not None
    assert "T" in record.timestamp  # ISO format


def test_task_state_contract():
    """Verify TaskState structure owned by Orchestrator."""
    state = TaskState(
        task_id="baf-experiment-01",
        dataset_variant="Base",
        current_stage="data_validation",
    )
    assert state.task_id == "baf-experiment-01"
    assert state.dataset_variant == "Base"
    assert state.current_stage == "data_validation"
    assert state.status == "running"
    assert state.stage_history == []
    assert state.retry_counts == {}
    assert state.final_report is None

    # Test independent defaults across instances
    state2 = TaskState(
        task_id="baf-experiment-02",
        dataset_variant="Variant_I",
        current_stage="feature_engineering",
    )
    state.retry_counts["data_validation"] = 1
    assert "data_validation" not in state2.retry_counts


def test_tool_spec_to_schema():
    """Verify ToolSpec converts to function schema dictionary."""
    def dummy_handler(x: int) -> int:
        return x * 2

    spec = ToolSpec(
        name="double_int",
        description="Doubles an integer",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
        handler=dummy_handler,
    )
    schema = spec.to_function_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "double_int"
    assert schema["function"]["parameters"]["properties"]["x"]["type"] == "integer"
