"""Smoke and integration tests for Argus Agent Runtime."""

import json
from typing import Any
import pytest

from agents.base import Agent
from agents.contracts import AgentInput, ToolSpec
from agents.llm import LLMMessage, LLMResponse, MockLLMClient, ToolCallRequest


def echo_tool(message: str) -> dict[str, str]:
    """Dummy echo tool returning input payload."""
    return {"message": message}


def test_agent_smoke_echo_tool():
    """Smoke test: Agent receives AgentInput, calls echo_tool, and returns valid AgentOutput."""
    dummy_tool_spec = ToolSpec(
        name="echo_tool",
        description="Echoes the provided message.",
        parameters={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        handler=echo_tool,
    )

    # Multi-turn mock client: Turn 1 requests echo_tool, Turn 2 completes with rationale & artifact
    mock_llm = MockLLMClient(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_echo_01",
                        name="echo_tool",
                        arguments={"message": "hello argus baf"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content=json.dumps({
                    "status": "success",
                    "artifact": {"echo_received": "hello argus baf"},
                    "rationale": "Successfully executed echo_tool and verified output payload.",
                }),
                finish_reason="stop",
            ),
        ]
    )

    agent = Agent(llm_client=mock_llm)
    agent_input = AgentInput(
        task_id="task-smoke-01",
        stage="smoke_test",
        system_prompt="You are a smoke test agent.",
        available_tools=[dummy_tool_spec],
        context={"experiment": "baf_baseline"},
    )

    output = agent.run(agent_input)

    # Assertions on AgentOutput
    assert output.status == "success"
    assert output.rationale == "Successfully executed echo_tool and verified output payload."
    assert output.artifact == {"echo_received": "hello argus baf"}
    assert output.errors == []
    assert len(output.tool_calls) == 1

    # Assertions on ToolCallLog
    tool_log = output.tool_calls[0]
    assert tool_log.tool_name == "echo_tool"
    assert tool_log.input == {"message": "hello argus baf"}
    assert tool_log.output == {"message": "hello argus baf"}
    assert tool_log.error is None
    assert tool_log.duration_ms >= 0.0

    # Verify LLM received feedback with tool output
    assert len(mock_llm.call_history) == 2
    turn2_messages = mock_llm.call_history[1]["messages"]
    tool_feedback_msg = [m for m in turn2_messages if m.role == "tool"]
    assert len(tool_feedback_msg) == 1
    assert "hello argus baf" in tool_feedback_msg[0].content


def test_agent_tool_failure_feedback_and_recovery():
    """Verify failing tool records error, feeds error back to LLM, and LLM can handle it.

    NOTE: the retry call intentionally varies its arguments. The base agent's
    repeated-call guard treats byte-identical repeated calls as no-progress and
    returns needs_retry; a real retry varies its inputs.
    """
    call_count = 0

    def flaky_tool(param: str) -> dict[str, str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("Simulated connection timeout to database")
        return {"status": "recovered", "param": param}

    flaky_tool_spec = ToolSpec(
        name="flaky_tool",
        description="A tool that fails on first call then succeeds on retry.",
        parameters={"type": "object", "properties": {"param": {"type": "string"}}},
        handler=flaky_tool,
    )

    # Turn 1: Calls flaky_tool (fails)
    # Turn 2: Retries flaky_tool with a varied argument (succeeds)
    # Turn 3: Completes execution
    mock_llm = MockLLMClient(
        responses=[
            LLMResponse(
                tool_calls=[
                    ToolCallRequest(id="call_flaky_1", name="flaky_tool", arguments={"param": "data_query"})
                ]
            ),
            LLMResponse(
                tool_calls=[
                    ToolCallRequest(id="call_flaky_2", name="flaky_tool", arguments={"param": "data_query_retry"})
                ]
            ),
            LLMResponse(
                content=json.dumps({
                    "status": "success",
                    "artifact": {"status": "recovered"},
                    "rationale": "Tool failed initially with connection error, retried and succeeded.",
                })
            ),
        ]
    )

    agent = Agent(llm_client=mock_llm)
    agent_input = AgentInput(
        task_id="task-retry-01",
        stage="data_fetch",
        system_prompt="You are a resilient data fetch agent.",
        available_tools=[flaky_tool_spec],
    )

    output = agent.run(agent_input)

    assert output.status == "success"
    assert "retried and succeeded" in output.rationale
    assert len(output.tool_calls) == 2
    assert output.tool_calls[0].error is not None
    assert "Simulated connection timeout" in output.tool_calls[0].error
    assert output.tool_calls[1].error is None
    assert output.tool_calls[1].output == {"status": "recovered", "param": "data_query_retry"}


def test_agent_unregistered_tool_handling():
    """Verify model requesting an unregistered tool is gracefully handled without crash."""
    mock_llm = MockLLMClient(
        responses=[
            LLMResponse(
                tool_calls=[
                    ToolCallRequest(id="call_unknown", name="non_existent_tool", arguments={})
                ]
            ),
            LLMResponse(
                content=json.dumps({
                    "status": "failure",
                    "rationale": "Requested tool is not available in environment.",
                })
            ),
        ]
    )

    agent = Agent(llm_client=mock_llm)
    agent_input = AgentInput(
        task_id="task-unknown-tool",
        stage="validation",
        system_prompt="Agent with no tools.",
        available_tools=[],
    )

    output = agent.run(agent_input)

    assert output.status == "failure"
    assert len(output.tool_calls) == 1
    assert "not registered" in (output.tool_calls[0].error or "")
    assert len(output.errors) >= 1


def test_agent_max_turns_safeguard():
    """Verify agent stops and returns failure when max_turns is exceeded.

    NOTE: the loop handler varies the tool arguments every turn so the
    repeated-call guard does not preempt the turn-cap path. This preserves
    the original intent of the test (exercise max_turns).
    """
    def endless_tool(**kwargs) -> dict[str, str]:
        return {"result": "ok"}

    tool_spec = ToolSpec(
        name="endless_tool",
        description="Always returns ok",
        handler=endless_tool,
    )

    turn_counter = {"n": 0}

    def loop_handler(messages: list[LLMMessage], tools: list[dict[str, Any]] | None) -> LLMResponse:
        turn_counter["n"] += 1
        return LLMResponse(
            tool_calls=[
                ToolCallRequest(
                    id=f"loop_call_{turn_counter['n']}",
                    name="endless_tool",
                    arguments={"n": turn_counter["n"]},
                )
            ]
        )

    mock_llm = MockLLMClient(handler=loop_handler)
    agent = Agent(llm_client=mock_llm, max_turns=4)

    agent_input = AgentInput(
        task_id="task-loop-test",
        stage="loop_test",
        system_prompt="Looping agent",
        available_tools=[tool_spec],
    )

    output = agent.run(agent_input)

    assert output.status == "failure"
    assert "maximum allowed turns" in output.errors[0]
    assert len(output.tool_calls) == 4


def test_agent_unhandled_exception_safety():
    """Verify Agent.run captures unexpected LLM client exceptions safely without crashing."""
    class CrashingLLMClient:
        def generate(self, messages: list[LLMMessage], tools: list[dict[str, Any]] | None = None, **kwargs: Any) -> LLMResponse:
            raise RuntimeError("Fatal connection reset by peer")

    agent = Agent(llm_client=CrashingLLMClient())
    agent_input = AgentInput(
        task_id="task-crash-test",
        stage="crash_test",
        system_prompt="Crashing client test",
    )

    output = agent.run(agent_input)

    assert output.status == "failure"
    assert "Fatal connection reset by peer" in output.rationale
    assert len(output.errors) == 1
    assert "CrashingLLMClient" in output.errors[0] or "Fatal connection reset" in output.errors[0]


def test_agent_honors_failure_status_in_json():
    """Agent must propagate an explicit 'failure' status from the LLM JSON."""
    mock = MockLLMClient(responses=[LLMResponse(
        content='{"status": "failure", "rationale": "critical severity", "artifact": null}',
        tool_calls=[], finish_reason="stop",
    )])
    out = Agent(llm_client=mock).run(AgentInput(task_id="t", stage="s", system_prompt="x"))
    assert out.status == "failure"


def test_agent_rejects_non_json_output_as_needs_retry():
    """An unparseable final message is a contract violation, never a silent success."""
    mock = MockLLMClient(responses=[LLMResponse(content="I could not complete this.", tool_calls=[], finish_reason="stop")])
    out = Agent(llm_client=mock).run(AgentInput(task_id="t", stage="s", system_prompt="x"))
    assert out.status in ("needs_retry", "failure")  # NOT "success"


def test_agent_repeated_tool_call_fails_fast():
    """Identical repeated tool calls are treated as no-progress and fail fast."""
    tc = ToolCallRequest(id="1", name="echo", arguments={"x": 1})
    mock = MockLLMClient(handler=lambda m, t: LLMResponse(tool_calls=[tc], finish_reason="tool_calls"))
    tool = ToolSpec(name="echo", description="e", parameters={}, handler=lambda **kw: {"ok": True})
    out = Agent(llm_client=mock, max_turns=10).run(
        AgentInput(task_id="t", stage="s", system_prompt="x", available_tools=[tool]))
    assert out.status == "needs_retry"
    assert any("repeated_tool_call" in e for e in out.errors)
    assert len(out.tool_calls) <= 2  # failed fast, not 10 turns