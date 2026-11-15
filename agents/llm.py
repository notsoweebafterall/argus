"""LLM abstraction and client implementations for Argus Agent Runtime."""

from __future__ import annotations

import json
from typing import Any, Callable, Protocol
from pydantic import BaseModel, Field


class ToolCallRequest(BaseModel):
    """Represents a tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMMessage(BaseModel):
    """Message in the LLM conversation history."""

    role: str  # "system", "user", "assistant", "tool"
    content: str | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


class LLMResponse(BaseModel):
    """Response returned from an LLM call."""

    content: str | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    finish_reason: str = "stop"  # "stop", "tool_calls", "length", etc.


class LLMClient(Protocol):
    """Protocol for LLM clients used by Agents."""

    def generate(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from the LLM given conversation messages and optional tool declarations."""
        ...


class MockLLMClient:
    """Mock LLM client for deterministic testing of Agent tool calling and error recovery."""

    def __init__(
        self,
        responses: list[LLMResponse] | None = None,
        handler: Callable[[list[LLMMessage], list[dict[str, Any]] | None], LLMResponse] | None = None,
    ) -> None:
        self._responses = list(responses) if responses else []
        self._handler = handler
        self.call_history: list[dict[str, Any]] = []

    def add_response(self, response: LLMResponse) -> None:
        """Queue a scripted response."""
        self._responses.append(response)

    def generate(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Return scripted response or invoke handler."""
        self.call_history.append({"messages": messages, "tools": tools, "kwargs": kwargs})

        if self._handler is not None:
            return self._handler(messages, tools)

        if self._responses:
            return self._responses.pop(0)

        # Default fallback response if queue is empty
        return LLMResponse(
            content=json.dumps({
                "status": "success",
                "rationale": "Mock LLM completed execution without further actions.",
                "artifact": None,
            }),
            finish_reason="stop",
        )
