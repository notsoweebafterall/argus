"""Groq-backed LLM client implementing the Argus LLMClient protocol."""
from __future__ import annotations
import json, os
from typing import Any
from groq import Groq
from agents.llm import LLMClient, LLMMessage, LLMResponse, ToolCallRequest


def _to_groq_msg(m: LLMMessage) -> dict[str, Any]:
    """Serialize an LLMMessage into the exact shape Groq accepts.

    Only emit fields that are meaningful for the message's role:
    - `content` only when non-empty
    - `tool_calls` only when present (and only valid on assistant messages)
    - `tool_call_id` only on tool messages
    - `name` only when set
    Empty lists (e.g. `tool_calls=[]`) must NOT be sent — Groq rejects
    `tool_calls` on user/system/tool messages with a 400.
    """
    d: dict[str, Any] = {"role": m.role}
    if m.content is not None and m.content != "":
        d["content"] = m.content
    if m.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in m.tool_calls
        ]
    if m.tool_call_id is not None:
        d["tool_call_id"] = m.tool_call_id
    if m.name is not None:
        d["name"] = m.name
    return d


class GroqLLMClient:
    def __init__(self, model: str = "openai/gpt-oss-120b") -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        msgs = [_to_groq_msg(m) for m in messages]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            tools=tools,
            tool_choice="auto" if tools else "none",
        )
        choice = resp.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCallRequest(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            )
            for tc in (msg.tool_calls or [])
        ]
        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
        )