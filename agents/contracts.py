"""Data contracts for Argus ML Agent Runtime."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Literal
from pydantic import BaseModel, ConfigDict, Field


class ToolSpec(BaseModel):
    """Specification of a tool available to an agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    handler: Callable[..., Any] | None = None

    def to_function_schema(self) -> dict[str, Any]:
        """Convert ToolSpec to standard OpenAI/Gemini function declaration schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolCallLog(BaseModel):
    """Log record of a single tool execution."""

    tool_name: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: float


class AgentInput(BaseModel):
    """Input payload provided to an Agent for execution."""

    task_id: str
    stage: str
    system_prompt: str
    available_tools: list[ToolSpec] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    retry_count: int = 0


class AgentOutput(BaseModel):
    """Structured result returned by an Agent execution."""

    status: Literal["success", "failure", "needs_retry"]
    artifact: dict[str, Any] | None = None
    rationale: str  # required even on success
    tool_calls: list[ToolCallLog] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def _current_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StageRecord(BaseModel):
    """Execution history record for a single pipeline stage."""

    stage: str
    agent: str
    output: AgentOutput
    critic_verdict: Literal["approve", "reject"] | None = None
    timestamp: str = Field(default_factory=_current_utc_iso)


class TaskState(BaseModel):
    """Global task state owned and updated by the Orchestrator."""

    task_id: str
    dataset_variant: str
    current_stage: str
    stage_history: list[StageRecord] = Field(default_factory=list)
    retry_counts: dict[str, int] = Field(default_factory=dict)
    status: Literal["running", "halted", "completed", "failed"] = "running"
    final_report: dict[str, Any] | None = None
