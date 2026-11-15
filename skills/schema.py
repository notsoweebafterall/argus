"""Skill Manifest schema definition for Argus ML."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class SkillManifest(BaseModel):
    """Declarative specification of a repeatable multi-tool workflow skill."""

    name: str = Field(description="Unique identifier for the skill")
    purpose: str = Field(description="Summary of what the skill accomplishes")
    when_to_use: str = Field(description="Context and triggers for invoking this skill")
    inputs: list[str] = Field(default_factory=list, description="Required or expected input fields")
    outputs: list[str] = Field(default_factory=list, description="Produced artifact fields or results")
    mcp_tools_called: list[str] = Field(
        default_factory=list,
        description="Ordered list of MCP tools invoked during skill execution",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Operational boundaries, data privacy, or leakage constraints",
    )
    failure_conditions: list[str] = Field(
        default_factory=list,
        description="Known conditions under which the skill will fail or abort",
    )
