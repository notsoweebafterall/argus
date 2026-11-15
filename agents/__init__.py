"""Argus ML Agents package."""

from .contracts import (
    AgentInput,
    AgentOutput,
    StageRecord,
    TaskState,
    ToolCallLog,
    ToolSpec,
)
from .base import Agent
from .llm import (
    LLMClient,
    LLMMessage,
    LLMResponse,
    MockLLMClient,
    ToolCallRequest,
)


def __getattr__(name: str):
    if name == "DataAgent":
        from .data_agent import DataAgent
        return DataAgent
    elif name == "FeatureAgent":
        from .feature_agent import FeatureAgent
        return FeatureAgent
    elif name == "MLAgent":
        from .ml_agent import MLAgent
        return MLAgent
    elif name == "CriticAgent":
        from .critic_agent import CriticAgent
        return CriticAgent
    elif name == "MLOpsAgent":
        from .mlops_agent import MLOpsAgent
        return MLOpsAgent
    elif name == "DATA_AGENT_SYSTEM_PROMPT":
        from .data_agent import DATA_AGENT_SYSTEM_PROMPT
        return DATA_AGENT_SYSTEM_PROMPT
    elif name == "FEATURE_AGENT_SYSTEM_PROMPT":
        from .feature_agent import FEATURE_AGENT_SYSTEM_PROMPT
        return FEATURE_AGENT_SYSTEM_PROMPT
    elif name == "ML_AGENT_SYSTEM_PROMPT":
        from .ml_agent import ML_AGENT_SYSTEM_PROMPT
        return ML_AGENT_SYSTEM_PROMPT
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "Agent",
    "DataAgent",
    "FeatureAgent",
    "MLAgent",
    "CriticAgent",
    "MLOpsAgent",
    "AgentInput",
    "AgentOutput",
    "StageRecord",
    "TaskState",
    "ToolCallLog",
    "ToolSpec",
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "MockLLMClient",
    "ToolCallRequest",
]
