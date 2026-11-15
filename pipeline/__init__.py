"""Argus ML Pipeline package."""

from .config import MAX_RETRIES_PER_STAGE, Settings, settings
from .logging import get_logger, setup_logging


def __getattr__(name: str):
    if name in ("Orchestrator", "run_task"):
        from .orchestrator import Orchestrator, run_task
        mapping = {
            "Orchestrator": Orchestrator,
            "run_task": run_task,
        }
        return mapping[name]
    elif name in ("DEFAULT_TASK_GRAPH", "TaskGraph", "PIPELINE_STAGES", "PipelineStage"):
        from .task_graph import (
            DEFAULT_TASK_GRAPH,
            PIPELINE_STAGES,
            PipelineStage,
            TaskGraph,
        )
        mapping = {
            "DEFAULT_TASK_GRAPH": DEFAULT_TASK_GRAPH,
            "TaskGraph": TaskGraph,
            "PIPELINE_STAGES": PIPELINE_STAGES,
            "PipelineStage": PipelineStage,
        }
        return mapping[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "MAX_RETRIES_PER_STAGE",
    "Settings",
    "settings",
    "get_logger",
    "setup_logging",
    "Orchestrator",
    "run_task",
]
