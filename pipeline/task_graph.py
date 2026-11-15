"""Task graph and stage dependency definitions for Argus ML."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class PipelineStage:
    """Represents a discrete stage in the Argus ML pipeline task graph."""

    name: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    required_context_keys: list[str] = field(default_factory=list)
    output_artifact_key: str = ""


# Declarative five-stage specification for the Argus pipeline
PIPELINE_STAGES: list[PipelineStage] = [
    PipelineStage(
        name="data_validation",
        description="Dataset loading, schema validation, sentinel missingness check, and exploratory data analysis",
        dependencies=[],
        required_context_keys=["dataset_variant"],
        output_artifact_key="data_validation",
    ),
    PipelineStage(
        name="feature_engineering",
        description="Temporal train/eval splitting, multicollinearity reduction, and leakage defense",
        dependencies=["data_validation"],
        required_context_keys=["dataset_path"],
        output_artifact_key="feature_engineering",
    ),
    PipelineStage(
        name="model_training",
        description="Candidate model training, hyperparameter search, and numbers-grounded model selection",
        dependencies=["feature_engineering"],
        required_context_keys=["train_set_path", "eval_set_path"],
        output_artifact_key="model_training",
    ),
    PipelineStage(
        name="critic_review",
        description="Deterministic multi-category audit across data quality, modeling, experiment, and fairness",
        dependencies=["model_training"],
        required_context_keys=["eval_set_path", "model_id"],
        output_artifact_key="critic_review",
    ),
    PipelineStage(
        name="mlops_registration",
        description="MLflow model registry publication conditional on approved Critic audit verdict",
        dependencies=["critic_review"],
        required_context_keys=["model_id", "run_id"],
        output_artifact_key="mlops_registration",
    ),
]


class TaskGraph:
    """Defines and validates the sequence and dependencies of pipeline execution stages."""

    def __init__(self, stages: Sequence[PipelineStage] | None = None) -> None:
        self.stages = list(stages or PIPELINE_STAGES)
        self._stage_map = {s.name: s for s in self.stages}

    @property
    def stage_names(self) -> list[str]:
        """Return ordered list of stage names in the task graph."""
        return [s.name for s in self.stages]

    def get_stage(self, stage_name: str) -> PipelineStage | None:
        """Get stage definition by name."""
        return self._stage_map.get(stage_name)

    def get_next_stage(self, current_stage: str) -> str | None:
        """Determine next stage in the linear pipeline progression."""
        names = self.stage_names
        if current_stage not in names:
            return None
        idx = names.index(current_stage)
        if idx < len(names) - 1:
            return names[idx + 1]
        return None

    def get_dependencies(self, stage_name: str) -> list[str]:
        """Get direct upstream stage dependencies for a given stage."""
        stage = self.get_stage(stage_name)
        return list(stage.dependencies) if stage else []

    def validate_dependencies_satisfied(
        self, stage_name: str, completed_stages: Sequence[str]
    ) -> bool:
        """Check whether all prerequisites for a stage have been executed."""
        deps = self.get_dependencies(stage_name)
        return all(dep in completed_stages for dep in deps)


# Default singleton instance exported as the single source of truth
DEFAULT_TASK_GRAPH = TaskGraph()

