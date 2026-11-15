"""Orchestrator state machine and pipeline execution for Argus ML."""

from __future__ import annotations

import os
import time
import uuid
from typing import TYPE_CHECKING, Any, Mapping

from agents.base import Agent
from agents.contracts import AgentInput, StageRecord, TaskState
from pipeline.config import MAX_RETRIES_PER_STAGE
from pipeline.logging import get_logger
from pipeline.task_graph import DEFAULT_TASK_GRAPH, TaskGraph

if TYPE_CHECKING:
    from agents.llm import LLMClient

logger = get_logger(__name__)


class Orchestrator:
    """Orchestrator managing task state, stage transitions, agent routing, retry budgets, and execution timeout."""

    def __init__(
        self,
        llm_client: LLMClient,
        stages: list[str] | None = None,
        agent_registry: Mapping[str, Agent] | None = None,
        max_retries_per_stage: int = MAX_RETRIES_PER_STAGE,
        timeout_seconds: float = 1800.0,
        task_graph: TaskGraph | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.max_retries = max_retries_per_stage
        self.timeout_seconds = timeout_seconds

        base_graph = task_graph or DEFAULT_TASK_GRAPH

        # If a custom subset/order of stages is requested, restrict the graph to exactly
        # those stages so get_next_stage / get_dependencies / output_artifact_key all agree
        # with self.stages. Otherwise use the graph as-is.
        if stages is not None:
            restricted = [base_graph.get_stage(s) for s in stages]
            missing = [s for s, st in zip(stages, restricted) if st is None]
            if missing:
                raise ValueError(f"Unknown stage(s) requested: {missing}")
            self.task_graph = TaskGraph([st for st in restricted if st is not None])
        else:
            self.task_graph = base_graph

        self.stages = list(stages or self.task_graph.stage_names)

        if agent_registry is not None:
            self.agent_registry: dict[str, Agent] = dict(agent_registry)
        else:
            # Lazy import to avoid circular dependency
            from agents.data_agent import DataAgent
            from agents.feature_agent import FeatureAgent
            from agents.ml_agent import MLAgent
            from agents.critic_agent import CriticAgent
            from agents.mlops_agent import MLOpsAgent

            self.agent_registry = {
                "data_validation": DataAgent(llm_client=self.llm_client),
                "feature_engineering": FeatureAgent(llm_client=self.llm_client),
                "model_training": MLAgent(llm_client=self.llm_client),
                "critic_review": CriticAgent(llm_client=self.llm_client),
                "mlops_registration": MLOpsAgent(llm_client=self.llm_client),
            }

    def run(
        self,
        dataset_variant: str,
        task_id: str | None = None,
        initial_context: dict[str, Any] | None = None,
    ) -> TaskState:
        """Execute the pipeline task graph from start to completion or failure."""
        tid = task_id or f"task_{uuid.uuid4().hex[:8]}"
        initial_stage = self.stages[0] if self.stages else "data_validation"
        start_time = time.monotonic()

        state = TaskState(
            task_id=tid,
            dataset_variant=dataset_variant,
            current_stage=initial_stage,
            status="running",
            stage_history=[],
            retry_counts={stage: 0 for stage in self.stages},
            final_report=None,
        )

        stage_contexts: dict[str, dict[str, Any]] = {
            stage: dict(initial_context or {}) for stage in self.stages
        }
        stage_artifacts: dict[str, dict[str, Any]] = {}

        logger.info(
            f"Starting Orchestrator pipeline for task={state.task_id} variant={state.dataset_variant}",
            extra={"task_id": state.task_id, "stage": state.current_stage},
        )

        while state.status == "running":
            curr_stage = state.current_stage

            # Wall-clock timeout guard (evaluated at stage loop boundaries).
            # Note: This prevents runaway retry-loop-style pathological duration across stage
            # transitions; it does not preemptively interrupt an in-flight synchronous call within a single stage.
            elapsed = time.monotonic() - start_time
            if elapsed > self.timeout_seconds:
                err = (
                    f"Pipeline execution exceeded maximum wall-clock timeout of {self.timeout_seconds:.1f}s "
                    f"(elapsed: {elapsed:.1f}s). Halting run."
                )
                logger.error(err, extra={"task_id": state.task_id, "stage": curr_stage})
                state.status = "failed"
                break

            agent = self.agent_registry.get(curr_stage)

            if not agent:
                err = f"No agent registered for pipeline stage '{curr_stage}'."
                logger.error(err, extra={"task_id": state.task_id, "stage": curr_stage})
                state.status = "failed"
                break

            # Dependency pre-check before invoking an agent
            completed_stages = [
                rec.stage for rec in state.stage_history if rec.output.status == "success"
            ]
            if not self.task_graph.validate_dependencies_satisfied(curr_stage, completed_stages):
                deps = self.task_graph.get_dependencies(curr_stage)
                missing_deps = [d for d in deps if d not in completed_stages]
                err = (
                    f"Cannot execute stage '{curr_stage}': missing prerequisite completed "
                    f"dependencies: {missing_deps}."
                )
                logger.error(err, extra={"task_id": state.task_id, "stage": curr_stage})
                state.status = "failed"
                break

            # Build stage context incorporating prior stage outputs
            ctx = dict(stage_contexts.get(curr_stage, {}))
            ctx["dataset_variant"] = state.dataset_variant
            ctx["task_id"] = state.task_id

            if "data_validation" in stage_artifacts:
                data_art = stage_artifacts["data_validation"]
                ctx["data_validation_artifact"] = data_art
                if "dataset_path" in data_art:
                    ctx["dataset_path"] = data_art["dataset_path"]

            if "feature_engineering" in stage_artifacts:
                feat_art = stage_artifacts["feature_engineering"]
                ctx["feature_engineering_artifact"] = feat_art
                if "train_set_path" in feat_art:
                    ctx["train_set_path"] = feat_art["train_set_path"]
                if "eval_set_path" in feat_art:
                    ctx["eval_set_path"] = feat_art["eval_set_path"]
                if "feature_set_path" in feat_art:
                    ctx["feature_set_path"] = feat_art["feature_set_path"]

            if "model_training" in stage_artifacts:
                model_art = stage_artifacts["model_training"]
                ctx["model_training_artifact"] = model_art
                if "model_id" in model_art:
                    ctx["model_id"] = model_art["model_id"]
                if "run_id" in model_art:
                    ctx["run_id"] = model_art["run_id"]
                elif "mlflow_run_id" in model_art:
                    ctx["run_id"] = model_art["mlflow_run_id"]

            if "critic_review" in stage_artifacts:
                ctx["critic_review_artifact"] = stage_artifacts["critic_review"]
                if "verdict" in stage_artifacts["critic_review"]:
                    ctx["critic_verdict"] = stage_artifacts["critic_review"]["verdict"]

            # Construct AgentInput
            agent_input = AgentInput(
                task_id=state.task_id,
                stage=curr_stage,
                system_prompt=getattr(agent, "system_prompt", f"You are the {curr_stage} agent."),
                available_tools=getattr(agent, "tools", []),
                context=ctx,
                retry_count=state.retry_counts.get(curr_stage, 0),
            )

            # Execute Agent
            agent_name = type(agent).__name__
            logger.info(
                f"Invoking {agent_name} for stage '{curr_stage}' (attempt {agent_input.retry_count + 1})",
                extra={"task_id": state.task_id, "stage": curr_stage, "agent": agent_name},
            )
            agent_output = agent.run(agent_input)

            # Extract critic verdict if available and tag existing MLflow run
            critic_verdict: str | None = None
            if curr_stage == "critic_review" and agent_output.artifact:
                verdict = agent_output.artifact.get("verdict")
                if verdict in ("approve", "reject"):
                    critic_verdict = verdict
                    run_id = (
                        ctx.get("run_id")
                        or (stage_artifacts.get("model_training") or {}).get("run_id")
                        or (stage_artifacts.get("model_training") or {}).get("mlflow_run_id")
                    )
                    if run_id:
                        try:
                            from tools.ml_tools import update_critic_verdict_tag
                            update_critic_verdict_tag(run_id=run_id, verdict=verdict)
                        except Exception as exc:
                            logger.warning(f"Could not update critic_verdict tag on MLflow run '{run_id}': {exc}")

            # Record stage execution
            record = StageRecord(
                stage=curr_stage,
                agent=agent_name,
                output=agent_output,
                critic_verdict=critic_verdict,
            )
            state.stage_history.append(record)

            # Evaluate Critic rejection routing or needs_retry
            is_critic_rejection = (
                curr_stage == "critic_review"
                and (
                    agent_output.status == "needs_retry"
                    or (critic_verdict == "reject")
                )
            )

            if is_critic_rejection:
                failed_checks = []
                if agent_output.artifact and isinstance(agent_output.artifact, dict):
                    failed_checks = agent_output.artifact.get("failed_checks", [])

                target_stage = self._determine_critic_retry_target(failed_checks)

                current_retries = state.retry_counts.get(target_stage, 0) + 1
                state.retry_counts[target_stage] = current_retries

                logger.warning(
                    f"Critic stage requested retry/reject. Routing back to '{target_stage}' ({current_retries}/{self.max_retries}).",
                    extra={"task_id": state.task_id, "stage": curr_stage, "target_stage": target_stage},
                )

                if current_retries > self.max_retries:
                    logger.error(
                        f"Stage '{target_stage}' exceeded retry budget ({self.max_retries}) via Critic rejection. Halting pipeline.",
                        extra={"task_id": state.task_id, "stage": curr_stage, "target_stage": target_stage},
                    )
                    state.status = "failed"
                    break

                # Route back to target_stage and enrich context with detailed failed_checks
                state.current_stage = target_stage
                target_retry_ctx = stage_contexts.setdefault(target_stage, {})
                target_retry_ctx["previous_attempt_errors"] = list(agent_output.errors)
                target_retry_ctx["previous_attempt_rationale"] = agent_output.rationale
                target_retry_ctx["previous_attempt_artifact"] = agent_output.artifact
                target_retry_ctx["critic_verdict"] = "reject"
                target_retry_ctx["critic_feedback"] = agent_output.artifact
                target_retry_ctx["failed_checks"] = failed_checks
                target_retry_ctx["retry_attempt"] = current_retries
                continue

            # Evaluate AgentOutput status
            if agent_output.status == "success":
                logger.info(
                    f"Stage '{curr_stage}' succeeded with rationale: {agent_output.rationale[:100]}...",
                    extra={"task_id": state.task_id, "stage": curr_stage},
                )
                if agent_output.artifact:
                    stage_def = self.task_graph.get_stage(curr_stage)
                    art_key = stage_def.output_artifact_key if stage_def else curr_stage
                    stage_artifacts[art_key] = agent_output.artifact

                next_stage = self.task_graph.get_next_stage(curr_stage)
                if next_stage is None:
                    # Final stage completed successfully
                    state.status = "completed"
                    state.final_report = self._build_final_report(state, stage_artifacts)
                    logger.info(
                        f"Pipeline completed successfully for task={state.task_id}",
                        extra={"task_id": state.task_id, "stage": curr_stage},
                    )
                    break
                else:
                    # Advance to next stage in graph
                    state.current_stage = next_stage
                    logger.info(
                        f"Advancing pipeline from '{curr_stage}' to '{next_stage}'",
                        extra={"task_id": state.task_id, "stage": next_stage},
                    )

            elif agent_output.status == "needs_retry":
                current_retries = state.retry_counts.get(curr_stage, 0) + 1
                state.retry_counts[curr_stage] = current_retries

                logger.warning(
                    f"Stage '{curr_stage}' requested retry ({current_retries}/{self.max_retries})",
                    extra={"task_id": state.task_id, "stage": curr_stage},
                )

                if current_retries > self.max_retries:
                    logger.error(
                        f"Stage '{curr_stage}' exceeded maximum retry budget ({self.max_retries}). Halting pipeline.",
                        extra={"task_id": state.task_id, "stage": curr_stage},
                    )
                    state.status = "failed"
                    break

                # Update context for the next retry attempt with previous errors and rationale
                retry_ctx = stage_contexts.setdefault(curr_stage, {})
                retry_ctx["previous_attempt_errors"] = list(agent_output.errors)
                retry_ctx["previous_attempt_rationale"] = agent_output.rationale
                retry_ctx["previous_attempt_artifact"] = agent_output.artifact
                retry_ctx["retry_attempt"] = current_retries

            elif agent_output.status == "failure":
                logger.error(
                    f"Stage '{curr_stage}' failed definitively. Halting pipeline.",
                    extra={"task_id": state.task_id, "stage": curr_stage},
                )
                state.status = "failed"
                break

            else:
                logger.error(
                    f"Unrecognized status '{agent_output.status}' returned by agent. Halting.",
                    extra={"task_id": state.task_id, "stage": curr_stage},
                )
                state.status = "failed"
                break

        return state

    def _determine_critic_retry_target(self, failed_checks: list[dict[str, Any]] | None) -> str:
        """Determine target stage to retry based on failed checks in priority order:
        1. Any category == 'data' -> 'data_validation'
        2. Any category == 'modeling' and check_name in ('month_not_in_feature_columns', 'fraud_bool_not_in_feature_columns') -> 'feature_engineering'
        3. Else (other modeling, experiment, fairness) -> 'model_training'
        """
        if not failed_checks:
            return "model_training"

        # 1. Data category failures take highest priority
        has_data_failure = any(fc.get("category") == "data" for fc in failed_checks)
        if has_data_failure:
            return "data_validation"

        # 2. Split-key or target leakage in feature columns routes to feature_engineering
        has_feature_leakage = any(
            fc.get("category") == "modeling"
            and fc.get("check_name") in ("month_not_in_feature_columns", "fraud_bool_not_in_feature_columns")
            for fc in failed_checks
        )
        if has_feature_leakage:
            return "feature_engineering"

        # 3. Default fallback for other modeling, experiment, or fairness failures
        return "model_training"

    def _build_final_report(
        self,
        state: TaskState,
        stage_artifacts: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate artifacts and execution metadata into the final task report."""
        return {
            "task_id": state.task_id,
            "dataset_variant": state.dataset_variant,
            "status": state.status,
            "stages_executed": [record.stage for record in state.stage_history],
            "total_stages": len(state.stage_history),
            "data_validation": stage_artifacts.get("data_validation"),
            "feature_engineering": stage_artifacts.get("feature_engineering"),
            "model_training": stage_artifacts.get("model_training"),
            "critic_review": stage_artifacts.get("critic_review"),
            "mlops_registration": stage_artifacts.get("mlops_registration"),
        }


def run_task(
    dataset_variant: str,
    task_id: str | None = None,
    llm_client: LLMClient | None = None,
) -> TaskState:
    """Convenience functional entry point to run the Argus multi-agent task pipeline."""
    if llm_client is None:
        if os.getenv("ARGUS_LLM") == "real":
            from dotenv import load_dotenv
            load_dotenv()
            from agents.llm_groq import GroqLLMClient
            llm_client = GroqLLMClient()
        else:
            from agents.llm import MockLLMClient
            llm_client = MockLLMClient()

    orchestrator = Orchestrator(llm_client=llm_client)
    return orchestrator.run(dataset_variant=dataset_variant, task_id=task_id)