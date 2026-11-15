"""Base Agent implementation for Argus ML."""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from pydantic import ValidationError

from .contracts import AgentInput, AgentOutput, ToolCallLog, ToolSpec
from .llm import LLMClient, LLMMessage, ToolCallRequest

logger = logging.getLogger(__name__)


class Agent:
    """Base Agent class managing the LLM tool-calling loop and structured execution."""

    def __init__(
        self,
        llm_client: LLMClient,
        max_turns: int = 10,
    ) -> None:
        self.llm_client = llm_client
        self.max_turns = max_turns

    def run(self, agent_input: AgentInput) -> AgentOutput:
        """Execute the agent loop for a given AgentInput.

        Guarantees that no unhandled exception escapes and always returns a valid AgentOutput.
        """
        tool_call_logs: list[ToolCallLog] = []
        errors: list[str] = []
        tool_map: dict[str, ToolSpec] = {tool.name: tool for tool in agent_input.available_tools}
        tool_declarations = [tool.to_function_schema() for tool in agent_input.available_tools] if agent_input.available_tools else None

        logger.info(
            f"Starting agent execution for task={agent_input.task_id} stage={agent_input.stage}",
            extra={"task_id": agent_input.task_id, "stage": agent_input.stage},
        )

        try:
            system_content = agent_input.system_prompt
            if agent_input.context:
                system_content += f"\n\nExecution Context:\n{json.dumps(agent_input.context, indent=2, default=str)}"

            messages: list[LLMMessage] = [
                LLMMessage(role="system", content=system_content),
                LLMMessage(
                    role="user",
                    content=(
                        f"Execute stage '{agent_input.stage}' for task '{agent_input.task_id}'. "
                        f"Retry count: {agent_input.retry_count}."
                    ),
                ),
            ]

            turn = 0
            final_content: str | None = None
            seen_calls: set[str] = set()

            while turn < self.max_turns:
                turn += 1
                logger.debug(
                    f"Agent turn {turn}/{self.max_turns} for task={agent_input.task_id}",
                    extra={"task_id": agent_input.task_id, "stage": agent_input.stage},
                )

                response = self.llm_client.generate(
                    messages=messages,
                    tools=tool_declarations,
                )

                assistant_msg = LLMMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
                messages.append(assistant_msg)

                if response.tool_calls:
                    for tool_request in response.tool_calls:
                        # --- Loop guard: identical tool call twice => no progress ---
                        call_sig = tool_request.name + "|" + json.dumps(
                            tool_request.arguments, sort_keys=True, default=str
                        )
                        if call_sig in seen_calls:
                            msg = (
                                f"repeated_tool_call: '{tool_request.name}' called twice with "
                                f"identical arguments — no progress"
                            )
                            errors.append(msg)
                            logger.warning(
                                msg, extra={"task_id": agent_input.task_id, "stage": agent_input.stage}
                            )
                            return AgentOutput(
                                status="needs_retry",
                                artifact=None,
                                rationale=(
                                    f"Agent stalled: repeated identical tool call "
                                    f"'{tool_request.name}'. Returning needs_retry."
                                ),
                                tool_calls=tool_call_logs,
                                errors=errors,
                            )
                        seen_calls.add(call_sig)

                        log_entry, tool_result_msg = self._execute_tool(
                            tool_request=tool_request,
                            tool_map=tool_map,
                            task_id=agent_input.task_id,
                            stage=agent_input.stage,
                        )
                        tool_call_logs.append(log_entry)
                        if log_entry.error:
                            errors.append(f"Tool '{tool_request.name}' error: {log_entry.error}")
                        messages.append(tool_result_msg)
                    continue

                final_content = response.content
                break

            if turn >= self.max_turns and final_content is None:
                msg = f"Agent exceeded maximum allowed turns ({self.max_turns}) without completing."
                errors.append(msg)
                logger.warning(msg, extra={"task_id": agent_input.task_id, "stage": agent_input.stage})
                return AgentOutput(
                    status="failure",
                    artifact=None,
                    rationale="Execution halted: maximum interaction turns exceeded.",
                    tool_calls=tool_call_logs,
                    errors=errors,
                )

            return self._parse_final_output(
                content=final_content,
                tool_call_logs=tool_call_logs,
                errors=errors,
                stage=agent_input.stage,
            )

        except Exception as exc:
            err_msg = f"Unhandled exception during agent execution: {type(exc).__name__}: {str(exc)}"
            logger.error(err_msg, exc_info=True, extra={"task_id": agent_input.task_id, "stage": agent_input.stage})
            errors.append(err_msg)
            return AgentOutput(
                status="failure",
                artifact=None,
                rationale=f"Execution failed due to internal error: {str(exc)}",
                tool_calls=tool_call_logs,
                errors=errors,
            )

    def _execute_tool(
        self,
        tool_request: ToolCallRequest,
        tool_map: dict[str, ToolSpec],
        task_id: str,
        stage: str,
    ) -> tuple[ToolCallLog, LLMMessage]:
        """Execute a single tool call, capture timing and errors, and format the result for LLM feedback."""
        tool_name = tool_request.name
        tool_args = tool_request.arguments
        start_time = time.perf_counter()

        logger.info(
            f"Executing tool '{tool_name}'",
            extra={"task_id": task_id, "stage": stage, "tool_name": tool_name},
        )

        if tool_name not in tool_map:
            duration_ms = (time.perf_counter() - start_time) * 1000
            err = f"Tool '{tool_name}' is not registered in available tools."
            logger.error(err, extra={"task_id": task_id, "stage": stage, "tool_name": tool_name})
            log_entry = ToolCallLog(
                tool_name=tool_name,
                input=tool_args,
                output=None,
                error=err,
                duration_ms=round(duration_ms, 2),
            )
            result_msg = LLMMessage(
                role="tool",
                name=tool_name,
                tool_call_id=tool_request.id,
                content=json.dumps({"error": err}),
            )
            return log_entry, result_msg

        tool_spec = tool_map[tool_name]
        if tool_spec.handler is None:
            duration_ms = (time.perf_counter() - start_time) * 1000
            err = f"Tool '{tool_name}' has no executable handler defined."
            logger.error(err, extra={"task_id": task_id, "stage": stage, "tool_name": tool_name})
            log_entry = ToolCallLog(
                tool_name=tool_name,
                input=tool_args,
                output=None,
                error=err,
                duration_ms=round(duration_ms, 2),
            )
            result_msg = LLMMessage(
                role="tool",
                name=tool_name,
                tool_call_id=tool_request.id,
                content=json.dumps({"error": err}),
            )
            return log_entry, result_msg

        try:
            raw_output = tool_spec.handler(**tool_args)
            duration_ms = (time.perf_counter() - start_time) * 1000

            output_dict: dict[str, Any]
            if isinstance(raw_output, dict):
                output_dict = raw_output
            else:
                output_dict = {"result": raw_output}

            log_entry = ToolCallLog(
                tool_name=tool_name,
                input=tool_args,
                output=output_dict,
                error=None,
                duration_ms=round(duration_ms, 2),
            )
            result_msg = LLMMessage(
                role="tool",
                name=tool_name,
                tool_call_id=tool_request.id,
                content=json.dumps(output_dict, default=str),
            )
            return log_entry, result_msg

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            err = f"Tool execution failed with {type(exc).__name__}: {str(exc)}"
            logger.warning(
                err,
                extra={"task_id": task_id, "stage": stage, "tool_name": tool_name, "duration_ms": duration_ms},
            )
            log_entry = ToolCallLog(
                tool_name=tool_name,
                input=tool_args,
                output=None,
                error=err,
                duration_ms=round(duration_ms, 2),
            )
            result_msg = LLMMessage(
                role="tool",
                name=tool_name,
                tool_call_id=tool_request.id,
                content=json.dumps({"error": err}),
            )
            return log_entry, result_msg

    def _parse_final_output(
        self,
        content: str | None,
        tool_call_logs: list[ToolCallLog],
        errors: list[str],
        stage: str,
    ) -> AgentOutput:
        """Parse LLM final output into a validated AgentOutput.

        Contract: the agent MUST emit JSON with a 'rationale' key (and a 'status'
        in {success, failure, needs_retry}). Anything else is a contract violation
        and is reported as needs_retry/failure — never silently as success.
        """
        if not content:
            return AgentOutput(
                status="failure" if errors else "needs_retry",
                artifact=None,
                rationale=f"Stage '{stage}' completed without text output.",
                tool_calls=tool_call_logs,
                errors=errors,
            )

        cleaned = content.strip()
        if cleaned.startswith("```"):
            first_nl = cleaned.find("\n")
            cleaned = cleaned[first_nl + 1:] if first_nl != -1 else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        parsed: Any = None
        try:
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, ValidationError):
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end > start:
                try:
                    parsed = json.loads(cleaned[start:end + 1])
                except (json.JSONDecodeError, ValidationError):
                    parsed = None

        if isinstance(parsed, dict) and "rationale" in parsed:
            status = parsed.get("status", "success" if not errors else "failure")
            if status not in ("success", "failure", "needs_retry"):
                errors.append(f"invalid_status_from_llm: {status!r}")
                status = "failure"
            rationale = str(parsed.get("rationale", "")).strip() or f"Stage '{stage}' executed."
            extra_errors = parsed.get("errors", [])
            if isinstance(extra_errors, list):
                errors.extend([str(e) for e in extra_errors if str(e) not in errors])
            artifact = parsed.get("artifact")
            return AgentOutput(
                status=status,
                artifact=artifact if isinstance(artifact, dict) else None,
                rationale=rationale,
                tool_calls=tool_call_logs,
                errors=errors,
            )

        return AgentOutput(
            status="needs_retry" if not errors else "failure",
            artifact=None,
            rationale=(
                f"Stage '{stage}' did not return the required JSON contract. "
                f"Raw output (truncated): {cleaned[:500]}"
            ),
            tool_calls=tool_call_logs,
            errors=errors + ["output_contract_violation: no parseable JSON with 'rationale' key"],
        )