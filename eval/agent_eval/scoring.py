"""Scoring schemas, Rubric definitions, and report aggregation for Argus Adversarial Eval."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class ScenarioResult(BaseModel):
    """Scoring result record for a single adversarial evaluation scenario."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario: str
    expected_terminal_state: str
    actual_terminal_state: str
    passed: bool
    notes: str
    latency_s: float
    rubric_a: dict[str, Any] | None = Field(
        default=None,
        description="Rubric A (Model Performance): PR-AUC, recall@5%FPR, predictive_equality_ratio",
    )
    rubric_b: dict[str, Any] = Field(
        default_factory=dict,
        description="Rubric B (System Reliability): Terminal state match, tool-selection/retry, no hallucination, reproducibility",
    )

    def to_summary_dict(self) -> dict[str, Any]:
        """Convert to the standard scenario scoring dictionary format."""
        return {
            "scenario": self.scenario,
            "expected_terminal_state": self.expected_terminal_state,
            "actual_terminal_state": self.actual_terminal_state,
            "passed": self.passed,
            "notes": self.notes,
            "latency_s": round(self.latency_s, 4),
        }


class EvalReportCollector:
    """Collector aggregating ScenarioResult records across adversarial evaluation runs."""

    def __init__(self) -> None:
        self._results: dict[str, ScenarioResult] = {}

    def record(self, result: ScenarioResult) -> None:
        """Record or overwrite a scenario result by its scenario name."""
        self._results[result.scenario] = result

    def get_results(self) -> list[ScenarioResult]:
        """Return all recorded scenario results sorted by scenario name."""
        return list(self._results.values())

    def clear(self) -> None:
        """Clear all stored results."""
        self._results.clear()

    def format_summary_table(self) -> str:
        """Format a markdown/ASCII summary table suitable for the Stage 8d defense doc."""
        return format_summary_table(self.get_results())


def format_summary_table(results: list[ScenarioResult]) -> str:
    """Format a list of ScenarioResults into a structured ASCII/Markdown summary table."""
    if not results:
        return "No evaluation scenarios recorded."

    headers = ["Scenario", "Status", "Expected State", "Actual State", "Latency (s)", "Notes"]
    col_widths = [45, 8, 16, 16, 12, 40]

    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    divider_row = "-+-".join("-" * w for w in col_widths)

    rows = []
    total_passed = 0
    total_latency = 0.0

    for r in results:
        status_str = "PASSED" if r.passed else "FAILED"
        if r.passed:
            total_passed += 1
        total_latency += r.latency_s

        scenario_col = (r.scenario[:42] + "...") if len(r.scenario) > 45 else r.scenario
        notes_col = (r.notes[:37] + "...") if len(r.notes) > 40 else r.notes

        row = " | ".join([
            scenario_col.ljust(col_widths[0]),
            status_str.ljust(col_widths[1]),
            r.expected_terminal_state.ljust(col_widths[2]),
            r.actual_terminal_state.ljust(col_widths[3]),
            f"{r.latency_s:.3f}".rjust(col_widths[4]),
            notes_col.ljust(col_widths[5]),
        ])
        rows.append(row)

    summary_divider = "=" * len(header_row)
    summary_line = (
        f"Summary: {total_passed}/{len(results)} Scenarios Passed | "
        f"Total Latency: {total_latency:.3f}s"
    )

    table = "\n".join([
        summary_divider,
        "Argus Adversarial Evaluation Harness — Execution Summary",
        summary_divider,
        header_row,
        divider_row,
        *rows,
        summary_divider,
        summary_line,
        summary_divider,
    ])
    return table


# Global instance for test session collection
eval_collector = EvalReportCollector()
