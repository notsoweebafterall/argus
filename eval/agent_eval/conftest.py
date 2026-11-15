"""Pytest configuration and session hooks for Argus Adversarial Evaluation Harness."""

from __future__ import annotations

import pytest
from eval.agent_eval.scoring import eval_collector
from serving.api import model_container
from tools.data_tools import load_dataset


@pytest.fixture(autouse=True)
def reset_model_serving_state():
    """Ensure serving model container is reset before and after each test."""
    model_container.reset()
    yield
    model_container.reset()


@pytest.fixture(scope="session")
def base_baf_dataset_path():
    """Resolve base BAF dataset path for evaluation scenarios."""
    res = load_dataset("base")
    return res["path"]


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the formatted adversarial evaluation summary table after all tests finish."""
    results = eval_collector.get_results()
    if results:
        terminalreporter.section("Adversarial Evaluation Summary Table")
        table_output = eval_collector.format_summary_table()
        terminalreporter.write_line(table_output)
