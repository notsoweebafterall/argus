"""Standalone execution runner and summary reporter for the Argus Adversarial Evaluation Harness."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from eval.agent_eval.scoring import ScenarioResult, eval_collector
from eval.agent_eval.test_adversarial_scenarios import (
    test_scenario_01_missing_dataset_variant,
    test_scenario_02_corrupted_truncated_csv,
    test_scenario_03_extreme_target_leakage_injected,
    test_scenario_04_malformed_tool_call_arguments,
    test_scenario_05_model_training_failure,
    test_scenario_06_critic_retry_loop_exhaustion,
    test_scenario_07_fairness_violation_on_skewed_variant,
    test_scenario_08_prediction_request_with_unavailable_model,
    test_scenario_09_drifted_input_at_serving_time,
    test_scenario_10_duplicate_model_registration,
)


def run_all_scenarios(print_table: bool = True) -> list[ScenarioResult]:
    """Execute all 10 adversarial evaluation scenarios and collect scored results."""
    eval_collector.clear()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Missing dataset variant
        test_scenario_01_missing_dataset_variant()

        # 2. Corrupted/truncated CSV
        test_scenario_02_corrupted_truncated_csv(tmp_path=tmp_path / "s2")

        # 3. Extreme target leakage injected
        test_scenario_03_extreme_target_leakage_injected()

        # 4. Malformed tool call arguments
        test_scenario_04_malformed_tool_call_arguments()

        # 5. Model training failure
        test_scenario_05_model_training_failure(tmp_path=tmp_path / "s5")

        # 6. Critic retry-loop exhaustion
        test_scenario_06_critic_retry_loop_exhaustion()

        # 7. Fairness violation on skewed variant
        test_scenario_07_fairness_violation_on_skewed_variant()

        # 8. Prediction request with unavailable model
        test_scenario_08_prediction_request_with_unavailable_model()

        # 9. Drifted input at serving time
        test_scenario_09_drifted_input_at_serving_time(tmp_path=tmp_path / "s9")

        # 10. Duplicate model registration
        test_scenario_10_duplicate_model_registration()

    results = eval_collector.get_results()

    if print_table:
        print("\n" + eval_collector.format_summary_table() + "\n")

    return results


if __name__ == "__main__":
    results = run_all_scenarios(print_table=True)
    all_passed = all(r.passed for r in results)
    sys.exit(0 if all_passed else 1)
