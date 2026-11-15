"""Unit tests for validate_dataset and run_eda tools."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from tools.data_tools import run_eda, validate_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(df: pd.DataFrame, tmp_path: Path, name: str = "test.csv") -> str:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return str(p)


def _base_df(n: int = 100) -> pd.DataFrame:
    """Minimal valid BAF-shaped dataframe with unique rows and all required columns."""
    import numpy as np
    rng = np.random.default_rng(42)
    data = {
        "fraud_bool": ([0] * (n - 2)) + [1, 1],
        "month": [i % 8 for i in range(n)],
        # Vary numeric columns per row to avoid duplicate rows
        "income": rng.uniform(0.1, 0.9, n).tolist(),
        "credit_risk_score": rng.integers(200, 800, n).tolist(),
        "customer_age": rng.integers(18, 80, n).tolist(),
        "prev_address_months_count": rng.integers(0, 60, n).tolist(),
        "current_address_months_count": rng.integers(0, 120, n).tolist(),
        "days_since_request": rng.uniform(0.0, 30.0, n).tolist(),
        "intended_balcon_amount": rng.uniform(-10.0, 500.0, n).tolist(),
        "zip_count_4w": rng.integers(1, 100, n).tolist(),
        "velocity_6h": rng.uniform(0.0, 10.0, n).tolist(),
        "velocity_24h": rng.uniform(0.0, 20.0, n).tolist(),
        "velocity_4w": rng.uniform(0.0, 50.0, n).tolist(),
        "bank_branch_count_8w": rng.integers(0, 20, n).tolist(),
        "date_of_birth_distinct_emails_4w": rng.integers(0, 5, n).tolist(),
        "email_is_free": rng.integers(0, 2, n).tolist(),
        "phone_home_valid": rng.integers(0, 2, n).tolist(),
        "phone_mobile_valid": rng.integers(0, 2, n).tolist(),
        "bank_months_count": rng.integers(0, 60, n).tolist(),
        "has_other_cards": rng.integers(0, 2, n).tolist(),
        "proposed_credit_limit": rng.uniform(200.0, 20000.0, n).tolist(),
        "foreign_request": rng.integers(0, 2, n).tolist(),
        "session_length_in_minutes": rng.uniform(0.0, 60.0, n).tolist(),
        "device_distinct_emails_8w": rng.integers(0, 5, n).tolist(),
        "name_email_similarity": rng.uniform(0.0, 1.0, n).tolist(),
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# validate_dataset — severity tests
# ---------------------------------------------------------------------------

class TestValidateDatasetSeverity:
    def test_severity_ok(self, tmp_path):
        """Dataset with clean data → severity == 'ok'."""
        df = _base_df(160)  # 8 months × 20 rows each, 2 fraud rows → ratio > 0.01
        path = _make_csv(df, tmp_path)
        result = validate_dataset(path)

        assert result["severity"] == "ok"
        assert result["temporal_integrity"]["out_of_range_count"] == 0
        assert result["temporal_integrity"]["gaps"] == []
        assert "missingness" in result
        assert "class_balance" in result

    def test_severity_warning_low_fraud_ratio(self, tmp_path):
        """Dataset with minority fraud ratio below 0.01 → severity == 'warning'."""
        df = _base_df(200)
        # Override fraud_bool: only 1 out of 200 is fraud (0.005 ratio)
        df["fraud_bool"] = 0
        df.loc[0, "fraud_bool"] = 1
        path = _make_csv(df, tmp_path)
        result = validate_dataset(path)

        assert result["severity"] == "warning"
        assert result["class_balance"]["ratio"] < 0.01

    def test_severity_warning_sentinel_above_10pct(self, tmp_path):
        """Dataset with >10% sentinel -1 in a column → severity == 'warning'."""
        df = _base_df(100)
        # Set 15% of rows to -1 in credit_risk_score (15 rows)
        df.loc[:14, "credit_risk_score"] = -1
        path = _make_csv(df, tmp_path)
        result = validate_dataset(path)

        assert result["severity"] == "warning"
        assert result["missingness"]["sentinel_missing"].get("credit_risk_score", 0) == 15

    def test_severity_critical_out_of_range_month(self, tmp_path):
        """Dataset with month values outside [0,7] → severity == 'critical'."""
        df = _base_df(100)
        df.loc[0, "month"] = 99  # out of range
        path = _make_csv(df, tmp_path)
        result = validate_dataset(path)

        assert result["severity"] == "critical"
        assert result["temporal_integrity"]["out_of_range_count"] > 0

    def test_severity_critical_sentinel_above_50pct(self, tmp_path):
        """Dataset with >50% sentinel -1 in a column → severity == 'critical'."""
        df = _base_df(100)
        # Set 60 rows to -1 in income (60%)
        df.loc[:59, "income"] = -1
        path = _make_csv(df, tmp_path)
        result = validate_dataset(path)

        assert result["severity"] == "critical"
        assert result["missingness"]["sentinel_missing"].get("income", 0) == 60

    def test_severity_critical_excess_duplicates(self, tmp_path):
        """Dataset with >10% duplicate rows → severity == 'critical'."""
        df = _base_df(80)
        # Append 20 copies of first row (25% duplication)
        dup_block = pd.concat([df.iloc[[0]]] * 20, ignore_index=True)
        df_with_dups = pd.concat([df, dup_block], ignore_index=True)
        path = _make_csv(df_with_dups, tmp_path)
        result = validate_dataset(path)

        assert result["severity"] == "critical"
        assert result["duplicate_row_count"] >= 20


# ---------------------------------------------------------------------------
# validate_dataset — sentinel -1 synthetic test
# ---------------------------------------------------------------------------

class TestSentinelMissingness:
    def test_sentinel_counted_not_nulls(self, tmp_path):
        """Pinned: sentinel -1 values in a numeric column are counted in sentinel_missing,
        NOT in true_nulls. This specifically validates BAF's encoding convention."""
        df = pd.DataFrame({
            "fraud_bool": [0, 0, 1, 0, 0],
            "month": [0, 1, 2, 3, 4],
            "income": [0.5, -1, -1, 0.8, 0.9],   # 2 sentinel -1 values
            "credit_risk_score": [300, 400, 500, 600, 700],
        })
        path = _make_csv(df, tmp_path)
        result = validate_dataset(path)

        # True nulls must show 0 for income (no actual NaNs)
        assert result["missingness"]["true_nulls"]["income"] == 0
        # Sentinel missing must count exactly 2
        assert result["missingness"]["sentinel_missing"]["income"] == 2

    def test_null_counted_in_true_nulls_not_sentinel(self, tmp_path):
        """Pinned: real NaN values appear in true_nulls and NOT in sentinel_missing."""
        df = pd.DataFrame({
            "fraud_bool": [0, 0, 1, 0, 0],
            "month": [0, 1, 2, 3, 4],
            "income": [0.5, None, None, 0.8, 0.9],  # 2 real NaN values
            "credit_risk_score": [300, 400, 500, 600, 700],
        })
        path = _make_csv(df, tmp_path)
        result = validate_dataset(path)

        assert result["missingness"]["true_nulls"]["income"] == 2
        # -1 sentinel count for income should be 0 (NaN != -1)
        assert result["missingness"]["sentinel_missing"].get("income", 0) == 0

    def test_temporal_gaps_detected(self, tmp_path):
        """Months 3 and 5 missing from data → detected in gaps list."""
        df = pd.DataFrame({
            "fraud_bool": [0, 1, 0, 0, 0, 0],
            "month": [0, 1, 2, 4, 6, 7],  # 3 and 5 are missing
            "income": [0.5] * 6,
            "credit_risk_score": [300] * 6,
        })
        path = _make_csv(df, tmp_path)
        result = validate_dataset(path)

        assert 3 in result["temporal_integrity"]["gaps"]
        assert 5 in result["temporal_integrity"]["gaps"]

    def test_missing_required_column_raises(self, tmp_path):
        """Dataset missing fraud_bool raises with dataset_not_loaded."""
        df = pd.DataFrame({"month": [0, 1, 2], "income": [0.5, 0.6, 0.7]})
        path = _make_csv(df, tmp_path)
        with pytest.raises((ValueError, FileNotFoundError, RuntimeError)) as exc_info:
            validate_dataset(path)
        assert "dataset_not_loaded" in str(exc_info.value)


# ---------------------------------------------------------------------------
# run_eda tests
# ---------------------------------------------------------------------------

class TestRunEda:
    def test_eda_basic_structure(self, tmp_path):
        """run_eda returns expected keys and non-empty distribution_summary."""
        df = _base_df(160)
        path = _make_csv(df, tmp_path)
        result = run_eda(path)

        assert "distribution_summary" in result
        assert "correlation_with_target" in result
        assert "high_correlation_pairs" in result
        assert "fraud_rate_by_month" in result

        # distribution_summary has entries for all numeric cols
        assert "income" in result["distribution_summary"]
        assert "fraud_bool" in result["distribution_summary"]
        assert all(
            k in result["distribution_summary"]["income"]
            for k in ["mean", "std", "min", "max", "q25", "q50", "q75"]
        )

    def test_eda_correlation_sorted_descending_by_abs(self, tmp_path):
        """correlation_with_target is sorted by absolute value descending."""
        df = _base_df(160)
        path = _make_csv(df, tmp_path)
        result = run_eda(path)

        corr_vals = list(result["correlation_with_target"].values())
        abs_corrs = [abs(v) for v in corr_vals]
        assert abs_corrs == sorted(abs_corrs, reverse=True)

    def test_eda_high_corr_pair_detected(self, tmp_path):
        """A near-perfectly correlated pair is detected in high_correlation_pairs."""
        import numpy as np
        df = _base_df(160)
        # Add a column that is ~99% correlated with income (which now varies per row)
        df["income_copy"] = df["income"] * 0.99 + 0.005
        # Verify income itself varies (not constant) — precondition for correlation
        assert df["income"].std() > 0, "income must vary for correlation test"
        path = _make_csv(df, tmp_path)
        result = run_eda(path)

        feature_pairs = [
            (p["feature_a"], p["feature_b"]) for p in result["high_correlation_pairs"]
        ]
        flat = [f for pair in feature_pairs for f in pair]
        assert "income" in flat or "income_copy" in flat, (
            f"Expected income/income_copy in high_correlation_pairs, got: {result['high_correlation_pairs']}"
        )

    def test_eda_fraud_rate_by_month_keys_are_month_strings(self, tmp_path):
        """fraud_rate_by_month keys are string-formatted month integers."""
        df = _base_df(160)
        path = _make_csv(df, tmp_path)
        result = run_eda(path)

        for k, v in result["fraud_rate_by_month"].items():
            assert isinstance(k, str)
            assert int(k) in range(0, 8)
            assert 0.0 <= v <= 1.0

    def test_eda_missing_file_raises(self, tmp_path):
        """run_eda raises FileNotFoundError with dataset_not_loaded for missing file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            run_eda(str(tmp_path / "nonexistent.csv"))
        assert "dataset_not_loaded" in str(exc_info.value)
