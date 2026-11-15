"""Unit tests for Argus Data Tools."""

import os
import tempfile
import pandas as pd
import pytest

from pipeline.config import settings
from tools.data_tools import inspect_dataset, load_dataset, run_sql


def test_load_dataset_success():
    """Verify loading Base BAF dataset."""
    result = load_dataset("base")
    assert result["rows"] > 0
    assert "fraud_bool" in result["columns"]
    assert "month" in result["columns"]
    assert os.path.exists(result["path"])


def test_load_dataset_file_not_found():
    """Verify loading non-existent variant raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_dataset("non_existent_variant_xyz")
    assert "file_not_found" in str(exc_info.value)


def test_load_dataset_schema_mismatch(tmp_path):
    """Verify schema mismatch error when required columns are absent."""
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]}).to_csv(bad_csv, index=False)

    # Temporary monkeypatch workspace root data path
    orig_workspace = settings.workspace_root
    try:
        settings.__dict__["workspace_root"] = tmp_path.parent
        bad_in_data = tmp_path.parent / "data"
        bad_in_data.mkdir(exist_ok=True)
        pd.DataFrame({"col_a": [1, 2]}).to_csv(bad_in_data / "Base.csv", index=False)

        with pytest.raises(ValueError) as exc_info:
            load_dataset("base")
        assert "schema_mismatch" in str(exc_info.value)
    finally:
        settings.__dict__["workspace_root"] = orig_workspace


def test_inspect_dataset_success():
    """Verify dataset inspection provides dtypes, nulls, and class balance."""
    load_res = load_dataset("base")
    inspect_res = inspect_dataset(load_res["path"])

    assert inspect_res["row_count"] > 0
    assert "fraud_bool" in inspect_res["dtypes"]
    assert "class_balance" in inspect_res
    assert "0" in inspect_res["class_balance"]["counts"]
    assert "1" in inspect_res["class_balance"]["counts"]
    assert "0" in inspect_res["class_balance"]["ratios"]
    assert "1" in inspect_res["class_balance"]["ratios"]


def test_inspect_dataset_missing_file():
    """Verify inspecting missing file raises dataset_not_loaded error."""
    with pytest.raises(FileNotFoundError) as exc_info:
        inspect_dataset("c:/invalid/path/to/nonexistent.csv")
    assert "dataset_not_loaded" in str(exc_info.value)


def test_run_sql_success():
    """Verify SQL query execution on SQLite table."""
    load_res = load_dataset("base")
    query = "SELECT COUNT(*) as total_count, AVG(fraud_bool) as avg_fraud FROM base"
    sql_res = run_sql(query, load_res["path"])

    assert sql_res["row_count"] == 1
    assert "total_count" in sql_res["rows"][0]
    assert sql_res["rows"][0]["total_count"] > 0


def test_run_sql_invalid_query():
    """Verify syntax error in SQL raises ValueError with invalid_query."""
    load_res = load_dataset("base")
    with pytest.raises(ValueError) as exc_info:
        run_sql("SELECT NON_EXISTING_SYNTAX FROM WHERE", load_res["path"])
    assert "invalid_query" in str(exc_info.value)


def test_run_sql_table_not_found():
    """Verify querying non-existent table raises RuntimeError with table_not_found."""
    load_res = load_dataset("base")
    with pytest.raises(RuntimeError) as exc_info:
        run_sql("SELECT * FROM non_existing_table_xyz", load_res["path"])
    assert "table_not_found" in str(exc_info.value)
