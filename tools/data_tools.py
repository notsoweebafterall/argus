"""Data tools for Argus ML."""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any
import pandas as pd

from agents.contracts import ToolSpec
from pipeline.config import settings

# In-memory SQLite connection cache for datasets
_SQLITE_CACHE: dict[str, sqlite3.Connection] = {}


def _resolve_variant_path(variant: str) -> Path:
    """Resolve a variant identifier to its corresponding CSV path."""
    v_clean = variant.strip().lower()
    variant_map = {
        "base": "Base.csv",
        "i": "Variant I.csv",
        "1": "Variant I.csv",
        "variant i": "Variant I.csv",
        "variant_i": "Variant I.csv",
        "ii": "Variant II.csv",
        "2": "Variant II.csv",
        "variant ii": "Variant II.csv",
        "variant_ii": "Variant II.csv",
        "iii": "Variant III.csv",
        "3": "Variant III.csv",
        "variant iii": "Variant III.csv",
        "variant_iii": "Variant III.csv",
        "iv": "Variant IV.csv",
        "4": "Variant IV.csv",
        "variant iv": "Variant IV.csv",
        "variant_iv": "Variant IV.csv",
        "v": "Variant V.csv",
        "5": "Variant V.csv",
        "variant v": "Variant V.csv",
        "variant_v": "Variant V.csv",
    }

    filename = variant_map.get(v_clean)
    if not filename:
        filename = f"{variant}.csv" if not variant.endswith(".csv") else variant

    return settings.workspace_root / "data" / filename


def load_dataset(variant: str) -> dict[str, Any]:
    """Load a BAF dataset variant and verify its schema.

    Returns rows count, columns list, and resolved file path.
    """
    path = _resolve_variant_path(variant)
    if not path.exists():
        raise FileNotFoundError(f"file_not_found: Dataset variant '{variant}' not found at {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"file_not_found: Unable to read dataset at {path}: {exc}") from exc

    required_columns = {"fraud_bool", "month"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"schema_mismatch: Dataset is missing required columns: {missing}")

    return {
        "rows": len(df),
        "columns": list(df.columns),
        "path": str(path.resolve()),
    }


def inspect_dataset(dataset_path: str) -> dict[str, Any]:
    """Inspect dataset statistics, data types, null counts, and class balance for fraud_bool."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"dataset_not_loaded: Dataset file does not exist at '{dataset_path}'")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"dataset_not_loaded: Failed to inspect dataset: {exc}") from exc

    if "fraud_bool" not in df.columns:
        raise ValueError("dataset_not_loaded: Required target column 'fraud_bool' missing from dataset")

    raw_counts = df["fraud_bool"].value_counts().to_dict()
    ratios = df["fraud_bool"].value_counts(normalize=True).to_dict()

    class_balance = {
        "counts": {str(k): int(v) for k, v in raw_counts.items()},
        "ratios": {str(k): float(round(v, 6)) for k, v in ratios.items()},
    }

    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    null_counts = {col: int(count) for col, count in df.isnull().sum().items()}

    return {
        "dtypes": dtypes,
        "null_counts": null_counts,
        "class_balance": class_balance,
        "row_count": len(df),
    }


def _get_sqlite_conn(dataset_path: str) -> tuple[sqlite3.Connection, str]:
    """Get or create in-memory SQLite connection with dataset loaded."""
    path = Path(dataset_path).resolve()
    path_key = str(path)

    if path_key in _SQLITE_CACHE:
        table_name = path.stem.lower().replace(" ", "_")
        return _SQLITE_CACHE[path_key], table_name

    if not path.exists():
        raise FileNotFoundError(f"table_not_found: Dataset file not found at '{dataset_path}'")

    df = pd.read_csv(path)
    conn = sqlite3.connect(":memory:", check_same_thread=False)

    # Derive primary table name, e.g. "base", "variant_i"
    table_name = path.stem.lower().replace(" ", "_")
    df.to_sql(table_name, conn, index=False, if_exists="replace")

    # Also register convenient aliases like "dataset", "baf", and stem without spaces
    aliases = {"dataset", "baf", path.stem.lower(), path.stem.replace(" ", "")}
    for alias in aliases:
        if alias != table_name:
            try:
                df.to_sql(alias, conn, index=False, if_exists="replace")
            except Exception:
                pass

    _SQLITE_CACHE[path_key] = conn
    return conn, table_name


def validate_dataset(dataset_path: str) -> dict[str, Any]:
    """Run deterministic validation checks on a BAF dataset.

    Checks true nulls, sentinel -1 missingness (numeric cols only), dtype issues,
    duplicate row count, class balance, and temporal integrity of the `month` column.
    Returns a severity level: 'ok', 'warning', or 'critical'.

    Note: BAF encodes unknown values as -1 in numeric columns per the dataset paper.
    isnull() alone is insufficient — this tool explicitly counts sentinel -1 values.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"dataset_not_loaded: Dataset file does not exist at '{dataset_path}'")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"dataset_not_loaded: Failed to load dataset: {exc}") from exc

    if "fraud_bool" not in df.columns or "month" not in df.columns:
        raise ValueError("dataset_not_loaded: Required columns 'fraud_bool' and/or 'month' missing")

    n_rows = len(df)

    # --- Missingness ---
    true_nulls = {col: int(df[col].isnull().sum()) for col in df.columns}
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    sentinel_missing = {
        col: int((df[col] == -1).sum())
        for col in numeric_cols
        if (df[col] == -1).any()
    }

    # --- Dtype issues: expected numeric types for known BAF columns ---
    BAF_EXPECTED_NUMERIC = {
        "fraud_bool", "income", "name_email_similarity", "prev_address_months_count",
        "current_address_months_count", "customer_age", "days_since_request",
        "intended_balcon_amount", "zip_count_4w", "velocity_6h", "velocity_24h",
        "velocity_4w", "bank_branch_count_8w", "date_of_birth_distinct_emails_4w",
        "credit_risk_score", "email_is_free", "phone_home_valid", "phone_mobile_valid",
        "bank_months_count", "has_other_cards", "proposed_credit_limit",
        "foreign_request", "session_length_in_minutes", "device_distinct_emails_8w",
        "month",
    }
    dtype_issues = [
        col for col in BAF_EXPECTED_NUMERIC
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
    ]

    # --- Duplicates ---
    duplicate_row_count = int(df.duplicated().sum())

    # --- Class balance ---
    counts = df["fraud_bool"].value_counts().to_dict()
    total = sum(counts.values())
    minority_count = counts.get(1, counts.get(1.0, 0))
    minority_ratio = float(minority_count / total) if total > 0 else 0.0
    class_balance = {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "ratio": round(minority_ratio, 6),
    }

    # --- Temporal integrity ---
    EXPECTED_RANGE = [0, 7]
    month_min = int(df["month"].min())
    month_max = int(df["month"].max())
    expected_months = set(range(EXPECTED_RANGE[0], EXPECTED_RANGE[1] + 1))
    actual_months = set(df["month"].dropna().astype(int).unique())
    gaps = sorted(expected_months - actual_months)
    out_of_range_count = int(((df["month"] < EXPECTED_RANGE[0]) | (df["month"] > EXPECTED_RANGE[1])).sum())
    temporal_integrity = {
        "month_min": month_min,
        "month_max": month_max,
        "expected_range": EXPECTED_RANGE,
        "gaps": gaps,
        "out_of_range_count": out_of_range_count,
    }

    # --- Severity (deterministic) ---
    sentinel_ratios = {
        col: count / n_rows for col, count in sentinel_missing.items()
    }
    max_sentinel_ratio = max(sentinel_ratios.values(), default=0.0)
    dup_ratio = duplicate_row_count / n_rows if n_rows > 0 else 0.0

    if out_of_range_count > 0 or max_sentinel_ratio > 0.5 or dup_ratio > 0.1:
        severity = "critical"
    elif max_sentinel_ratio > 0.1 or minority_ratio < 0.01:
        severity = "warning"
    else:
        severity = "ok"

    return {
        "missingness": {
            "true_nulls": true_nulls,
            "sentinel_missing": sentinel_missing,
        },
        "dtype_issues": dtype_issues,
        "duplicate_row_count": duplicate_row_count,
        "class_balance": class_balance,
        "temporal_integrity": temporal_integrity,
        "severity": severity,
    }


def run_eda(dataset_path: str) -> dict[str, Any]:
    """Run exploratory data analysis on a BAF dataset.

    Returns distribution summary per numeric column, correlation with fraud_bool
    (sorted by absolute value descending), high-correlation feature pairs
    (|corr| > 0.9, multicollinearity flag), and fraud rate by month.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"dataset_not_loaded: Dataset file does not exist at '{dataset_path}'")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"dataset_not_loaded: Failed to load dataset for EDA: {exc}") from exc

    if "fraud_bool" not in df.columns or "month" not in df.columns:
        raise ValueError("dataset_not_loaded: Required columns 'fraud_bool' and/or 'month' missing")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    feature_cols = [c for c in numeric_cols if c != "fraud_bool"]

    # --- Distribution summary ---
    distribution_summary: dict[str, Any] = {}
    for col in numeric_cols:
        s = df[col]
        q = s.quantile([0.25, 0.5, 0.75])
        distribution_summary[col] = {
            "mean": round(float(s.mean()), 6),
            "std": round(float(s.std()), 6),
            "min": round(float(s.min()), 6),
            "max": round(float(s.max()), 6),
            "q25": round(float(q[0.25]), 6),
            "q50": round(float(q[0.50]), 6),
            "q75": round(float(q[0.75]), 6),
        }

    # --- Correlation with target ---
    target = df["fraud_bool"]
    corr_with_target: dict[str, float] = {}
    for col in feature_cols:
        try:
            corr_val = float(df[col].corr(target))
            if not (corr_val != corr_val):  # NaN check
                corr_with_target[col] = round(corr_val, 6)
        except Exception:
            pass
    # Sort by absolute value descending
    corr_with_target = dict(
        sorted(corr_with_target.items(), key=lambda kv: abs(kv[1]), reverse=True)
    )

    # --- High-correlation pairs (multicollinearity) ---
    high_correlation_pairs: list[dict[str, Any]] = []
    if len(feature_cols) > 1:
        corr_matrix = df[feature_cols].corr()
        checked = set()
        for i, col_a in enumerate(feature_cols):
            for col_b in feature_cols[i + 1:]:
                pair_key = (col_a, col_b)
                if pair_key in checked:
                    continue
                checked.add(pair_key)
                try:
                    val = float(corr_matrix.loc[col_a, col_b])
                    if not (val != val) and abs(val) > 0.9:
                        high_correlation_pairs.append({
                            "feature_a": col_a,
                            "feature_b": col_b,
                            "correlation": round(val, 6),
                        })
                except Exception:
                    pass

    # --- Fraud rate by month ---
    fraud_rate_by_month: dict[str, float] = {}
    for month_val, group in df.groupby("month"):
        rate = float(group["fraud_bool"].mean())
        fraud_rate_by_month[str(int(month_val))] = round(rate, 6)

    return {
        "distribution_summary": distribution_summary,
        "correlation_with_target": corr_with_target,
        "high_correlation_pairs": high_correlation_pairs,
        "fraud_rate_by_month": fraud_rate_by_month,
    }


FORBIDDEN_SQL_KEYWORDS = {
    "ATTACH",
    "DETACH",
    "PRAGMA",
    "ALTER",
    "DROP",
    "INSERT",
    "UPDATE",
    "DELETE",
    "CREATE",
    "REPLACE",
    "VACUUM",
    "REINDEX",
    "GRANT",
    "REVOKE",
}


def _validate_safe_sql(query: str) -> None:
    """Validate that query is strictly a read-only SELECT or WITH statement without dangerous commands."""
    clean = query.strip()
    if not clean:
        raise ValueError("invalid_query: SQL query cannot be empty.")

    # Reject multiple statements
    if ";" in clean.rstrip(";"):
        raise ValueError("unauthorized_sql_statement: Multiple semicolon-separated SQL statements are prohibited.")

    # Must start with SELECT or WITH
    upper_clean = clean.upper()
    if not (upper_clean.startswith("SELECT") or upper_clean.startswith("WITH")):
        raise ValueError("unauthorized_sql_statement: Only SELECT or WITH queries are permitted in run_sql.")

    # Check for forbidden keywords as isolated word tokens
    for kw in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{kw}\b", clean, re.IGNORECASE):
            raise ValueError(f"unauthorized_sql_statement: Forbidden SQL statement/keyword '{kw}' detected in run_sql.")


def run_sql(query: str, dataset_path: str) -> dict[str, Any]:
    """Execute SQL query against dataset in SQLite, guarded by strict read-only statement restrictions."""
    _validate_safe_sql(query)

    try:
        conn, _ = _get_sqlite_conn(dataset_path)
    except FileNotFoundError as exc:
        raise exc
    except Exception as exc:
        raise RuntimeError(f"table_not_found: Could not initialize database table: {exc}") from exc

    try:
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows_data = cursor.fetchall()
        rows = [dict(zip(columns, row)) for row in rows_data]
        return {
            "rows": rows,
            "row_count": len(rows),
        }
    except sqlite3.OperationalError as exc:
        err_msg = str(exc)
        if "no such table" in err_msg.lower():
            raise RuntimeError(f"table_not_found: {err_msg}") from exc
        raise ValueError(f"invalid_query: {err_msg}") from exc
    except Exception as exc:
        err_str = str(exc)
        if "table_not_found" in err_str:
            raise exc
        raise ValueError(f"invalid_query: {err_str}") from exc


# Tool Specs
VALIDATE_DATASET_SPEC = ToolSpec(
    name="validate_dataset",
    description="Runs deterministic validation on a BAF dataset: sentinel-aware missingness, dtype checks, duplicates, class balance, and temporal integrity. Returns severity: ok/warning/critical.",
    parameters={
        "type": "object",
        "properties": {
            "dataset_path": {
                "type": "string",
                "description": "Path to the dataset CSV file.",
            }
        },
        "required": ["dataset_path"],
    },
    handler=validate_dataset,
)

RUN_EDA_SPEC = ToolSpec(
    name="run_eda",
    description="Runs EDA on a BAF dataset: distribution summary, correlation with fraud_bool, multicollinearity pairs, and fraud rate by month.",
    parameters={
        "type": "object",
        "properties": {
            "dataset_path": {
                "type": "string",
                "description": "Path to the dataset CSV file.",
            }
        },
        "required": ["dataset_path"],
    },
    handler=run_eda,
)

LOAD_DATASET_SPEC = ToolSpec(
    name="load_dataset",
    description="Loads a BAF dataset variant (base, I, II, III, IV, V) and verifies required columns.",
    parameters={
        "type": "object",
        "properties": {
            "variant": {
                "type": "string",
                "enum": ["base", "I", "II", "III", "IV", "V"],
                "description": "Variant name of the BAF dataset.",
            }
        },
        "required": ["variant"],
    },
    handler=load_dataset,
)

INSPECT_DATASET_SPEC = ToolSpec(
    name="inspect_dataset",
    description="Inspects schema, dtypes, missing values, total rows, and class balance of a dataset.",
    parameters={
        "type": "object",
        "properties": {
            "dataset_path": {
                "type": "string",
                "description": "Path to the dataset CSV file.",
            }
        },
        "required": ["dataset_path"],
    },
    handler=inspect_dataset,
)

RUN_SQL_SPEC = ToolSpec(
    name="run_sql",
    description="Runs a SQL query against an in-memory SQLite table loaded from a dataset CSV.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL query to execute.",
            },
            "dataset_path": {
                "type": "string",
                "description": "Path to the dataset CSV file.",
            },
        },
        "required": ["query", "dataset_path"],
    },
    handler=run_sql,
)
