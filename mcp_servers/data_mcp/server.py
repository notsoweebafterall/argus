"""Data MCP Server for Argus ML."""

from __future__ import annotations
from typing import Any
from mcp.server.mcpserver import MCPServer
from tools.data_tools import inspect_dataset, load_dataset, load_dataset_from_path, run_eda, run_sql, validate_dataset

data_mcp_server = MCPServer("data_mcp")


@data_mcp_server.tool(
    name="load_dataset",
    description="Loads a BAF dataset variant (base, I, II, III, IV, V) and verifies required columns.",
)
def mcp_load_dataset(variant: str) -> dict[str, Any]:
    """MCP tool wrapper for load_dataset."""
    return load_dataset(variant=variant)


@data_mcp_server.tool(
    name="load_dataset_from_path",
    description="Loads a BAF CSV from a path."
)
def mcp_load_dataset_from_path(dataset_path: str) -> dict[str, Any]:
    return load_dataset_from_path(dataset_path=dataset_path)


@data_mcp_server.tool(
    name="inspect_dataset",
    description="Inspects schema, dtypes, missing values, total rows, and class balance of a dataset.",
)
def mcp_inspect_dataset(dataset_path: str) -> dict[str, Any]:
    """MCP tool wrapper for inspect_dataset."""
    return inspect_dataset(dataset_path=dataset_path)


@data_mcp_server.tool(
    name="run_sql",
    description="Runs a SQL query against an in-memory SQLite table loaded from a dataset CSV.",
)
def mcp_run_sql(query: str, dataset_path: str) -> dict[str, Any]:
    """MCP tool wrapper for run_sql."""
    return run_sql(query=query, dataset_path=dataset_path)


@data_mcp_server.tool(
    name="validate_dataset",
    description="Runs deterministic validation on a BAF dataset: sentinel-aware missingness, dtype checks, duplicates, class balance, and temporal integrity. Returns severity: ok/warning/critical.",
)
def mcp_validate_dataset(dataset_path: str) -> dict[str, Any]:
    """MCP tool wrapper for validate_dataset."""
    return validate_dataset(dataset_path=dataset_path)


@data_mcp_server.tool(
    name="run_eda",
    description="Runs EDA on a BAF dataset: distribution summary, correlation with fraud_bool, multicollinearity pairs, and fraud rate by month.",
)
def mcp_run_eda(dataset_path: str) -> dict[str, Any]:
    """MCP tool wrapper for run_eda."""
    return run_eda(dataset_path=dataset_path)


if __name__ == "__main__":
    data_mcp_server.run(transport="stdio")
