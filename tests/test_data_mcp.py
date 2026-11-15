"""Direct unit tests for Data MCP Server."""

import asyncio
import json
import pytest
from mcp_servers.data_mcp.server import data_mcp_server


def _run_async(coro):
    return asyncio.run(coro)


def test_data_mcp_load_dataset():
    """Verify load_dataset executes through Data MCP Server."""
    result = _run_async(data_mcp_server.call_tool("load_dataset", {"variant": "base"}))
    assert not result.is_error
    data = json.loads(result.content[0].text)
    assert data["rows"] == 1000000
    assert "fraud_bool" in data["columns"]


def test_data_mcp_inspect_dataset():
    """Verify inspect_dataset executes through Data MCP Server."""
    load_res = _run_async(data_mcp_server.call_tool("load_dataset", {"variant": "base"}))
    load_data = json.loads(load_res.content[0].text)

    inspect_res = _run_async(data_mcp_server.call_tool("inspect_dataset", {"dataset_path": load_data["path"]}))
    assert not inspect_res.is_error
    inspect_data = json.loads(inspect_res.content[0].text)
    assert inspect_data["row_count"] == 1000000
    assert "class_balance" in inspect_data


def test_data_mcp_run_sql():
    """Verify run_sql executes through Data MCP Server."""
    load_res = _run_async(data_mcp_server.call_tool("load_dataset", {"variant": "base"}))
    load_data = json.loads(load_res.content[0].text)

    sql_res = _run_async(
        data_mcp_server.call_tool(
            "run_sql",
            {
                "query": "SELECT COUNT(*) as count FROM base",
                "dataset_path": load_data["path"],
            },
        )
    )
    assert not sql_res.is_error
    sql_data = json.loads(sql_res.content[0].text)
    assert sql_data["row_count"] == 1
    assert sql_data["rows"][0]["count"] == 1000000
