"""Unit tests for MCP Client router."""

import pytest
from mcp_servers.client import MCPClient, get_mcp_client
from tools.data_tools import LOAD_DATASET_SPEC


def test_mcp_client_routing_data_tools():
    """Verify MCP client routes data tools to data_mcp."""
    client = MCPClient()
    result = client.call_tool("load_dataset", {"variant": "base"})
    assert result["rows"] == 1000000
    assert "fraud_bool" in result["columns"]


def test_mcp_client_wrap_tool():
    """Verify wrap_tool produces an executable ToolSpec dispatched via MCP."""
    client = get_mcp_client()
    wrapped_spec = client.wrap_tool(LOAD_DATASET_SPEC)

    assert wrapped_spec.name == LOAD_DATASET_SPEC.name
    assert wrapped_spec.handler is not None
    res = wrapped_spec.handler(variant="base")
    assert res["rows"] == 1000000


def test_mcp_client_unknown_tool():
    """Verify MCP client raises ValueError for unregistered tool."""
    client = MCPClient()
    with pytest.raises(ValueError) as exc_info:
        client.call_tool("non_existent_tool_xyz", {})
    assert "not registered" in str(exc_info.value)
