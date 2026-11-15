"""Tests for System MCP Server and Scoped File Operations."""

from __future__ import annotations

import pytest
from mcp_servers.client import get_mcp_client
from mcp_servers.system_mcp.server import system_mcp_server


def test_system_mcp_run_python_success() -> None:
    """Test executing basic Python code via MCP client."""
    client = get_mcp_client()
    res = client.call_tool("run_python", {"code": "result = 42 * 2"})
    assert res.get("result") == 84


def test_system_mcp_run_python_socket_blocked() -> None:
    """Test that socket import is blocked in sandbox via MCP client."""
    client = get_mcp_client()
    with pytest.raises(Exception) as exc_info:
        client.call_tool("run_python", {"code": "import socket\ns = socket.socket()"})
    assert "Network access disabled" in str(exc_info.value) or "forbidden import" in str(exc_info.value) or "execution_error" in str(exc_info.value)


def test_system_mcp_run_python_timeout() -> None:
    """Test that execution timeout is enforced via MCP client."""
    client = get_mcp_client()
    with pytest.raises(Exception) as exc_info:
        client.call_tool("run_python", {"code": "import time\ntime.sleep(6)", "timeout": 0.5})
    assert "timeout" in str(exc_info.value).lower()


def test_system_mcp_write_and_read_file_scoped() -> None:
    """Test write_file and read_file within a designated run directory."""
    client = get_mcp_client()
    task_id = "test_task_mcp_123"
    relative_path = "artifacts/report.txt"
    content = "Argus Fraud Detection Run Report: All systems nominal."

    write_res = client.call_tool(
        "write_file",
        {
            "task_id": task_id,
            "relative_path": relative_path,
            "content": content,
        },
    )
    assert write_res.get("status") == "written"
    assert write_res.get("bytes_written") == len(content.encode("utf-8"))

    read_res = client.call_tool(
        "read_file",
        {
            "task_id": task_id,
            "relative_path": relative_path,
        },
    )
    assert read_res.get("content") == content
    assert read_res.get("size_bytes") == len(content.encode("utf-8"))


def test_system_mcp_path_traversal_rejection() -> None:
    """Test that path traversal outside the run directory is rejected."""
    client = get_mcp_client()
    task_id = "test_task_traversal"

    with pytest.raises(Exception) as exc_info:
        client.call_tool(
            "write_file",
            {
                "task_id": task_id,
                "relative_path": "../../evil.txt",
                "content": "hacked",
            },
        )
    assert "Path traversal detected" in str(exc_info.value) or "outside allowed" in str(exc_info.value)

    with pytest.raises(Exception) as exc_info:
        client.call_tool(
            "read_file",
            {
                "task_id": task_id,
                "relative_path": "../../../etc/passwd",
            },
        )
    assert "Path traversal detected" in str(exc_info.value) or "outside allowed" in str(exc_info.value)
