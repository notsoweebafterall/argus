"""System MCP Server for Argus ML."""

from __future__ import annotations

from typing import Any
from mcp.server.mcpserver import MCPServer
from tools.system_tools import read_file, run_python, write_file

system_mcp_server = MCPServer("system_mcp")


@system_mcp_server.tool(
    name="run_python",
    description="Executes a Python snippet in a subprocess with working directory pinned to scratch dir, hard timeout, and network imports blocked.",
)
def mcp_run_python(
    code: str,
    context_vars: dict[str, Any] | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """MCP tool wrapper for run_python."""
    return run_python(code=code, context_vars=context_vars, timeout=timeout)


@system_mcp_server.tool(
    name="write_file",
    description="Writes text content to a file scoped strictly to the designated run directory (experiments/runs/{task_id}/).",
)
def mcp_write_file(
    task_id: str,
    relative_path: str,
    content: str,
) -> dict[str, Any]:
    """MCP tool wrapper for write_file."""
    return write_file(task_id=task_id, relative_path=relative_path, content=content)


@system_mcp_server.tool(
    name="read_file",
    description="Reads text content from a file scoped strictly to the designated run directory (experiments/runs/{task_id}/).",
)
def mcp_read_file(
    task_id: str,
    relative_path: str,
) -> dict[str, Any]:
    """MCP tool wrapper for read_file."""
    return read_file(task_id=task_id, relative_path=relative_path)


if __name__ == "__main__":
    system_mcp_server.run(transport="stdio")
