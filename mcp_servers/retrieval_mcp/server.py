"""Retrieval MCP Server for Argus ML."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from tools.retrieval_tools import retrieve_similar_failures

retrieval_mcp_server = MCPServer("retrieval_mcp")


@retrieval_mcp_server.tool(
    name="retrieve_similar_failures",
    description=(
        "Retrieves past critic failures most similar to a given failure description. "
        "Returns run_id, similarity score, and the failed_checks from that run. "
        "Use when deciding whether a model's metrics/profile resembles a previously rejected run."
    ),
)
def mcp_retrieve_similar_failures(query: str, k: int = 3) -> dict[str, Any]:
    """MCP tool wrapper for retrieve_similar_failures."""
    return retrieve_similar_failures(query=query, k=k)


if __name__ == "__main__":
    retrieval_mcp_server.run(transport="stdio")