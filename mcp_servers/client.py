"""Unified MCP Client and Tool Router for Argus ML."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from agents.contracts import ToolSpec

logger = logging.getLogger(__name__)

# Registry mapping tool names to owning MCP server
TOOL_TO_SERVER_MAP: dict[str, str] = {
    "load_dataset": "data_mcp",
    "load_dataset_from_path": "data_mcp",
    "inspect_dataset": "data_mcp",
    "run_sql": "data_mcp",
    "validate_dataset": "data_mcp",
    "run_eda": "data_mcp",
    "create_features": "ml_mcp",
    "train_model": "ml_mcp",
    "search_hyperparams": "ml_mcp",
    "evaluate_model": "ml_mcp",
    "check_fairness": "ml_mcp",
    "save_model": "ml_mcp",
    "load_model": "ml_mcp",
    "register_model": "ml_mcp",
    "run_python": "system_mcp",
    "write_file": "system_mcp",
    "read_file": "system_mcp",
    "retrieve_similar_failures": "retrieval_mcp",
}


class MCPClient:
    """Client for routing tool calls across registered MCP servers."""

    def __init__(self) -> None:
        self.tool_map = dict(TOOL_TO_SERVER_MAP)
        self._servers: dict[str, Any] = {}

    def _get_server(self, server_name: str) -> Any:
        """Lazy load and return MCP server instance."""
        if server_name not in self._servers:
            if server_name == "data_mcp":
                from mcp_servers.data_mcp.server import data_mcp_server
                self._servers[server_name] = data_mcp_server
            elif server_name == "ml_mcp":
                from mcp_servers.ml_mcp.server import ml_mcp_server
                self._servers[server_name] = ml_mcp_server
            elif server_name == "system_mcp":
                from mcp_servers.system_mcp.server import system_mcp_server
                self._servers[server_name] = system_mcp_server
            elif server_name == "retrieval_mcp":
                from mcp_servers.retrieval_mcp.server import retrieval_mcp_server
                self._servers[server_name] = retrieval_mcp_server
            else:
                raise ValueError(f"Unknown MCP server '{server_name}'")
        return self._servers[server_name]

    def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Route tool invocation to the owning MCP server and unpack result."""
        args = arguments or {}
        server_name = self.tool_map.get(tool_name)

        if not server_name:
            raise ValueError(f"Tool '{tool_name}' is not registered with any MCP server.")

        server = self._get_server(server_name)
        logger.debug(f"Routing tool '{tool_name}' through MCP server '{server_name}'")

        try:
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, server.call_tool(tool_name, args))
                    result = future.result()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(server.call_tool(tool_name, args))
        except Exception as exc:
            if exc.__cause__ is not None:
                raise exc.__cause__ from None
            raise

        # Unpack MCP result content
        if hasattr(result, "is_error") and result.is_error:
            error_text = ""
            if hasattr(result, "content") and result.content:
                error_text = result.content[0].text
            raise RuntimeError(f"MCP tool error in '{tool_name}': {error_text}")

        if hasattr(result, "content") and result.content:
            text_content = result.content[0].text
            try:
                parsed = json.loads(text_content)
                if isinstance(parsed, dict):
                    return parsed
                return {"result": parsed}
            except (json.JSONDecodeError, TypeError):
                return {"result": text_content}

        if hasattr(result, "structured_content") and result.structured_content:
            return result.structured_content

        return {}

    def wrap_tool(self, tool_spec: Any) -> Any:
        """Create an MCP-wrapped ToolSpec whose handler executes via MCPClient."""
        from agents.contracts import ToolSpec
        tool_name = tool_spec.name

        def mcp_handler(**kwargs: Any) -> dict[str, Any]:
            return self.call_tool(tool_name, kwargs)

        return ToolSpec(
            name=tool_spec.name,
            description=tool_spec.description,
            parameters=tool_spec.parameters,
            handler=mcp_handler,
        )

    def wrap_tools(self, tools: list[Any]) -> list[Any]:
        """Wrap a list of ToolSpecs to route through MCP."""
        return [self.wrap_tool(tool) if tool.name in self.tool_map else tool for tool in tools]


# Global singleton instance
_GLOBAL_MCP_CLIENT: MCPClient | None = None


def get_mcp_client() -> MCPClient:
    """Get or create the global MCP client instance."""
    global _GLOBAL_MCP_CLIENT
    if _GLOBAL_MCP_CLIENT is None:
        _GLOBAL_MCP_CLIENT = MCPClient()
    return _GLOBAL_MCP_CLIENT
