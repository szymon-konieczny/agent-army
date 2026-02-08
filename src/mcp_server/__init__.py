"""AgentArmy MCP Server.

This package exposes AgentArmy capabilities via the Model Context Protocol (MCP),
enabling external AI tools (Claude Code, Claude Desktop, Cursor, etc.) to interact
with the agent fleet.
"""

from src.mcp_server.server import MCPServer

__all__ = ["MCPServer"]
