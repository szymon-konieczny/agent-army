"""MCP Client Manager — AgentArmy agents consuming external MCP servers.

This module handles the other direction of MCP integration: AgentArmy agents
can USE external MCP tools and resources from other MCP servers.

Example external MCP servers AgentArmy might connect to:
  - GitHub MCP (repository operations, issue management)
  - Filesystem MCP (file access and manipulation)
  - Database MCP (direct database queries)
  - Custom tool servers (domain-specific tools)
  - Slack MCP (messaging)
  - Linear MCP (project management)

This manager discovers, connects to, and manages tool calls across multiple
external MCP servers, making their capabilities available to agents.
"""

import asyncio
from enum import Enum
from typing import Any, Optional

import httpx
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class TransportType(str, Enum):
    """Supported MCP transport types."""

    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable-http"


class ToolDefinition(BaseModel):
    """Definition of an MCP tool."""

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    input_schema: dict[str, Any] = Field(description="JSON Schema for tool input")
    server_name: str = Field(description="Parent MCP server name")


class ResourceDefinition(BaseModel):
    """Definition of an MCP resource."""

    uri: str = Field(description="Resource URI")
    name: str = Field(description="Resource name")
    description: str = Field(description="Resource description")
    mime_type: str = Field(default="application/json", description="MIME type")
    server_name: str = Field(description="Parent MCP server name")


class MCPConnection(BaseModel):
    """Represents a connection to an external MCP server.

    Attributes:
        server_name: Unique name for this MCP server
        transport: Transport type (stdio, sse, streamable-http)
        config: Transport-specific configuration
        connected: Whether currently connected
        available_tools: List of available tools from this server
        available_resources: List of available resources
        last_heartbeat: Timestamp of last successful communication
    """

    server_name: str = Field(description="Unique server name")
    transport: TransportType = Field(description="Transport type")
    config: dict[str, Any] = Field(description="Transport config")
    connected: bool = Field(default=False, description="Connection status")
    available_tools: list[ToolDefinition] = Field(
        default_factory=list, description="Available tools"
    )
    available_resources: list[ResourceDefinition] = Field(
        default_factory=list, description="Available resources"
    )
    last_heartbeat: Optional[str] = Field(
        default=None, description="Last successful communication"
    )
    error_message: Optional[str] = Field(default=None, description="Last error if failed")


class MCPClientManager:
    """Manages connections to external MCP servers and tool discovery.

    This manager handles:
    - Connecting to external MCP servers
    - Tool and resource discovery
    - Tool invocation routing
    - Connection lifecycle management
    - Error handling and reconnection logic

    Attributes:
        connections: Map of server names to MCPConnection objects
        tools_cache: Cached tool definitions by server and tool name
    """

    def __init__(self) -> None:
        """Initialize the MCP client manager."""
        self.connections: dict[str, MCPConnection] = {}
        self.tools_cache: dict[str, dict[str, ToolDefinition]] = {}
        self._lock = asyncio.Lock()
        self._logger = structlog.get_logger(__name__)

    async def connect(
        self,
        server_name: str,
        transport: TransportType,
        config: dict[str, Any],
    ) -> MCPConnection:
        """Connect to an external MCP server.

        Establishes a connection to an MCP server and discovers available
        tools and resources. Supports multiple transport types.

        Args:
            server_name: Unique name for this server
            transport: Transport type (stdio, sse, streamable-http)
            config: Transport-specific configuration:
                For stdio: {"command": "...", "args": [...]}
                For sse: {"url": "http://..."}
                For streamable-http: {"url": "http://..."}

        Returns:
            MCPConnection object with tools and resources discovered

        Raises:
            ValueError: If server already connected or invalid config
            httpx.ConnectError: If connection fails
        """
        async with self._lock:
            if server_name in self.connections:
                raise ValueError(f"Server '{server_name}' already connected")

            self._logger.info("connecting_to_mcp_server", server=server_name, transport=transport.value)

            try:
                connection = await self._establish_connection(server_name, transport, config)
                await self._discover_tools(connection)
                await self._discover_resources(connection)

                self.connections[server_name] = connection
                self.tools_cache[server_name] = {
                    tool.name: tool for tool in connection.available_tools
                }

                self._logger.info(
                    "mcp_connected",
                    server=server_name,
                    tools_count=len(connection.available_tools),
                    resources_count=len(connection.available_resources),
                )

                return connection

            except Exception as e:
                self._logger.error("mcp_connection_failed", server=server_name, error=str(e))
                # Create failed connection record
                connection = MCPConnection(
                    server_name=server_name,
                    transport=transport,
                    config=config,
                    connected=False,
                    error_message=str(e),
                )
                self.connections[server_name] = connection
                raise

    async def disconnect(self, server_name: str) -> None:
        """Disconnect from an MCP server.

        Args:
            server_name: Name of server to disconnect

        Raises:
            ValueError: If server not connected
        """
        async with self._lock:
            if server_name not in self.connections:
                raise ValueError(f"Server '{server_name}' not connected")

            connection = self.connections[server_name]
            self._logger.info("disconnecting_from_mcp_server", server=server_name)

            # Clean up any transport-specific resources here
            if connection.transport == TransportType.STDIO:
                # Cleanup stdio process if any
                pass
            elif connection.transport == TransportType.SSE:
                # Close HTTP connection if any
                pass

            del self.connections[server_name]
            if server_name in self.tools_cache:
                del self.tools_cache[server_name]

            self._logger.info("mcp_disconnected", server=server_name)

    def list_connections(self) -> list[MCPConnection]:
        """List all current MCP server connections.

        Returns:
            List of MCPConnection objects
        """
        return list(self.connections.values())

    def get_connection(self, server_name: str) -> Optional[MCPConnection]:
        """Get a specific connection by server name.

        Args:
            server_name: Name of the server

        Returns:
            MCPConnection if found, None otherwise
        """
        return self.connections.get(server_name)

    def list_tools(
        self,
        server_name: Optional[str] = None,
    ) -> dict[str, list[str]]:
        """List available tools.

        Args:
            server_name: If specified, only tools from this server.
                        If None, tools from all servers.

        Returns:
            Dictionary mapping server names to lists of tool names
        """
        if server_name:
            if server_name not in self.tools_cache:
                return {server_name: []}
            return {
                server_name: list(self.tools_cache[server_name].keys())
            }

        return {
            server: list(tools.keys())
            for server, tools in self.tools_cache.items()
        }

    def discover_tools(self) -> dict[str, list[ToolDefinition]]:
        """Get full tool definitions for discovery.

        Returns:
            Dictionary mapping server names to lists of ToolDefinition objects
        """
        return {
            server: list(tools.values())
            for server, tools in self.tools_cache.items()
        }

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Call a tool on a specific MCP server.

        Invokes a tool on an external MCP server and returns the result.
        Handles error cases gracefully.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to call
            arguments: Arguments for the tool

        Returns:
            Tool execution result

        Raises:
            ValueError: If server or tool not found
            httpx.HTTPError: If tool invocation fails
        """
        if server_name not in self.connections:
            raise ValueError(f"Server '{server_name}' not connected")

        connection = self.connections[server_name]
        if not connection.connected:
            raise ValueError(f"Server '{server_name}' not connected")

        if server_name not in self.tools_cache or tool_name not in self.tools_cache[server_name]:
            raise ValueError(f"Tool '{tool_name}' not found on server '{server_name}'")

        self._logger.info(
            "calling_mcp_tool",
            server=server_name,
            tool=tool_name,
        )

        try:
            result = await self._invoke_tool(server_name, tool_name, arguments)
            self._logger.info("mcp_tool_success", server=server_name, tool=tool_name)
            return result
        except Exception as e:
            self._logger.error(
                "mcp_tool_failed",
                server=server_name,
                tool=tool_name,
                error=str(e),
            )
            raise

    # Private implementation methods

    async def _establish_connection(
        self,
        server_name: str,
        transport: TransportType,
        config: dict[str, Any],
    ) -> MCPConnection:
        """Establish transport connection.

        Args:
            server_name: Server name
            transport: Transport type
            config: Transport config

        Returns:
            MCPConnection object
        """
        connection = MCPConnection(
            server_name=server_name,
            transport=transport,
            config=config,
            connected=True,  # Mark as tentatively connected
        )

        # Simulate actual connection logic based on transport
        # In real implementation, this would:
        # - For stdio: Start subprocess
        # - For SSE: Create HTTP client and test connection
        # - For streamable-http: Create HTTP client and test connection

        if transport == TransportType.SSE:
            if "url" not in config:
                raise ValueError("SSE transport requires 'url' in config")
            # Test connection
            try:
                async with httpx.AsyncClient() as client:
                    await client.get(config["url"], timeout=5.0)
            except httpx.ConnectError:
                connection.connected = False
                raise

        return connection

    async def _discover_tools(self, connection: MCPConnection) -> None:
        """Discover available tools from an MCP server.

        Args:
            connection: MCPConnection to discover from
        """
        # In real implementation, send MCP protocol message:
        # {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}

        # For now, simulate discovery (would be overridden in real impl)
        connection.available_tools = [
            ToolDefinition(
                name="list_files",
                description="List files in a directory",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"],
                },
                server_name=connection.server_name,
            ),
        ]

    async def _discover_resources(self, connection: MCPConnection) -> None:
        """Discover available resources from an MCP server.

        Args:
            connection: MCPConnection to discover from
        """
        # In real implementation, send MCP protocol message:
        # {"jsonrpc": "2.0", "id": 1, "method": "resources/list", "params": {}}

        # For now, simulate discovery
        connection.available_resources = [
            ResourceDefinition(
                uri="file:///workspace",
                name="workspace",
                description="Workspace directory",
                server_name=connection.server_name,
            ),
        ]

    async def _invoke_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Invoke a tool on an MCP server.

        Args:
            server_name: Server name
            tool_name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result
        """
        connection = self.connections[server_name]

        # In real implementation, send MCP protocol message:
        # {"jsonrpc": "2.0", "id": n, "method": "tools/call", "params": {
        #   "name": tool_name, "arguments": arguments
        # }}

        # For now, simulate tool invocation
        return {
            "server": server_name,
            "tool": tool_name,
            "arguments": arguments,
            "result": "Tool execution would happen here",
        }

    async def health_check(self, server_name: str) -> bool:
        """Check if an MCP server connection is healthy.

        Args:
            server_name: Name of server to check

        Returns:
            True if healthy, False otherwise
        """
        if server_name not in self.connections:
            return False

        connection = self.connections[server_name]
        if not connection.connected:
            return False

        try:
            # In real implementation, send a heartbeat/ping
            # For now, just return connection status
            return True
        except Exception as e:
            self._logger.error("health_check_failed", server=server_name, error=str(e))
            connection.connected = False
            return False

    async def reconnect(self, server_name: str) -> bool:
        """Attempt to reconnect to a failed MCP server.

        Args:
            server_name: Name of server to reconnect to

        Returns:
            True if reconnection successful, False otherwise
        """
        if server_name not in self.connections:
            return False

        connection = self.connections[server_name]

        try:
            self._logger.info("reconnecting_to_mcp_server", server=server_name)
            await self.disconnect(server_name)
            await self.connect(server_name, connection.transport, connection.config)
            return True
        except Exception as e:
            self._logger.error("reconnect_failed", server=server_name, error=str(e))
            return False
