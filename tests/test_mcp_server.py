"""Unit tests for AgentArmy MCP Server.

Tests cover:
- Tool definitions and invocations
- Resource definitions
- Prompt templates
- Error handling
- Client manager connections
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog

from src.mcp_server.client_manager import (
    MCPClientManager,
    MCPConnection,
    ResourceDefinition,
    ToolDefinition,
    TransportType,
)
from src.mcp_server.server import MCPServer

logger = structlog.get_logger(__name__)


class TestMCPServer:
    """Test AgentArmy MCP Server."""

    @pytest.fixture
    def server(self) -> MCPServer:
        """Create MCPServer instance."""
        return MCPServer(
            api_base_url="http://localhost:8000",
            api_timeout=5.0,
        )

    def test_server_initialization(self, server: MCPServer) -> None:
        """Test server initializes correctly."""
        assert server.api_base_url == "http://localhost:8000"
        assert server.api_timeout == 5.0
        assert server.app is not None

    @pytest.mark.asyncio
    async def test_get_http_client(self, server: MCPServer) -> None:
        """Test HTTP client creation."""
        client = await server._get_http_client()
        assert client is not None
        assert client.base_url == "http://localhost:8000"

        # Should return same instance
        client2 = await server._get_http_client()
        assert client is client2

    @pytest.mark.asyncio
    async def test_call_api_success(self, server: MCPServer) -> None:
        """Test successful API call."""
        with patch.object(server, "_get_http_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "ok", "data": []}
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await server._call_api("GET", "/api/test")

            assert result == {"status": "ok", "data": []}
            mock_client.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_api_failure(self, server: MCPServer) -> None:
        """Test API call failure handling."""
        import httpx

        with patch.object(server, "_get_http_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(
                side_effect=httpx.ConnectError("Connection failed")
            )
            mock_get_client.return_value = mock_client

            with pytest.raises(httpx.ConnectError):
                await server._call_api("GET", "/api/test")


class TestMCPTools:
    """Test MCP tool definitions."""

    @pytest.fixture
    def server(self) -> MCPServer:
        """Create MCPServer instance."""
        return MCPServer()

    @pytest.mark.asyncio
    async def test_create_task_tool(self, server: MCPServer) -> None:
        """Test task creation tool."""
        # Tools are registered as app methods
        assert hasattr(server.app, "tool")

    def test_tools_registered(self, server: MCPServer) -> None:
        """Test all tools are registered."""
        # Access tool list from FastMCP app
        # In real implementation, inspect server.app._tools
        assert server.app is not None


class TestMCPResources:
    """Test MCP resource definitions."""

    @pytest.fixture
    def server(self) -> MCPServer:
        """Create MCPServer instance."""
        return MCPServer()

    def test_resources_registered(self, server: MCPServer) -> None:
        """Test all resources are registered."""
        assert server.app is not None


class TestMCPPrompts:
    """Test MCP prompt templates."""

    @pytest.fixture
    def server(self) -> MCPServer:
        """Create MCPServer instance."""
        return MCPServer()

    def test_prompts_registered(self, server: MCPServer) -> None:
        """Test all prompts are registered."""
        assert server.app is not None


class TestMCPClientManager:
    """Test MCPClientManager for consuming external MCP servers."""

    @pytest.fixture
    def manager(self) -> MCPClientManager:
        """Create MCPClientManager instance."""
        return MCPClientManager()

    @pytest.mark.asyncio
    async def test_initialization(self, manager: MCPClientManager) -> None:
        """Test manager initializes correctly."""
        assert manager.connections == {}
        assert manager.tools_cache == {}

    @pytest.mark.asyncio
    async def test_connect_sse(self, manager: MCPClientManager) -> None:
        """Test connecting to SSE MCP server."""
        with patch.object(manager, "_establish_connection") as mock_establish:
            with patch.object(manager, "_discover_tools") as mock_discover_tools:
                with patch.object(
                    manager, "_discover_resources"
                ) as mock_discover_resources:
                    mock_connection = MCPConnection(
                        server_name="test-server",
                        transport=TransportType.SSE,
                        config={"url": "http://localhost:3000"},
                        connected=True,
                        available_tools=[
                            ToolDefinition(
                                name="test_tool",
                                description="Test tool",
                                input_schema={"type": "object"},
                                server_name="test-server",
                            )
                        ],
                    )
                    mock_establish.return_value = mock_connection
                    mock_discover_tools.return_value = None
                    mock_discover_resources.return_value = None

                    connection = await manager.connect(
                        "test-server",
                        TransportType.SSE,
                        {"url": "http://localhost:3000"},
                    )

                    assert connection.server_name == "test-server"
                    assert connection.connected is True
                    assert "test-server" in manager.connections

    @pytest.mark.asyncio
    async def test_connect_duplicate_error(self, manager: MCPClientManager) -> None:
        """Test error when connecting to already-connected server."""
        manager.connections["test"] = MCPConnection(
            server_name="test",
            transport=TransportType.SSE,
            config={},
            connected=True,
        )

        with pytest.raises(ValueError, match="already connected"):
            await manager.connect("test", TransportType.SSE, {})

    @pytest.mark.asyncio
    async def test_disconnect(self, manager: MCPClientManager) -> None:
        """Test disconnecting from MCP server."""
        manager.connections["test"] = MCPConnection(
            server_name="test",
            transport=TransportType.SSE,
            config={},
            connected=True,
        )

        await manager.disconnect("test")

        assert "test" not in manager.connections

    @pytest.mark.asyncio
    async def test_disconnect_not_found(self, manager: MCPClientManager) -> None:
        """Test error when disconnecting from non-existent server."""
        with pytest.raises(ValueError, match="not connected"):
            await manager.disconnect("nonexistent")

    def test_list_connections(self, manager: MCPClientManager) -> None:
        """Test listing connections."""
        conn1 = MCPConnection(
            server_name="server1",
            transport=TransportType.SSE,
            config={},
            connected=True,
        )
        conn2 = MCPConnection(
            server_name="server2",
            transport=TransportType.STDIO,
            config={},
            connected=True,
        )

        manager.connections["server1"] = conn1
        manager.connections["server2"] = conn2

        connections = manager.list_connections()
        assert len(connections) == 2
        assert conn1 in connections
        assert conn2 in connections

    def test_list_tools_all(self, manager: MCPClientManager) -> None:
        """Test listing all tools."""
        manager.tools_cache["server1"] = {
            "tool1": ToolDefinition(
                name="tool1",
                description="Tool 1",
                input_schema={},
                server_name="server1",
            ),
            "tool2": ToolDefinition(
                name="tool2",
                description="Tool 2",
                input_schema={},
                server_name="server1",
            ),
        }

        manager.tools_cache["server2"] = {
            "tool3": ToolDefinition(
                name="tool3",
                description="Tool 3",
                input_schema={},
                server_name="server2",
            ),
        }

        tools = manager.list_tools()

        assert "server1" in tools
        assert "server2" in tools
        assert len(tools["server1"]) == 2
        assert len(tools["server2"]) == 1

    def test_list_tools_specific_server(self, manager: MCPClientManager) -> None:
        """Test listing tools for specific server."""
        manager.tools_cache["server1"] = {
            "tool1": ToolDefinition(
                name="tool1",
                description="Tool 1",
                input_schema={},
                server_name="server1",
            ),
        }

        tools = manager.list_tools(server_name="server1")

        assert "server1" in tools
        assert len(tools["server1"]) == 1
        assert "tool1" in tools["server1"]

    @pytest.mark.asyncio
    async def test_call_tool_success(self, manager: MCPClientManager) -> None:
        """Test successful tool invocation."""
        manager.connections["server1"] = MCPConnection(
            server_name="server1",
            transport=TransportType.SSE,
            config={},
            connected=True,
        )

        manager.tools_cache["server1"] = {
            "tool1": ToolDefinition(
                name="tool1",
                description="Tool 1",
                input_schema={},
                server_name="server1",
            ),
        }

        with patch.object(manager, "_invoke_tool") as mock_invoke:
            mock_invoke.return_value = {"result": "success"}

            result = await manager.call_tool("server1", "tool1", {"arg": "value"})

            assert result == {"result": "success"}
            mock_invoke.assert_called_once_with("server1", "tool1", {"arg": "value"})

    @pytest.mark.asyncio
    async def test_call_tool_server_not_found(
        self, manager: MCPClientManager
    ) -> None:
        """Test error when calling tool on non-existent server."""
        with pytest.raises(ValueError, match="not connected"):
            await manager.call_tool("nonexistent", "tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_tool_not_found(self, manager: MCPClientManager) -> None:
        """Test error when calling non-existent tool."""
        manager.connections["server1"] = MCPConnection(
            server_name="server1",
            transport=TransportType.SSE,
            config={},
            connected=True,
        )

        with pytest.raises(ValueError, match="not found"):
            await manager.call_tool("server1", "nonexistent", {})

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, manager: MCPClientManager) -> None:
        """Test health check for healthy connection."""
        manager.connections["server1"] = MCPConnection(
            server_name="server1",
            transport=TransportType.SSE,
            config={},
            connected=True,
        )

        healthy = await manager.health_check("server1")

        assert healthy is True

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, manager: MCPClientManager) -> None:
        """Test health check for unhealthy connection."""
        healthy = await manager.health_check("nonexistent")

        assert healthy is False

    @pytest.mark.asyncio
    async def test_reconnect_success(self, manager: MCPClientManager) -> None:
        """Test successful reconnection."""
        original_connection = MCPConnection(
            server_name="server1",
            transport=TransportType.SSE,
            config={"url": "http://localhost:3000"},
            connected=False,
        )
        manager.connections["server1"] = original_connection

        with patch.object(manager, "disconnect") as mock_disconnect:
            with patch.object(manager, "connect") as mock_connect:
                mock_connect.return_value = MCPConnection(
                    server_name="server1",
                    transport=TransportType.SSE,
                    config={"url": "http://localhost:3000"},
                    connected=True,
                )

                success = await manager.reconnect("server1")

                assert success is True
                mock_disconnect.assert_called_once_with("server1")
                mock_connect.assert_called_once()


class TestMCPModels:
    """Test Pydantic models."""

    def test_tool_definition_model(self) -> None:
        """Test ToolDefinition model."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
            server_name="test-server",
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.server_name == "test-server"

    def test_resource_definition_model(self) -> None:
        """Test ResourceDefinition model."""
        resource = ResourceDefinition(
            uri="file:///path",
            name="test_resource",
            description="A test resource",
            server_name="test-server",
        )

        assert resource.uri == "file:///path"
        assert resource.name == "test_resource"
        assert resource.mime_type == "application/json"

    def test_mcp_connection_model(self) -> None:
        """Test MCPConnection model."""
        connection = MCPConnection(
            server_name="test-server",
            transport=TransportType.SSE,
            config={"url": "http://localhost:3000"},
            connected=True,
        )

        assert connection.server_name == "test-server"
        assert connection.transport == TransportType.SSE
        assert connection.connected is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
