"""Worker client — manages the worker-side lifecycle in cluster mode.

When Code Horde runs in ``worker`` mode, this module:
  1. Connects to the Command Center's MCP server (streamable-HTTP).
  2. Registers this worker (sends platform, capabilities, agents).
  3. Runs a heartbeat loop to keep the Center informed.
  4. Starts a local MCP server so the Center can push tasks to us.
  5. Handles graceful shutdown and reconnection.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import socket
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog
from mcp.server.fastmcp import FastMCP

from src.platform import detect_platform, get_desktop_adapter, get_calendar_adapter

logger = structlog.get_logger(__name__)


class WorkerClient:
    """Manages the worker side of the Center ↔ Worker MCP connection.

    Args:
        center_url: The Command Center's MCP server URL.
        center_token: Shared authentication token.
        worker_name: Human-readable name for this worker.
        worker_port: Port for this worker's local MCP server.
        heartbeat_interval: Seconds between heartbeat pings.
        agents: List of agent instances running locally.
        task_manager: The local TaskManager for routing received tasks.
    """

    def __init__(
        self,
        center_url: str,
        center_token: str = "",
        worker_name: str = "",
        worker_port: int = 8002,
        heartbeat_interval: int = 30,
        agents: Optional[list] = None,
        task_manager: Optional[Any] = None,
    ) -> None:
        self.center_url = center_url.rstrip("/")
        self.center_token = center_token
        self.worker_name = worker_name or f"{platform.node()}"
        self.worker_port = worker_port
        self.heartbeat_interval = heartbeat_interval
        self.agents = agents or []
        self.task_manager = task_manager

        self.worker_id: Optional[str] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._mcp_task: Optional[asyncio.Task] = None
        self._running = False
        self._reconnect_delay = 5.0  # seconds, doubles on failure

        # Build capability + agent lists from current platform
        self._platform = detect_platform()
        self._capabilities = self._collect_capabilities()
        self._agent_names = [a.identity.role for a in self.agents] if self.agents else []

        # Local MCP server for receiving dispatched tasks
        self._local_mcp = FastMCP("codehorde-worker")
        self._register_worker_tools()

    def _collect_capabilities(self) -> list[str]:
        """Collect available platform capabilities."""
        caps = []
        try:
            for cap in get_desktop_adapter().capabilities():
                if cap.available:
                    caps.append(cap.name)
        except Exception:
            pass
        try:
            for cap in get_calendar_adapter().capabilities():
                if cap.available:
                    caps.append(cap.name)
        except Exception:
            pass
        return caps

    def _get_local_mcp_url(self) -> str:
        """Build the URL the Center should use to reach this worker's MCP server."""
        # Use the machine's hostname or first non-loopback IP
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
        except Exception:
            ip = "127.0.0.1"
        return f"http://{ip}:{self.worker_port}"

    # ── Lifecycle ─────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect to Center, register, start heartbeat + local MCP server."""
        self._running = True
        await logger.ainfo(
            "worker_starting",
            center_url=self.center_url,
            name=self.worker_name,
            platform=self._platform,
            capabilities=self._capabilities,
            agents=self._agent_names,
        )

        # Register with Center
        await self._register()

        # Start background tasks
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._mcp_task = asyncio.create_task(self._run_local_mcp())

    async def stop(self) -> None:
        """Gracefully disconnect from the Center."""
        self._running = False

        # Cancel background tasks
        for task in [self._heartbeat_task, self._mcp_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Unregister from Center
        if self.worker_id:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(
                        f"{self.center_url}/tools/call",
                        json={
                            "name": "worker_unregister",
                            "arguments": {
                                "worker_id": self.worker_id,
                                "auth_token": self.center_token,
                            },
                        },
                    )
            except Exception as e:
                await logger.awarning("unregister_failed", error=str(e))

        await logger.ainfo("worker_stopped", worker_id=self.worker_id)

    # ── Registration ──────────────────────────────────────────────

    async def _register(self) -> None:
        """Register with the Command Center via its MCP tools."""
        mcp_url = self._get_local_mcp_url()
        payload = {
            "name": "worker_register",
            "arguments": {
                "name": self.worker_name,
                "mcp_url": mcp_url,
                "platform": self._platform,
                "capabilities": self._capabilities,
                "agents": self._agent_names,
                "auth_token": self.center_token,
            },
        }

        while self._running:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    response = await client.post(
                        f"{self.center_url}/tools/call",
                        json=payload,
                    )
                    response.raise_for_status()
                    result = response.json()

                if result.get("registered"):
                    self.worker_id = result["worker_id"]
                    self._reconnect_delay = 5.0  # reset backoff
                    await logger.ainfo(
                        "worker_registered",
                        worker_id=self.worker_id,
                        center_url=self.center_url,
                    )
                    return
                else:
                    error = result.get("error", "Unknown registration error")
                    await logger.aerror("registration_rejected", error=error)
                    # Don't retry if token is invalid
                    if "token" in error.lower():
                        self._running = False
                        return

            except Exception as e:
                await logger.awarning(
                    "registration_failed",
                    error=str(e),
                    retry_in=self._reconnect_delay,
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 120)

    # ── Heartbeat ─────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """Periodically send heartbeat to the Center."""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                if not self.worker_id:
                    continue

                load = self._calculate_load()
                active = self._count_active_tasks()

                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.post(
                        f"{self.center_url}/tools/call",
                        json={
                            "name": "worker_heartbeat",
                            "arguments": {
                                "worker_id": self.worker_id,
                                "load": load,
                                "active_tasks": active,
                                "auth_token": self.center_token,
                            },
                        },
                    )
                    result = response.json()

                if not result.get("alive"):
                    # Center doesn't recognise us — re-register
                    await logger.awarning("heartbeat_rejected", result=result)
                    await self._register()

            except asyncio.CancelledError:
                return
            except Exception as e:
                await logger.awarning("heartbeat_error", error=str(e))

    def _calculate_load(self) -> float:
        """Estimate current worker load (0.0–1.0)."""
        if not self.task_manager:
            return 0.0
        try:
            stats = self.task_manager.get_stats()
            active = stats.get("in_progress", 0) + stats.get("assigned", 0)
            # Assume max 5 concurrent tasks as "fully loaded"
            return min(active / 5.0, 1.0)
        except Exception:
            return 0.0

    def _count_active_tasks(self) -> int:
        """Count currently active tasks."""
        if not self.task_manager:
            return 0
        try:
            stats = self.task_manager.get_stats()
            return stats.get("in_progress", 0) + stats.get("assigned", 0)
        except Exception:
            return 0

    # ── Local MCP server (receives tasks from Center) ─────────────

    def _register_worker_tools(self) -> None:
        """Register MCP tools on this worker's local server."""

        agents_by_role = {}

        @self._local_mcp.tool()
        async def accept_task(task_payload: str) -> dict[str, Any]:
            """Accept a task dispatched by the Command Center.

            The Center calls this tool to push a task to this worker.
            The worker routes it to the appropriate local agent and
            returns the result.

            Args:
                task_payload: JSON-encoded task payload with agent_id and message

            Returns:
                Task result from the local agent
            """
            try:
                payload = json.loads(task_payload)
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON payload: {e}"}

            target_role = payload.get("agent_role", "")
            message = payload.get("message", "")

            # Find the matching local agent
            agent = None
            for a in self.agents:
                if a.identity.role == target_role:
                    agent = a
                    break

            if not agent:
                # Try to find any suitable agent
                for a in self.agents:
                    agent = a
                    break

            if not agent:
                return {"error": "No agents available on this worker"}

            await logger.ainfo(
                "accepting_task",
                agent=agent.identity.role,
                message_preview=message[:100],
            )

            try:
                task = {
                    "id": payload.get("task_id", "remote-task"),
                    "type": "chat",
                    "context": {"message": message, **payload.get("context", {})},
                }
                result = await asyncio.wait_for(
                    agent.process_task(task),
                    timeout=payload.get("timeout", 120),
                )
                return {
                    "status": "completed",
                    "agent_id": agent.identity.id,
                    "agent_role": agent.identity.role,
                    "result": result,
                }
            except asyncio.TimeoutError:
                return {"error": "Task timed out", "agent_role": target_role}
            except Exception as e:
                return {"error": str(e), "agent_role": target_role}

        @self._local_mcp.tool()
        async def worker_status() -> dict[str, Any]:
            """Report this worker's current health and load.

            Returns:
                Worker status including load, active tasks, platform info
            """
            return {
                "worker_id": self.worker_id,
                "name": self.worker_name,
                "platform": self._platform,
                "load": self._calculate_load(),
                "active_tasks": self._count_active_tasks(),
                "agents": self._agent_names,
                "capabilities": self._capabilities,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        @self._local_mcp.tool()
        async def cancel_task(task_id: str) -> dict[str, Any]:
            """Cancel a running task on this worker.

            Args:
                task_id: The task ID to cancel

            Returns:
                Cancellation confirmation
            """
            if self.task_manager:
                try:
                    self.task_manager.cancel_task(task_id)
                    return {"cancelled": True, "task_id": task_id}
                except Exception as e:
                    return {"cancelled": False, "error": str(e)}
            return {"cancelled": False, "error": "No task manager available"}

    async def _run_local_mcp(self) -> None:
        """Start the worker's local MCP server for receiving dispatched tasks."""
        try:
            await logger.ainfo(
                "worker_mcp_starting",
                port=self.worker_port,
            )
            # Use SSE transport for the worker's local server
            # The Center connects to this as an HTTP client
            import uvicorn
            from starlette.applications import Starlette
            from starlette.routing import Mount

            # For now, run a minimal HTTP server that handles tool calls
            from starlette.requests import Request
            from starlette.responses import JSONResponse
            from starlette.routing import Route

            async def handle_tool_call(request: Request) -> JSONResponse:
                """Handle incoming MCP tool call from Center."""
                body = await request.json()
                tool_name = body.get("name", "")
                arguments = body.get("arguments", {})

                # Find and call the tool
                tools = {
                    "accept_task": self._local_mcp._tool_manager._tools.get("accept_task"),
                    "worker_status": self._local_mcp._tool_manager._tools.get("worker_status"),
                    "cancel_task": self._local_mcp._tool_manager._tools.get("cancel_task"),
                }

                tool = tools.get(tool_name)
                if not tool:
                    return JSONResponse({"error": f"Unknown tool: {tool_name}"}, status_code=404)

                try:
                    result = await tool.fn(**arguments)
                    return JSONResponse(result)
                except Exception as e:
                    return JSONResponse({"error": str(e)}, status_code=500)

            async def health(request: Request) -> JSONResponse:
                return JSONResponse({"status": "ok", "worker_id": self.worker_id})

            app = Starlette(
                routes=[
                    Route("/tools/call", handle_tool_call, methods=["POST"]),
                    Route("/health", health, methods=["GET"]),
                ],
            )

            config = uvicorn.Config(
                app=app,
                host="0.0.0.0",
                port=self.worker_port,
                log_level="warning",
                access_log=False,
            )
            server = uvicorn.Server(config)
            await server.serve()

        except asyncio.CancelledError:
            return
        except Exception as e:
            await logger.aerror("worker_mcp_error", error=str(e))
