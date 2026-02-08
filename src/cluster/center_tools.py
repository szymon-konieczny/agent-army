"""Center-mode MCP tools — worker registration, heartbeat, dispatch.

These tools are registered on the existing MCPServer when AgentArmy
runs in ``center`` mode.  Workers call these tools over streamable-HTTP
to register themselves, send heartbeats, and receive dispatched tasks.

Usage::

    from src.cluster.center_tools import register_center_tools
    register_center_tools(mcp_server.app, registry, token="shared-secret")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog

from mcp.server.fastmcp import FastMCP

from src.cluster.registry import WorkerRegistry

logger = structlog.get_logger(__name__)


def register_center_tools(
    app: FastMCP,
    registry: WorkerRegistry,
    token: str = "",
) -> None:
    """Register cluster management tools on the MCP server app.

    Args:
        app: The FastMCP application instance (from MCPServer.app).
        registry: The WorkerRegistry for tracking connected workers.
        token: Shared cluster authentication token.  If set, workers must
               provide this token when registering.
    """

    def _validate_token(provided: str) -> bool:
        """Check that the provided token matches the configured one."""
        if not token:
            return True  # No token configured → allow all
        return provided == token

    # ── Worker → Center tools ─────────────────────────────────────

    @app.tool()
    async def worker_register(
        name: str,
        mcp_url: str,
        platform: str,
        capabilities: list[str],
        agents: list[str],
        auth_token: str = "",
    ) -> dict[str, Any]:
        """Register a new Agent Army worker with the Command Center.

        Workers call this tool once on startup to announce their presence,
        platform capabilities, and available agents.

        Args:
            name: Human-readable worker name (e.g. "Szymon's MacBook")
            mcp_url: This worker's MCP server URL for receiving dispatched tasks
            platform: OS platform ("macos", "windows", "linux")
            capabilities: List of platform capabilities (e.g. ["applescript", "clipboard"])
            agents: List of agent role names (e.g. ["builder", "designer", "automator"])
            auth_token: Cluster authentication token

        Returns:
            Worker ID and registration confirmation
        """
        if not _validate_token(auth_token):
            return {"error": "Invalid cluster token", "registered": False}

        worker = await registry.register(
            name=name,
            mcp_url=mcp_url,
            platform=platform,
            capabilities=capabilities,
            agents=agents,
            token=auth_token,
        )
        return {
            "registered": True,
            "worker_id": worker.worker_id,
            "name": worker.name,
            "registered_at": worker.registered_at.isoformat(),
        }

    @app.tool()
    async def worker_heartbeat(
        worker_id: str,
        load: float = 0.0,
        active_tasks: int = 0,
        auth_token: str = "",
    ) -> dict[str, Any]:
        """Send a heartbeat from a worker to the Command Center.

        Workers call this tool periodically (every 30s by default) to
        confirm they are still alive and report their current load.

        Args:
            worker_id: The worker ID received from worker_register
            load: Current CPU/task utilisation (0.0=idle, 1.0=fully busy)
            active_tasks: Number of tasks currently being processed
            auth_token: Cluster authentication token

        Returns:
            Confirmation with current server time
        """
        if not _validate_token(auth_token):
            return {"error": "Invalid cluster token", "alive": False}

        ok = await registry.heartbeat(worker_id, load=load, active_tasks=active_tasks)
        if not ok:
            return {"alive": False, "error": f"Worker {worker_id} not found — re-register"}

        return {
            "alive": True,
            "server_time": datetime.now(timezone.utc).isoformat(),
        }

    @app.tool()
    async def worker_unregister(
        worker_id: str,
        auth_token: str = "",
    ) -> dict[str, Any]:
        """Gracefully unregister a worker from the Command Center.

        Workers call this on shutdown to remove themselves from the registry.

        Args:
            worker_id: The worker ID to unregister
            auth_token: Cluster authentication token

        Returns:
            Confirmation of removal
        """
        if not _validate_token(auth_token):
            return {"error": "Invalid cluster token"}

        removed = await registry.unregister(worker_id)
        return {"unregistered": removed, "worker_id": worker_id}

    # ── Dashboard / admin tools ───────────────────────────────────

    @app.tool()
    async def worker_list(include_offline: bool = False) -> dict[str, Any]:
        """List all connected workers and their status.

        Used by the Command Center dashboard to display the cluster overview.

        Args:
            include_offline: Whether to include workers marked as offline

        Returns:
            List of workers with their details and cluster health metrics
        """
        workers = registry.list_workers(include_offline=include_offline)
        return {
            "workers": [w.to_dict() for w in workers],
            "cluster_health": registry.cluster_health(),
        }

    @app.tool()
    async def worker_capabilities(worker_id: str) -> dict[str, Any]:
        """Query a specific worker's capabilities and status.

        Args:
            worker_id: The worker ID to query

        Returns:
            Worker details including platform, capabilities, agents, and load
        """
        worker = registry.get_worker(worker_id)
        if not worker:
            return {"error": f"Worker {worker_id} not found"}
        return worker.to_dict()

    @app.tool()
    async def dispatch_task(
        worker_id: str,
        task_payload: str,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Dispatch a task to a specific worker via its MCP server.

        The Command Center calls this to push a task to a remote worker.
        It connects to the worker's MCP server and calls ``accept_task``.

        Args:
            worker_id: Target worker ID
            task_payload: JSON-serialised task payload
            timeout: Maximum seconds to wait for result

        Returns:
            Task result from the worker, or error details
        """
        worker = registry.get_worker(worker_id)
        if not worker:
            return {"error": f"Worker {worker_id} not found"}

        if worker.status == "offline":
            return {"error": f"Worker {worker.name} is offline"}

        await logger.ainfo(
            "dispatching_task",
            worker_id=worker_id,
            worker_name=worker.name,
            mcp_url=worker.mcp_url,
        )

        try:
            # Call the worker's MCP server accept_task tool via HTTP
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{worker.mcp_url}/tools/call",
                    json={
                        "name": "accept_task",
                        "arguments": {"task_payload": task_payload},
                    },
                )
                response.raise_for_status()
                result = response.json()
                return {
                    "dispatched": True,
                    "worker_id": worker_id,
                    "worker_name": worker.name,
                    "result": result,
                }
        except httpx.TimeoutException:
            await logger.awarning("dispatch_timeout", worker_id=worker_id)
            return {"error": "Task dispatch timed out", "worker_id": worker_id}
        except httpx.HTTPError as e:
            await logger.aerror("dispatch_failed", worker_id=worker_id, error=str(e))
            return {"error": f"Dispatch failed: {e}", "worker_id": worker_id}

    # ── MCP Resources ─────────────────────────────────────────────

    @app.resource("agentarmy://workers")
    def get_workers_resource() -> str:
        """List all connected workers as a JSON resource.

        Returns:
            JSON-formatted list of connected workers with status
        """
        workers = registry.list_workers(include_offline=True)
        return json.dumps([w.to_dict() for w in workers], indent=2)

    @app.resource("agentarmy://cluster-health")
    def get_cluster_health_resource() -> str:
        """Cluster-wide health metrics as a JSON resource.

        Returns:
            JSON with worker count, total agents, aggregate load
        """
        return json.dumps(registry.cluster_health(), indent=2)
