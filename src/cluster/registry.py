"""Worker registry — Center-side tracking of connected Agent Army workers.

The Command Center maintains a `WorkerRegistry` that tracks every connected
worker's identity, capabilities, platform, load, and heartbeat status.
When a task arrives that needs remote execution, the registry picks the
best worker using capability matching + lowest-load selection.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class WorkerInfo:
    """Describes a single connected Agent Army worker.

    Attributes:
        worker_id: UUID assigned by the Center on registration.
        name: Human-readable name (e.g. "Szymon's MacBook").
        mcp_url: The worker's MCP server URL for task dispatch callbacks.
        platform: Normalised OS name ("macos", "windows", "linux").
        capabilities: Platform capabilities (from DesktopAdapter + CalendarAdapter).
        agents: List of agent role names running on this worker.
        registered_at: UTC timestamp of initial registration.
        last_heartbeat: UTC timestamp of most recent heartbeat.
        load: Current utilisation 0.0 (idle) → 1.0 (fully busy).
        status: "online", "busy", or "offline".
        active_tasks: Number of tasks currently being processed.
    """

    worker_id: str
    name: str
    mcp_url: str
    platform: str
    capabilities: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    load: float = 0.0
    status: str = "online"
    active_tasks: int = 0

    def to_dict(self) -> dict:
        """Serialise to JSON-safe dictionary."""
        return {
            "worker_id": self.worker_id,
            "name": self.name,
            "mcp_url": self.mcp_url,
            "platform": self.platform,
            "capabilities": self.capabilities,
            "agents": self.agents,
            "registered_at": self.registered_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "load": self.load,
            "status": self.status,
            "active_tasks": self.active_tasks,
        }


class WorkerRegistry:
    """Thread-safe registry of connected workers.

    Used by the Command Center to track and select workers for task dispatch.
    """

    def __init__(self) -> None:
        self._workers: dict[str, WorkerInfo] = {}
        self._lock = asyncio.Lock()

    # ── Registration ──────────────────────────────────────────────

    async def register(
        self,
        name: str,
        mcp_url: str,
        platform: str,
        capabilities: list[str],
        agents: list[str],
        token: str = "",
    ) -> WorkerInfo:
        """Register a new worker. Returns the WorkerInfo with assigned worker_id."""
        async with self._lock:
            worker_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)
            worker = WorkerInfo(
                worker_id=worker_id,
                name=name or f"worker-{worker_id[:8]}",
                mcp_url=mcp_url,
                platform=platform,
                capabilities=capabilities,
                agents=agents,
                registered_at=now,
                last_heartbeat=now,
            )
            self._workers[worker_id] = worker
            await logger.ainfo(
                "worker_registered",
                worker_id=worker_id,
                name=worker.name,
                platform=platform,
                agents_count=len(agents),
            )
            return worker

    async def unregister(self, worker_id: str) -> bool:
        """Remove a worker from the registry. Returns True if found and removed."""
        async with self._lock:
            if worker_id in self._workers:
                worker = self._workers.pop(worker_id)
                await logger.ainfo("worker_unregistered", worker_id=worker_id, name=worker.name)
                return True
            return False

    # ── Heartbeat ─────────────────────────────────────────────────

    async def heartbeat(self, worker_id: str, load: float = 0.0,
                        active_tasks: int = 0) -> bool:
        """Update a worker's heartbeat timestamp and load. Returns False if not found."""
        async with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return False
            worker.last_heartbeat = datetime.now(timezone.utc)
            worker.load = max(0.0, min(1.0, load))
            worker.active_tasks = active_tasks
            worker.status = "busy" if load > 0.8 else "online"
            return True

    async def cleanup_stale(self, timeout_seconds: int = 90) -> list[str]:
        """Mark workers as offline if their heartbeat is older than timeout.

        Returns list of worker_ids that were marked offline.
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            stale = []
            for wid, worker in self._workers.items():
                elapsed = (now - worker.last_heartbeat).total_seconds()
                if elapsed > timeout_seconds and worker.status != "offline":
                    worker.status = "offline"
                    stale.append(wid)
                    await logger.awarning(
                        "worker_stale",
                        worker_id=wid,
                        name=worker.name,
                        last_heartbeat_ago=elapsed,
                    )
            return stale

    # ── Queries ───────────────────────────────────────────────────

    def list_workers(self, *, include_offline: bool = False) -> list[WorkerInfo]:
        """Return all workers, optionally filtering out offline ones."""
        if include_offline:
            return list(self._workers.values())
        return [w for w in self._workers.values() if w.status != "offline"]

    def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get a specific worker by ID."""
        return self._workers.get(worker_id)

    def get_best_worker(
        self,
        required_capabilities: Optional[list[str]] = None,
        preferred_platform: Optional[str] = None,
    ) -> Optional[WorkerInfo]:
        """Select the best online worker for a task.

        Selection criteria (in order):
        1. Must be online (not offline or stale)
        2. Must have all required_capabilities (if specified)
        3. Prefer matching platform (if specified)
        4. Lowest load wins

        Returns None if no suitable worker is found.
        """
        candidates = [w for w in self._workers.values() if w.status in ("online", "busy")]

        if required_capabilities:
            req_set = set(required_capabilities)
            candidates = [w for w in candidates if req_set.issubset(set(w.capabilities))]

        if not candidates:
            return None

        # Prefer matching platform
        if preferred_platform:
            platform_match = [w for w in candidates if w.platform == preferred_platform]
            if platform_match:
                candidates = platform_match

        # Pick lowest load
        return min(candidates, key=lambda w: w.load)

    @property
    def worker_count(self) -> int:
        """Total number of registered workers (including offline)."""
        return len(self._workers)

    @property
    def online_count(self) -> int:
        """Number of online workers."""
        return sum(1 for w in self._workers.values() if w.status != "offline")

    def cluster_health(self) -> dict:
        """Return cluster-wide health metrics."""
        workers = list(self._workers.values())
        online = [w for w in workers if w.status != "offline"]
        return {
            "total_workers": len(workers),
            "online_workers": len(online),
            "total_agents": sum(len(w.agents) for w in online),
            "aggregate_load": sum(w.load for w in online) / max(len(online), 1),
            "platforms": list({w.platform for w in online}),
            "total_active_tasks": sum(w.active_tasks for w in online),
        }
