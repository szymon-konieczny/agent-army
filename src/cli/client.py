"""HTTP client for communicating with the Code Horde API from the terminal."""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Optional

import httpx
import structlog


logger = structlog.get_logger(__name__)


class APIError(Exception):
    """Raised when the Code Horde API returns an error."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class Code HordeCLI:
    """Client for the Code Horde REST API.

    Provides typed methods for every API endpoint,
    with connection pooling and automatic retries.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    # ── lifecycle ────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Create the HTTP connection pool."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
            headers={"Content-Type": "application/json"},
        )

    async def close(self) -> None:
        """Close the HTTP connection pool."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Call connect() first")
        return self._client

    # ── low-level request ────────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Send a request with automatic retry."""
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await self.client.request(
                    method,
                    path,
                    json=json_body,
                    params=params,
                )
                if resp.status_code >= 400:
                    detail = resp.text
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except Exception:
                        pass
                    raise APIError(resp.status_code, detail)
                return resp.json()

            except httpx.ConnectError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * attempt)
            except httpx.ReadTimeout as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(1.0 * attempt)

        raise ConnectionError(
            f"Cannot reach Code Horde at {self.base_url} "
            f"after {self.max_retries} attempts: {last_exc}"
        )

    # ── health ───────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """GET /health — check system health."""
        return await self._request("GET", "/health")

    async def is_alive(self) -> bool:
        """Quick connectivity check (returns True/False, never raises)."""
        try:
            h = await self.health()
            return h.get("status") == "healthy"
        except Exception:
            return False

    # ── agents ───────────────────────────────────────────────────────

    async def list_agents(self) -> dict[str, Any]:
        """GET /agents — list all registered agents with their status."""
        return await self._request("GET", "/agents")

    # ── tasks ────────────────────────────────────────────────────────

    async def submit_task(
        self,
        description: str,
        *,
        priority: int = 3,
        tags: Optional[list[str]] = None,
        context: Optional[dict[str, Any]] = None,
        timeout_seconds: int = 3600,
    ) -> dict[str, Any]:
        """POST /tasks — submit a new task to the orchestrator.

        Returns:
            {"task_id": "...", "status": "submitted", "assigned_agent": "..."}
        """
        body: dict[str, Any] = {
            "description": description,
            "priority": priority,
            "tags": tags or [],
            "timeout_seconds": timeout_seconds,
        }
        if context:
            body["context"] = context
        return await self._request("POST", "/tasks", json_body=body)

    async def get_task(self, task_id: str) -> dict[str, Any]:
        """GET /tasks/{task_id} — poll task status."""
        return await self._request("GET", f"/tasks/{task_id}")

    async def wait_for_task(
        self,
        task_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        """Poll a task until it reaches a terminal state (completed/failed)."""
        start = time.monotonic()
        while True:
            task = await self.get_task(task_id)
            status = task.get("status", "")
            if status in ("completed", "failed", "cancelled"):
                return task
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Task {task_id} did not finish within {timeout}s (status: {status})"
                )
            await asyncio.sleep(poll_interval)

    # ── chat (send free-form text to orchestrator) ───────────────────

    async def chat(
        self,
        message: str,
        *,
        agent_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send a free-form chat message.

        If the API has a /chat endpoint this uses it directly;
        otherwise it falls back to submitting as a task.
        """
        # Try dedicated chat endpoint first
        try:
            return await self._request(
                "POST",
                "/chat",
                json_body={
                    "message": message,
                    "agent_id": agent_id,
                },
            )
        except APIError as exc:
            if exc.status_code == 404:
                # Fallback: submit as a task
                return await self.submit_task(
                    description=message,
                    priority=3,
                    tags=["cli-chat"],
                    context={"source": "cli", "agent_id": agent_id},
                )
            raise
