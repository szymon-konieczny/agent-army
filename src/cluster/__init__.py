"""Code Horde Cluster — multi-machine coordination via MCP.

Three modes:
  - standalone: Default single-instance (no clustering).
  - center: Command Center on VPS/cloud — accepts worker registrations,
            dispatches tasks to the best worker.
  - worker: Local Agent Army instance connected to a Command Center.
"""

from src.cluster.registry import WorkerInfo, WorkerRegistry

__all__ = ["WorkerInfo", "WorkerRegistry"]
