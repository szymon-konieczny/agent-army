"""Task lifecycle management for Code Horde."""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class TaskStatus(str, Enum):
    """Task lifecycle status.

    Attributes:
        PENDING: Task created, not yet queued.
        QUEUED: Task in queue, awaiting assignment.
        ASSIGNED: Task assigned to agent.
        IN_PROGRESS: Agent actively processing.
        AWAITING_APPROVAL: Requires human approval before proceeding.
        COMPLETED: Task completed successfully.
        FAILED: Task failed.
        CANCELLED: Task cancelled by user/system.
    """

    PENDING = "pending"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskResult(BaseModel):
    """Result of a completed or failed task.

    Attributes:
        task_id: ID of the task.
        status: Final task status.
        output: Task output/result data.
        error: Error message (if failed).
        execution_time_seconds: How long the task took to execute.
        agent_id: ID of agent that executed the task.
        completed_at: Completion timestamp.
    """

    task_id: str = Field(description="Task ID")
    status: TaskStatus = Field(description="Final task status")
    output: Optional[dict[str, Any]] = Field(default=None, description="Task output")
    error: Optional[str] = Field(default=None, description="Error message")
    execution_time_seconds: float = Field(ge=0, description="Execution time")
    agent_id: str = Field(description="Agent that executed task")
    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Completion time",
    )


class TaskManager:
    """Manages task lifecycle: creation, assignment, execution, completion.

    Handles:
    - Task creation and registration
    - Task assignment to agents
    - Task status updates
    - Retry logic with exponential backoff
    - Timeout handling
    - Task history tracking

    Attributes:
        tasks: Registry of all tasks.
        task_history: Historical record of task executions.
        priority_queue: Priority queue for task assignment.
    """

    def __init__(self) -> None:
        """Initialize the task manager."""
        self.tasks: dict[str, dict[str, Any]] = {}
        self.task_history: dict[str, list[TaskResult]] = {}
        self.priority_queue: asyncio.PriorityQueue[tuple[int, str]] = asyncio.PriorityQueue()
        self._logger = structlog.get_logger(__name__)

    async def create_task(
        self,
        description: str,
        priority: int = 3,
        payload: Optional[dict[str, Any]] = None,
        timeout_seconds: int = 3600,
        max_retries: int = 3,
        tags: Optional[list[str]] = None,
    ) -> str:
        """Create a new task.

        Args:
            description: Task description.
            priority: Priority level (1=critical, 5=deferred).
            payload: Task data/parameters.
            timeout_seconds: Task timeout in seconds.
            max_retries: Maximum retry attempts (0-10).
            tags: Optional tags for categorization.

        Returns:
            Task ID of created task.
        """
        task_id = str(uuid.uuid4())

        self.tasks[task_id] = {
            "id": task_id,
            "description": description,
            "priority": priority,
            "payload": payload or {},
            "status": TaskStatus.PENDING,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "assigned_agent": None,
            "started_at": None,
            "completed_at": None,
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries,
            "retry_count": 0,
            "tags": tags or [],
            "error": None,
            "result": None,
        }

        await self.priority_queue.put((priority, task_id))

        await self._logger.ainfo(
            "task_created",
            task_id=task_id,
            description=description,
            priority=priority,
        )

        return task_id

    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        error: Optional[str] = None,
        result: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update task status.

        Args:
            task_id: ID of task to update.
            status: New status.
            error: Error message (if failed).
            result: Task result (if completed).

        Raises:
            KeyError: If task not found.
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        old_status = task["status"]
        task["status"] = status
        task["updated_at"] = datetime.now(timezone.utc)

        if error:
            task["error"] = error

        if result:
            task["result"] = result

        if status == TaskStatus.IN_PROGRESS and not task["started_at"]:
            task["started_at"] = datetime.now(timezone.utc)

        if status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            task["completed_at"] = datetime.now(timezone.utc)

        await self._logger.ainfo(
            "task_status_updated",
            task_id=task_id,
            old_status=old_status,
            new_status=status.value,
        )

    async def assign_task(self, task_id: str, agent_id: str) -> None:
        """Assign a task to an agent.

        Args:
            task_id: ID of task to assign.
            agent_id: ID of agent to assign to.

        Raises:
            KeyError: If task not found.
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        task["assigned_agent"] = agent_id
        task["status"] = TaskStatus.ASSIGNED
        task["updated_at"] = datetime.now(timezone.utc)

        await self._logger.ainfo(
            "task_assigned",
            task_id=task_id,
            agent_id=agent_id,
        )

    async def complete_task(
        self,
        task_id: str,
        result: dict[str, Any],
    ) -> TaskResult:
        """Mark a task as completed successfully.

        Args:
            task_id: ID of task to complete.
            result: Task result data.

        Returns:
            TaskResult with execution summary.

        Raises:
            KeyError: If task not found.
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        await self.update_task_status(task_id, TaskStatus.COMPLETED, result=result)

        # Calculate execution time
        started_at = task.get("started_at", task["created_at"])
        completed_at = task["completed_at"]
        execution_time = (completed_at - started_at).total_seconds()

        # Create result record
        task_result = TaskResult(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            output=result,
            execution_time_seconds=execution_time,
            agent_id=task["assigned_agent"] or "unknown",
            completed_at=completed_at,
        )

        # Add to history
        if task_id not in self.task_history:
            self.task_history[task_id] = []
        self.task_history[task_id].append(task_result)

        await self._logger.ainfo(
            "task_completed",
            task_id=task_id,
            execution_time_seconds=execution_time,
            agent_id=task["assigned_agent"],
        )

        return task_result

    async def fail_task(
        self,
        task_id: str,
        error: str,
    ) -> TaskResult:
        """Mark a task as failed.

        Args:
            task_id: ID of task that failed.
            error: Error message/description.

        Returns:
            TaskResult with failure information.

        Raises:
            KeyError: If task not found.
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        await self.update_task_status(task_id, TaskStatus.FAILED, error=error)

        # Calculate execution time
        started_at = task.get("started_at", task["created_at"])
        completed_at = task["completed_at"]
        execution_time = (completed_at - started_at).total_seconds()

        # Create result record
        task_result = TaskResult(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error=error,
            execution_time_seconds=execution_time,
            agent_id=task["assigned_agent"] or "unknown",
            completed_at=completed_at,
        )

        # Add to history
        if task_id not in self.task_history:
            self.task_history[task_id] = []
        self.task_history[task_id].append(task_result)

        await self._logger.aerror(
            "task_failed",
            task_id=task_id,
            error=error,
            agent_id=task["assigned_agent"],
        )

        return task_result

    async def retry_task(
        self,
        task_id: str,
        delay_seconds: float = 5.0,
    ) -> bool:
        """Retry a failed task with exponential backoff.

        Args:
            task_id: ID of task to retry.
            delay_seconds: Initial delay before retry.

        Returns:
            True if retry was scheduled, False if max retries exceeded.

        Raises:
            KeyError: If task not found.
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self.tasks[task_id]

        if task["retry_count"] >= task["max_retries"]:
            await self._logger.awarning(
                "task_max_retries_exceeded",
                task_id=task_id,
                max_retries=task["max_retries"],
                retry_count=task["retry_count"],
            )
            return False

        # Exponential backoff: delay * (2 ^ retry_count)
        backoff_delay = delay_seconds * (2 ** task["retry_count"])

        task["retry_count"] += 1
        task["status"] = TaskStatus.QUEUED
        task["updated_at"] = datetime.now(timezone.utc)
        task["assigned_agent"] = None

        await self._logger.ainfo(
            "task_retry_scheduled",
            task_id=task_id,
            retry_count=task["retry_count"],
            backoff_delay_seconds=backoff_delay,
        )

        # Add back to queue after delay
        await asyncio.sleep(backoff_delay)
        await self.priority_queue.put((task["priority"], task_id))

        return True

    async def cancel_task(self, task_id: str) -> None:
        """Cancel a task (any non-terminal status).

        Args:
            task_id: ID of task to cancel.

        Raises:
            KeyError: If task not found.
            ValueError: If task already in a terminal state.
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self.tasks[task_id]

        terminal = (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
        if task["status"] in terminal:
            raise ValueError(f"Cannot cancel task in {task['status']} status")

        task["status"] = TaskStatus.CANCELLED
        task["updated_at"] = datetime.now(timezone.utc)
        task["completed_at"] = datetime.now(timezone.utc)

        await self._logger.ainfo(
            "task_cancelled",
            task_id=task_id,
        )

    def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        """Get task information.

        Args:
            task_id: ID of task to retrieve.

        Returns:
            Task data if found, None otherwise.
        """
        return self.tasks.get(task_id)

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        agent_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """List tasks with optional filtering.

        Args:
            status: Filter by status.
            agent_id: Filter by assigned agent.
            tags: Filter by tags (all tags must match).

        Returns:
            List of matching tasks.
        """
        results = []

        for task in self.tasks.values():
            if status and task["status"] != status:
                continue

            if agent_id and task["assigned_agent"] != agent_id:
                continue

            if tags and not all(tag in task["tags"] for tag in tags):
                continue

            results.append(task)

        return results

    def get_task_history(self, task_id: str) -> list[TaskResult]:
        """Get execution history for a task.

        Args:
            task_id: ID of task.

        Returns:
            List of TaskResult records for all executions (retries).
        """
        return self.task_history.get(task_id, [])

    async def cleanup_old_tasks(self, days: int = 30) -> int:
        """Clean up old completed/failed tasks.

        Args:
            days: Delete tasks older than this many days.

        Returns:
            Number of tasks deleted.
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        task_ids_to_delete = []

        for task_id, task in self.tasks.items():
            if task["status"] in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                if task["completed_at"] and task["completed_at"] < cutoff_time:
                    task_ids_to_delete.append(task_id)

        for task_id in task_ids_to_delete:
            del self.tasks[task_id]

        await self._logger.ainfo(
            "old_tasks_cleaned_up",
            deleted_count=len(task_ids_to_delete),
            days=days,
        )

        return len(task_ids_to_delete)

    async def get_stats(self) -> dict[str, Any]:
        """Get task statistics.

        Returns:
            Dictionary with task statistics.
        """
        stats = {
            "total_tasks": len(self.tasks),
            "by_status": {},
            "by_priority": {},
            "average_execution_time": 0.0,
            "total_retries": 0,
        }

        execution_times = []
        total_retries = 0

        for task in self.tasks.values():
            status = task["status"].value
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

            priority = task["priority"]
            stats["by_priority"][priority] = stats["by_priority"].get(priority, 0) + 1

            total_retries += task["retry_count"]

            if task["started_at"] and task["completed_at"]:
                execution_time = (task["completed_at"] - task["started_at"]).total_seconds()
                execution_times.append(execution_time)

        if execution_times:
            stats["average_execution_time"] = sum(execution_times) / len(execution_times)

        stats["total_retries"] = total_retries

        return stats
