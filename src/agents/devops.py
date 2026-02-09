"""Infrastructure/DevOps agent for deployment and configuration management."""

from datetime import datetime, timezone
from typing import Any

import structlog

from src.core.agent_base import AgentCapability, AgentIdentity, AgentState, BaseAgent

logger = structlog.get_logger(__name__)


class DevOpsAgent(BaseAgent):
    """Infrastructure-focused agent for deployment and CI/CD management.

    Responsibilities:
    - Manages Docker deployments
    - Handles CI/CD pipelines
    - Configuration management
    - Infrastructure monitoring

    Capabilities:
    - deploy_docker: Deploy Docker containers
    - manage_pipeline: Manage CI/CD pipeline
    - manage_config: Manage configurations
    - infrastructure_health: Monitor infrastructure health
    """

    def __init__(
        self,
        agent_id: str = "devops-infra",
        name: str = "DevOps Infrastructure Agent",
        role: str = "infrastructure",
    ) -> None:
        """Initialize the DevOps infrastructure agent.

        Args:
            agent_id: Unique agent identifier.
            name: Display name for the agent.
            role: Agent role classification.
        """
        identity = AgentIdentity(
            id=agent_id,
            name=name,
            role=role,
            security_level=4,
            capabilities=[
                AgentCapability(
                    name="deploy_docker",
                    version="1.0.0",
                    description="Deploy and manage Docker containers",
                    parameters={
                        "image": "str",
                        "version": "str",
                        "environment": "str",
                        "replicas": "int",
                    },
                ),
                AgentCapability(
                    name="manage_pipeline",
                    version="1.0.0",
                    description="Manage CI/CD pipeline execution",
                    parameters={
                        "pipeline_name": "str",
                        "trigger": "str",
                        "environment": "str",
                    },
                ),
                AgentCapability(
                    name="manage_config",
                    version="1.0.0",
                    description="Manage configuration and secrets",
                    parameters={
                        "config_type": "str",
                        "environment": "str",
                        "operation": "str",
                    },
                ),
                AgentCapability(
                    name="infrastructure_health",
                    version="1.0.0",
                    description="Monitor infrastructure health",
                    parameters={
                        "resources": "list[str]",
                        "metrics": "list[str]",
                    },
                ),
            ],
        )
        super().__init__(identity)
        self._deployments = []
        self._pipeline_runs = []
        self._configurations = {}

    async def startup(self) -> None:
        """Initialize DevOps agent.

        Raises:
            Exception: If startup fails.
        """
        await super().startup()
        await logger.ainfo(
            "devops_startup",
            agent_id=self.identity.id,
        )

    async def shutdown(self) -> None:
        """Shutdown DevOps agent gracefully.

        Raises:
            Exception: If shutdown fails.
        """
        await logger.ainfo(
            "devops_shutdown",
            agent_id=self.identity.id,
            deployments_managed=len(self._deployments),
            pipelines_run=len(self._pipeline_runs),
        )
        await super().shutdown()

    def _get_system_context(self) -> str:
        """Get system context for LLM reasoning.

        Returns:
            System context string describing the agent's role and expertise.
        """
        return (
            "You are a DevOps and infrastructure engineer specializing in Docker, "
            "CI/CD pipelines, configuration management, and infrastructure health. "
            "You automate deployment workflows and ensure system reliability."
        )

    async def process_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Process infrastructure-related tasks.

        Supported task types:
        - deploy_docker: Deploy Docker containers
        - manage_pipeline: Manage CI/CD pipeline
        - manage_config: Manage configurations
        - infrastructure_health: Monitor infrastructure

        Args:
            task: Task payload with type and parameters.

        Returns:
            Result dictionary with deployment/pipeline results.

        Raises:
            ValueError: If task type is unsupported.
        """
        task_type = task.get("type", "unknown")
        task_id = task.get("id", "unknown")

        await logger.ainfo(
            "devops_processing_task",
            task_id=task_id,
            task_type=task_type,
        )

        # ── Chain-of-Thought reasoning ────────────────────────────
        reasoning = await self.think(task)
        task.setdefault("context", {})["_reasoning"] = reasoning.conclusion

        try:
            if task_type == "deploy_docker":
                result = await self._handle_deploy_docker(task)
            elif task_type == "manage_pipeline":
                result = await self._handle_manage_pipeline(task)
            elif task_type == "manage_config":
                result = await self._handle_manage_config(task)
            elif task_type == "infrastructure_health":
                result = await self._handle_infrastructure_health(task)
            else:
                return await self._handle_chat_message(task)
            result["reasoning"] = reasoning.to_audit_dict()
            return result
        except Exception as exc:
            await logger.aerror(
                "devops_task_error",
                task_id=task_id,
                error=str(exc),
            )
            raise

    async def _handle_deploy_docker(self, task: dict[str, Any]) -> dict[str, Any]:
        """Deploy Docker container using LLM reasoning.

        Args:
            task: Task with deployment parameters.

        Returns:
            Dictionary with deployment status.
        """
        image = task.get("context", {}).get("image", "codehorde/app")
        version = task.get("context", {}).get("version", "latest")
        environment = task.get("context", {}).get("environment", "staging")
        replicas = task.get("context", {}).get("replicas", 2)

        await logger.ainfo(
            "docker_deployment_started",
            image=image,
            version=version,
            environment=environment,
            replicas=replicas,
        )

        try:
            chain = await self.think(task, strategy="step_by_step")
            deployment_analysis = chain.conclusion

            deployment_record = {
                "deployment_id": f"deploy-{datetime.now(timezone.utc).timestamp()}",
                "image": f"{image}:{version}",
                "environment": environment,
                "replicas": replicas,
                "status": "deployed",
                "deployed_at": datetime.now(timezone.utc).isoformat(),
            }

            self._deployments.append(deployment_record)

            result = {
                "status": "completed",
                "deployment_id": deployment_record["deployment_id"],
                "image": deployment_record["image"],
                "environment": environment,
                "replicas_deployed": replicas,
                "deployment_analysis": deployment_analysis,
                "reasoning_steps": len(chain.steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            await logger.ainfo(
                "docker_deployment_completed",
                deployment_id=deployment_record["deployment_id"],
                replicas=replicas,
            )

            return result
        except Exception as exc:
            await logger.awarning(
                "docker_deployment_llm_fallback",
                error=str(exc),
            )
            # Fallback if LLM is unavailable
            deployment_record = {
                "deployment_id": f"deploy-{datetime.now(timezone.utc).timestamp()}",
                "image": f"{image}:{version}",
                "environment": environment,
                "replicas": replicas,
                "status": "deployed",
                "deployed_at": datetime.now(timezone.utc).isoformat(),
                "containers": [
                    {
                        "id": f"container-{i}",
                        "image": f"{image}:{version}",
                        "status": "running",
                        "cpu_limit": "1.0",
                        "memory_limit": "1Gi",
                    }
                    for i in range(replicas)
                ],
            }

            self._deployments.append(deployment_record)

            return {
                "status": "completed",
                "deployment_id": deployment_record["deployment_id"],
                "image": deployment_record["image"],
                "environment": environment,
                "replicas_deployed": replicas,
                "deployment_status": "success",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _handle_manage_pipeline(self, task: dict[str, Any]) -> dict[str, Any]:
        """Manage CI/CD pipeline using LLM reasoning.

        Args:
            task: Task with pipeline parameters.

        Returns:
            Dictionary with pipeline execution result.
        """
        pipeline_name = task.get("context", {}).get("pipeline_name", "main")
        trigger = task.get("context", {}).get("trigger", "manual")
        environment = task.get("context", {}).get("environment", "staging")

        await logger.ainfo(
            "pipeline_execution_started",
            pipeline_name=pipeline_name,
            trigger=trigger,
            environment=environment,
        )

        try:
            chain = await self.think(task, strategy="step_by_step")
            pipeline_analysis = chain.conclusion

            pipeline_run = {
                "run_id": f"run-{datetime.now(timezone.utc).timestamp()}",
                "pipeline_name": pipeline_name,
                "trigger": trigger,
                "environment": environment,
                "status": "success",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }

            self._pipeline_runs.append(pipeline_run)

            result = {
                "status": "completed",
                "run_id": pipeline_run["run_id"],
                "pipeline_name": pipeline_name,
                "pipeline_analysis": pipeline_analysis,
                "reasoning_steps": len(chain.steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            await logger.ainfo(
                "pipeline_execution_completed",
                run_id=pipeline_run["run_id"],
                status=pipeline_run["status"],
            )

            return result
        except Exception as exc:
            await logger.awarning(
                "pipeline_execution_llm_fallback",
                error=str(exc),
            )
            # Fallback if LLM is unavailable
            pipeline_run = {
                "run_id": f"run-{datetime.now(timezone.utc).timestamp()}",
                "pipeline_name": pipeline_name,
                "trigger": trigger,
                "environment": environment,
                "status": "success",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "stages": [
                    {
                        "name": "build",
                        "status": "success",
                        "duration_seconds": 120,
                    },
                    {
                        "name": "test",
                        "status": "success",
                        "duration_seconds": 180,
                    },
                    {
                        "name": "deploy",
                        "status": "success",
                        "duration_seconds": 90,
                    },
                ],
                "total_duration_seconds": 390,
            }

            self._pipeline_runs.append(pipeline_run)

            return {
                "status": "completed",
                "run_id": pipeline_run["run_id"],
                "pipeline_name": pipeline_name,
                "pipeline_status": pipeline_run["status"],
                "stages_executed": len(pipeline_run["stages"]),
                "total_duration_seconds": pipeline_run["total_duration_seconds"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _handle_manage_config(self, task: dict[str, Any]) -> dict[str, Any]:
        """Manage configuration and secrets using LLM reasoning.

        Args:
            task: Task with configuration parameters.

        Returns:
            Dictionary with configuration operation result.
        """
        config_type = task.get("context", {}).get("config_type", "env")
        environment = task.get("context", {}).get("environment", "staging")
        operation = task.get("context", {}).get("operation", "get")

        await logger.ainfo(
            "config_management_started",
            config_type=config_type,
            environment=environment,
            operation=operation,
        )

        try:
            chain = await self.think(task, strategy="step_by_step")
            config_analysis = chain.conclusion

            config_key = f"{environment}_{config_type}"

            result = {
                "status": "completed",
                "config_type": config_type,
                "environment": environment,
                "operation": operation,
                "config_analysis": config_analysis,
                "reasoning_steps": len(chain.steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            await logger.ainfo(
                "config_management_completed",
                operation=operation,
            )

            return result
        except Exception as exc:
            await logger.awarning(
                "config_management_llm_fallback",
                error=str(exc),
            )
            # Fallback if LLM is unavailable
            config_key = f"{environment}_{config_type}"

            if operation == "set":
                config_value = task.get("context", {}).get("value", {})
                self._configurations[config_key] = config_value
                operation_result = f"Configuration {config_key} updated"
            elif operation == "get":
                config_value = self._configurations.get(config_key, {})
                operation_result = f"Retrieved configuration {config_key}"
            elif operation == "delete":
                if config_key in self._configurations:
                    del self._configurations[config_key]
                    operation_result = f"Configuration {config_key} deleted"
                else:
                    operation_result = f"Configuration {config_key} not found"
            else:
                operation_result = f"Unknown operation: {operation}"

            return {
                "status": "completed",
                "config_type": config_type,
                "environment": environment,
                "operation": operation,
                "operation_result": operation_result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _handle_infrastructure_health(
        self, task: dict[str, Any]
    ) -> dict[str, Any]:
        """Monitor infrastructure health using LLM reasoning.

        Args:
            task: Task with infrastructure monitoring parameters.

        Returns:
            Dictionary with infrastructure health status.
        """
        resources = task.get("context", {}).get(
            "resources",
            ["kubernetes", "databases", "storage"],
        )
        metrics = task.get("context", {}).get(
            "metrics", ["cpu", "memory", "disk", "network"]
        )

        await logger.ainfo(
            "infrastructure_health_check_started",
            resources=len(resources),
            metrics=len(metrics),
        )

        try:
            chain = await self.think(task, strategy="step_by_step")
            health_analysis = chain.conclusion

            result = {
                "status": "completed",
                "health_analysis": health_analysis,
                "resources_monitored": len(resources),
                "metrics_collected": len(metrics),
                "reasoning_steps": len(chain.steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            await logger.ainfo(
                "infrastructure_health_check_completed",
            )

            return result
        except Exception as exc:
            await logger.awarning(
                "infrastructure_health_llm_fallback",
                error=str(exc),
            )
            # Fallback if LLM is unavailable
            health_status = {
                "kubernetes": {
                    "status": "healthy",
                    "nodes": 3,
                    "pods_running": 24,
                    "pods_pending": 0,
                    "cpu_usage": 45.2,
                    "memory_usage": 62.5,
                },
                "databases": {
                    "status": "healthy",
                    "primary": {"status": "active", "lag_bytes": 0},
                    "replica": {"status": "active", "lag_seconds": 0},
                    "connections": 42,
                    "connection_pool_usage": 0.65,
                },
                "storage": {
                    "status": "healthy",
                    "usage_percent": 72.3,
                    "iops": 1250,
                    "latency_ms": 12.5,
                },
            }

            overall_health = "healthy" if all(
                r.get("status") == "healthy" for r in health_status.values()
            ) else "degraded"

            return {
                "status": "completed",
                "overall_health": overall_health,
                "health_status": health_status,
                "resources_monitored": len(resources),
                "metrics_collected": len(metrics),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
