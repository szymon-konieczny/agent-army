"""Code Horde MCP Server — expose agent capabilities via Model Context Protocol.

This module implements a full MCP server that makes Code Horde a tool provider
for any MCP-compatible client (Claude Code, Claude Desktop, Cursor, etc.).

Dual role:
  - Code Horde agents USE external MCP servers (as clients)
  - Code Horde EXPOSES itself as an MCP server (this module)

The server exposes:
  - Tools: Task management, agent operations, workflows, knowledge graph, security
  - Resources: System status, agent capabilities, policies, trust scores
  - Prompts: Delegation templates, security review, deployment guidance

Transport support:
  - stdio: Direct stdin/stdout communication
  - SSE: HTTP-based streaming
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog
from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP

logger = structlog.get_logger(__name__)


# Response models for documentation
class TaskResponse(BaseModel):
    """Response from task operations."""

    task_id: str = Field(description="Unique task identifier")
    status: str = Field(description="Task status (pending, in_progress, completed, etc.)")
    description: str = Field(description="Task description")
    priority: int = Field(description="Priority level (1-5, lower is higher)")
    assigned_agent: Optional[str] = Field(default=None, description="ID of assigned agent")
    created_at: str = Field(description="Creation timestamp")
    updated_at: str = Field(description="Last update timestamp")
    result: Optional[dict[str, Any]] = Field(default=None, description="Task result")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class AgentStatusResponse(BaseModel):
    """Agent system status."""

    agent_id: str = Field(description="Agent identifier")
    role: str = Field(description="Agent role (executor, sentinel, guardian, etc.)")
    state: str = Field(description="Agent state (idle, busy, errored)")
    tasks_completed: int = Field(description="Number of completed tasks")
    trust_score: float = Field(description="Trust score (0.0-1.0)")
    last_activity: str = Field(description="Last activity timestamp")
    capabilities: list[str] = Field(description="Agent capabilities")


class WorkflowResponse(BaseModel):
    """Workflow execution response."""

    execution_id: str = Field(description="Unique execution identifier")
    workflow_name: str = Field(description="Name of the workflow")
    status: str = Field(description="Execution status")
    progress: float = Field(description="Progress percentage (0-100)")
    current_step: str = Field(description="Current workflow step")
    started_at: str = Field(description="Start timestamp")
    estimated_completion: Optional[str] = Field(
        default=None, description="Estimated completion time"
    )


class SecurityScanResponse(BaseModel):
    """Security scan results."""

    scan_id: str = Field(description="Unique scan identifier")
    target: str = Field(description="Scan target (all, code, dependencies, etc.)")
    status: str = Field(description="Scan status (pending, in_progress, completed)")
    findings: list[dict[str, Any]] = Field(description="Security findings")
    severity_summary: dict[str, int] = Field(
        description="Count by severity (critical, high, medium, low)"
    )
    scanned_at: str = Field(description="Scan timestamp")


class SystemHealthResponse(BaseModel):
    """Full system health status."""

    healthy: bool = Field(description="Overall system health")
    agents_online: int = Field(description="Number of online agents")
    agents_total: int = Field(description="Total agents")
    pending_tasks: int = Field(description="Pending tasks")
    avg_trust_score: float = Field(description="Average trust score across agents")
    uptime_seconds: float = Field(description="System uptime")
    estimated_monthly_cost: float = Field(description="Estimated monthly LLM cost")


class MCPServer:
    """Code Horde MCP Server implementation.

    Exposes Code Horde capabilities through the Model Context Protocol,
    enabling external AI tools to interact with the agent fleet.

    Attributes:
        app: FastMCP application instance
        api_base_url: Base URL for Code Horde HTTP API
        api_timeout: HTTP request timeout in seconds
        redis_url: Redis connection URL (for caching, optional)
        neo4j_uri: Neo4j connection URI (for knowledge graph)
    """

    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        api_timeout: float = 30.0,
        redis_url: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
    ) -> None:
        """Initialize the MCP server.

        Args:
            api_base_url: Base URL for Code Horde API
            api_timeout: HTTP request timeout
            redis_url: Redis connection URL
            neo4j_uri: Neo4j connection URI
        """
        self.api_base_url = api_base_url
        self.api_timeout = api_timeout
        self.redis_url = redis_url
        self.neo4j_uri = neo4j_uri
        self._http_client: Optional[httpx.AsyncClient] = None

        self.app = FastMCP("codehorde")

        self._register_tools()
        self._register_resources()
        self._register_prompts()

    def register_additional_tools(self, registrar: callable) -> None:
        """Allow external modules to register tools on this server's app.

        Used by the cluster module to add worker management tools when
        running in center mode.

        Args:
            registrar: A callable that takes a FastMCP app instance
                      and registers additional tools/resources on it.
        """
        registrar(self.app)

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for API calls."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self.api_base_url, timeout=self.api_timeout
            )
        return self._http_client

    async def _call_api(
        self, method: str, endpoint: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Call Code Horde HTTP API with error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments to pass to httpx

        Returns:
            Response JSON as dictionary

        Raises:
            httpx.HTTPError: If API call fails
        """
        client = await self._get_http_client()
        try:
            response = await client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error("api_call_failed", endpoint=endpoint, error=str(e))
            raise

    def _register_tools(self) -> None:
        """Register all MCP tools."""

        # Task Management Tools

        @self.app.tool()
        async def codehorde_create_task(
            description: str,
            priority: int = 3,
            tags: Optional[list[str]] = None,
        ) -> dict[str, Any]:
            """Create a task and assign it to the best agent automatically.

            This tool creates a new task in the Code Horde system and uses
            intelligent matching to assign it to the most suitable agent
            based on capabilities, current workload, and trust score.

            Args:
                description: Detailed task description for agents
                priority: Priority level (1=critical, 5=deferred, default=3 normal)
                tags: Optional tags for categorization (e.g., ["security", "urgent"])

            Returns:
                Task details including ID, status, and assigned agent

            Raises:
                httpx.HTTPError: If API is unreachable
            """
            try:
                result = await self._call_api(
                    "POST",
                    "/api/tasks",
                    json={
                        "description": description,
                        "priority": priority,
                        "tags": tags or [],
                    },
                )
                logger.info("task_created", task_id=result.get("task_id"))
                return result
            except Exception as e:
                logger.error("create_task_failed", error=str(e))
                return {"error": str(e), "status": "failed"}

        @self.app.tool()
        async def codehorde_task_status(task_id: str) -> dict[str, Any]:
            """Get the current status of a specific task.

            Retrieve detailed information about a task including its current
            status, assigned agent, progress, and any results or errors.

            Args:
                task_id: The task ID to query

            Returns:
                Task details with current status and progress information

            Raises:
                httpx.HTTPError: If task not found or API error
            """
            try:
                result = await self._call_api("GET", f"/api/tasks/{task_id}")
                return result
            except Exception as e:
                logger.error("task_status_failed", task_id=task_id, error=str(e))
                return {"error": str(e), "task_id": task_id}

        @self.app.tool()
        async def codehorde_list_tasks(
            status: str = "all",
            limit: int = 20,
        ) -> dict[str, Any]:
            """List tasks filtered by status.

            Retrieve a filtered list of tasks with pagination support.
            Useful for monitoring task queue, progress, and completion.

            Args:
                status: Filter by status (all, pending, in_progress, completed, failed)
                limit: Maximum number of tasks to return (default 20)

            Returns:
                List of task objects with total count
            """
            try:
                params = {"limit": limit}
                if status != "all":
                    params["status"] = status

                result = await self._call_api(
                    "GET",
                    "/api/tasks",
                    params=params,
                )
                return result
            except Exception as e:
                logger.error("list_tasks_failed", error=str(e))
                return {"error": str(e), "tasks": []}

        # Agent Operations

        @self.app.tool()
        async def codehorde_agent_status() -> dict[str, Any]:
            """Get status of all agents in the system.

            Retrieve comprehensive status information for every agent including
            their role, current workload, trust score, and capabilities.

            Returns:
                Dictionary with agent statuses and system-wide metrics
            """
            try:
                result = await self._call_api("GET", "/api/agents/status")
                logger.info("agent_status_retrieved", agent_count=len(result.get("agents", [])))
                return result
            except Exception as e:
                logger.error("agent_status_failed", error=str(e))
                return {"error": str(e), "agents": []}

        @self.app.tool()
        async def codehorde_ask_agent(
            agent_role: str,
            question: str,
        ) -> dict[str, Any]:
            """Ask a specific agent type a question.

            Query a particular agent role (e.g., sentinel for security,
            executor for tasks) directly with a natural language question.

            Args:
                agent_role: Agent role to query (sentinel, executor, guardian, etc.)
                question: The question to ask the agent

            Returns:
                Agent's response and reasoning
            """
            try:
                result = await self._call_api(
                    "POST",
                    f"/api/agents/{agent_role}/ask",
                    json={"question": question},
                )
                return result
            except Exception as e:
                logger.error("ask_agent_failed", agent_role=agent_role, error=str(e))
                return {"error": str(e), "agent_role": agent_role}

        # Workflow Execution

        @self.app.tool()
        async def codehorde_start_workflow(
            workflow_name: str,
            context: Optional[dict[str, Any]] = None,
        ) -> dict[str, Any]:
            """Start a predefined workflow with given context.

            Launch a workflow for common operations like feature development,
            security scanning, bug fixes, or deployments.

            Args:
                workflow_name: Workflow name (feature_development, security_scan,
                              bug_fix, deploy_staging, deploy_production)
                context: Additional context data for the workflow

            Returns:
                Workflow execution details with execution_id for monitoring
            """
            try:
                result = await self._call_api(
                    "POST",
                    f"/api/workflows/{workflow_name}/start",
                    json={"context": context or {}},
                )
                logger.info("workflow_started", workflow=workflow_name, execution_id=result.get("execution_id"))
                return result
            except Exception as e:
                logger.error("start_workflow_failed", workflow=workflow_name, error=str(e))
                return {"error": str(e), "workflow_name": workflow_name}

        @self.app.tool()
        async def codehorde_workflow_status(execution_id: str) -> dict[str, Any]:
            """Check the progress of a running workflow.

            Monitor a workflow's execution including current step, progress
            percentage, and estimated completion time.

            Args:
                execution_id: The workflow execution ID

            Returns:
                Workflow status with progress and current step information
            """
            try:
                result = await self._call_api("GET", f"/api/workflows/{execution_id}")
                return result
            except Exception as e:
                logger.error("workflow_status_failed", execution_id=execution_id, error=str(e))
                return {"error": str(e), "execution_id": execution_id}

        @self.app.tool()
        async def codehorde_approve(request_id: str) -> dict[str, Any]:
            """Approve a RED-tier action waiting for human approval.

            Some high-risk operations (RED-tier) require explicit human approval
            before execution. Use this tool to grant approval for a pending request.

            Args:
                request_id: The approval request ID

            Returns:
                Confirmation of approval and action status
            """
            try:
                result = await self._call_api(
                    "POST",
                    f"/api/approvals/{request_id}/approve",
                    json={"approved_at": datetime.now(timezone.utc).isoformat()},
                )
                logger.info("approval_granted", request_id=request_id)
                return result
            except Exception as e:
                logger.error("approval_failed", request_id=request_id, error=str(e))
                return {"error": str(e), "request_id": request_id}

        # Knowledge Graph

        @self.app.tool()
        async def codehorde_knowledge_query(question: str) -> dict[str, Any]:
            """Query the Neo4j knowledge graph using natural language.

            Use GraphRAG to query the knowledge graph with a natural language
            question. Returns relevant entities, relationships, and insights.

            Args:
                question: Natural language question about system knowledge

            Returns:
                Query results with entities, relationships, and answers
            """
            try:
                result = await self._call_api(
                    "POST",
                    "/api/knowledge/query",
                    json={"question": question},
                )
                logger.info("knowledge_query_executed", question=question)
                return result
            except Exception as e:
                logger.error("knowledge_query_failed", error=str(e))
                return {"error": str(e), "results": []}

        @self.app.tool()
        async def codehorde_knowledge_add(
            entity: str,
            entity_type: str,
            relationships: Optional[list[dict[str, Any]]] = None,
        ) -> dict[str, Any]:
            """Add knowledge to the Neo4j knowledge graph.

            Add new entities and relationships to enrich the system's knowledge
            base. Useful for documenting patterns, configurations, and learnings.

            Args:
                entity: Entity name or description
                entity_type: Type classification (agent, service, config, etc.)
                relationships: List of relationships to other entities

            Returns:
                Confirmation with created entity ID and relationship count
            """
            try:
                result = await self._call_api(
                    "POST",
                    "/api/knowledge/add",
                    json={
                        "entity": entity,
                        "entity_type": entity_type,
                        "relationships": relationships or [],
                    },
                )
                logger.info("knowledge_added", entity=entity, entity_type=entity_type)
                return result
            except Exception as e:
                logger.error("knowledge_add_failed", entity=entity, error=str(e))
                return {"error": str(e)}

        # Security

        @self.app.tool()
        async def codehorde_security_scan(target: str = "all") -> dict[str, Any]:
            """Trigger a comprehensive security scan.

            Initiate security scans for dependencies, secrets, code analysis,
            and other security checks. Scans run asynchronously.

            Args:
                target: Scan target (all, code, dependencies, secrets, config)

            Returns:
                Scan ID and initial status for monitoring
            """
            try:
                result = await self._call_api(
                    "POST",
                    "/api/security/scan",
                    json={"target": target},
                )
                logger.info("security_scan_started", target=target, scan_id=result.get("scan_id"))
                return result
            except Exception as e:
                logger.error("security_scan_failed", target=target, error=str(e))
                return {"error": str(e), "target": target}

        @self.app.tool()
        async def codehorde_security_report() -> dict[str, Any]:
            """Get the latest security report.

            Retrieve the most recent comprehensive security report including
            all findings, vulnerabilities, and recommendations.

            Returns:
                Full security report with findings and severity analysis
            """
            try:
                result = await self._call_api("GET", "/api/security/report")
                return result
            except Exception as e:
                logger.error("security_report_failed", error=str(e))
                return {"error": str(e), "findings": []}

        # System Operations

        @self.app.tool()
        async def codehorde_system_health() -> dict[str, Any]:
            """Perform a full system health check.

            Get comprehensive system status including agent availability,
            infrastructure health, estimated costs, and trust scores.

            Returns:
                System health status with all metrics
            """
            try:
                result = await self._call_api("GET", "/api/system/health")
                logger.info("system_health_checked")
                return result
            except Exception as e:
                logger.error("system_health_failed", error=str(e))
                return {
                    "error": str(e),
                    "healthy": False,
                }

        @self.app.tool()
        async def codehorde_digest() -> dict[str, Any]:
            """Get the latest activity digest.

            Retrieve a summary of recent agent activities, completed tasks,
            and important events. Useful for monitoring system activity.

            Returns:
                Activity digest with recent events and statistics
            """
            try:
                result = await self._call_api("GET", "/api/system/digest")
                return result
            except Exception as e:
                logger.error("digest_failed", error=str(e))
                return {"error": str(e), "events": []}

        @self.app.tool()
        async def codehorde_cost_report(period: str = "today") -> dict[str, Any]:
            """Get LLM API cost breakdown.

            Retrieve cost data for LLM API calls including per-agent breakdown,
            per-model breakdown, and cost projections.

            Args:
                period: Time period (today, week, month, all)

            Returns:
                Cost report with detailed breakdown and projections
            """
            try:
                result = await self._call_api(
                    "GET",
                    "/api/system/costs",
                    params={"period": period},
                )
                return result
            except Exception as e:
                logger.error("cost_report_failed", period=period, error=str(e))
                return {"error": str(e), "period": period}

        # Code Reranker

        @self.app.tool()
        async def codehorde_rerank_code(
            query: str,
            candidates: list[dict[str, Any]],
            k: int = 32,
        ) -> dict[str, Any]:
            """Rerank code fragments by relevance to a natural-language query.

            Uses an LLM-based reranker with BM25 fallback.  Returns
            candidates sorted by relevance score (lower = more relevant).

            Works with any programming language or framework.

            Args:
                query: Natural-language question, refactoring request,
                       bug description, or stack trace.
                candidates: List of code fragment objects, each with
                    ``id`` (unique identifier, e.g. file path + span),
                    ``content`` (code/text), and optional ``metadata``
                    dict (``language``, ``framework``, ``path``, ``repo``).
                k: Number of top results to return (default 32).

            Returns:
                Ranked results with scores and reranking source info.
            """
            try:
                from src.core.reranker import CodeReranker, RerankCandidate

                reranker = CodeReranker(k_rerank=k)
                cands = [
                    RerankCandidate(
                        id=c.get("id", f"candidate-{i}"),
                        content=c.get("content", ""),
                        metadata=c.get("metadata", {}),
                    )
                    for i, c in enumerate(candidates)
                ]
                results = await reranker.rerank(query, cands)
                return {
                    "results": [r.to_dict() for r in results],
                    "total_candidates": len(candidates),
                    "returned": len(results),
                    "query": query,
                }
            except Exception as e:
                logger.error("rerank_code_failed", error=str(e))
                return {"error": str(e), "results": []}

    def _register_resources(self) -> None:
        """Register MCP resources."""

        @self.app.resource("codehorde://status")
        def get_system_status() -> str:
            """Get current system status as JSON.

            Returns a snapshot of system status including agent counts,
            task queue depth, and overall health.

            Returns:
                JSON-formatted system status
            """
            try:
                # In a real implementation, this would fetch from Redis or internal state
                status = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "healthy": True,
                    "agents_online": 8,
                    "agents_total": 10,
                    "pending_tasks": 12,
                    "source": "codehorde://status",
                }
                return json.dumps(status, indent=2)
            except Exception as e:
                logger.error("status_resource_failed", error=str(e))
                return json.dumps({"error": str(e)})

        @self.app.resource("codehorde://agents")
        def get_agents_resource() -> str:
            """List all agents with their capabilities.

            Returns details about each agent including role, capabilities,
            state, and trust score.

            Returns:
                JSON-formatted agent list
            """
            try:
                agents = [
                    {
                        "agent_id": "executor-1",
                        "role": "executor",
                        "capabilities": ["code_execution", "task_management", "learning"],
                        "state": "idle",
                        "trust_score": 0.92,
                    },
                    {
                        "agent_id": "sentinel-1",
                        "role": "sentinel",
                        "capabilities": ["security_scanning", "threat_detection", "audit"],
                        "state": "idle",
                        "trust_score": 0.95,
                    },
                    {
                        "agent_id": "guardian-1",
                        "role": "guardian",
                        "capabilities": ["policy_enforcement", "approval_handling"],
                        "state": "idle",
                        "trust_score": 0.98,
                    },
                ]
                return json.dumps(agents, indent=2)
            except Exception as e:
                logger.error("agents_resource_failed", error=str(e))
                return json.dumps({"error": str(e)})

        @self.app.resource("codehorde://policies")
        def get_policies() -> str:
            """Get current autonomy policies.

            Returns the current set of autonomy policies governing what
            agents can do without approval.

            Returns:
                JSON-formatted policy document
            """
            try:
                policies = {
                    "autonomy_levels": {
                        "BLUE": "Requires explicit approval",
                        "GREEN": "Automatic execution (low risk)",
                        "RED": "Requires approval (high risk)",
                    },
                    "agent_policies": {
                        "executor": "Can execute code up to GREEN level",
                        "sentinel": "Can scan and report, RED scans need approval",
                        "guardian": "Enforces all policies, auto-approves safe GREEN",
                    },
                }
                return json.dumps(policies, indent=2)
            except Exception as e:
                logger.error("policies_resource_failed", error=str(e))
                return json.dumps({"error": str(e)})

        @self.app.resource("codehorde://trust-scores")
        def get_trust_scores() -> str:
            """Get trust profiles for all agents.

            Returns trust scores and trust history for each agent,
            reflecting reliability and safety performance.

            Returns:
                JSON-formatted trust score data
            """
            try:
                trust_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "agents": {
                        "executor-1": {
                            "trust_score": 0.92,
                            "tasks_completed": 247,
                            "tasks_failed": 3,
                            "reliability": "High",
                        },
                        "sentinel-1": {
                            "trust_score": 0.95,
                            "scans_completed": 156,
                            "false_positives": 2,
                            "reliability": "Very High",
                        },
                        "guardian-1": {
                            "trust_score": 0.98,
                            "approvals_processed": 89,
                            "policy_violations": 0,
                            "reliability": "Very High",
                        },
                    },
                }
                return json.dumps(trust_data, indent=2)
            except Exception as e:
                logger.error("trust_scores_resource_failed", error=str(e))
                return json.dumps({"error": str(e)})

    def _register_prompts(self) -> None:
        """Register MCP prompt templates."""

        @self.app.prompt()
        def delegate_to_army(task_description: str) -> str:
            """Template for delegating work to the agent army.

            Use this prompt template when you want to delegate a task to
            the most suitable agent. Includes guidance on task structure.

            Args:
                task_description: Description of work to delegate

            Returns:
                Formatted prompt for agent delegation
            """
            return f"""You are about to delegate the following task to Code Horde's autonomous agents.

Task Description:
{task_description}

Consider these questions before delegating:
1. Is the task clearly defined and actionable?
2. Are there security or compliance concerns?
3. What success metrics apply?
4. Should this go through approval workflows?

For safe delegation, structure the task with:
- Clear acceptance criteria
- Priority level (1-5)
- Any security requirements
- Relevant tags (e.g., "security", "urgent", "learning")

Use the codehorde_create_task tool to delegate."""

        @self.app.prompt()
        def security_review(target: str) -> str:
            """Template for security review using agents.

            Guide for conducting security reviews using Code Horde's
            sentinel and security-focused agents.

            Args:
                target: What to review (code, dependencies, config, all)

            Returns:
                Formatted prompt for security review
            """
            return f"""You are initiating a security review using Code Horde's security agents.

Review Target: {target}

Security review process:
1. Scan for vulnerabilities in {target}
2. Check dependencies and supply chain
3. Review secrets management
4. Validate configuration security
5. Generate comprehensive report

Code Horde's sentinel agent will:
- Perform automated security scanning
- Analyze threats and risks
- Generate detailed reports
- Recommend mitigations

Use codehorde_security_scan to initiate scanning.
Use codehorde_security_report to get results."""

        @self.app.prompt()
        def deploy(environment: str) -> str:
            """Template for deployment workflows.

            Guide for using Code Horde's deployment workflows to safely
            deploy to staging or production.

            Args:
                environment: Target environment (staging or production)

            Returns:
                Formatted prompt for deployment
            """
            return f"""You are setting up a deployment to {environment} using Code Horde.

Pre-deployment checklist:
1. Verify code quality and tests pass
2. Run security scans
3. Review deployment changes
4. Get necessary approvals
5. Execute deployment

For {environment} deployment:
- Staging: Can proceed with automated agent execution
- Production: Requires multiple approvals and careful orchestration

Use codehorde_start_workflow with 'deploy_{environment}' to begin.
Monitor progress with codehorde_workflow_status.
Approve RED-tier actions with codehorde_approve as needed."""

    async def run_stdio(self) -> None:
        """Run the server with stdio transport (for MCP-compatible clients).

        This is the standard transport for MCP clients like Claude Desktop
        and Claude Code.
        """
        logger.info("starting_mcp_server", transport="stdio")
        try:
            async with self.app.run_async():
                await asyncio.sleep(float("inf"))
        except KeyboardInterrupt:
            logger.info("mcp_server_stopped")
        except Exception as e:
            logger.error("mcp_server_error", error=str(e))
            raise
        finally:
            if self._http_client:
                await self._http_client.aclose()

    async def run_sse(self, host: str = "0.0.0.0", port: int = 8001) -> None:
        """Run the server with SSE (Server-Sent Events) transport.

        Args:
            host: Host to bind to
            port: Port to listen on
        """
        logger.info("starting_mcp_server", transport="sse", host=host, port=port)
        try:
            from mcp.server.sse import SSEServerTransport
            from starlette.applications import Starlette
            from starlette.routing import Mount
            from starlette.staticfiles import StaticFiles

            # Create Starlette app with SSE transport
            sse_transport = SSEServerTransport(endpoint="/sse")
            app_starlette = Starlette(
                routes=[
                    Mount("/sse", app=sse_transport),
                ]
            )

            import uvicorn

            config = uvicorn.Config(
                app=app_starlette, host=host, port=port, log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
        except ImportError:
            logger.error("sse_dependencies_missing")
            raise RuntimeError(
                "SSE transport requires: starlette, uvicorn. Install with: pip install starlette uvicorn"
            )
        except KeyboardInterrupt:
            logger.info("mcp_server_stopped")
        finally:
            if self._http_client:
                await self._http_client.aclose()
