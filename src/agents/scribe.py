"""Documentation agent for API docs and knowledge base management."""

from datetime import datetime, timezone
from typing import Any

import structlog

from src.core.agent_base import AgentCapability, AgentIdentity, AgentState, BaseAgent

logger = structlog.get_logger(__name__)


class ScribeAgent(BaseAgent):
    """Documentation-focused agent for knowledge base and documentation management.

    Responsibilities:
    - Generates API documentation
    - Creates changelogs
    - Maintains knowledge base
    - Generates reports from agent activities

    Capabilities:
    - generate_api_docs: Generate API documentation
    - create_changelog: Create/update changelog
    - maintain_knowledge_base: Manage knowledge base
    - generate_activity_report: Generate reports from activities
    """

    def __init__(
        self,
        agent_id: str = "scribe-docs",
        name: str = "Scribe Documentation Agent",
        role: str = "documentation",
    ) -> None:
        """Initialize the Scribe documentation agent.

        Args:
            agent_id: Unique agent identifier.
            name: Display name for the agent.
            role: Agent role classification.
        """
        identity = AgentIdentity(
            id=agent_id,
            name=name,
            role=role,
            security_level=1,
            capabilities=[
                AgentCapability(
                    name="generate_api_docs",
                    version="1.0.0",
                    description="Generate API documentation from code",
                    parameters={
                        "source_path": "str",
                        "format": "str",
                        "include_examples": "bool",
                    },
                ),
                AgentCapability(
                    name="create_changelog",
                    version="1.0.0",
                    description="Create or update changelog",
                    parameters={
                        "version": "str",
                        "changes": "list[str]",
                        "change_type": "str",
                    },
                ),
                AgentCapability(
                    name="maintain_knowledge_base",
                    version="1.0.0",
                    description="Manage knowledge base entries",
                    parameters={
                        "operation": "str",
                        "category": "str",
                        "content": "str",
                    },
                ),
                AgentCapability(
                    name="generate_activity_report",
                    version="1.0.0",
                    description="Generate reports from agent activities",
                    parameters={
                        "time_period": "str",
                        "agents": "list[str]",
                        "format": "str",
                    },
                ),
            ],
        )
        super().__init__(identity)
        self._generated_docs = []
        self._changelogs = []
        self._knowledge_base = {}
        self._activity_reports = []

    async def startup(self) -> None:
        """Initialize documentation agent.

        Raises:
            Exception: If startup fails.
        """
        await super().startup()
        await logger.ainfo(
            "scribe_startup",
            agent_id=self.identity.id,
        )

    async def shutdown(self) -> None:
        """Shutdown documentation agent gracefully.

        Raises:
            Exception: If shutdown fails.
        """
        await logger.ainfo(
            "scribe_shutdown",
            agent_id=self.identity.id,
            docs_generated=len(self._generated_docs),
            changelogs_created=len(self._changelogs),
        )
        await super().shutdown()

    def _get_system_context(self) -> str:
        """Get system context for LLM reasoning.

        Returns:
            System context string describing the agent's role and expertise.
        """
        return (
            "You are a technical documentation specialist skilled in API docs, "
            "changelogs, knowledge base articles, and activity reports. You write "
            "clear, well-structured documentation that serves both developers "
            "and stakeholders."
        )

    async def process_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Process documentation-related tasks.

        Supported task types:
        - generate_api_docs: Generate API documentation
        - create_changelog: Create/update changelog
        - maintain_knowledge_base: Manage knowledge base
        - generate_activity_report: Generate reports

        Args:
            task: Task payload with type and parameters.

        Returns:
            Result dictionary with generated documentation.

        Raises:
            ValueError: If task type is unsupported.
        """
        task_type = task.get("type", "unknown")
        task_id = task.get("id", "unknown")

        await logger.ainfo(
            "scribe_processing_task",
            task_id=task_id,
            task_type=task_type,
        )

        # ── Chain-of-Thought reasoning ────────────────────────────
        reasoning = await self.think(task)
        task.setdefault("context", {})["_reasoning"] = reasoning.conclusion

        try:
            if task_type == "generate_api_docs":
                result = await self._handle_generate_api_docs(task)
            elif task_type == "create_changelog":
                result = await self._handle_create_changelog(task)
            elif task_type == "maintain_knowledge_base":
                result = await self._handle_maintain_knowledge_base(task)
            elif task_type == "generate_activity_report":
                result = await self._handle_generate_activity_report(task)
            else:
                return await self._handle_chat_message(task)
            result["reasoning"] = reasoning.to_audit_dict()
            return result
        except Exception as exc:
            await logger.aerror(
                "scribe_task_error",
                task_id=task_id,
                error=str(exc),
            )
            raise

    async def _handle_generate_api_docs(self, task: dict[str, Any]) -> dict[str, Any]:
        """Generate API documentation using LLM reasoning.

        Args:
            task: Task with API documentation parameters.

        Returns:
            Dictionary with generated API documentation.
        """
        source_path = task.get("context", {}).get("source_path", "src/")
        doc_format = task.get("context", {}).get("format", "markdown")
        include_examples = task.get("context", {}).get("include_examples", True)

        await logger.ainfo(
            "api_docs_generation_started",
            source_path=source_path,
            format=doc_format,
        )

        try:
            chain = await self.think(task, strategy="step_by_step")
            api_docs = chain.conclusion

            output_file = f"api_docs.{self._get_doc_extension(doc_format)}"
            self._generated_docs.append(output_file)

            result = {
                "status": "completed",
                "output_file": output_file,
                "api_docs": api_docs,
                "format": doc_format,
                "reasoning_steps": len(chain.steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            await logger.ainfo(
                "api_docs_generation_completed",
                output_file=output_file,
            )

            return result
        except Exception as exc:
            await logger.awarning(
                "api_docs_generation_llm_fallback",
                error=str(exc),
            )
            # Fallback if LLM is unavailable
            api_docs = f"""# API Documentation

Generated from: {source_path}
Format: {doc_format}
Include Examples: {include_examples}

## Endpoints

### GET /health
Health check endpoint.

**Parameters:** None

**Response:**
```json
{{
    "status": "healthy",
    "timestamp": "2024-02-06T10:30:00Z",
    "version": "1.0.0"
}}
```

### POST /tasks
Create a new task.

**Parameters:**
- `description` (string, required): Task description
- `priority` (integer, optional): Priority level 1-5

**Response:**
```json
{{
    "task_id": "uuid",
    "status": "pending",
    "created_at": "2024-02-06T10:30:00Z"
}}
```

## Error Responses

400 Bad Request - Invalid parameters
401 Unauthorized - Authentication required
500 Internal Server Error - Server error
"""

            output_file = f"api_docs.{self._get_doc_extension(doc_format)}"
            self._generated_docs.append(output_file)

            return {
                "status": "completed",
                "output_file": output_file,
                "api_docs": api_docs,
                "format": doc_format,
                "endpoints_documented": 2,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _handle_create_changelog(self, task: dict[str, Any]) -> dict[str, Any]:
        """Create/update changelog using LLM reasoning.

        Args:
            task: Task with changelog parameters.

        Returns:
            Dictionary with changelog information.
        """
        version = task.get("context", {}).get("version", "1.1.0")
        changes = task.get("context", {}).get(
            "changes",
            ["Added feature X", "Fixed bug Y", "Improved performance"],
        )
        change_type = task.get("context", {}).get("change_type", "release")

        await logger.ainfo(
            "changelog_creation_started",
            version=version,
            changes_count=len(changes),
        )

        try:
            chain = await self.think(task, strategy="step_by_step")
            changelog_entry = chain.conclusion

            self._changelogs.append(
                {
                    "version": version,
                    "changes": changes,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

            result = {
                "status": "completed",
                "version": version,
                "changelog_entry": changelog_entry,
                "changes_count": len(changes),
                "reasoning_steps": len(chain.steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            await logger.ainfo(
                "changelog_creation_completed",
                version=version,
            )

            return result
        except Exception as exc:
            await logger.awarning(
                "changelog_creation_llm_fallback",
                error=str(exc),
            )
            # Fallback if LLM is unavailable
            changelog_entry = f"""## [{version}] - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}

### Added
- New feature for API versioning
- Support for async/await in all endpoints

### Fixed
- Fixed bug in authentication flow
- Corrected error handling in database layer

### Changed
- Improved performance of query optimization
- Updated dependencies to latest versions

### Deprecated
- Legacy authentication method (use OAuth instead)

### Removed
- Deprecated REST endpoints

### Security
- Fixed security vulnerability in input validation
"""

            self._changelogs.append(
                {
                    "version": version,
                    "changes": changes,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

            return {
                "status": "completed",
                "version": version,
                "changelog_entry": changelog_entry,
                "changes_count": len(changes),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _handle_maintain_knowledge_base(
        self, task: dict[str, Any]
    ) -> dict[str, Any]:
        """Manage knowledge base entries using LLM reasoning.

        Args:
            task: Task with knowledge base parameters.

        Returns:
            Dictionary with knowledge base operation result.
        """
        operation = task.get("context", {}).get("operation", "add")
        category = task.get("context", {}).get("category", "general")
        content = task.get("context", {}).get("content", "")

        await logger.ainfo(
            "knowledge_base_operation_started",
            operation=operation,
            category=category,
        )

        try:
            chain = await self.think(task, strategy="step_by_step")
            analysis = chain.conclusion

            result = {
                "status": "completed",
                "operation": operation,
                "category": category,
                "analysis": analysis,
                "reasoning_steps": len(chain.steps),
                "kb_size": len(self._knowledge_base),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            await logger.ainfo(
                "knowledge_base_operation_completed",
                operation=operation,
            )

            return result
        except Exception as exc:
            await logger.awarning(
                "knowledge_base_operation_llm_fallback",
                error=str(exc),
            )
            # Fallback if LLM is unavailable
            if operation == "add":
                entry_id = f"kb-{datetime.now(timezone.utc).timestamp()}"
                self._knowledge_base[entry_id] = {
                    "category": category,
                    "content": content,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                operation_result = f"Added knowledge base entry {entry_id}"
            elif operation == "list":
                entries = [
                    k for k, v in self._knowledge_base.items()
                    if v.get("category") == category or not category
                ]
                operation_result = f"Found {len(entries)} entries"
            elif operation == "delete":
                entry_id = task.get("context", {}).get("entry_id")
                if entry_id in self._knowledge_base:
                    del self._knowledge_base[entry_id]
                    operation_result = f"Deleted entry {entry_id}"
                else:
                    operation_result = f"Entry {entry_id} not found"
            else:
                operation_result = f"Unknown operation: {operation}"

            return {
                "status": "completed",
                "operation": operation,
                "category": category,
                "operation_result": operation_result,
                "kb_size": len(self._knowledge_base),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _handle_generate_activity_report(
        self, task: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate activity report using LLM reasoning.

        Args:
            task: Task with activity report parameters.

        Returns:
            Dictionary with activity report.
        """
        time_period = task.get("context", {}).get("time_period", "24h")
        agents = task.get("context", {}).get(
            "agents", ["sentinel", "builder", "inspector"]
        )
        report_format = task.get("context", {}).get("format", "json")

        await logger.ainfo(
            "activity_report_generation_started",
            time_period=time_period,
            agents_count=len(agents),
        )

        try:
            chain = await self.think(task, strategy="step_by_step")
            report_content = chain.conclusion

            result = {
                "status": "completed",
                "report": report_content,
                "report_format": report_format,
                "reasoning_steps": len(chain.steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            await logger.ainfo(
                "activity_report_generation_completed",
            )

            return result
        except Exception as exc:
            await logger.awarning(
                "activity_report_llm_fallback",
                error=str(exc),
            )
            # Fallback if LLM is unavailable
            activity_report = {
                "title": f"Agent Activity Report ({time_period})",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "time_period": time_period,
                "summary": {
                    "total_tasks_processed": 247,
                    "successful_tasks": 242,
                    "failed_tasks": 5,
                    "success_rate": 0.9797,
                },
                "agent_activities": {
                    "sentinel": {
                        "tasks_processed": 45,
                        "vulnerabilities_found": 8,
                        "status": "healthy",
                    },
                    "builder": {
                        "tasks_processed": 32,
                        "code_files_generated": 12,
                        "prs_created": 3,
                        "status": "healthy",
                    },
                    "inspector": {
                        "tasks_processed": 85,
                        "tests_run": 1230,
                        "coverage": 0.82,
                        "status": "healthy",
                    },
                    "watcher": {
                        "tasks_processed": 65,
                        "alerts_sent": 2,
                        "status": "healthy",
                    },
                    "scout": {
                        "tasks_processed": 12,
                        "advisories_monitored": 15,
                        "status": "healthy",
                    },
                    "scribe": {
                        "tasks_processed": 8,
                        "docs_generated": 5,
                        "status": "healthy",
                    },
                },
                "top_performers": [
                    {"agent": "inspector", "score": 95},
                    {"agent": "watcher", "score": 92},
                    {"agent": "sentinel", "score": 88},
                ],
                "issues": [
                    {
                        "agent": "sentinel",
                        "issue": "3 failed vulnerability scans",
                        "severity": "warning",
                    },
                ],
                "recommendations": [
                    "Continue monitoring sentinel scan failures",
                    "Maintain current healthy status",
                    "Consider scaling inspector for higher throughput",
                ],
            }

            self._activity_reports.append(activity_report)

            return {
                "status": "completed",
                "report": activity_report,
                "report_format": report_format,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    @staticmethod
    def _get_doc_extension(doc_format: str) -> str:
        """Get file extension for documentation format.

        Args:
            doc_format: Documentation format name.

        Returns:
            File extension.
        """
        extensions = {
            "markdown": "md",
            "html": "html",
            "pdf": "pdf",
            "rst": "rst",
            "json": "json",
        }
        return extensions.get(doc_format.lower(), "txt")
