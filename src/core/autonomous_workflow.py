"""Autonomous Workflow Engine — end-to-end task execution without human intervention.

Defines workflow templates that agents can execute completely autonomously.
Each step has its own policy evaluation, so the workflow self-governs.
If any step hits a RED policy, the workflow pauses AT THAT POINT (not from start).

Think: CI/CD pipelines but for any agent task.

A workflow is a DAG of steps where each step:
1. Depends on completion of prior steps
2. Has its own policy evaluation
3. Can retry on failure
4. Can pause for human approval
5. Can be skipped on failure

Workflows support complex patterns:
- Sequential steps (A → B → C)
- Parallel steps (A → [B, C] → D)
- Conditional execution (based on step results)
- Failure handling (retry, skip, abort, escalate)
- Cost tracking (sum of all step costs)
- Timeout management (overall and per-step)
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow execution status."""

    PENDING = "pending"  # Waiting to start
    RUNNING = "running"  # Steps executing
    PAUSED_FOR_APPROVAL = "paused_for_approval"  # Waiting for RED approval
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"  # Failed, cannot continue
    ABORTED = "aborted"  # Aborted by user or system


class StepStatus(str, Enum):
    """Individual step execution status."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ABORTED = "aborted"


class WorkflowStep(BaseModel):
    """Definition of a single step in a workflow.

    Attributes:
        id: Unique step identifier
        name: Human-readable step name
        agent_role: Which agent type handles this ("builder", "sentinel", etc.)
        action: Action identifier (e.g., "git.create_pull_request")
        parameters: Parameters to pass to the action
        depends_on: List of step IDs that must complete first
        on_failure: What to do if step fails (retry/skip/abort/escalate)
        max_retries: Maximum number of retries
        timeout_seconds: Step timeout (0 = no timeout)
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    agent_role: str
    action: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    on_failure: str = Field(default="abort")  # retry/skip/abort/escalate
    max_retries: int = Field(default=1)
    timeout_seconds: int = Field(default=0)

    def is_ready(self, completed_steps: set[str]) -> bool:
        """Check if all dependencies are satisfied.

        Args:
            completed_steps: Set of step IDs that have completed

        Returns:
            True if all dependencies are met
        """
        return all(dep in completed_steps for dep in self.depends_on)


class StepResult(BaseModel):
    """Result from executing a single step.

    Attributes:
        step_id: Which step this is for
        status: Step status (completed, failed, skipped, etc.)
        output: Step output data
        error: Error message if failed
        cost: Estimated/actual cost of this step
        duration_seconds: How long step took
        retry_count: How many retries occurred
        approval_request_id: ID if waiting for approval
    """

    step_id: str
    status: StepStatus
    output: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    cost: float = Field(default=0.0)
    duration_seconds: float = Field(default=0.0)
    retry_count: int = Field(default=0)
    approval_request_id: Optional[str] = None


class WorkflowExecution(BaseModel):
    """Execution instance of a workflow.

    Attributes:
        id: Unique execution ID
        template_name: Name of template this is executing
        status: Current workflow status
        steps: List of step definitions
        step_results: Dict of step_id -> StepResult
        current_step_index: Index of step currently executing
        paused_at_step: Which step we're paused at (if any)
        approval_request_id: ID of approval request (if waiting)
        started_at: When execution started
        completed_at: When execution completed (if done)
        total_cost: Sum of all step costs
        error: Error message if failed
        context: Context data available to all steps
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    template_name: str
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    steps: list[WorkflowStep]
    step_results: dict[str, StepResult] = Field(default_factory=dict)
    current_step_index: int = Field(default=0)
    paused_at_step: Optional[str] = None
    approval_request_id: Optional[str] = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    total_cost: float = Field(default=0.0)
    error: Optional[str] = None
    context: dict[str, Any] = Field(default_factory=dict)

    def get_completed_step_ids(self) -> set[str]:
        """Get set of completed step IDs.

        Returns:
            Set of step IDs with status COMPLETED
        """
        return {
            sid
            for sid, result in self.step_results.items()
            if result.status == StepStatus.COMPLETED
        }

    def get_duration_seconds(self) -> float:
        """Get total execution duration.

        Returns:
            Seconds elapsed (or to completion if done)
        """
        end = self.completed_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()


class WorkflowTemplate(BaseModel):
    """Reusable workflow template.

    Attributes:
        name: Unique template name
        description: Human-readable description
        steps: List of step definitions
        max_total_cost: Budget cap for this workflow
        notification_preference: none/digest/realtime
    """

    name: str
    description: str
    steps: list[WorkflowStep]
    max_total_cost: float = Field(default=100.0)
    notification_preference: str = Field(default="digest")


class AutonomousWorkflowEngine:
    """Orchestrates autonomous workflow execution.

    Manages workflow templates, executions, and step-by-step policy evaluation.
    """

    def __init__(self, policy_engine: Optional[Any] = None) -> None:
        """Initialize workflow engine.

        Args:
            policy_engine: PolicyEngine instance for evaluating steps
        """
        self.policy_engine = policy_engine
        self.templates: dict[str, WorkflowTemplate] = {}
        self.executions: dict[str, WorkflowExecution] = {}

        logger.info("Autonomous workflow engine initialized")

    def register_template(self, template: WorkflowTemplate) -> None:
        """Register a workflow template.

        Args:
            template: WorkflowTemplate to register
        """
        self.templates[template.name] = template
        logger.info(
            "Workflow template registered",
            name=template.name,
            step_count=len(template.steps),
        )

    def start_workflow(
        self, template_name: str, context: dict[str, Any] = {}
    ) -> WorkflowExecution:
        """Start a new workflow execution.

        Args:
            template_name: Name of template to execute
            context: Context data for all steps

        Returns:
            WorkflowExecution instance

        Raises:
            ValueError: If template not found
        """
        if template_name not in self.templates:
            raise ValueError(f"Template not found: {template_name}")

        template = self.templates[template_name]

        # Create execution
        execution = WorkflowExecution(
            template_name=template_name,
            steps=template.steps.copy(),
            context=context,
        )

        self.executions[execution.id] = execution

        logger.info(
            "Workflow execution started",
            execution_id=execution.id,
            template=template_name,
            steps=len(execution.steps),
        )

        return execution

    def get_status(self, execution_id: str) -> WorkflowExecution:
        """Get current status of a workflow execution.

        Args:
            execution_id: Execution ID

        Returns:
            WorkflowExecution with current status

        Raises:
            ValueError: If execution not found
        """
        if execution_id not in self.executions:
            raise ValueError(f"Execution not found: {execution_id}")
        return self.executions[execution_id]

    async def resume_workflow(self, execution_id: str, approval: bool) -> WorkflowExecution:
        """Resume a workflow that was paused waiting for approval.

        Args:
            execution_id: Execution ID
            approval: True to approve and continue, False to abort

        Returns:
            Updated WorkflowExecution

        Raises:
            ValueError: If execution not found or not paused
        """
        if execution_id not in self.executions:
            raise ValueError(f"Execution not found: {execution_id}")

        execution = self.executions[execution_id]

        if execution.status != WorkflowStatus.PAUSED_FOR_APPROVAL:
            raise ValueError(f"Execution not paused: {execution_id}")

        if not approval:
            execution.status = WorkflowStatus.ABORTED
            execution.error = "Workflow aborted by user"
            logger.info("Workflow aborted", execution_id=execution_id)
            return execution

        # Continue execution
        await self._run_steps(execution)
        return execution

    async def _run_steps(self, execution: WorkflowExecution) -> None:
        """Run workflow steps sequentially.

        Starts from current_step_index, executing steps in order.
        Respects dependencies and policy decisions.

        Args:
            execution: WorkflowExecution to continue
        """
        execution.status = WorkflowStatus.RUNNING

        while execution.current_step_index < len(execution.steps):
            step = execution.steps[execution.current_step_index]

            # Check dependencies
            completed = execution.get_completed_step_ids()
            if not step.is_ready(completed):
                logger.debug(f"Step {step.id} waiting for dependencies")
                await asyncio.sleep(0.1)
                continue

            # Execute step
            result = await self._execute_step(execution, step)
            execution.step_results[step.id] = result

            # Update costs
            execution.total_cost += result.cost

            if execution.total_cost > execution.steps[0]:  # Placeholder - use template max
                logger.warning(
                    "Workflow cost budget exceeded",
                    execution_id=execution.id,
                    cost=execution.total_cost,
                )

            # Handle results
            if result.status == StepStatus.COMPLETED:
                execution.current_step_index += 1
            elif result.status == StepStatus.WAITING_FOR_APPROVAL:
                execution.status = WorkflowStatus.PAUSED_FOR_APPROVAL
                execution.paused_at_step = step.id
                execution.approval_request_id = result.approval_request_id
                logger.info(
                    "Workflow paused for approval",
                    execution_id=execution.id,
                    step_id=step.id,
                )
                return
            elif result.status == StepStatus.FAILED:
                if step.on_failure == "skip":
                    execution.current_step_index += 1
                elif step.on_failure == "abort":
                    execution.status = WorkflowStatus.FAILED
                    execution.error = f"Step {step.id} failed: {result.error}"
                    execution.completed_at = datetime.now(timezone.utc)
                    logger.error(
                        "Workflow failed",
                        execution_id=execution.id,
                        step_id=step.id,
                        error=result.error,
                    )
                    return
                elif step.on_failure == "retry":
                    if result.retry_count < step.max_retries:
                        logger.info(
                            "Retrying step",
                            step_id=step.id,
                            attempt=result.retry_count + 1,
                        )
                        await asyncio.sleep(1)  # Back off before retry
                    else:
                        execution.status = WorkflowStatus.FAILED
                        execution.error = f"Step {step.id} exhausted retries"
                        execution.completed_at = datetime.now(timezone.utc)
                        return
                else:
                    execution.current_step_index += 1
            else:
                execution.current_step_index += 1

        # All steps completed
        execution.status = WorkflowStatus.COMPLETED
        execution.completed_at = datetime.now(timezone.utc)
        logger.info(
            "Workflow completed",
            execution_id=execution.id,
            duration_seconds=execution.get_duration_seconds(),
            total_cost=execution.total_cost,
        )

    async def _execute_step(
        self, execution: WorkflowExecution, step: WorkflowStep
    ) -> StepResult:
        """Execute a single workflow step.

        Evaluates policy, runs guardrails, and executes the step action.

        Args:
            execution: Workflow execution context
            step: Step to execute

        Returns:
            StepResult with outcome
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Prepare context for policy evaluation
            context = {
                **execution.context,
                **step.parameters,
                "agent_id": step.agent_role,
                "execution_id": execution.id,
            }

            # Evaluate policy
            if self.policy_engine:
                decision = await self.policy_engine.evaluate(
                    step.agent_role, step.action, context
                )

                if decision.tier == "black":
                    return StepResult(
                        step_id=step.id,
                        status=StepStatus.FAILED,
                        error="Action forbidden by policy (BLACK tier)",
                    )

                if decision.requires_approval:
                    approval_request_id = str(uuid.uuid4())
                    return StepResult(
                        step_id=step.id,
                        status=StepStatus.WAITING_FOR_APPROVAL,
                        approval_request_id=approval_request_id,
                    )
            else:
                logger.warning("No policy engine set for workflow")

            # Execute step
            # In real implementation, this would call the agent
            output = await self._call_agent(step.agent_role, step.action, step.parameters)

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            return StepResult(
                step_id=step.id,
                status=StepStatus.COMPLETED,
                output=output,
                cost=output.get("cost", 0.0) if isinstance(output, dict) else 0.0,
                duration_seconds=duration,
            )

        except asyncio.TimeoutError:
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error=f"Step timeout after {step.timeout_seconds} seconds",
            )
        except Exception as e:
            logger.error(
                "Step execution failed",
                step_id=step.id,
                error=str(e),
            )
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error=str(e),
            )

    async def _call_agent(
        self, agent_role: str, action: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Call an agent to execute a step.

        This is a placeholder. Real implementation would call the agent system.

        Args:
            agent_role: Agent type
            action: Action to perform
            parameters: Action parameters

        Returns:
            Action output
        """
        # Simulate execution
        await asyncio.sleep(0.1)
        return {"status": "success", "cost": 0.01}


def create_builtin_templates() -> dict[str, WorkflowTemplate]:
    """Create built-in workflow templates.

    Returns:
        Dictionary of template name -> WorkflowTemplate
    """
    templates = {}

    # Feature development workflow
    feature_template = WorkflowTemplate(
        name="feature_development",
        description="Complete workflow for adding a new feature",
        steps=[
            WorkflowStep(
                id="research",
                name="Research and design",
                agent_role="sentinel",
                action="research.design_feature",
                parameters={"feature_description": ""},
            ),
            WorkflowStep(
                id="implement",
                name="Implement feature",
                agent_role="builder",
                action="code.write",
                depends_on=["research"],
            ),
            WorkflowStep(
                id="test",
                name="Write and run tests",
                agent_role="builder",
                action="test.run_suite",
                depends_on=["implement"],
            ),
            WorkflowStep(
                id="review",
                name="Code review",
                agent_role="inspector",
                action="code.review",
                depends_on=["test"],
            ),
            WorkflowStep(
                id="create_pr",
                name="Create pull request",
                agent_role="builder",
                action="git.create_pull_request",
                depends_on=["review"],
            ),
        ],
    )
    templates["feature_development"] = feature_template

    # Security scan workflow
    security_template = WorkflowTemplate(
        name="security_scan",
        description="Full security audit of codebase",
        steps=[
            WorkflowStep(
                id="dep_scan",
                name="Dependency vulnerability scan",
                agent_role="sentinel",
                action="scan.dependencies",
            ),
            WorkflowStep(
                id="secret_scan",
                name="Secret/credential scan",
                agent_role="sentinel",
                action="scan.secrets",
            ),
            WorkflowStep(
                id="code_analysis",
                name="Static code analysis",
                agent_role="inspector",
                action="analyze.code_quality",
            ),
            WorkflowStep(
                id="report",
                name="Generate security report",
                agent_role="scribe",
                action="docs.generate_report",
                depends_on=["dep_scan", "secret_scan", "code_analysis"],
            ),
        ],
    )
    templates["security_scan"] = security_template

    # Bug fix workflow
    bugfix_template = WorkflowTemplate(
        name="bug_fix",
        description="Workflow for investigating and fixing bugs",
        steps=[
            WorkflowStep(
                id="investigate",
                name="Investigate issue",
                agent_role="sentinel",
                action="research.investigate_bug",
            ),
            WorkflowStep(
                id="reproduce",
                name="Reproduce bug",
                agent_role="builder",
                action="test.reproduce_bug",
                depends_on=["investigate"],
            ),
            WorkflowStep(
                id="fix",
                name="Implement fix",
                agent_role="builder",
                action="code.write",
                depends_on=["reproduce"],
            ),
            WorkflowStep(
                id="test",
                name="Test fix",
                agent_role="builder",
                action="test.run_suite",
                depends_on=["fix"],
            ),
            WorkflowStep(
                id="create_pr",
                name="Create PR with fix",
                agent_role="builder",
                action="git.create_pull_request",
                depends_on=["test"],
            ),
        ],
    )
    templates["bug_fix"] = bugfix_template

    # Staging deployment workflow
    staging_template = WorkflowTemplate(
        name="deploy_staging",
        description="Deploy to staging environment",
        steps=[
            WorkflowStep(
                id="test",
                name="Run full test suite",
                agent_role="builder",
                action="test.run_suite",
            ),
            WorkflowStep(
                id="build",
                name="Build artifacts",
                agent_role="builder",
                action="deploy.build",
                depends_on=["test"],
            ),
            WorkflowStep(
                id="deploy",
                name="Deploy to staging",
                agent_role="devops",
                action="deploy.staging",
                depends_on=["build"],
            ),
            WorkflowStep(
                id="smoke_test",
                name="Smoke tests on staging",
                agent_role="inspector",
                action="test.smoke",
                depends_on=["deploy"],
            ),
        ],
    )
    templates["deploy_staging"] = staging_template

    # Production deployment workflow
    production_template = WorkflowTemplate(
        name="deploy_production",
        description="Deploy to production (with approval)",
        steps=[
            WorkflowStep(
                id="test",
                name="Run full test suite",
                agent_role="builder",
                action="test.run_suite",
            ),
            WorkflowStep(
                id="build",
                name="Build artifacts",
                agent_role="builder",
                action="deploy.build",
                depends_on=["test"],
            ),
            WorkflowStep(
                id="deploy_staging",
                name="Deploy to staging first",
                agent_role="devops",
                action="deploy.staging",
                depends_on=["build"],
            ),
            WorkflowStep(
                id="smoke_test",
                name="Smoke tests on staging",
                agent_role="inspector",
                action="test.smoke",
                depends_on=["deploy_staging"],
            ),
            WorkflowStep(
                id="deploy_prod",
                name="Deploy to production",
                agent_role="devops",
                action="deploy.production",
                depends_on=["smoke_test"],
                on_failure="abort",
            ),
            WorkflowStep(
                id="verify",
                name="Verify production health",
                agent_role="sentinel",
                action="monitor.verify_health",
                depends_on=["deploy_prod"],
            ),
        ],
    )
    templates["deploy_production"] = production_template

    return templates
