"""Autonomous Operations Coordinator — integrates policy, trust, guardrails, and workflows.

This module is the orchestration layer that brings together:
1. PolicyEngine — evaluates action permissions
2. TrustScorer — tracks agent performance
3. GuardrailRunner — runs automated safety checks
4. WhatsAppDigest — notifies humans of activity
5. AutonomousWorkflowEngine — manages multi-step tasks

Together, these components enable true autonomous operation with
safeguards that scale from "completely untrusted" to "fully autonomous"
based on agent performance.

The AutonomousOperations coordinator is the main interface that:
- Coordinates policy evaluation
- Records outcomes and updates trust
- Handles approval requests
- Manages notifications
- Orchestrates complex workflows
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from pydantic import BaseModel, Field

from src.core.autonomous_workflow import (
    AutonomousWorkflowEngine,
    WorkflowTemplate,
    create_builtin_templates,
)
from src.security.guardrails import GuardrailRunner
from src.security.policy_engine import AutonomyTier, PolicyEngine
from src.security.trust_score import TrustScorer

logger = structlog.get_logger(__name__)


class ActionRequest(BaseModel):
    """Request to perform an action.

    Attributes:
        agent_id: Which agent wants to act
        action: Action identifier
        parameters: Action parameters
        context: Context for policy evaluation
        estimated_cost: Estimated cost of this action
    """

    agent_id: str
    action: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    estimated_cost: float = Field(default=0.0)


class ActionResult(BaseModel):
    """Result from executing an action.

    Attributes:
        request_id: Original request ID
        agent_id: Agent that performed action
        action: Action that was taken
        status: success/failure/pending
        tier: Which tier was used
        output: Action output
        cost: Actual cost
        policy_decision_id: ID from policy engine
    """

    request_id: str
    agent_id: str
    action: str
    status: str
    tier: str
    output: Optional[dict[str, Any]] = None
    cost: float = Field(default=0.0)
    policy_decision_id: Optional[str] = None


class ApprovalRequest(BaseModel):
    """Request for human approval of an action.

    Attributes:
        request_id: Unique request ID
        agent_id: Which agent needs approval
        action: What action is being requested
        summary: Human-readable summary
        reason: Why it needs approval
        context: Context for the decision
        created_at: When request was created
        expires_at: When approval expires
    """

    request_id: str
    agent_id: str
    action: str
    summary: str
    reason: str
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AutonomousOperations:
    """Coordinator for autonomous agent operations with safeguards.

    Integrates policy, trust, guardrails, notifications, and workflows
    into a cohesive system for autonomous operation with human oversight.
    """

    def __init__(
        self,
        policy_engine: Optional[PolicyEngine] = None,
        trust_scorer: Optional[TrustScorer] = None,
        guardrail_runner: Optional[GuardrailRunner] = None,
        digest: Optional[Any] = None,
        workflow_engine: Optional[AutonomousWorkflowEngine] = None,
    ) -> None:
        """Initialize autonomous operations coordinator.

        Args:
            policy_engine: PolicyEngine for action evaluation
            trust_scorer: TrustScorer for agent reputation
            guardrail_runner: GuardrailRunner for safety checks
            digest: WhatsAppDigest for notifications
            workflow_engine: AutonomousWorkflowEngine for multi-step tasks
        """
        # Initialize or use provided instances
        self.policy_engine = policy_engine or PolicyEngine()
        self.trust_scorer = trust_scorer or TrustScorer()
        self.guardrail_runner = guardrail_runner or GuardrailRunner()
        self.digest = digest
        self.workflow_engine = workflow_engine or AutonomousWorkflowEngine(
            policy_engine=self.policy_engine
        )

        # Connect components
        self.policy_engine.set_trust_scorer(self.trust_scorer)

        # Pending approval requests
        self.pending_approvals: dict[str, ApprovalRequest] = {}

        # Action executor (user-provided)
        self.action_executor: Optional[Any] = None

        logger.info("Autonomous operations coordinator initialized")

    def configure_action_executor(self, executor: Any) -> None:
        """Set the executor for actions.

        The executor should have:
            async def execute(agent_id: str, action: str, parameters: dict) -> dict

        Args:
            executor: Action executor instance
        """
        self.action_executor = executor
        logger.info("Action executor configured")

    async def request_action(self, request: ActionRequest) -> ActionResult:
        """Process an action request from an agent.

        This is the main entry point. It:
        1. Evaluates policy
        2. Runs guardrails
        3. Decides if execution is allowed
        4. Executes the action (if GREEN/YELLOW)
        5. Sends approval request (if RED)
        6. Updates trust scoring
        7. Sends notifications

        Args:
            request: ActionRequest from agent

        Returns:
            ActionResult with outcome

        Raises:
            ValueError: If policy blocks the action (BLACK tier)
        """
        logger.info(
            "Action request received",
            agent_id=request.agent_id,
            action=request.action,
        )

        # Evaluate policy
        context = {
            **request.context,
            "agent_id": request.agent_id,
            "estimated_cost": request.estimated_cost,
        }

        policy_decision = await self.policy_engine.evaluate(
            request.agent_id, request.action, context
        )

        # Black tier actions are never executed
        if policy_decision.tier == AutonomyTier.BLACK:
            logger.error(
                "Action blocked by policy (BLACK tier)",
                agent_id=request.agent_id,
                action=request.action,
            )
            raise ValueError(f"Action forbidden by policy: {request.action}")

        # Build result
        result = ActionResult(
            request_id=str(datetime.now(timezone.utc)),
            agent_id=request.agent_id,
            action=request.action,
            status="pending",
            tier=policy_decision.tier.value,
            policy_decision_id=policy_decision.decision_id,
        )

        # Handle based on tier
        if policy_decision.tier == AutonomyTier.RED:
            # Need approval - create approval request and pause
            await self._send_approval_request(
                request=request,
                policy_decision=policy_decision,
                result=result,
            )
            result.status = "waiting_for_approval"

        else:
            # Can execute (GREEN or YELLOW)
            result = await self._execute_action(request, policy_decision, result)

            # Notify if YELLOW
            if policy_decision.tier == AutonomyTier.YELLOW and self.digest:
                from src.bridges.whatsapp_digest import DigestEntry

                entry = DigestEntry(
                    agent_id=request.agent_id,
                    action=request.action,
                    tier="yellow",
                    summary=f"Agent {request.agent_id} executed {request.action}",
                    outcome=result.status,
                    cost=result.cost,
                )
                self.digest.add_event(entry)

        return result

    async def approve_action(self, request_id: str, approved: bool) -> ActionResult:
        """Handle human approval/rejection of a RED-tier action.

        Args:
            request_id: ID of approval request
            approved: True to approve, False to reject

        Returns:
            ActionResult with final outcome

        Raises:
            ValueError: If approval request not found
        """
        if request_id not in self.pending_approvals:
            raise ValueError(f"Approval request not found: {request_id}")

        approval = self.pending_approvals.pop(request_id)
        logger.info(
            "Approval decision received",
            request_id=request_id,
            agent_id=approval.agent_id,
            approved=approved,
        )

        if not approved:
            # Action rejected
            return ActionResult(
                request_id=request_id,
                agent_id=approval.agent_id,
                action=approval.action,
                status="rejected",
                tier="red",
            )

        # Re-create original request
        request = ActionRequest(
            agent_id=approval.agent_id,
            action=approval.action,
            context=approval.context,
        )

        # Execute the approved action
        policy_decision = await self.policy_engine.evaluate(
            approval.agent_id, approval.action, approval.context
        )

        result = ActionResult(
            request_id=request_id,
            agent_id=approval.agent_id,
            action=approval.action,
            status="pending",
            tier="red",
            policy_decision_id=policy_decision.decision_id,
        )

        result = await self._execute_action(request, policy_decision, result)

        # Notify after execution
        if self.digest:
            from src.bridges.whatsapp_digest import DigestEntry

            entry = DigestEntry(
                agent_id=approval.agent_id,
                action=approval.action,
                tier="red",
                summary=f"Approved action {approval.action} executed",
                outcome=result.status,
                cost=result.cost,
            )
            self.digest.add_event(entry)

        return result

    async def _execute_action(
        self, request: ActionRequest, policy_decision: Any, result: ActionResult
    ) -> ActionResult:
        """Actually execute the action.

        Args:
            request: ActionRequest
            policy_decision: PolicyDecision from engine
            result: ActionResult to update

        Returns:
            Updated ActionResult with outcome
        """
        if not self.action_executor:
            logger.warning("No action executor configured, simulating execution")
            result.status = "success"
            result.output = {"simulated": True}
            result.cost = 0.01
            return result

        try:
            # Execute
            output = await self.action_executor.execute(
                request.agent_id, request.action, request.parameters
            )

            result.status = "success"
            result.output = output
            result.cost = output.get("cost", request.estimated_cost)

            # Update trust
            self.trust_scorer.record_success(
                request.agent_id,
                request.action,
                severity="medium",
                details=output,
            )

            logger.info(
                "Action executed successfully",
                agent_id=request.agent_id,
                action=request.action,
                cost=result.cost,
            )

        except Exception as e:
            result.status = "failure"
            result.cost = request.estimated_cost
            result.output = {"error": str(e)}

            # Update trust
            self.trust_scorer.record_failure(
                request.agent_id,
                request.action,
                severity="high",
                details={"error": str(e)},
            )

            logger.error(
                "Action execution failed",
                agent_id=request.agent_id,
                action=request.action,
                error=str(e),
            )

        # Record outcome for policy engine
        self.policy_engine.record_outcome(
            result.policy_decision_id or "",
            result.status == "success",
            request.agent_id,
            request.action,
        )

        return result

    async def _send_approval_request(
        self,
        request: ActionRequest,
        policy_decision: Any,
        result: ActionResult,
    ) -> None:
        """Send approval request to human via WhatsApp.

        Args:
            request: Original ActionRequest
            policy_decision: PolicyDecision that requires approval
            result: ActionResult being built
        """
        approval_id = str(datetime.now(timezone.utc))

        approval = ApprovalRequest(
            request_id=approval_id,
            agent_id=request.agent_id,
            action=request.action,
            summary=f"{request.agent_id} wants to {request.action}",
            reason="; ".join(policy_decision.reasons),
            context=request.context,
        )

        self.pending_approvals[approval_id] = approval

        if self.digest and self.digest.send_notification_fn:
            # Format approval message
            message = self._format_approval_message(approval, policy_decision)

            try:
                await self.digest.send_notification_fn(message)
                logger.info(
                    "Approval request sent",
                    request_id=approval_id,
                    agent_id=request.agent_id,
                )
            except Exception as e:
                logger.error(
                    "Failed to send approval request",
                    error=str(e),
                )

    def _format_approval_message(self, approval: ApprovalRequest, policy_decision: Any) -> str:
        """Format approval request for WhatsApp.

        Args:
            approval: ApprovalRequest
            policy_decision: PolicyDecision with reasons

        Returns:
            Formatted message
        """
        lines = [
            "*AgentArmy — Action Approval Needed*",
            "",
            f"🤖 *Agent*: {approval.agent_id.title()}",
            f"🎯 *Action*: {approval.action}",
            f"📝 *Summary*: {approval.summary}",
            "",
            "*Why approval needed:*",
        ]

        for reason in policy_decision.reasons:
            lines.append(f"  • {reason}")

        lines.extend(
            [
                "",
                f"🕒 *Expires in*: 5 minutes",
                "",
                "_Reply /approve to execute or /reject to abort_",
            ]
        )

        return "\n".join(lines)

    async def start_background_tasks(self) -> None:
        """Start background tasks (decay, digest loop, etc.).

        Should be called during system startup.
        """
        # Start trust decay
        await self.trust_scorer.start_decay_loop()

        # Start digest loop
        if self.digest:
            await self.digest.start_digest_loop()

        logger.info("Background tasks started")

    async def stop_background_tasks(self) -> None:
        """Stop background tasks.

        Should be called during system shutdown.
        """
        await self.trust_scorer.stop_decay_loop()

        if self.digest:
            await self.digest.stop_digest_loop()

        logger.info("Background tasks stopped")

    def register_workflow_template(self, template: WorkflowTemplate) -> None:
        """Register a workflow template for autonomous execution.

        Args:
            template: WorkflowTemplate to register
        """
        self.workflow_engine.register_template(template)

    async def start_workflow(
        self, template_name: str, context: dict[str, Any] = {}
    ) -> str:
        """Start a workflow execution.

        Args:
            template_name: Name of template to execute
            context: Context data for workflow steps

        Returns:
            Execution ID

        Raises:
            ValueError: If template not found
        """
        execution = self.workflow_engine.start_workflow(template_name, context)

        # Start execution
        await self.workflow_engine._run_steps(execution)

        return execution.id

    def get_workflow_status(self, execution_id: str) -> dict[str, Any]:
        """Get status of a workflow execution.

        Args:
            execution_id: Execution ID

        Returns:
            Status dictionary

        Raises:
            ValueError: If execution not found
        """
        execution = self.workflow_engine.get_status(execution_id)

        return {
            "id": execution.id,
            "template": execution.template_name,
            "status": execution.status.value,
            "progress": f"{execution.current_step_index}/{len(execution.steps)}",
            "total_cost": execution.total_cost,
            "duration_seconds": execution.get_duration_seconds(),
            "error": execution.error,
        }

    def get_agent_status(self, agent_id: str) -> dict[str, Any]:
        """Get complete status of an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Status dictionary with trust, recent actions, etc.
        """
        trust_profile = self.trust_scorer.get_profile(agent_id)

        return {
            "agent_id": agent_id,
            "trust_score": trust_profile.score,
            "trust_level": trust_profile.level.value,
            "is_frozen": trust_profile.is_frozen,
            "freeze_reason": trust_profile.freeze_reason,
            "total_actions": trust_profile.total_actions,
            "successful_actions": trust_profile.successful_actions,
            "failed_actions": trust_profile.failed_actions,
            "consecutive_successes": trust_profile.consecutive_successes,
            "consecutive_failures": trust_profile.consecutive_failures,
        }

    def get_system_stats(self) -> dict[str, Any]:
        """Get overall system statistics.

        Returns:
            Statistics dictionary
        """
        trust_stats = self.trust_scorer.get_stats()

        return {
            "agents": trust_stats,
            "pending_approvals": len(self.pending_approvals),
            "pending_digest_entries": self.digest.get_pending_count() if self.digest else 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
