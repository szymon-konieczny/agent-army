"""Policy Engine — tiered autonomy for agent operations.

Instead of asking the human for every action, the policy engine classifies
each operation into one of four autonomy tiers:

    GREEN  — Fully autonomous. Agent executes, action is logged.
    YELLOW — Autonomous + notify. Agent executes, human gets a digest.
    RED    — Requires approval. Agent pauses, asks human via WhatsApp.
    BLACK  — Forbidden. Never executed, even with approval.

The tier is determined by combining:
    1. Static rules (from YAML policy files)
    2. Agent trust score (earned through successful operations)
    3. Context risk assessment (what's being changed, in what environment)
    4. Guardrail results (automated pre-checks)

The policy engine is the decision-making heart of autonomous operations.
Every agent action flows through evaluate() before execution.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import structlog
import yaml
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class AutonomyTier(str, Enum):
    """Autonomy levels for agent actions.

    GREEN:  Fully autonomous. Executes immediately, logged.
    YELLOW: Autonomous + notify. Executes, but human gets digest notification.
    RED:    Approval required. Agent pauses, sends WhatsApp request to human.
    BLACK:  Forbidden. Never executed, terminates workflow.
    """

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    BLACK = "black"


class PolicyCondition(BaseModel):
    """Condition that overrides tier based on context.

    Examples:
        - If environment == "production", upgrade from YELLOW to RED
        - If file_path matches "secrets/*", downgrade to BLACK
        - If estimated_cost > 10.0, upgrade from YELLOW to RED

    Attributes:
        field: Context field name (e.g., "environment", "file_path", "cost_estimate")
        operator: Comparison operator (eq, ne, gt, lt, gte, lte, contains, matches)
        value: Value to compare against
        override_tier: Tier to apply if condition is true
    """

    field: str = Field(..., description="Context field name to check")
    operator: str = Field(..., description="Comparison operator")
    value: Any = Field(..., description="Value to compare against")
    override_tier: AutonomyTier = Field(
        ..., description="Tier to apply if condition matches"
    )

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Check if this condition is met in the given context.

        Args:
            context: Context dictionary with field values

        Returns:
            True if condition is satisfied, False otherwise
        """
        if self.field not in context:
            return False

        context_value = context[self.field]

        if self.operator == "eq":
            return context_value == self.value
        elif self.operator == "ne":
            return context_value != self.value
        elif self.operator == "gt":
            return float(context_value) > float(self.value)
        elif self.operator == "lt":
            return float(context_value) < float(self.value)
        elif self.operator == "gte":
            return float(context_value) >= float(self.value)
        elif self.operator == "lte":
            return float(context_value) <= float(self.value)
        elif self.operator == "contains":
            return str(self.value) in str(context_value)
        elif self.operator == "matches":
            pattern = re.compile(self.value)
            return bool(pattern.search(str(context_value)))
        else:
            logger.warning(
                "Unknown operator",
                operator=self.operator,
                field=self.field,
            )
            return False


class PolicyRule(BaseModel):
    """Declares autonomy policy for a class of actions.

    A rule matches actions via glob patterns and defines their default tier.
    Conditions can override the tier based on context (environment, cost, etc.).

    Attributes:
        action_pattern: Glob pattern matching action names (e.g., "git.*", "deploy.*")
        default_tier: Default autonomy tier for matching actions
        conditions: List of conditions that can override the tier
        description: Human-readable description of this rule
    """

    action_pattern: str = Field(..., description="Glob pattern for action names")
    default_tier: AutonomyTier = Field(..., description="Default tier for this action")
    conditions: list[PolicyCondition] = Field(
        default_factory=list, description="Conditions that override tier"
    )
    description: str = Field(..., description="Human-readable rule description")

    def matches(self, action: str) -> bool:
        """Check if action matches this rule's pattern.

        Args:
            action: Action name to check

        Returns:
            True if action matches the pattern
        """
        pattern = self.action_pattern.replace(".", r"\.").replace("*", ".*")
        return bool(re.match(f"^{pattern}$", action))

    def get_tier(self, context: dict[str, Any]) -> AutonomyTier:
        """Determine tier for this action given context.

        Evaluates conditions in order. First matching condition's tier is used.
        If no conditions match, returns default_tier.

        Args:
            context: Context dictionary for condition evaluation

        Returns:
            Appropriate autonomy tier
        """
        for condition in self.conditions:
            if condition.evaluate(context):
                return condition.override_tier
        return self.default_tier


class GuardrailResult(BaseModel):
    """Result of running a single guardrail check.

    Attributes:
        name: Guardrail name
        passed: Whether guardrail passed
        severity: severity level (info, warning, error, critical)
        message: Human-readable result message
        details: Additional structured details
        execution_time_ms: How long the guardrail took to run
    """

    name: str
    passed: bool
    severity: str = Field(default="info")
    message: str
    details: Optional[dict[str, Any]] = None
    execution_time_ms: float


class PolicyDecision(BaseModel):
    """The result of policy evaluation for an agent action.

    This is passed to the executor to determine what actually happens.
    The executor checks requires_approval and either runs (GREEN/YELLOW)
    or pauses for human approval (RED) or terminates (BLACK).

    Attributes:
        tier: Determined autonomy tier
        action: The action being evaluated
        agent_id: Which agent is taking this action
        reasons: List of strings explaining why this tier was chosen
        guardrail_results: Results from guardrail checks
        trust_score: Agent's current trust score (affects tier elevation)
        requires_approval: Computed from tier (RED=True, others=False)
        auto_approved: True if tier is GREEN/YELLOW
        timestamp: When decision was made
        decision_id: Unique ID for tracking
    """

    tier: AutonomyTier
    action: str
    agent_id: str
    reasons: list[str] = Field(default_factory=list)
    guardrail_results: list[GuardrailResult] = Field(default_factory=list)
    trust_score: float
    requires_approval: bool = Field(default=False)
    auto_approved: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    decision_id: str = Field(default_factory=lambda: str(datetime.now(timezone.utc)))

    def model_post_init(self, __context: Any) -> None:
        """Compute derived fields after initialization."""
        self.requires_approval = self.tier == AutonomyTier.RED
        self.auto_approved = self.tier in (AutonomyTier.GREEN, AutonomyTier.YELLOW)


class PolicyEngine:
    """Evaluates agent actions against policies and trust scores.

    The policy engine is stateless for policy evaluation, but maintains
    guardrail registry and can integrate with trust scorer.
    """

    def __init__(self, default_tier: AutonomyTier = AutonomyTier.YELLOW) -> None:
        """Initialize the policy engine.

        Args:
            default_tier: Fallback tier if no rule matches
        """
        self.default_tier = default_tier
        self.rules: list[PolicyRule] = []
        self.guardrails: dict[str, Callable] = {}
        self.trust_scorer: Optional[Any] = None

        logger.info(
            "Policy engine initialized",
            default_tier=default_tier.value,
        )

    def load_policies(self, yaml_path: str | Path) -> None:
        """Load policies from YAML configuration file.

        The YAML file should have structure:
            tiers:
              green:
                actions: [list of action patterns]
              yellow:
                actions: [list of action patterns]
              red:
                actions: [list of action patterns]
              black:
                actions: [list of action patterns]

        Args:
            yaml_path: Path to YAML policy file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is malformed
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Policy file not found: {yaml_path}")

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        if "tiers" not in config:
            raise ValueError("Policy file must contain 'tiers' section")

        # Load rules from each tier
        for tier_name, tier_config in config["tiers"].items():
            try:
                tier = AutonomyTier(tier_name)
            except ValueError:
                logger.warning(f"Unknown tier: {tier_name}")
                continue

            if isinstance(tier_config, dict) and "actions" in tier_config:
                for action_pattern in tier_config["actions"]:
                    rule = PolicyRule(
                        action_pattern=action_pattern,
                        default_tier=tier,
                        description=f"Default {tier_name} action",
                    )
                    self.rules.append(rule)
                    logger.debug(
                        "Loaded policy rule",
                        pattern=action_pattern,
                        tier=tier.value,
                    )

        logger.info(
            "Policies loaded from YAML",
            path=str(yaml_path),
            rule_count=len(self.rules),
        )

    def load_default_policies(self) -> None:
        """Load sensible default policies for a dev team.

        These policies reflect what a real development team would want:
        - Read operations and tests are GREEN (fully autonomous)
        - Pull requests and staging deploys are YELLOW (execute + notify)
        - Production deploys require approval (RED)
        - Secret exposure is always forbidden (BLACK)
        """
        # GREEN: Fully autonomous operations
        green_rules = [
            PolicyRule(
                action_pattern="git.read*",
                default_tier=AutonomyTier.GREEN,
                description="Reading git repository is always safe",
            ),
            PolicyRule(
                action_pattern="git.create_branch",
                default_tier=AutonomyTier.GREEN,
                description="Creating feature branches is autonomous",
            ),
            PolicyRule(
                action_pattern="git.push_commits",
                default_tier=AutonomyTier.GREEN,
                conditions=[
                    PolicyCondition(
                        field="branch",
                        operator="in",
                        value=["main", "master"],
                        override_tier=AutonomyTier.RED,
                    )
                ],
                description="Push to non-main branches is autonomous",
            ),
            PolicyRule(
                action_pattern="test.run*",
                default_tier=AutonomyTier.GREEN,
                description="Running tests is always safe",
            ),
            PolicyRule(
                action_pattern="test.generate*",
                default_tier=AutonomyTier.GREEN,
                description="Generating test cases is autonomous",
            ),
            PolicyRule(
                action_pattern="code.review",
                default_tier=AutonomyTier.GREEN,
                description="Code review is autonomous",
            ),
            PolicyRule(
                action_pattern="code.write",
                default_tier=AutonomyTier.GREEN,
                conditions=[
                    PolicyCondition(
                        field="file_path",
                        operator="matches",
                        value=r"^(config|secrets|\.env)",
                        override_tier=AutonomyTier.RED,
                    )
                ],
                description="Writing source code is autonomous",
            ),
            PolicyRule(
                action_pattern="docs.generate",
                default_tier=AutonomyTier.GREEN,
                description="Generating documentation is autonomous",
            ),
            PolicyRule(
                action_pattern="research.*",
                default_tier=AutonomyTier.GREEN,
                description="Research tasks are fully autonomous",
            ),
            PolicyRule(
                action_pattern="monitor.*",
                default_tier=AutonomyTier.GREEN,
                description="Monitoring and observability is autonomous",
            ),
            PolicyRule(
                action_pattern="analyze.*",
                default_tier=AutonomyTier.GREEN,
                description="Analysis tasks are autonomous",
            ),
        ]

        # YELLOW: Autonomous but notify
        yellow_rules = [
            PolicyRule(
                action_pattern="git.create_pull_request",
                default_tier=AutonomyTier.YELLOW,
                description="Creating PRs is autonomous + notify",
            ),
            PolicyRule(
                action_pattern="git.merge",
                default_tier=AutonomyTier.YELLOW,
                conditions=[
                    PolicyCondition(
                        field="branch",
                        operator="in",
                        value=["main", "master"],
                        override_tier=AutonomyTier.RED,
                    )
                ],
                description="Merging non-main branches notifies user",
            ),
            PolicyRule(
                action_pattern="dependency.update",
                default_tier=AutonomyTier.YELLOW,
                conditions=[
                    PolicyCondition(
                        field="update_type",
                        operator="eq",
                        value="major",
                        override_tier=AutonomyTier.RED,
                    )
                ],
                description="Minor/patch updates are autonomous + notify",
            ),
            PolicyRule(
                action_pattern="deploy.staging",
                default_tier=AutonomyTier.YELLOW,
                description="Staging deployments are autonomous + notify",
            ),
            PolicyRule(
                action_pattern="scan.security",
                default_tier=AutonomyTier.YELLOW,
                description="Security scans are autonomous + notify",
            ),
        ]

        # RED: Requires approval
        red_rules = [
            PolicyRule(
                action_pattern="deploy.production",
                default_tier=AutonomyTier.RED,
                description="Production deployments require approval",
            ),
            PolicyRule(
                action_pattern="git.merge",
                default_tier=AutonomyTier.RED,
                conditions=[
                    PolicyCondition(
                        field="branch",
                        operator="in",
                        value=["main", "master"],
                        override_tier=AutonomyTier.RED,
                    )
                ],
                description="Merging to main/master requires approval",
            ),
            PolicyRule(
                action_pattern="git.force_push",
                default_tier=AutonomyTier.RED,
                description="Force push requires approval",
            ),
            PolicyRule(
                action_pattern="git.rewrite_history",
                default_tier=AutonomyTier.RED,
                description="History rewrite requires approval",
            ),
            PolicyRule(
                action_pattern="secrets.rotate",
                default_tier=AutonomyTier.RED,
                description="Rotating secrets requires approval",
            ),
            PolicyRule(
                action_pattern="infrastructure.modify",
                default_tier=AutonomyTier.RED,
                description="Infrastructure changes require approval",
            ),
        ]

        # BLACK: Never allowed
        black_rules = [
            PolicyRule(
                action_pattern="secrets.expose",
                default_tier=AutonomyTier.BLACK,
                description="Exposing secrets is absolutely forbidden",
            ),
            PolicyRule(
                action_pattern="data.exfiltrate",
                default_tier=AutonomyTier.BLACK,
                description="Data exfiltration is absolutely forbidden",
            ),
            PolicyRule(
                action_pattern="audit_log.delete",
                default_tier=AutonomyTier.BLACK,
                description="Deleting audit logs is absolutely forbidden",
            ),
            PolicyRule(
                action_pattern="audit_log.modify",
                default_tier=AutonomyTier.BLACK,
                description="Modifying audit logs is absolutely forbidden",
            ),
        ]

        self.rules = green_rules + yellow_rules + red_rules + black_rules
        logger.info(
            "Default policies loaded",
            rule_count=len(self.rules),
            green=len(green_rules),
            yellow=len(yellow_rules),
            red=len(red_rules),
            black=len(black_rules),
        )

    def register_guardrail(self, name: str, fn: Callable) -> None:
        """Register a guardrail check function.

        The function should have signature:
            async def guardrail_fn(action: str, context: dict) -> GuardrailResult

        Args:
            name: Unique name for this guardrail
            fn: Async callable that performs the check
        """
        self.guardrails[name] = fn
        logger.info("Guardrail registered", name=name)

    def set_trust_scorer(self, scorer: Any) -> None:
        """Integrate with a trust scorer for elevating tiers.

        The scorer should have method: can_elevate_tier(agent_id, from_tier) -> bool

        Args:
            scorer: Trust scorer instance
        """
        self.trust_scorer = scorer
        logger.info("Trust scorer integrated")

    async def evaluate(
        self, agent_id: str, action: str, context: dict[str, Any]
    ) -> PolicyDecision:
        """Evaluate a proposed action and return the policy decision.

        This is the main entry point. Given an agent and action, it:
        1. Finds matching policy rule
        2. Gets agent trust score (if scorer available)
        3. Applies context conditions
        4. Runs guardrails
        5. Potentially elevates tier based on trust
        6. Returns full decision

        Args:
            agent_id: ID of agent taking the action
            action: Action identifier (e.g., "git.push_commits")
            context: Context dictionary (environment, file_path, cost, etc.)

        Returns:
            PolicyDecision describing what should happen
        """
        # Find matching rule
        matching_rule = None
        for rule in self.rules:
            if rule.matches(action):
                matching_rule = rule
                break

        if matching_rule is None:
            # No matching rule - use default tier
            tier = self.default_tier
            reasons = [f"No matching rule found, using default tier: {self.default_tier.value}"]
        else:
            # Get tier from rule (including context conditions)
            tier = matching_rule.get_tier(context)
            reasons = [f"Matched rule: {matching_rule.description}"]

        # Get trust score if available
        trust_score = 50.0
        if self.trust_scorer:
            profile = self.trust_scorer.get_profile(agent_id)
            trust_score = profile.score
            reasons.append(f"Agent trust: {profile.level.value} ({trust_score:.0f}/100)")

        # Run guardrails
        guardrail_results = []
        if self.guardrails:
            for name, guardrail_fn in self.guardrails.items():
                try:
                    result = await guardrail_fn(action, context)
                    guardrail_results.append(result)

                    if not result.passed:
                        reasons.append(f"Guardrail {name}: {result.message}")

                        # Critical guardrail failure escalates tier
                        if result.severity == "critical":
                            tier = AutonomyTier.RED
                except Exception as e:
                    logger.error(
                        "Guardrail execution failed",
                        name=name,
                        error=str(e),
                        agent_id=agent_id,
                        action=action,
                    )

        # Trust-based tier elevation
        # High-trust agents can elevate YELLOW to GREEN for non-critical actions
        if (
            self.trust_scorer
            and tier == AutonomyTier.YELLOW
            and self.trust_scorer.can_elevate_tier(agent_id, tier)
        ):
            tier = AutonomyTier.GREEN
            reasons.append("Elevated from YELLOW to GREEN based on high trust")

        # Build decision
        decision = PolicyDecision(
            tier=tier,
            action=action,
            agent_id=agent_id,
            reasons=reasons,
            guardrail_results=guardrail_results,
            trust_score=trust_score,
        )

        logger.info(
            "Policy decision made",
            agent_id=agent_id,
            action=action,
            tier=tier.value,
            requires_approval=decision.requires_approval,
            trust_score=trust_score,
        )

        return decision

    def can_auto_execute(self, decision: PolicyDecision) -> bool:
        """Check if an action can execute without human approval.

        Args:
            decision: PolicyDecision from evaluate()

        Returns:
            True if tier is GREEN or YELLOW, False if RED or BLACK
        """
        return decision.tier in (AutonomyTier.GREEN, AutonomyTier.YELLOW)

    def record_outcome(
        self, decision_id: str, success: bool, agent_id: str = "", action: str = ""
    ) -> None:
        """Record the outcome of an executed action for trust scoring.

        Args:
            decision_id: ID from PolicyDecision
            success: Whether action succeeded
            agent_id: Agent ID (for logging)
            action: Action name (for logging)
        """
        if self.trust_scorer:
            if success:
                self.trust_scorer.record_success(agent_id, action, severity="medium")
            else:
                self.trust_scorer.record_failure(agent_id, action, severity="medium")

        logger.info(
            "Outcome recorded",
            decision_id=decision_id,
            success=success,
            agent_id=agent_id,
            action=action,
        )
