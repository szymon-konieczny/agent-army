"""Guardrails — automated pre-flight checks before agent actions.

Guardrails are like pre-commit hooks but for ALL agent operations.
They run automatically BEFORE any action. If guardrails pass, the action
proceeds at its policy tier. If they fail, the tier escalates.

Guardrails don't ask for permission — they VERIFY automatically.
Think: a compiler for agent safety.

Built-in guardrails check:
- Cost (won't spend >$10 without approval)
- Secret leaks (scans for exposed API keys/passwords)
- Protected paths (won't write to .env, secrets/, etc.)
- Branch protection (won't force-push main)
- Deployment safety (tests must pass first)
- Rate limits (prevent runaway loops)
- Diff size (warn if >500 lines changed)
"""

import asyncio
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class GuardrailResult(BaseModel):
    """Result of running a single guardrail check.

    Attributes:
        name: Guardrail name
        passed: Whether guardrail passed
        severity: severity level (info/warning/error/critical)
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


class GuardrailSuite(BaseModel):
    """Aggregate results from running all guardrails.

    Attributes:
        results: Individual guardrail results
        all_passed: Whether all guardrails passed
        critical_failures: Count of critical-severity failures
        warnings: Count of warning-severity issues
        total_time_ms: Total execution time
        should_escalate: True if action should be escalated to RED
        escalate_to: Specific tier to escalate to if needed
    """

    results: list[GuardrailResult] = Field(default_factory=list)
    all_passed: bool = Field(default=True)
    critical_failures: int = Field(default=0)
    warnings: int = Field(default=0)
    total_time_ms: float = Field(default=0.0)
    should_escalate: bool = Field(default=False)
    escalate_to: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        """Compute derived fields after initialization."""
        self.critical_failures = sum(1 for r in self.results if r.severity == "critical")
        self.warnings = sum(1 for r in self.results if r.severity == "warning")
        self.all_passed = all(r.passed for r in self.results)

        # Escalate to RED if any critical failures or >3 warnings
        if self.critical_failures > 0 or self.warnings >= 3:
            self.should_escalate = True
            self.escalate_to = "red"


class Guardrail(ABC):
    """Abstract base class for guardrail implementations.

    Subclasses implement specific safety checks.
    """

    def __init__(self, name: str, enabled: bool = True) -> None:
        """Initialize guardrail.

        Args:
            name: Unique name for this guardrail
            enabled: Whether to run this guardrail
        """
        self.name = name
        self.enabled = enabled

    @abstractmethod
    async def check(self, action: str, context: dict[str, Any]) -> GuardrailResult:
        """Run the guardrail check.

        Args:
            action: Action being evaluated
            context: Context data for the action

        Returns:
            GuardrailResult with pass/fail and details
        """
        pass


class CostGuardrail(Guardrail):
    """Checks estimated LLM cost against budget limits.

    Warns if >$1, errors if >$10, critical if >$50
    """

    def __init__(
        self,
        warn_threshold: float = 1.0,
        error_threshold: float = 10.0,
        critical_threshold: float = 50.0,
    ) -> None:
        """Initialize cost guardrail.

        Args:
            warn_threshold: Cost to trigger warning
            error_threshold: Cost to trigger error
            critical_threshold: Cost to trigger critical
        """
        super().__init__("cost", enabled=True)
        self.warn_threshold = warn_threshold
        self.error_threshold = error_threshold
        self.critical_threshold = critical_threshold

    async def check(self, action: str, context: dict[str, Any]) -> GuardrailResult:
        """Check estimated cost in context.

        Context should contain 'estimated_cost' field.

        Args:
            action: Action being evaluated
            context: Should contain 'estimated_cost'

        Returns:
            GuardrailResult
        """
        start_time = time.time()

        estimated_cost = context.get("estimated_cost", 0.0)

        if estimated_cost >= self.critical_threshold:
            return GuardrailResult(
                name=self.name,
                passed=False,
                severity="critical",
                message=f"Estimated cost ${estimated_cost:.2f} exceeds critical threshold of ${self.critical_threshold:.2f}",
                details={"estimated_cost": estimated_cost},
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        elif estimated_cost >= self.error_threshold:
            return GuardrailResult(
                name=self.name,
                passed=False,
                severity="error",
                message=f"Estimated cost ${estimated_cost:.2f} exceeds error threshold of ${self.error_threshold:.2f}",
                details={"estimated_cost": estimated_cost},
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        elif estimated_cost >= self.warn_threshold:
            return GuardrailResult(
                name=self.name,
                passed=True,
                severity="warning",
                message=f"Estimated cost ${estimated_cost:.2f} exceeds warning threshold of ${self.warn_threshold:.2f}",
                details={"estimated_cost": estimated_cost},
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        else:
            return GuardrailResult(
                name=self.name,
                passed=True,
                severity="info",
                message=f"Cost OK: ${estimated_cost:.2f}",
                details={"estimated_cost": estimated_cost},
                execution_time_ms=(time.time() - start_time) * 1000,
            )


class SecretLeakGuardrail(Guardrail):
    """Scans action payload for exposed API keys, passwords, tokens."""

    # Patterns for common secret formats
    SECRET_PATTERNS = {
        "api_key": r"['\"]?[a-zA-Z0-9_-]*[aA]pi[_-]?key['\"]?\s*[:=]\s*['\"]?[a-zA-Z0-9_\-/+=]{20,}",
        "aws_key": r"AKIA[0-9A-Z]{16}",
        "private_key": r"-----BEGIN (?:RSA |DSA |EC )?PRIVATE KEY",
        "password": r"['\"]?password['\"]?\s*[:=]\s*['\"]?[^\s'\"]{8,}",
        "token": r"['\"]?(?:access_token|auth_token|token)['\"]?\s*[:=]\s*['\"]?[a-zA-Z0-9_\-/+=]{20,}",
        "github_token": r"ghp_[a-zA-Z0-9_]{36,255}",
        "slack_token": r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*",
    }

    def __init__(self) -> None:
        """Initialize secret leak guardrail."""
        super().__init__("secret_leak", enabled=True)
        self.compiled_patterns = {
            name: re.compile(pattern) for name, pattern in self.SECRET_PATTERNS.items()
        }

    async def check(self, action: str, context: dict[str, Any]) -> GuardrailResult:
        """Scan context for exposed secrets.

        Checks context['payload'] or converts entire context to string for scanning.

        Args:
            action: Action being evaluated
            context: Action context to scan

        Returns:
            GuardrailResult with pass/fail
        """
        start_time = time.time()

        # Get payload to scan
        payload = context.get("payload", "")
        if not isinstance(payload, str):
            payload = str(context)

        # Scan for secrets
        found_secrets = []
        for secret_type, pattern in self.compiled_patterns.items():
            matches = pattern.finditer(payload)
            for match in matches:
                found_secrets.append(secret_type)

        if found_secrets:
            unique_types = list(set(found_secrets))
            return GuardrailResult(
                name=self.name,
                passed=False,
                severity="critical",
                message=f"Potential secrets detected: {', '.join(unique_types)}",
                details={"secret_types_found": unique_types, "count": len(found_secrets)},
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        return GuardrailResult(
            name=self.name,
            passed=True,
            severity="info",
            message="No secrets detected",
            execution_time_ms=(time.time() - start_time) * 1000,
        )


class ProtectedPathGuardrail(Guardrail):
    """Prevents writes to sensitive files/directories."""

    DEFAULT_PROTECTED_PATHS = [
        r"^\.env",
        r"^secrets/",
        r".*\.pem$",
        r".*\.key$",
        r".*credentials\.",
        r"^\.aws/",
        r"^\.ssh/",
    ]

    def __init__(self, protected_paths: Optional[list[str]] = None) -> None:
        """Initialize protected path guardrail.

        Args:
            protected_paths: List of regex patterns for protected paths
        """
        super().__init__("protected_path", enabled=True)
        patterns = protected_paths or self.DEFAULT_PROTECTED_PATHS
        self.compiled_patterns = [re.compile(p) for p in patterns]

    async def check(self, action: str, context: dict[str, Any]) -> GuardrailResult:
        """Check if action would write to protected paths.

        Args:
            action: Action being evaluated
            context: Should contain 'file_path' or 'paths'

        Returns:
            GuardrailResult with pass/fail
        """
        start_time = time.time()

        paths_to_check = []
        if "file_path" in context:
            paths_to_check.append(context["file_path"])
        if "paths" in context:
            paths = context["paths"]
            if isinstance(paths, list):
                paths_to_check.extend(paths)
            else:
                paths_to_check.append(str(paths))

        # Check against protected patterns
        protected_violations = []
        for path in paths_to_check:
            for pattern in self.compiled_patterns:
                if pattern.match(str(path)):
                    protected_violations.append(path)
                    break

        if protected_violations:
            return GuardrailResult(
                name=self.name,
                passed=False,
                severity="critical",
                message=f"Attempt to write to protected paths: {protected_violations[:3]}",
                details={"violations": protected_violations},
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        return GuardrailResult(
            name=self.name,
            passed=True,
            severity="info",
            message="Path check OK",
            execution_time_ms=(time.time() - start_time) * 1000,
        )


class BranchProtectionGuardrail(Guardrail):
    """Prevents dangerous git operations on main/master branches."""

    DEFAULT_PROTECTED_BRANCHES = ["main", "master", "production", "staging"]

    def __init__(self, protected_branches: Optional[list[str]] = None) -> None:
        """Initialize branch protection guardrail.

        Args:
            protected_branches: List of branch names to protect
        """
        super().__init__("branch_protection", enabled=True)
        self.protected_branches = protected_branches or self.DEFAULT_PROTECTED_BRANCHES

    async def check(self, action: str, context: dict[str, Any]) -> GuardrailResult:
        """Check git operations on protected branches.

        Critical failures for: force_push, rebase, history rewrite on protected branches

        Args:
            action: Action being evaluated
            context: Should contain 'branch' and possibly 'operation'

        Returns:
            GuardrailResult with pass/fail
        """
        start_time = time.time()

        branch = context.get("branch", "")
        operation = context.get("operation", "")

        # Check if operating on protected branch
        if branch not in self.protected_branches:
            return GuardrailResult(
                name=self.name,
                passed=True,
                severity="info",
                message=f"Branch '{branch}' not protected",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Check for dangerous operations
        dangerous_ops = ["force_push", "rebase", "rewrite_history", "delete"]
        if operation in dangerous_ops:
            return GuardrailResult(
                name=self.name,
                passed=False,
                severity="critical",
                message=f"Dangerous operation '{operation}' on protected branch '{branch}'",
                details={"branch": branch, "operation": operation},
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        return GuardrailResult(
            name=self.name,
            passed=True,
            severity="info",
            message=f"Safe operation on protected branch '{branch}'",
            execution_time_ms=(time.time() - start_time) * 1000,
        )


class DeploymentSafetyGuardrail(Guardrail):
    """Checks deployment preconditions before deployment."""

    def __init__(self) -> None:
        """Initialize deployment safety guardrail."""
        super().__init__("deployment_safety", enabled=True)

    async def check(self, action: str, context: dict[str, Any]) -> GuardrailResult:
        """Check deployment preconditions.

        Requirements:
        - Tests must have passed recently
        - No critical security issues open
        - Must be within deployment window

        Args:
            action: Action being evaluated
            context: Should contain test/security/window info

        Returns:
            GuardrailResult with pass/fail
        """
        start_time = time.time()

        if "deploy" not in action:
            return GuardrailResult(
                name=self.name,
                passed=True,
                severity="info",
                message="Not a deployment action",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        issues = []

        # Check tests
        if not context.get("tests_passed", False):
            issues.append("Tests have not passed")

        # Check security
        if context.get("critical_security_issues", 0) > 0:
            issues.append(f"{context.get('critical_security_issues', 0)} critical security issues")

        # Check deployment window
        if "deployment_window_start" in context:
            now = datetime.now(timezone.utc)
            start = context["deployment_window_start"]
            end = context.get("deployment_window_end")
            if not (start <= now <= end) if end else now < start:
                issues.append("Outside deployment window")

        if issues:
            severity = "error" if len(issues) > 1 else "warning"
            return GuardrailResult(
                name=self.name,
                passed=False,
                severity=severity,
                message="Deployment preconditions not met: " + "; ".join(issues),
                details={"issues": issues},
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        return GuardrailResult(
            name=self.name,
            passed=True,
            severity="info",
            message="Deployment preconditions OK",
            execution_time_ms=(time.time() - start_time) * 1000,
        )


class RateLimitGuardrail(Guardrail):
    """Prevents runaway agent loops with rate limiting."""

    def __init__(
        self, per_agent_per_minute: int = 30, global_per_minute: int = 100
    ) -> None:
        """Initialize rate limit guardrail.

        Args:
            per_agent_per_minute: Max actions per agent per minute
            global_per_minute: Max actions globally per minute
        """
        super().__init__("rate_limit", enabled=True)
        self.per_agent_per_minute = per_agent_per_minute
        self.global_per_minute = global_per_minute

        # Track action counts: (agent_id, timestamp) -> count
        self.agent_action_times: dict[str, list[float]] = {}
        self.global_action_times: list[float] = []

    async def check(self, action: str, context: dict[str, Any]) -> GuardrailResult:
        """Check rate limits for agent.

        Args:
            action: Action being evaluated
            context: Should contain 'agent_id'

        Returns:
            GuardrailResult with pass/fail
        """
        start_time = time.time()

        agent_id = context.get("agent_id", "unknown")
        now = time.time()
        cutoff = now - 60  # 1 minute ago

        # Clean old entries
        if agent_id not in self.agent_action_times:
            self.agent_action_times[agent_id] = []

        self.agent_action_times[agent_id] = [t for t in self.agent_action_times[agent_id] if t > cutoff]
        self.global_action_times = [t for t in self.global_action_times if t > cutoff]

        # Check limits
        agent_count = len(self.agent_action_times[agent_id])
        global_count = len(self.global_action_times)

        if agent_count >= self.per_agent_per_minute:
            return GuardrailResult(
                name=self.name,
                passed=False,
                severity="error",
                message=f"Agent rate limit exceeded: {agent_count}/{self.per_agent_per_minute} per minute",
                details={"agent_id": agent_id, "count": agent_count},
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        if global_count >= self.global_per_minute:
            return GuardrailResult(
                name=self.name,
                passed=False,
                severity="error",
                message=f"Global rate limit exceeded: {global_count}/{self.global_per_minute} per minute",
                details={"count": global_count},
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Record this action
        self.agent_action_times[agent_id].append(now)
        self.global_action_times.append(now)

        severity = "warning" if agent_count >= self.per_agent_per_minute * 0.8 else "info"
        return GuardrailResult(
            name=self.name,
            passed=True,
            severity=severity,
            message=f"Rate OK: {agent_count}/{self.per_agent_per_minute} agent, {global_count}/{self.global_per_minute} global",
            execution_time_ms=(time.time() - start_time) * 1000,
        )


class DiffSizeGuardrail(Guardrail):
    """Prevents accidentally large code changes in single action."""

    def __init__(self, warn_lines: int = 500, error_lines: int = 2000, max_files: int = 10) -> None:
        """Initialize diff size guardrail.

        Args:
            warn_lines: Warn if >N lines changed
            error_lines: Error if >N lines changed
            max_files: Error if >N files modified
        """
        super().__init__("diff_size", enabled=True)
        self.warn_lines = warn_lines
        self.error_lines = error_lines
        self.max_files = max_files

    async def check(self, action: str, context: dict[str, Any]) -> GuardrailResult:
        """Check diff size metrics.

        Args:
            action: Action being evaluated
            context: Should contain 'lines_added' and/or 'files_changed'

        Returns:
            GuardrailResult with pass/fail
        """
        start_time = time.time()

        lines_changed = context.get("lines_added", 0) + context.get("lines_removed", 0)
        files_changed = len(context.get("files_changed", []))

        if files_changed > self.max_files:
            return GuardrailResult(
                name=self.name,
                passed=False,
                severity="error",
                message=f"{files_changed} files changed exceeds limit of {self.max_files}",
                details={"files_changed": files_changed, "limit": self.max_files},
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        if lines_changed >= self.error_lines:
            return GuardrailResult(
                name=self.name,
                passed=False,
                severity="error",
                message=f"{lines_changed} lines changed exceeds error limit of {self.error_lines}",
                details={"lines_changed": lines_changed},
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        severity = "warning" if lines_changed >= self.warn_lines else "info"
        return GuardrailResult(
            name=self.name,
            passed=True,
            severity=severity,
            message=f"Diff OK: {lines_changed} lines, {files_changed} files",
            details={"lines_changed": lines_changed, "files_changed": files_changed},
            execution_time_ms=(time.time() - start_time) * 1000,
        )


class GuardrailRunner:
    """Orchestrates running guardrails and aggregating results."""

    def __init__(self) -> None:
        """Initialize guardrail runner."""
        self.guardrails: dict[str, Guardrail] = {}
        logger.info("Guardrail runner initialized")

    def register(self, guardrail: Guardrail) -> None:
        """Register a guardrail.

        Args:
            guardrail: Guardrail instance to register
        """
        self.guardrails[guardrail.name] = guardrail
        logger.info("Guardrail registered", name=guardrail.name)

    async def run_all(self, action: str, context: dict[str, Any]) -> GuardrailSuite:
        """Run all registered guardrails.

        Args:
            action: Action being evaluated
            context: Action context

        Returns:
            GuardrailSuite with results from all guardrails
        """
        start_time = time.time()

        # Run all guardrails in parallel
        tasks = [
            g.check(action, context)
            for g in self.guardrails.values()
            if g.enabled
        ]

        if not tasks:
            return GuardrailSuite()

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        checked_results = []
        for result in results:
            if isinstance(result, Exception):
                checked_results.append(
                    GuardrailResult(
                        name="unknown",
                        passed=False,
                        severity="error",
                        message=f"Guardrail execution failed: {str(result)}",
                        execution_time_ms=0.0,
                    )
                )
            else:
                checked_results.append(result)

        total_time = (time.time() - start_time) * 1000

        suite = GuardrailSuite(
            results=checked_results,
            total_time_ms=total_time,
        )

        logger.info(
            "Guardrail suite executed",
            action=action,
            total_guardrails=len(checked_results),
            all_passed=suite.all_passed,
            critical_failures=suite.critical_failures,
            warnings=suite.warnings,
            total_time_ms=round(total_time, 2),
        )

        return suite

    async def run_subset(
        self, names: list[str], action: str, context: dict[str, Any]
    ) -> GuardrailSuite:
        """Run a specific subset of guardrails.

        Args:
            names: List of guardrail names to run
            action: Action being evaluated
            context: Action context

        Returns:
            GuardrailSuite with results
        """
        start_time = time.time()

        tasks = [
            self.guardrails[name].check(action, context)
            for name in names
            if name in self.guardrails and self.guardrails[name].enabled
        ]

        if not tasks:
            return GuardrailSuite()

        results = await asyncio.gather(*tasks)
        total_time = (time.time() - start_time) * 1000

        return GuardrailSuite(
            results=results,
            total_time_ms=total_time,
        )
