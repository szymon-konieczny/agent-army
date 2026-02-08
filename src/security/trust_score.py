"""Trust scoring — agents earn autonomy through consistent performance.

Each agent starts with a base trust score. Successful actions increase it,
failures decrease it. Higher trust unlocks more autonomy (YELLOW→GREEN).
Catastrophic failures can trigger a trust freeze (all actions become RED
until human reviews).

Inspired by reputation systems: trust is hard to build, easy to lose.
An agent that consistently succeeds should be able to operate more
autonomously. But a single critical failure can freeze all operations.
"""

import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class TrustLevel(str, Enum):
    """Trust levels determine how much autonomy an agent has."""

    PROBATION = "probation"  # 0-30: New agents or those in recovery
    STANDARD = "standard"  # 30-60: Normal operating level
    TRUSTED = "trusted"  # 60-85: Can elevate YELLOW to GREEN
    AUTONOMOUS = "autonomous"  # 85-100: Maximum autonomy


class TrustEvent(BaseModel):
    """A single event that affects trust score.

    Attributes:
        agent_id: Which agent this event is for
        action: What action was taken
        success: Whether it succeeded
        severity: Impact level (low/medium/high/critical)
        impact: Calculated score impact
        timestamp: When this happened
        details: Additional context
    """

    agent_id: str
    action: str
    success: bool
    severity: str = Field(default="medium")
    impact: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: Optional[dict] = None


class TrustProfile(BaseModel):
    """Complete trust profile for an agent.

    Attributes:
        agent_id: Agent identifier
        score: Current trust score (0-100)
        level: Derived trust level
        total_actions: Total actions taken
        successful_actions: How many succeeded
        failed_actions: How many failed
        consecutive_successes: Current success streak
        consecutive_failures: Current failure streak
        is_frozen: If True, all actions become RED until unfrozen
        freeze_reason: Why agent is frozen
        history: Deque of recent events (last 100)
        created_at: When profile was created
        updated_at: Last update timestamp
    """

    agent_id: str
    score: float = Field(default=50.0, ge=0.0, le=100.0)
    level: TrustLevel = Field(default=TrustLevel.STANDARD)
    total_actions: int = Field(default=0)
    successful_actions: int = Field(default=0)
    failed_actions: int = Field(default=0)
    consecutive_successes: int = Field(default=0)
    consecutive_failures: int = Field(default=0)
    is_frozen: bool = Field(default=False)
    freeze_reason: Optional[str] = None
    history: deque = Field(default_factory=lambda: deque(maxlen=100))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        """Pydantic config allowing deque."""

        arbitrary_types_allowed = True

    def compute_level(self) -> TrustLevel:
        """Derive trust level from score.

        Returns:
            Appropriate trust level
        """
        if self.score < 30:
            return TrustLevel.PROBATION
        elif self.score < 60:
            return TrustLevel.STANDARD
        elif self.score < 85:
            return TrustLevel.TRUSTED
        else:
            return TrustLevel.AUTONOMOUS

    def update_level(self) -> None:
        """Update trust level based on current score."""
        self.level = self.compute_level()


class TrustScorer:
    """Manages trust profiles and scoring for all agents.

    Trust is calculated as follows:
    - Base score starts at 50
    - Successful action: +0.5 to +3.0 depending on severity
    - Bonus for consecutive successes: +0.2 per streak (max +2.0)
    - Failed action: -1.0 to -20.0 depending on severity
    - 3+ consecutive failures: auto-freeze
    - Critical failure: immediate freeze
    - Score slowly decays toward 50 over time (use-it-or-lose-it)
    """

    def __init__(
        self,
        initial_score: float = 50.0,
        max_score: float = 100.0,
        probation_threshold: float = 30.0,
        trusted_threshold: float = 60.0,
        autonomous_threshold: float = 85.0,
        freeze_on_failures: int = 3,
    ) -> None:
        """Initialize the trust scorer.

        Args:
            initial_score: Starting score for new agents
            max_score: Maximum possible score
            probation_threshold: Below this = PROBATION
            trusted_threshold: Above this = TRUSTED (can elevate YELLOW→GREEN)
            autonomous_threshold: Above this = AUTONOMOUS
            freeze_on_failures: Auto-freeze after N consecutive failures
        """
        self.initial_score = initial_score
        self.max_score = max_score
        self.probation_threshold = probation_threshold
        self.trusted_threshold = trusted_threshold
        self.autonomous_threshold = autonomous_threshold
        self.freeze_on_failures = freeze_on_failures

        # Storage: agent_id -> TrustProfile
        self.profiles: dict[str, TrustProfile] = {}

        # Decay task
        self._decay_task: Optional[asyncio.Task] = None

        logger.info(
            "Trust scorer initialized",
            initial_score=initial_score,
            max_score=max_score,
            probation_threshold=probation_threshold,
            trusted_threshold=trusted_threshold,
            autonomous_threshold=autonomous_threshold,
        )

    def get_profile(self, agent_id: str) -> TrustProfile:
        """Get (or create) the trust profile for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            TrustProfile for the agent
        """
        if agent_id not in self.profiles:
            profile = TrustProfile(
                agent_id=agent_id,
                score=self.initial_score,
                level=self._score_to_level(self.initial_score),
            )
            self.profiles[agent_id] = profile
            logger.info("New trust profile created", agent_id=agent_id, score=self.initial_score)
        return self.profiles[agent_id]

    def record_success(
        self, agent_id: str, action: str, severity: str = "medium", details: Optional[dict] = None
    ) -> None:
        """Record a successful action, increasing trust.

        Args:
            agent_id: Agent that succeeded
            action: What action succeeded
            severity: Impact level (low/medium/high/critical)
            details: Optional context about the success
        """
        profile = self.get_profile(agent_id)

        # If frozen, unfreezing requires human action
        if profile.is_frozen:
            logger.info("Action success on frozen agent (not unfreezing)", agent_id=agent_id)
            return

        # Calculate impact based on severity
        severity_impacts = {"low": 0.5, "medium": 1.0, "high": 2.0, "critical": 3.0}
        base_impact = severity_impacts.get(severity, 1.0)

        # Consecutive success bonus
        profile.consecutive_successes += 1
        success_bonus = min(0.2 * profile.consecutive_failures, 2.0)
        profile.consecutive_failures = 0  # Reset failure streak

        total_impact = base_impact + success_bonus

        # Apply score increase (capped at max)
        profile.score = min(profile.score + total_impact, self.max_score)
        profile.total_actions += 1
        profile.successful_actions += 1
        profile.updated_at = datetime.now(timezone.utc)
        profile.update_level()

        # Record event
        event = TrustEvent(
            agent_id=agent_id,
            action=action,
            success=True,
            severity=severity,
            impact=total_impact,
            details=details,
        )
        profile.history.append(event)

        logger.info(
            "Trust increased (success)",
            agent_id=agent_id,
            action=action,
            severity=severity,
            impact=total_impact,
            new_score=profile.score,
            level=profile.level.value,
        )

    def record_failure(
        self, agent_id: str, action: str, severity: str = "medium", details: Optional[dict] = None
    ) -> None:
        """Record a failed action, decreasing trust.

        Critical failures and 3+ consecutive failures trigger freeze.

        Args:
            agent_id: Agent that failed
            action: What action failed
            severity: Impact level (low/medium/high/critical)
            details: Optional context about the failure
        """
        profile = self.get_profile(agent_id)

        # Calculate impact based on severity
        severity_impacts = {"low": -1.0, "medium": -3.0, "high": -8.0, "critical": -20.0}
        base_impact = severity_impacts.get(severity, -3.0)

        profile.consecutive_failures += 1
        profile.consecutive_successes = 0  # Reset success streak

        # Apply score decrease (floored at 0)
        profile.score = max(profile.score + base_impact, 0.0)
        profile.total_actions += 1
        profile.failed_actions += 1
        profile.updated_at = datetime.now(timezone.utc)
        profile.update_level()

        # Record event
        event = TrustEvent(
            agent_id=agent_id,
            action=action,
            success=False,
            severity=severity,
            impact=base_impact,
            details=details,
        )
        profile.history.append(event)

        # Check for freeze conditions
        should_freeze = False
        freeze_reason = ""

        if severity == "critical":
            should_freeze = True
            freeze_reason = f"Critical failure in action: {action}"
        elif profile.consecutive_failures >= self.freeze_on_failures:
            should_freeze = True
            freeze_reason = f"{profile.consecutive_failures} consecutive failures"

        if should_freeze and not profile.is_frozen:
            self.freeze_agent(agent_id, freeze_reason)

        logger.info(
            "Trust decreased (failure)",
            agent_id=agent_id,
            action=action,
            severity=severity,
            impact=base_impact,
            new_score=profile.score,
            level=profile.level.value,
            is_frozen=profile.is_frozen,
        )

    def can_elevate_tier(self, agent_id: str, from_tier: str) -> bool:
        """Check if agent can be elevated from one tier to a higher one.

        YELLOW→GREEN elevation requires TRUSTED level (60+)
        Other elevations allowed for AUTONOMOUS level (85+)

        Args:
            agent_id: Agent to check
            from_tier: Current tier string

        Returns:
            True if elevation is allowed
        """
        if from_tier == "yellow":
            profile = self.get_profile(agent_id)
            return profile.score >= self.trusted_threshold and not profile.is_frozen

        return False

    def freeze_agent(self, agent_id: str, reason: str) -> None:
        """Freeze an agent, making all actions RED until unfrozen.

        Args:
            agent_id: Agent to freeze
            reason: Why the agent is being frozen
        """
        profile = self.get_profile(agent_id)
        profile.is_frozen = True
        profile.freeze_reason = reason
        profile.updated_at = datetime.now(timezone.utc)

        logger.warning(
            "Agent frozen",
            agent_id=agent_id,
            reason=reason,
            score=profile.score,
            level=profile.level.value,
        )

    def unfreeze_agent(self, agent_id: str) -> None:
        """Unfreeze an agent, restoring normal policy evaluation.

        Only called after human review.

        Args:
            agent_id: Agent to unfreeze
        """
        profile = self.get_profile(agent_id)
        profile.is_frozen = False
        profile.freeze_reason = None
        profile.consecutive_failures = 0
        profile.updated_at = datetime.now(timezone.utc)

        logger.info(
            "Agent unfrozen",
            agent_id=agent_id,
            score=profile.score,
            level=profile.level.value,
        )

    def decay(self) -> None:
        """Decay trust scores toward baseline (use-it-or-lose-it).

        Called periodically. Agents with activity maintain higher scores.
        Inactive agents decay toward the initial score.
        """
        now = datetime.now(timezone.utc)

        for agent_id, profile in self.profiles.items():
            # Check how long since last action
            time_since_update = now - profile.updated_at
            days_inactive = time_since_update.total_seconds() / 86400.0

            if days_inactive > 0:
                # Decay rate: 0.1 points per day inactive
                decay_amount = 0.1 * days_inactive
                new_score = profile.score - decay_amount

                # Don't decay below initial score
                new_score = max(new_score, self.initial_score)

                if new_score < profile.score:
                    old_level = profile.level
                    profile.score = new_score
                    profile.update_level()

                    if profile.level != old_level:
                        logger.info(
                            "Trust decayed",
                            agent_id=agent_id,
                            old_score=profile.score + decay_amount,
                            new_score=profile.score,
                            old_level=old_level.value,
                            new_level=profile.level.value,
                            days_inactive=round(days_inactive, 1),
                        )

    async def start_decay_loop(self, interval_seconds: int = 3600) -> None:
        """Start background task that decays scores periodically.

        Args:
            interval_seconds: How often to decay (default: 1 hour)
        """

        async def decay_loop() -> None:
            while True:
                await asyncio.sleep(interval_seconds)
                self.decay()

        self._decay_task = asyncio.create_task(decay_loop())
        logger.info("Trust decay loop started", interval_seconds=interval_seconds)

    async def stop_decay_loop(self) -> None:
        """Stop the background decay task."""
        if self._decay_task:
            self._decay_task.cancel()
            try:
                await self._decay_task
            except asyncio.CancelledError:
                pass
            logger.info("Trust decay loop stopped")

    def _score_to_level(self, score: float) -> TrustLevel:
        """Convert score to trust level.

        Args:
            score: Trust score (0-100)

        Returns:
            Corresponding trust level
        """
        if score < self.probation_threshold:
            return TrustLevel.PROBATION
        elif score < self.trusted_threshold:
            return TrustLevel.STANDARD
        elif score < self.autonomous_threshold:
            return TrustLevel.TRUSTED
        else:
            return TrustLevel.AUTONOMOUS

    def get_stats(self) -> dict:
        """Get aggregate statistics across all agents.

        Returns:
            Dictionary with overall stats
        """
        if not self.profiles:
            return {
                "total_agents": 0,
                "average_score": 0.0,
                "frozen_agents": 0,
                "by_level": {},
            }

        profiles = list(self.profiles.values())
        scores = [p.score for p in profiles]
        frozen_count = sum(1 for p in profiles if p.is_frozen)

        levels = {level.value: 0 for level in TrustLevel}
        for profile in profiles:
            levels[profile.level.value] += 1

        return {
            "total_agents": len(profiles),
            "average_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "frozen_agents": frozen_count,
            "by_level": levels,
        }
