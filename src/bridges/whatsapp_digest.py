"""WhatsApp Digest — batch notifications instead of per-action alerts.

Instead of flooding WhatsApp with every agent action, the digest system:
1. Collects YELLOW-tier events during a time window
2. Sends a formatted summary at configurable intervals (default: every 30 min)
3. RED events still send IMMEDIATELY
4. User can configure: realtime / digest / daily / silent

The digest is the key to autonomous operation feeling right —
you know what's happening without being interrupted.

This module is async-compatible and sends notifications via WhatsApp API.
It handles batching, formatting, and delivery of status summaries.
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class DigestEntry(BaseModel):
    """Single event in a digest.

    Attributes:
        timestamp: When the event occurred
        agent_id: Which agent took the action
        action: What action was taken
        tier: Autonomy tier (green/yellow/red)
        summary: Brief description of what happened
        outcome: Result status (success/failure/pending)
        cost: Estimated cost of the action
        details: Additional details
    """

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_id: str
    action: str
    tier: str
    summary: str
    outcome: str = Field(default="pending")
    cost: Optional[float] = None
    details: Optional[dict[str, Any]] = None


class DigestConfig(BaseModel):
    """Configuration for digest behavior.

    Attributes:
        mode: none/digest/daily/realtime
        digest_interval_minutes: How often to send digest (default 30)
        quiet_hours: Tuple of (start_hour, end_hour) for no notifications (24h format)
        always_notify_on: List of action patterns that always notify immediately
        include_cost_summary: Include cost totals in digest
        include_trust_changes: Include trust score changes
    """

    mode: str = Field(default="digest")
    digest_interval_minutes: int = Field(default=30, ge=5, le=1440)
    quiet_hours: Optional[tuple[int, int]] = None
    always_notify_on: list[str] = Field(default_factory=list)
    include_cost_summary: bool = Field(default=True)
    include_trust_changes: bool = Field(default=True)


class DigestStats(BaseModel):
    """Statistics for a digest batch.

    Attributes:
        total_actions: Total actions in digest
        by_agent: Count of actions per agent
        by_outcome: Count of actions per outcome
        successful: Count of successful actions
        failed: Count of failed actions
        total_cost: Sum of all costs
        trust_changes: List of agent trust level changes
    """

    total_actions: int = Field(default=0)
    by_agent: dict[str, int] = Field(default_factory=dict)
    by_outcome: dict[str, int] = Field(default_factory=dict)
    successful: int = Field(default=0)
    failed: int = Field(default=0)
    total_cost: float = Field(default=0.0)
    trust_changes: list[dict[str, Any]] = Field(default_factory=list)


class WhatsAppDigest:
    """Manages batched WhatsApp notifications for agent operations.

    Collects YELLOW-tier events and sends periodic digests.
    RED events are sent immediately.
    Respects quiet hours and user preferences.
    """

    def __init__(
        self,
        config: Optional[DigestConfig] = None,
        send_notification_fn: Optional[Callable] = None,
    ) -> None:
        """Initialize WhatsApp digest system.

        Args:
            config: DigestConfig for behavior customization
            send_notification_fn: Async function to send WhatsApp messages
        """
        self.config = config or DigestConfig()
        self.send_notification_fn = send_notification_fn

        # Buffer for pending digest entries
        self.pending_entries: list[DigestEntry] = []

        # Background task
        self._digest_task: Optional[asyncio.Task] = None

        logger.info(
            "WhatsApp digest initialized",
            mode=self.config.mode,
            interval_minutes=self.config.digest_interval_minutes,
        )

    def add_event(self, entry: DigestEntry) -> None:
        """Add an event to the digest queue.

        If event.tier is RED or action matches always_notify_on, sends immediately.
        Otherwise, adds to pending queue for next digest flush.

        Args:
            entry: DigestEntry to add
        """
        if self._should_notify_immediately(entry):
            asyncio.create_task(self._send_immediate(entry))
        else:
            self.pending_entries.append(entry)
            logger.debug(
                "Digest entry queued",
                agent=entry.agent_id,
                action=entry.action,
                pending_count=len(self.pending_entries),
            )

    async def flush(self) -> None:
        """Send accumulated digest to WhatsApp.

        Respects quiet hours. Does nothing if:
        - No pending entries
        - config.mode is "none"
        - Currently in quiet hours
        """
        if not self.pending_entries:
            return

        if self.config.mode == "none":
            logger.debug("Digest mode is 'none', skipping flush")
            return

        if not self._should_send_at_this_hour():
            logger.debug("In quiet hours, skipping digest flush")
            return

        if self.config.mode == "daily":
            # Only send once per day - check if already sent today
            # (simplified implementation - in production track daily state)
            pass

        entries = self.pending_entries.copy()
        self.pending_entries.clear()

        message = self._format_digest(entries)

        if self.send_notification_fn:
            try:
                await self.send_notification_fn(message)
                logger.info(
                    "Digest sent to WhatsApp",
                    entry_count=len(entries),
                )
            except Exception as e:
                logger.error(
                    "Failed to send digest",
                    error=str(e),
                )
                # Re-queue entries on failure
                self.pending_entries.extend(entries)
        else:
            logger.debug("No send function configured, digest would be:")
            logger.debug(message)

    def _format_digest(self, entries: list[DigestEntry]) -> str:
        """Format entries into WhatsApp message.

        Uses WhatsApp markdown formatting:
        - *bold*
        - _italic_
        - ```monospace```

        Args:
            entries: List of entries to format

        Returns:
            Formatted WhatsApp message
        """
        # Compute statistics
        stats = self._compute_stats(entries)

        lines = []
        lines.append("*AgentArmy — Status Digest*")
        lines.append("")

        # Group by agent
        by_agent: dict[str, list[DigestEntry]] = defaultdict(list)
        for entry in entries:
            by_agent[entry.agent_id].append(entry)

        # Format each agent's actions
        for agent_id in sorted(by_agent.keys()):
            agent_entries = by_agent[agent_id]
            action_count = len(agent_entries)
            success_count = sum(1 for e in agent_entries if e.outcome == "success")
            fail_count = sum(1 for e in agent_entries if e.outcome == "failure")

            emoji = "🟢" if fail_count == 0 else "🟡" if success_count > 0 else "🔴"
            lines.append(f"{emoji} *{agent_id.title()}* ({action_count} actions)")

            # Show action summaries
            for entry in agent_entries[:3]:  # Show first 3
                outcome_emoji = "✓" if entry.outcome == "success" else "✗" if entry.outcome == "failure" else "⏳"
                cost_str = f" ${entry.cost:.2f}" if entry.cost else ""
                lines.append(
                    f"  {outcome_emoji} {entry.summary}{cost_str}"
                )

            if len(agent_entries) > 3:
                lines.append(f"  ... and {len(agent_entries) - 3} more")

            # Show agent stats
            if fail_count > 0:
                lines.append(f"  ⚠️  {fail_count} failed")

        lines.append("")

        # Cost summary
        if self.config.include_cost_summary and stats.total_cost > 0:
            lines.append(f"💰 *Cost*: ${stats.total_cost:.2f}")

        # Overall stats
        lines.append(f"📊 *Summary*: {stats.successful}/{stats.total_actions} succeeded")

        # Trust changes
        if self.config.include_trust_changes and stats.trust_changes:
            lines.append("")
            lines.append("*Trust Updates*:")
            for change in stats.trust_changes[:3]:
                direction = "📈" if change.get("increase", False) else "📉"
                lines.append(
                    f"  {direction} {change.get('agent', 'Unknown')}: {change.get('old_level')} → {change.get('new_level')}"
                )

        # Footer
        lines.append("")
        lines.append("_Reply /details for full log_")

        return "\n".join(lines)

    def _compute_stats(self, entries: list[DigestEntry]) -> DigestStats:
        """Compute statistics from entries.

        Args:
            entries: List of entries

        Returns:
            DigestStats with computed values
        """
        stats = DigestStats(total_actions=len(entries))

        for entry in entries:
            # Count by agent
            stats.by_agent[entry.agent_id] = stats.by_agent.get(entry.agent_id, 0) + 1

            # Count by outcome
            stats.by_outcome[entry.outcome] = stats.by_outcome.get(entry.outcome, 0) + 1

            # Track success/failure
            if entry.outcome == "success":
                stats.successful += 1
            elif entry.outcome == "failure":
                stats.failed += 1

            # Sum costs
            if entry.cost:
                stats.total_cost += entry.cost

        return stats

    def _should_notify_immediately(self, entry: DigestEntry) -> bool:
        """Check if entry should be notified immediately.

        Returns True if:
        - entry.tier is RED
        - action matches any pattern in always_notify_on

        Args:
            entry: DigestEntry to check

        Returns:
            True if immediate notification needed
        """
        if entry.tier == "red":
            return True

        for pattern in self.config.always_notify_on:
            if entry.action.startswith(pattern.replace("*", "")):
                return True

        return False

    async def _send_immediate(self, entry: DigestEntry) -> None:
        """Send immediate notification for RED-tier or critical actions.

        Args:
            entry: DigestEntry to send
        """
        if not self.send_notification_fn:
            return

        message = self._format_immediate_message(entry)

        try:
            await self.send_notification_fn(message)
            logger.info(
                "Immediate notification sent",
                agent=entry.agent_id,
                action=entry.action,
                tier=entry.tier,
            )
        except Exception as e:
            logger.error(
                "Failed to send immediate notification",
                error=str(e),
            )

    def _format_immediate_message(self, entry: DigestEntry) -> str:
        """Format immediate notification message.

        Args:
            entry: DigestEntry to format

        Returns:
            WhatsApp formatted message
        """
        emoji = "🔴" if entry.tier == "red" else "⚠️"

        lines = [
            f"{emoji} *Action requires approval*",
            "",
            f"*Agent*: {entry.agent_id.title()}",
            f"*Action*: {entry.action}",
            f"*Description*: {entry.summary}",
        ]

        if entry.cost:
            lines.append(f"*Cost*: ${entry.cost:.2f}")

        if entry.details:
            lines.append("")
            lines.append("*Details*:")
            for key, value in entry.details.items():
                lines.append(f"  {key}: {value}")

        lines.append("")
        lines.append("_Reply /approve to continue or /reject to abort_")

        return "\n".join(lines)

    def _should_send_at_this_hour(self) -> bool:
        """Check if current time is within sending hours.

        Returns False if currently in quiet hours.

        Returns:
            True if OK to send
        """
        if not self.config.quiet_hours:
            return True

        start_hour, end_hour = self.config.quiet_hours
        now = datetime.now(timezone.utc)
        current_hour = now.hour

        # Handle wrap-around (e.g., 23 to 7 means 11pm to 7am)
        if start_hour < end_hour:
            return not (start_hour <= current_hour < end_hour)
        else:
            return not (current_hour >= start_hour or current_hour < end_hour)

    def get_pending_count(self) -> int:
        """Get number of pending digest entries.

        Returns:
            Count of entries waiting to be sent
        """
        return len(self.pending_entries)

    async def start_digest_loop(self) -> None:
        """Start background task that flushes digest periodically.

        Runs forever, sending digest every interval_minutes.
        """

        async def digest_loop() -> None:
            while True:
                await asyncio.sleep(self.config.digest_interval_minutes * 60)
                await self.flush()

        if self._digest_task:
            self._digest_task.cancel()

        self._digest_task = asyncio.create_task(digest_loop())
        logger.info(
            "Digest loop started",
            interval_minutes=self.config.digest_interval_minutes,
        )

    async def stop_digest_loop(self) -> None:
        """Stop the background digest task."""
        if self._digest_task:
            self._digest_task.cancel()
            try:
                await self._digest_task
            except asyncio.CancelledError:
                pass
            self._digest_task = None
            logger.info("Digest loop stopped")

    def set_send_function(self, fn: Callable) -> None:
        """Set the function used to send WhatsApp messages.

        Args:
            fn: Async callable that takes a message string
        """
        self.send_notification_fn = fn
        logger.info("WhatsApp send function configured")

    def configure(self, config: DigestConfig) -> None:
        """Update digest configuration at runtime.

        Args:
            config: New DigestConfig
        """
        self.config = config
        logger.info(
            "Digest configuration updated",
            mode=config.mode,
            interval_minutes=config.digest_interval_minutes,
        )
