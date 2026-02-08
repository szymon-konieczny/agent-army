"""
Threat detection module for AgentArmy.

Detects and alerts on security threats including prompt injection,
anomalous behavior, secret leakage, and suspicious rate patterns.
"""

import hashlib
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ThreatLevel(str, Enum):
    """Threat severity levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatReport(BaseModel):
    """Report of detected security threat.

    Attributes:
        id: Unique report ID
        timestamp: When threat was detected
        threat_level: Severity level
        threat_type: Type of threat detected
        agent_id: Agent involved
        description: Threat description
        confidence_score: Confidence 0-1
        details: Additional details
        remediation: Suggested remediation
        escalated: Whether threat was escalated
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Report ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Detection timestamp",
    )
    threat_level: ThreatLevel = Field(
        ..., description="Threat severity"
    )
    threat_type: str = Field(
        ..., description="Type of threat"
    )
    agent_id: Optional[str] = Field(
        default=None, description="Associated agent"
    )
    description: str = Field(
        ..., description="Threat description"
    )
    confidence_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score"
    )
    details: dict = Field(
        default_factory=dict, description="Additional details"
    )
    remediation: str = Field(
        default="", description="Suggested remediation"
    )
    escalated: bool = Field(
        default=False, description="Was escalated"
    )


class ThreatDetector:
    """Detects security threats and anomalies in agent behavior.

    Monitors for:
    - Prompt injection attempts
    - Anomalous agent behavior
    - Secret leakage in outputs
    - Suspicious rate patterns
    - Unauthorized resource access
    """

    def __init__(
        self,
        common_secrets_file: Optional[str] = None,
    ) -> None:
        """Initialize threat detector.

        Args:
            common_secrets_file: Path to file with common secret patterns
        """
        # Threat reports
        self._reports: list[ThreatReport] = []

        # Agent behavior baselines
        self._agent_baselines: dict[str, dict] = {}

        # Rate limiting per agent (for anomaly detection)
        self._agent_rates: dict[str, list[datetime]] = {}

        # Common secret patterns
        self._secret_patterns = self._compile_secret_patterns()

        # Injection attempt patterns
        self._injection_patterns = self._compile_injection_patterns()

        # Alert thresholds
        self._thresholds = {
            "request_rate_per_minute": 100,
            "error_rate_percent": 10,
            "response_time_stddev_factor": 3.0,
            "secret_leakage_confidence": 0.7,
        }

    @staticmethod
    def _compile_secret_patterns() -> list[re.Pattern]:
        """Compile regex patterns for common secrets.

        Returns:
            List of compiled patterns
        """
        patterns = [
            # API keys
            r"api[_-]?key[:\s]*['\"]?([a-zA-Z0-9\-_]{20,})['\"]?",
            # AWS keys
            r"AKIA[0-9A-Z]{16}",
            r"aws[_-]?secret[_-]?access[_-]?key[:\s]*['\"]?([a-zA-Z0-9/+=]{40})['\"]?",
            # GitHub tokens
            r"gh[pousr]{1}_[a-zA-Z0-9_]{36,255}",
            # JWT tokens
            r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.?[a-zA-Z0-9_-]*",
            # Private keys
            r"-----BEGIN[A-Z ]*PRIVATE KEY-----",
            # Database URLs
            r"(postgres|mysql|mongodb)[+://]+[^\s]+:[^\s]+@[^\s]+",
            # OAuth tokens
            r"oauth[_-]?token[:\s]*['\"]?([a-zA-Z0-9_-]{40,})['\"]?",
            # Passwords in connection strings
            r"password[:\s]*['\"]?([^\s'\"]+)['\"]?",
        ]

        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    @staticmethod
    def _compile_injection_patterns() -> list[re.Pattern]:
        """Compile regex patterns for prompt injection attempts.

        Returns:
            List of compiled patterns
        """
        patterns = [
            # Ignore/override attempts
            r"ignore[s]? (previous|earlier|prior|all|my previous|the above)",
            r"disregard",
            r"override[s]? (rule|instruction|constraint|policy)",
            r"bypass[s]?",
            r"jailbreak",
            # Role switching
            r"(you are now|pretend you are|act as|assume the role of|forget you are)",
            r"new instructions?[:\s]",
            # Token/injection markers
            r"<|im_start|>",
            r"<|im_end|>",
            r"\[SYSTEM\]",
            r"\[PROMPT\]",
            # Command injection
            r"(execute|run|eval)[s]? (code|command|bash|sh|python)",
            # Data extraction
            r"(reveal|show|tell|dump|extract) (system|secret|private|hidden|password|key|token)",
        ]

        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    async def detect_prompt_injection(
        self, input_text: str, agent_id: Optional[str] = None
    ) -> Optional[ThreatReport]:
        """Detect prompt injection attempts.

        Args:
            input_text: Text to analyze
            agent_id: Associated agent ID

        Returns:
            ThreatReport if threat detected, None otherwise
        """
        threat_score = 0.0
        matched_patterns = []

        for pattern in self._injection_patterns:
            matches = pattern.findall(input_text)
            if matches:
                threat_score += 0.15
                matched_patterns.append(pattern.pattern)

        # Check for multiple suspicious indicators
        if len(matched_patterns) > 2:
            threat_score += 0.2

        # Check for unusual characters/encoding
        if re.search(r"[^\w\s\-.,!?\"'();:/\\]", input_text):
            threat_score += 0.1

        if threat_score > 0.5:
            threat_level = (
                ThreatLevel.CRITICAL
                if threat_score > 0.85
                else ThreatLevel.HIGH
                if threat_score > 0.75
                else ThreatLevel.MEDIUM
                if threat_score > 0.6
                else ThreatLevel.LOW
            )

            report = ThreatReport(
                threat_level=threat_level,
                threat_type="prompt_injection",
                agent_id=agent_id,
                description=f"Potential prompt injection detected in input",
                confidence_score=threat_score,
                details={
                    "matched_patterns": matched_patterns,
                    "suspicious_length": len(input_text) > 1000,
                    "input_preview": input_text[:100],
                },
                remediation="Review input for malicious intent. Consider blocking or sanitizing.",
            )

            self._reports.append(report)
            logger.warning(
                "prompt_injection_detected",
                report_id=report.id,
                agent_id=agent_id,
                threat_level=threat_level.value,
                confidence=threat_score,
            )

            return report

        return None

    async def detect_secret_leakage(
        self, output_text: str, agent_id: Optional[str] = None
    ) -> Optional[ThreatReport]:
        """Detect potential secret leakage in output.

        Args:
            output_text: Output text to analyze
            agent_id: Associated agent ID

        Returns:
            ThreatReport if threat detected, None otherwise
        """
        leaked_secrets = []
        confidence_scores = []

        for pattern in self._secret_patterns:
            matches = pattern.finditer(output_text)
            for match in matches:
                leaked_secrets.append(match.group(0)[:20] + "...")
                confidence_scores.append(0.8)

        # Check for credential-like patterns
        if re.search(
            r"(password|passphrase|secret|token|key)\s*=\s*[^\s]+",
            output_text,
            re.IGNORECASE,
        ):
            confidence_scores.append(0.7)
            leaked_secrets.append("credential_assignment_pattern")

        if leaked_secrets:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)

            if avg_confidence > self._thresholds["secret_leakage_confidence"]:
                threat_level = (
                    ThreatLevel.CRITICAL
                    if avg_confidence > 0.95
                    else ThreatLevel.HIGH
                    if avg_confidence > 0.85
                    else ThreatLevel.MEDIUM
                )

                report = ThreatReport(
                    threat_level=threat_level,
                    threat_type="secret_leakage",
                    agent_id=agent_id,
                    description=f"Potential secret exposure in agent output",
                    confidence_score=avg_confidence,
                    details={
                        "secret_count": len(leaked_secrets),
                        "detected_types": list(set(leaked_secrets[:5])),
                    },
                    remediation=(
                        "Immediately revoke exposed secrets. Review agent output "
                        "handling and implement output filtering."
                    ),
                )

                self._reports.append(report)
                logger.critical(
                    "secret_leakage_detected",
                    report_id=report.id,
                    agent_id=agent_id,
                    secret_count=len(leaked_secrets),
                    confidence=avg_confidence,
                )

                return report

        return None

    async def detect_anomalous_behavior(
        self,
        agent_id: str,
        request_count: int = 0,
        error_count: int = 0,
        response_time_ms: float = 0,
    ) -> Optional[ThreatReport]:
        """Detect anomalous agent behavior.

        Args:
            agent_id: Agent identifier
            request_count: Number of recent requests
            error_count: Number of recent errors
            response_time_ms: Average response time

        Returns:
            ThreatReport if anomaly detected, None otherwise
        """
        # Initialize baseline if not exists
        if agent_id not in self._agent_baselines:
            self._agent_baselines[agent_id] = {
                "request_count": request_count,
                "error_rate": (
                    error_count / request_count
                    if request_count > 0
                    else 0
                ),
                "avg_response_time": response_time_ms,
                "first_seen": datetime.now(timezone.utc),
            }
            return None

        baseline = self._agent_baselines[agent_id]

        # Check for rate anomalies
        rate_increase = (
            request_count / baseline["request_count"]
            if baseline["request_count"] > 0
            else 1
        )

        anomaly_score = 0.0

        # Detect rate spike (> 3x normal)
        if rate_increase > 3.0:
            anomaly_score += 0.3
            logger.info(
                "rate_spike_detected",
                agent_id=agent_id,
                increase_factor=rate_increase,
            )

        # Detect error rate spike
        current_error_rate = (
            error_count / request_count if request_count > 0 else 0
        )
        error_rate_change = (
            abs(current_error_rate - baseline["error_rate"])
            / baseline["error_rate"]
            if baseline["error_rate"] > 0
            else current_error_rate
        )

        if error_rate_change > 2.0:
            anomaly_score += 0.3
            logger.warning(
                "error_rate_anomaly_detected",
                agent_id=agent_id,
                current_rate=current_error_rate,
                baseline_rate=baseline["error_rate"],
            )

        # Detect response time anomaly
        if baseline["avg_response_time"] > 0:
            response_time_deviation = (
                response_time_ms / baseline["avg_response_time"]
            )
            if response_time_deviation > 3.0 or response_time_deviation < 0.33:
                anomaly_score += 0.2

        if anomaly_score > 0.4:
            threat_level = (
                ThreatLevel.CRITICAL
                if anomaly_score > 0.8
                else ThreatLevel.HIGH
                if anomaly_score > 0.6
                else ThreatLevel.MEDIUM
            )

            report = ThreatReport(
                threat_level=threat_level,
                threat_type="anomalous_behavior",
                agent_id=agent_id,
                description=f"Anomalous behavior pattern detected for agent",
                confidence_score=anomaly_score,
                details={
                    "rate_increase_factor": rate_increase,
                    "error_rate_change": error_rate_change,
                    "baseline_response_time_ms": baseline["avg_response_time"],
                    "current_response_time_ms": response_time_ms,
                },
                remediation="Monitor agent closely. Consider rate limiting or disabling if anomaly continues.",
            )

            self._reports.append(report)
            logger.warning(
                "anomalous_behavior_detected",
                report_id=report.id,
                agent_id=agent_id,
                anomaly_score=anomaly_score,
            )

            return report

        # Update baseline
        baseline["request_count"] = request_count
        baseline["error_rate"] = current_error_rate
        baseline["avg_response_time"] = response_time_ms

        return None

    async def detect_rate_anomaly(
        self, agent_id: str, window_seconds: int = 60
    ) -> Optional[ThreatReport]:
        """Detect unusual request rate patterns.

        Args:
            agent_id: Agent identifier
            window_seconds: Time window for rate analysis

        Returns:
            ThreatReport if anomaly detected, None otherwise
        """
        if agent_id not in self._agent_rates:
            self._agent_rates[agent_id] = []

        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - window_seconds

        # Remove old timestamps
        self._agent_rates[agent_id] = [
            ts for ts in self._agent_rates[agent_id] if ts.timestamp() > cutoff
        ]

        # Add current timestamp
        self._agent_rates[agent_id].append(now)

        request_count = len(self._agent_rates[agent_id])
        rate_per_minute = (request_count / window_seconds) * 60

        if rate_per_minute > self._thresholds["request_rate_per_minute"]:
            report = ThreatReport(
                threat_level=ThreatLevel.HIGH,
                threat_type="rate_anomaly",
                agent_id=agent_id,
                description=f"Unusually high request rate detected",
                confidence_score=0.85,
                details={
                    "request_count": request_count,
                    "rate_per_minute": rate_per_minute,
                    "threshold": self._thresholds["request_rate_per_minute"],
                },
                remediation="Implement rate limiting. Check for runaway agent or attack.",
            )

            self._reports.append(report)
            logger.critical(
                "rate_anomaly_detected",
                report_id=report.id,
                agent_id=agent_id,
                rate_per_minute=rate_per_minute,
            )

            return report

        return None

    def get_threat_report(self, report_id: str) -> Optional[ThreatReport]:
        """Get threat report by ID.

        Args:
            report_id: Report identifier

        Returns:
            ThreatReport if found, None otherwise
        """
        for report in self._reports:
            if report.id == report_id:
                return report
        return None

    def get_recent_threats(
        self,
        agent_id: Optional[str] = None,
        threat_level: Optional[ThreatLevel] = None,
        limit: int = 100,
    ) -> list[ThreatReport]:
        """Get recent threat reports.

        Args:
            agent_id: Filter by agent
            threat_level: Filter by threat level
            limit: Maximum results

        Returns:
            List of threat reports
        """
        results = []

        for report in reversed(self._reports):
            if agent_id and report.agent_id != agent_id:
                continue
            if threat_level and report.threat_level != threat_level:
                continue

            results.append(report)

            if len(results) >= limit:
                break

        return results

    def escalate_threat(self, report_id: str, reason: str = "") -> bool:
        """Escalate threat report for human review.

        Args:
            report_id: Report identifier
            reason: Escalation reason

        Returns:
            True if escalated successfully
        """
        report = self.get_threat_report(report_id)
        if not report:
            return False

        report.escalated = True

        logger.critical(
            "threat_escalated",
            report_id=report_id,
            threat_type=report.threat_type,
            threat_level=report.threat_level.value,
            reason=reason,
        )

        return True

    def get_statistics(self) -> dict:
        """Get threat detection statistics.

        Returns:
            Dict with stats
        """
        stats = {
            "total_threats": len(self._reports),
            "by_level": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
            },
            "by_type": {},
            "escalated_count": 0,
            "agents_with_threats": set(),
        }

        for report in self._reports:
            stats["by_level"][report.threat_level.value] += 1
            stats["by_type"][report.threat_type] = (
                stats["by_type"].get(report.threat_type, 0) + 1
            )
            if report.escalated:
                stats["escalated_count"] += 1
            if report.agent_id:
                stats["agents_with_threats"].add(report.agent_id)

        stats["agents_with_threats"] = list(stats["agents_with_threats"])

        return stats
