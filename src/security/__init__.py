"""
Security subsystem for AgentArmy.

This module provides comprehensive security features for multi-agent systems:
- Authentication & Authorization (JWT, API keys, rate limiting)
- Capability-based permissions (resource-level access control)
- Immutable audit trails (hash-chained event logging)
- Secret management (encrypted storage with rotation)
- Execution sandboxing (resource isolation and limits)
- Threat detection (injection, anomaly, leakage detection)

All components are designed for production use with cryptographic integrity,
async support, and compliance requirements in mind.
"""

from src.security.audit import AuditEntry, AuditTrail
from src.security.auth import AuthManager, TokenPayload
from src.security.capabilities import (
    Capability,
    CapabilityManager,
    Permission,
    ResourceType,
)
from src.security.sandbox import Sandbox, SandboxConfig
from src.security.secrets import SecretReference, SecretsManager
from src.security.threat_detector import ThreatDetector, ThreatLevel, ThreatReport

__all__ = [
    # Authentication & Authorization
    "TokenPayload",
    "AuthManager",
    # Capabilities
    "Permission",
    "ResourceType",
    "Capability",
    "CapabilityManager",
    # Audit Trail
    "AuditEntry",
    "AuditTrail",
    # Secrets
    "SecretReference",
    "SecretsManager",
    # Sandbox
    "SandboxConfig",
    "Sandbox",
    # Threat Detection
    "ThreatLevel",
    "ThreatReport",
    "ThreatDetector",
]
