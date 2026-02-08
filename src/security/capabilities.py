"""
Capability-based permission system for AgentArmy.

Implements fine-grained, resource-level access control with support for
elevated permissions, capability manifests, and comprehensive audit logging.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

import structlog
import yaml
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class Permission(str, Enum):
    """Permission types for resources."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"


class ResourceType(str, Enum):
    """Resource types that can be controlled."""

    GIT = "git"
    FILESYSTEM = "filesystem"
    NETWORK = "network"
    LLM = "llm"
    DATABASE = "database"
    SECRETS = "secrets"
    DEPLOYMENT = "deployment"


@dataclass
class CapabilityConstraint:
    """Constraint on a capability usage.

    Attributes:
        name: Constraint name (e.g., "paths", "domains", "timeout")
        value: Constraint value (single value or list)
    """

    name: str
    value: str | list[str] | int | float


class Capability(BaseModel):
    """A capability that allows an agent to perform an action on a resource.

    Attributes:
        id: Unique capability identifier
        agent_id: Agent that has this capability
        resource_type: Type of resource (git, filesystem, etc.)
        permissions: List of allowed permissions
        constraints: Optional constraints on capability usage
        requires_approval: If True, elevated actions need human approval
        created_at: When capability was granted
        expires_at: Optional expiration time
    """

    id: str = Field(default_factory=lambda: str(uuid4()), description="Capability ID")
    agent_id: str = Field(..., description="Agent ID")
    resource_type: ResourceType = Field(..., description="Resource type")
    permissions: list[Permission] = Field(
        default_factory=list, description="Allowed permissions"
    )
    constraints: list[CapabilityConstraint] = Field(
        default_factory=list, description="Usage constraints"
    )
    requires_approval: bool = Field(
        default=False, description="Requires human approval for use"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    expires_at: Optional[datetime] = Field(
        default=None, description="Expiration timestamp"
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


@dataclass
class ApprovalRequest:
    """Request for elevated capability approval.

    Attributes:
        id: Unique request ID
        capability_id: Capability being requested
        agent_id: Agent requesting access
        action: Action being attempted
        target: Target of action
        reason: Reason for elevated access
        timestamp: When request was created
        approved_by: Who approved it (None if pending)
        approved_at: When it was approved
    """

    id: str = None
    capability_id: str = None
    agent_id: str = None
    action: str = None
    target: str = None
    reason: str = None
    timestamp: datetime = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Initialize defaults."""
        if self.id is None:
            self.id = str(uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class CapabilityManager:
    """Manages agent capabilities and permissions.

    Handles capability manifests, permission checks, elevation requests,
    and audit logging of all permission-related activities.
    """

    def __init__(self) -> None:
        """Initialize capability manager."""
        # agent_id -> [capabilities]
        self._capabilities: dict[str, list[Capability]] = {}

        # approval_id -> approval_request
        self._approval_requests: dict[str, ApprovalRequest] = {}

        # Audit log of permission checks
        self._audit_log: list[dict] = []

    def load_from_yaml(self, yaml_path: str) -> bool:
        """Load capabilities from YAML manifest.

        Example manifest:
            agents:
              - id: "agent-1"
                capabilities:
                  - resource_type: "git"
                    permissions: ["read", "write"]
                    constraints:
                      - name: "repositories"
                        value: ["repo1", "repo2"]
                  - resource_type: "filesystem"
                    permissions: ["read"]
                    constraints:
                      - name: "paths"
                        value: ["/data", "/logs"]
                    requires_approval: true

        Args:
            yaml_path: Path to YAML manifest file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(yaml_path, "r") as f:
                manifest = yaml.safe_load(f)

            if not manifest or "agents" not in manifest:
                logger.warning("invalid_manifest_format", path=yaml_path)
                return False

            for agent_config in manifest["agents"]:
                agent_id = agent_config.get("id")
                if not agent_id:
                    logger.warning("agent_missing_id")
                    continue

                self._capabilities[agent_id] = []

                for cap_config in agent_config.get("capabilities", []):
                    try:
                        constraints = [
                            CapabilityConstraint(
                                name=c.get("name"),
                                value=c.get("value"),
                            )
                            for c in cap_config.get("constraints", [])
                        ]

                        capability = Capability(
                            agent_id=agent_id,
                            resource_type=ResourceType(
                                cap_config.get("resource_type")
                            ),
                            permissions=[
                                Permission(p)
                                for p in cap_config.get("permissions", [])
                            ],
                            constraints=constraints,
                            requires_approval=cap_config.get(
                                "requires_approval", False
                            ),
                            expires_at=(
                                datetime.fromisoformat(
                                    cap_config.get("expires_at")
                                )
                                if cap_config.get("expires_at")
                                else None
                            ),
                        )

                        self._capabilities[agent_id].append(capability)

                    except (ValueError, KeyError) as e:
                        logger.warning(
                            "invalid_capability_config",
                            agent_id=agent_id,
                            error=str(e),
                        )

            logger.info(
                "capabilities_loaded_from_yaml",
                path=yaml_path,
                agent_count=len(self._capabilities),
            )
            return True

        except Exception as e:
            logger.error("failed_to_load_yaml_manifest", path=yaml_path, error=str(e))
            return False

    def grant_capability(self, capability: Capability) -> bool:
        """Grant a capability to an agent.

        Args:
            capability: Capability to grant

        Returns:
            True if granted successfully
        """
        if capability.agent_id not in self._capabilities:
            self._capabilities[capability.agent_id] = []

        self._capabilities[capability.agent_id].append(capability)

        logger.info(
            "capability_granted",
            capability_id=capability.id,
            agent_id=capability.agent_id,
            resource_type=capability.resource_type.value,
            permissions=[p.value for p in capability.permissions],
        )

        return True

    def has_capability(
        self,
        agent_id: str,
        resource_type: ResourceType,
        permission: Permission,
        target: Optional[str] = None,
    ) -> bool:
        """Check if agent has capability for action.

        Args:
            agent_id: Agent identifier
            resource_type: Resource type
            permission: Required permission
            target: Optional target (checked against constraints)

        Returns:
            True if agent has capability
        """
        if agent_id not in self._capabilities:
            logger.debug("no_capabilities_for_agent", agent_id=agent_id)
            self._audit_log.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "agent_id": agent_id,
                    "action": "capability_check",
                    "result": "denied",
                    "reason": "no_capabilities",
                    "resource_type": resource_type.value,
                    "permission": permission.value,
                    "target": target,
                }
            )
            return False

        for capability in self._capabilities[agent_id]:
            # Check resource type match
            if capability.resource_type != resource_type:
                continue

            # Check if capability has expired
            if capability.expires_at:
                now = datetime.now(timezone.utc)
                if now > capability.expires_at:
                    logger.debug(
                        "capability_expired",
                        capability_id=capability.id,
                        agent_id=agent_id,
                    )
                    continue

            # Check permission
            if permission not in capability.permissions:
                continue

            # Check constraints
            if target and capability.constraints:
                if not self._check_constraints(capability.constraints, target):
                    continue

            # Found matching capability
            self._audit_log.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "agent_id": agent_id,
                    "action": "capability_check",
                    "result": "granted",
                    "capability_id": capability.id,
                    "resource_type": resource_type.value,
                    "permission": permission.value,
                    "target": target,
                }
            )

            logger.debug(
                "capability_check_passed",
                agent_id=agent_id,
                capability_id=capability.id,
                resource_type=resource_type.value,
                permission=permission.value,
            )

            return True

        # No matching capability found
        self._audit_log.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_id": agent_id,
                "action": "capability_check",
                "result": "denied",
                "reason": "no_matching_capability",
                "resource_type": resource_type.value,
                "permission": permission.value,
                "target": target,
            }
        )

        logger.warning(
            "capability_denied",
            agent_id=agent_id,
            resource_type=resource_type.value,
            permission=permission.value,
            target=target,
        )

        return False

    def requires_approval(
        self,
        agent_id: str,
        resource_type: ResourceType,
        permission: Permission,
        target: Optional[str] = None,
    ) -> bool:
        """Check if action requires human approval.

        Args:
            agent_id: Agent identifier
            resource_type: Resource type
            permission: Required permission
            target: Optional target

        Returns:
            True if approval is required
        """
        if agent_id not in self._capabilities:
            return False

        for capability in self._capabilities[agent_id]:
            if (
                capability.resource_type == resource_type
                and permission in capability.permissions
            ):
                if capability.requires_approval:
                    logger.info(
                        "approval_required",
                        agent_id=agent_id,
                        resource_type=resource_type.value,
                        permission=permission.value,
                    )
                    return True

        return False

    def request_elevated_access(
        self,
        agent_id: str,
        capability_id: str,
        action: str,
        target: str,
        reason: str,
    ) -> str:
        """Request elevated access for a capability.

        Args:
            agent_id: Agent requesting access
            capability_id: Capability ID being requested
            action: Action to perform
            target: Target of action
            reason: Reason for elevated access

        Returns:
            Approval request ID
        """
        request = ApprovalRequest(
            capability_id=capability_id,
            agent_id=agent_id,
            action=action,
            target=target,
            reason=reason,
        )

        self._approval_requests[request.id] = request

        logger.info(
            "elevated_access_requested",
            request_id=request.id,
            agent_id=agent_id,
            capability_id=capability_id,
            action=action,
            target=target,
        )

        return request.id

    def approve_elevated_access(
        self, request_id: str, approved_by: str
    ) -> bool:
        """Approve an elevated access request.

        Args:
            request_id: Approval request ID
            approved_by: User ID of approver

        Returns:
            True if approved successfully
        """
        if request_id not in self._approval_requests:
            logger.warning("approval_request_not_found", request_id=request_id)
            return False

        request = self._approval_requests[request_id]
        request.approved_by = approved_by
        request.approved_at = datetime.now(timezone.utc)

        logger.info(
            "elevated_access_approved",
            request_id=request_id,
            agent_id=request.agent_id,
            approved_by=approved_by,
            action=request.action,
            target=request.target,
        )

        return True

    def deny_elevated_access(self, request_id: str, reason: str = "") -> bool:
        """Deny an elevated access request.

        Args:
            request_id: Approval request ID
            reason: Reason for denial

        Returns:
            True if denied successfully
        """
        if request_id not in self._approval_requests:
            logger.warning("approval_request_not_found", request_id=request_id)
            return False

        request = self._approval_requests[request_id]
        del self._approval_requests[request_id]

        logger.info(
            "elevated_access_denied",
            request_id=request_id,
            agent_id=request.agent_id,
            reason=reason,
        )

        return True

    def get_agent_capabilities(self, agent_id: str) -> list[Capability]:
        """Get all capabilities for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            List of capabilities
        """
        return self._capabilities.get(agent_id, [])

    def get_audit_log(self, agent_id: Optional[str] = None) -> list[dict]:
        """Get audit log of permission checks.

        Args:
            agent_id: Filter by agent ID (optional)

        Returns:
            List of audit entries
        """
        if agent_id:
            return [e for e in self._audit_log if e.get("agent_id") == agent_id]
        return self._audit_log

    @staticmethod
    def _check_constraints(
        constraints: list[CapabilityConstraint], target: str
    ) -> bool:
        """Check if target matches constraints.

        Args:
            constraints: List of constraints
            target: Target to check

        Returns:
            True if target matches all constraints
        """
        for constraint in constraints:
            if constraint.name in ("paths", "repositories", "domains"):
                allowed_values = (
                    constraint.value
                    if isinstance(constraint.value, list)
                    else [constraint.value]
                )
                if target not in allowed_values:
                    return False

        return True
