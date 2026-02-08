"""
Secret management module for AgentArmy.

Provides secure secret storage with encryption at rest, rotation,
access logging, and Vault integration support.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

import structlog
from cryptography.fernet import Fernet
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class SecretReference(BaseModel):
    """Reference to a secret without exposing its value.

    Attributes:
        name: Secret name (e.g., "github_token", "aws_key")
        version: Version number of secret
        accessed_by: Agent ID that accessed it
        accessed_at: When it was accessed
        purpose: Purpose of access
    """

    name: str = Field(..., description="Secret name")
    version: int = Field(default=1, description="Secret version")
    accessed_by: str = Field(..., description="Agent that accessed it")
    accessed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Access timestamp",
    )
    purpose: str = Field(default="", description="Purpose of access")


class SecretMetadata(BaseModel):
    """Metadata about a stored secret.

    Attributes:
        name: Secret name
        version: Current version
        created_at: When first created
        updated_at: When last updated
        created_by: Agent/user that created it
        rotation_interval: Days between rotations (0 = no rotation)
        last_rotated_at: When last rotated
        access_count: Number of times accessed
        next_rotation_at: When next rotation is due
    """

    name: str = Field(..., description="Secret name")
    version: int = Field(default=1, description="Current version")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Update timestamp",
    )
    created_by: str = Field(..., description="Creator ID")
    rotation_interval: int = Field(
        default=0, description="Days between rotations"
    )
    last_rotated_at: Optional[datetime] = Field(
        default=None, description="Last rotation timestamp"
    )
    access_count: int = Field(default=0, description="Total accesses")
    next_rotation_at: Optional[datetime] = Field(
        default=None, description="Next scheduled rotation"
    )


class SecretsManager:
    """Manages secret storage, rotation, and access control.

    Provides encryption at rest, version control, audit logging,
    and integration points for HashiCorp Vault.
    """

    def __init__(self, master_key: Optional[str] = None) -> None:
        """Initialize secrets manager.

        Args:
            master_key: Encryption master key (generates new if not provided)
        """
        # Generate or use provided encryption key
        if master_key:
            self._cipher = Fernet(master_key.encode())
            self._master_key = master_key
        else:
            self._master_key = Fernet.generate_key().decode()
            self._cipher = Fernet(self._master_key.encode())

        # Secret storage: {name: {version: encrypted_value}}
        self._secrets: dict[str, dict[int, bytes]] = {}

        # Metadata per secret
        self._metadata: dict[str, SecretMetadata] = {}

        # Access audit log
        self._access_log: list[SecretReference] = []

        # Rotation schedule
        self._rotation_schedule: dict[str, datetime] = {}

        logger.info("secrets_manager_initialized")

    def store_secret(
        self,
        name: str,
        value: str,
        created_by: str,
        rotation_interval_days: int = 0,
    ) -> bool:
        """Store a secret securely.

        Args:
            name: Secret name
            value: Secret value
            created_by: Creator identifier
            rotation_interval_days: Days between rotations (0 = no rotation)

        Returns:
            True if stored successfully
        """
        # Encrypt the secret
        encrypted_value = self._cipher.encrypt(value.encode())

        # Initialize secret if new
        if name not in self._secrets:
            self._secrets[name] = {}
            version = 1

            metadata = SecretMetadata(
                name=name,
                version=version,
                created_by=created_by,
                rotation_interval=rotation_interval_days,
            )

            if rotation_interval_days > 0:
                metadata.next_rotation_at = (
                    datetime.now(timezone.utc)
                    + timedelta(days=rotation_interval_days)
                )

            self._metadata[name] = metadata

        else:
            # Update existing secret version
            version = max(self._secrets[name].keys()) + 1
            self._metadata[name].version = version
            self._metadata[name].updated_at = datetime.now(timezone.utc)

        # Store encrypted value
        self._secrets[name][version] = encrypted_value

        logger.info(
            "secret_stored",
            name=name,
            version=version,
            created_by=created_by,
            rotation_interval_days=rotation_interval_days,
        )

        return True

    def retrieve_secret(
        self, name: str, accessed_by: str, purpose: str = ""
    ) -> Optional[str]:
        """Retrieve a secret by name.

        Args:
            name: Secret name
            accessed_by: Agent/user accessing the secret
            purpose: Purpose of access (logged)

        Returns:
            Decrypted secret value, or None if not found
        """
        if name not in self._secrets:
            logger.warning(
                "secret_not_found",
                name=name,
                accessed_by=accessed_by,
            )
            self._log_access_attempt(name, accessed_by, purpose, success=False)
            return None

        # Get latest version
        versions = self._secrets[name]
        latest_version = max(versions.keys())
        encrypted_value = versions[latest_version]

        try:
            # Decrypt
            decrypted_value = self._cipher.decrypt(encrypted_value).decode()

            # Log access
            self._log_access_attempt(
                name, accessed_by, purpose, success=True
            )

            # Update access count
            if name in self._metadata:
                self._metadata[name].access_count += 1

            logger.info(
                "secret_retrieved",
                name=name,
                version=latest_version,
                accessed_by=accessed_by,
                purpose=purpose,
            )

            return decrypted_value

        except Exception as e:
            logger.error(
                "secret_decryption_failed",
                name=name,
                accessed_by=accessed_by,
                error=str(e),
            )
            self._log_access_attempt(name, accessed_by, purpose, success=False)
            return None

    def rotate_secret(
        self,
        name: str,
        new_value: str,
        rotated_by: str,
    ) -> bool:
        """Rotate a secret to a new value.

        Args:
            name: Secret name
            new_value: New secret value
            rotated_by: User/agent performing rotation

        Returns:
            True if rotation successful
        """
        if name not in self._secrets:
            logger.warning("cannot_rotate_nonexistent_secret", name=name)
            return False

        # Store new version
        version = max(self._secrets[name].keys()) + 1
        encrypted_value = self._cipher.encrypt(new_value.encode())
        self._secrets[name][version] = encrypted_value

        # Update metadata
        if name in self._metadata:
            self._metadata[name].last_rotated_at = datetime.now(timezone.utc)
            self._metadata[name].version = version

            if self._metadata[name].rotation_interval > 0:
                self._metadata[name].next_rotation_at = (
                    datetime.now(timezone.utc)
                    + timedelta(
                        days=self._metadata[name].rotation_interval
                    )
                )

        logger.info(
            "secret_rotated",
            name=name,
            new_version=version,
            rotated_by=rotated_by,
        )

        return True

    def get_rotation_due_secrets(self) -> list[str]:
        """Get list of secrets due for rotation.

        Returns:
            List of secret names due for rotation
        """
        now = datetime.now(timezone.utc)
        due_secrets = []

        for name, metadata in self._metadata.items():
            if (
                metadata.rotation_interval > 0
                and metadata.next_rotation_at
                and now >= metadata.next_rotation_at
            ):
                due_secrets.append(name)

        logger.debug(
            "rotation_due_secrets_check",
            count=len(due_secrets),
            secrets=due_secrets,
        )

        return due_secrets

    def schedule_rotation(
        self, name: str, rotation_interval_days: int
    ) -> bool:
        """Schedule automatic rotation for a secret.

        Args:
            name: Secret name
            rotation_interval_days: Days between rotations

        Returns:
            True if scheduled successfully
        """
        if name not in self._metadata:
            logger.warning("secret_not_found_for_scheduling", name=name)
            return False

        self._metadata[name].rotation_interval = rotation_interval_days

        if rotation_interval_days > 0:
            self._metadata[name].next_rotation_at = (
                datetime.now(timezone.utc)
                + timedelta(days=rotation_interval_days)
            )

        logger.info(
            "secret_rotation_scheduled",
            name=name,
            interval_days=rotation_interval_days,
            next_rotation=self._metadata[name].next_rotation_at.isoformat(),
        )

        return True

    def get_access_log(
        self,
        name: Optional[str] = None,
        accessed_by: Optional[str] = None,
        limit: int = 100,
    ) -> list[SecretReference]:
        """Get audit log of secret accesses.

        Args:
            name: Filter by secret name
            accessed_by: Filter by accessor
            limit: Maximum results

        Returns:
            List of access log entries
        """
        results = []

        for entry in reversed(self._access_log):
            if name and entry.name != name:
                continue
            if accessed_by and entry.accessed_by != accessed_by:
                continue

            results.append(entry)

            if len(results) >= limit:
                break

        logger.debug(
            "access_log_queried",
            filters={"name": name, "accessed_by": accessed_by},
            result_count=len(results),
        )

        return results

    def get_secret_metadata(self, name: str) -> Optional[SecretMetadata]:
        """Get metadata for a secret.

        Args:
            name: Secret name

        Returns:
            SecretMetadata if found, None otherwise
        """
        return self._metadata.get(name)

    def get_master_key(self) -> str:
        """Get the master encryption key.

        WARNING: This should only be accessed in secure contexts.
        Never log or expose this key.

        Returns:
            Master key string
        """
        logger.warning("master_key_accessed")
        return self._master_key

    def export_rotation_schedule(self) -> dict:
        """Export rotation schedule for all secrets.

        Returns:
            Dict mapping secret names to next rotation times
        """
        schedule = {}

        for name, metadata in self._metadata.items():
            if metadata.next_rotation_at:
                schedule[name] = metadata.next_rotation_at.isoformat()

        return schedule

    def validate_secret_health(self) -> dict:
        """Validate health and compliance of stored secrets.

        Returns:
            Dict with health checks
        """
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_secrets": len(self._secrets),
            "secrets_needing_rotation": len(self.get_rotation_due_secrets()),
            "issues": [],
        }

        # Check for secrets without rotation schedules
        for name, metadata in self._metadata.items():
            if metadata.rotation_interval == 0:
                results["issues"].append(
                    f"Secret '{name}' has no rotation schedule"
                )

        # Check for old secrets
        old_threshold = datetime.now(timezone.utc) - timedelta(days=365)
        for name, metadata in self._metadata.items():
            if metadata.last_rotated_at and metadata.last_rotated_at < old_threshold:
                results["issues"].append(
                    f"Secret '{name}' last rotated over 1 year ago"
                )

        logger.info(
            "secret_health_validated",
            total_secrets=results["total_secrets"],
            issues_count=len(results["issues"]),
        )

        return results

    def _log_access_attempt(
        self,
        name: str,
        accessed_by: str,
        purpose: str,
        success: bool,
    ) -> None:
        """Log secret access attempt.

        Args:
            name: Secret name
            accessed_by: Accessor ID
            purpose: Purpose of access
            success: Whether access was successful
        """
        reference = SecretReference(
            name=name,
            accessed_by=accessed_by,
            purpose=purpose,
        )

        self._access_log.append(reference)

        log_level = "info" if success else "warning"
        logger_func = getattr(logger, log_level)
        logger_func(
            f"secret_access_{('success' if success else 'failed')}",
            name=name,
            accessed_by=accessed_by,
            purpose=purpose,
        )
