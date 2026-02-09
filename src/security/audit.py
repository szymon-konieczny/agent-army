"""
Immutable audit trail module for Code Horde.

Implements hash-chained audit logging inspired by Prooflog's blockchain-style
integrity verification. Each audit entry is cryptographically linked to the
previous entry, forming an immutable chain that can be verified at any time.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class AuditEntry(BaseModel):
    """Immutable audit log entry with cryptographic integrity.

    Attributes:
        entry_id: Unique entry identifier
        timestamp: When action occurred
        agent_id: Agent that performed action
        action: Action type (e.g., "file_read", "secret_access")
        target: Target of action (e.g., file path, resource ID)
        input_hash: SHA256 hash of input data
        output_hash: SHA256 hash of output data
        prev_hash: Hash of previous entry (forms chain)
        entry_hash: Self-hash (depends on all above fields)
        signature: Ed25519 signature by agent's private key
        metadata: Additional context
    """

    entry_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique entry ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Action timestamp",
    )
    agent_id: str = Field(..., description="Agent ID")
    action: str = Field(..., description="Action type")
    target: str = Field(..., description="Action target")
    input_hash: str = Field(
        default="", description="SHA256 hash of input data"
    )
    output_hash: str = Field(
        default="", description="SHA256 hash of output data"
    )
    prev_hash: str = Field(
        default="", description="Hash of previous entry"
    )
    entry_hash: str = Field(
        default="", description="Self hash of this entry"
    )
    signature: str = Field(
        default="", description="Ed25519 signature"
    )
    metadata: dict = Field(
        default_factory=dict, description="Additional metadata"
    )

    def compute_hash(self) -> str:
        """Compute SHA256 hash of entry contents.

        Returns:
            Hex digest of hash
        """
        # Create stable JSON representation
        content = {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "action": self.action,
            "target": self.target,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "prev_hash": self.prev_hash,
            "metadata": self.metadata,
        }

        json_str = json.dumps(content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode()).hexdigest()

    def sign(self, private_key_pem: bytes) -> None:
        """Sign entry with Ed25519 private key.

        Args:
            private_key_pem: Private key in PEM format
        """
        # Compute entry hash first
        self.entry_hash = self.compute_hash()

        # Sign the entry hash
        private_key = serialization.load_pem_private_key(
            private_key_pem, password=None
        )
        signature_bytes = private_key.sign(self.entry_hash.encode())
        self.signature = signature_bytes.hex()

    def verify_signature(self, public_key_pem: bytes) -> bool:
        """Verify Ed25519 signature.

        Args:
            public_key_pem: Public key in PEM format

        Returns:
            True if signature is valid
        """
        if not self.signature:
            return False

        try:
            public_key = serialization.load_pem_public_key(public_key_pem)
            public_key.verify(
                bytes.fromhex(self.signature),
                self.entry_hash.encode(),
            )
            return True
        except Exception as e:
            logger.warning(
                "signature_verification_failed",
                entry_id=self.entry_id,
                error=str(e),
            )
            return False

    @staticmethod
    def hash_data(data: str | bytes) -> str:
        """Hash arbitrary data using SHA256.

        Args:
            data: Data to hash

        Returns:
            Hex digest of hash
        """
        if isinstance(data, str):
            data = data.encode()
        return hashlib.sha256(data).hexdigest()


class AuditTrail:
    """Immutable, hash-chained audit trail for security events.

    Maintains cryptographic integrity of audit log by linking each entry
    to the previous one via hash chains and cryptographic signatures.
    """

    def __init__(self) -> None:
        """Initialize audit trail."""
        self._entries: list[AuditEntry] = []
        self._entry_by_id: dict[str, AuditEntry] = {}
        self._agent_keys: dict[str, tuple[bytes, bytes]] = {}  # pub, priv key pairs

    def register_agent(self, agent_id: str) -> tuple[str, str]:
        """Register agent with Ed25519 key pair.

        Args:
            agent_id: Agent identifier

        Returns:
            Tuple of (public_key_pem, private_key_pem) as strings
        """
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        self._agent_keys[agent_id] = (public_pem, private_pem)

        logger.info(
            "agent_registered_for_audit",
            agent_id=agent_id,
            key_id=public_pem[:32].hex(),
        )

        return (public_pem.decode(), private_pem.decode())

    def append_entry(
        self,
        agent_id: str,
        action: str,
        target: str,
        input_data: str = "",
        output_data: str = "",
        metadata: Optional[dict] = None,
    ) -> str:
        """Append entry to audit trail.

        Entry is automatically linked to previous entry via hash chain
        and signed with agent's private key.

        Args:
            agent_id: Agent performing action
            action: Action type
            target: Action target
            input_data: Input data (will be hashed)
            output_data: Output data (will be hashed)
            metadata: Additional metadata

        Returns:
            Entry ID
        """
        if agent_id not in self._agent_keys:
            logger.error(
                "agent_not_registered", agent_id=agent_id, action=action
            )
            raise ValueError(f"Agent {agent_id} not registered for audit")

        # Create entry
        entry = AuditEntry(
            agent_id=agent_id,
            action=action,
            target=target,
            input_hash=AuditEntry.hash_data(input_data),
            output_hash=AuditEntry.hash_data(output_data),
            prev_hash=self._entries[-1].entry_hash if self._entries else "",
            metadata=metadata or {},
        )

        # Sign entry
        public_pem, private_pem = self._agent_keys[agent_id]
        entry.sign(private_pem)

        # Append to trail
        self._entries.append(entry)
        self._entry_by_id[entry.entry_id] = entry

        logger.info(
            "audit_entry_appended",
            entry_id=entry.entry_id,
            agent_id=agent_id,
            action=action,
            target=target,
            chain_length=len(self._entries),
        )

        return entry.entry_id

    def verify_chain_integrity(self) -> bool:
        """Verify integrity of entire audit chain.

        Checks that:
        1. Each entry's hash is correct
        2. Each entry's signature is valid
        3. Hash chain is unbroken

        Returns:
            True if chain is valid, False if tampered
        """
        for i, entry in enumerate(self._entries):
            # Verify self-hash
            expected_hash = entry.compute_hash()
            if entry.entry_hash != expected_hash:
                logger.error(
                    "chain_integrity_check_failed",
                    entry_id=entry.entry_id,
                    reason="entry_hash_mismatch",
                    index=i,
                )
                return False

            # Verify signature
            if entry.agent_id in self._agent_keys:
                public_pem, _ = self._agent_keys[entry.agent_id]
                if not entry.verify_signature(public_pem):
                    logger.error(
                        "chain_integrity_check_failed",
                        entry_id=entry.entry_id,
                        reason="signature_invalid",
                        index=i,
                    )
                    return False

            # Verify hash chain link (except first entry)
            if i > 0:
                if entry.prev_hash != self._entries[i - 1].entry_hash:
                    logger.error(
                        "chain_integrity_check_failed",
                        entry_id=entry.entry_id,
                        reason="prev_hash_mismatch",
                        index=i,
                    )
                    return False

        logger.info(
            "chain_integrity_verified",
            entry_count=len(self._entries),
        )
        return True

    def query_entries(
        self,
        agent_id: Optional[str] = None,
        action: Optional[str] = None,
        target: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit log with filters.

        Args:
            agent_id: Filter by agent ID
            action: Filter by action type
            target: Filter by target
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp
            limit: Maximum results

        Returns:
            List of matching entries
        """
        results = []

        for entry in self._entries:
            # Apply filters
            if agent_id and entry.agent_id != agent_id:
                continue
            if action and entry.action != action:
                continue
            if target and entry.target != target:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue

            results.append(entry)

            if len(results) >= limit:
                break

        logger.debug(
            "audit_query_executed",
            filters={
                "agent_id": agent_id,
                "action": action,
                "target": target,
            },
            result_count=len(results),
        )

        return results

    def export_for_compliance(self, format: str = "json") -> str:
        """Export audit trail for compliance/audit purposes.

        Args:
            format: Export format (json or csv)

        Returns:
            Exported audit trail as string
        """
        if format == "json":
            entries = [
                {
                    "entry_id": e.entry_id,
                    "timestamp": e.timestamp.isoformat(),
                    "agent_id": e.agent_id,
                    "action": e.action,
                    "target": e.target,
                    "prev_hash": e.prev_hash,
                    "entry_hash": e.entry_hash,
                    "signature": e.signature,
                    "metadata": e.metadata,
                }
                for e in self._entries
            ]
            return json.dumps(entries, indent=2)

        elif format == "csv":
            lines = [
                "entry_id,timestamp,agent_id,action,target,prev_hash,entry_hash"
            ]
            for e in self._entries:
                lines.append(
                    f'"{e.entry_id}","{e.timestamp.isoformat()}","{e.agent_id}",'
                    f'"{e.action}","{e.target}","{e.prev_hash}","{e.entry_hash}"'
                )
            return "\n".join(lines)

        else:
            logger.warning("unsupported_export_format", format=format)
            return ""

    def periodic_integrity_check(self) -> dict:
        """Perform periodic integrity check of audit trail.

        Returns:
            Dict with check results
        """
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_entries": len(self._entries),
            "chain_valid": self.verify_chain_integrity(),
            "tamper_detected": False,
            "details": [],
        }

        if not results["chain_valid"]:
            results["tamper_detected"] = True
            logger.critical("tamper_detected_in_audit_trail")

        logger.info(
            "audit_trail_integrity_check_completed",
            total_entries=len(self._entries),
            chain_valid=results["chain_valid"],
            tamper_detected=results["tamper_detected"],
        )

        return results

    def get_entry(self, entry_id: str) -> Optional[AuditEntry]:
        """Get specific entry by ID.

        Args:
            entry_id: Entry identifier

        Returns:
            Entry if found, None otherwise
        """
        return self._entry_by_id.get(entry_id)

    def get_entries_since(self, entry_id: str) -> list[AuditEntry]:
        """Get all entries after a specific entry in chain.

        Args:
            entry_id: Reference entry ID

        Returns:
            List of entries after reference
        """
        if entry_id not in self._entry_by_id:
            return []

        found = False
        results = []

        for entry in self._entries:
            if entry.entry_id == entry_id:
                found = True
                continue

            if found:
                results.append(entry)

        return results
