"""Agent-to-Agent (A2A) communication protocol for secure inter-agent messaging."""

import hmac
import hashlib
import json
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional, Any, Callable, Awaitable
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field


logger = structlog.get_logger(__name__)


class A2AMessageType(str, Enum):
    """Types of messages in the A2A protocol."""

    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_RESPONSE = "capability_response"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class A2AMessage(BaseModel):
    """Agent-to-Agent message with security and tracking.

    All messages are signed to ensure authenticity and integrity.
    """

    protocol_version: str = Field(
        default="1.0", description="A2A protocol version"
    )
    message_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique message identifier",
    )
    from_agent: str = Field(..., description="ID of the sending agent")
    to_agent: str = Field(..., description="ID of the receiving agent")
    type: A2AMessageType = Field(..., description="Type of message")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Message-specific data"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When message was created",
    )
    signature: Optional[str] = Field(
        default=None,
        description="HMAC signature for verification",
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="ID of message this is a response to",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (priority, tags, etc.)",
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "protocol_version": "1.0",
                "message_id": "550e8400-e29b-41d4-a716-446655440000",
                "from_agent": "agent-coordinator",
                "to_agent": "agent-worker-1",
                "type": "task_request",
                "payload": {
                    "task_id": "task-123",
                    "task_type": "data_processing",
                    "parameters": {"input_file": "data.csv"},
                },
                "timestamp": "2024-01-15T10:30:00Z",
                "signature": "sha256=abc123...",
                "metadata": {"priority": "high", "retry_count": 0},
            }
        }


class A2ACapability(BaseModel):
    """Describes an agent's capabilities."""

    name: str = Field(..., description="Capability name")
    version: str = Field(..., description="Capability version")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Input parameters the capability accepts",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description",
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "name": "process_data",
                "version": "1.0",
                "parameters": {
                    "input_format": "csv",
                    "output_format": "json",
                },
                "description": "Process CSV data and convert to JSON",
            }
        }


class A2AProtocol:
    """Agent-to-Agent communication protocol.

    Manages secure message exchange, capability discovery, and task delegation
    between agents with signature verification and correlation tracking.
    """

    def __init__(
        self,
        agent_id: str,
        secret_key: str,
        message_ttl_seconds: int = 3600,
    ) -> None:
        """Initialize A2A protocol handler.

        Args:
            agent_id: This agent's unique identifier.
            secret_key: Secret key for signing/verifying messages.
            message_ttl_seconds: How long messages are considered valid.
        """
        self.agent_id = agent_id
        self.secret_key = secret_key
        self.message_ttl_seconds = message_ttl_seconds

        # Capability registry
        self.capabilities: dict[str, A2ACapability] = {}

        # Message handlers by type
        self.handlers: dict[A2AMessageType, list[Callable]] = {
            msg_type: [] for msg_type in A2AMessageType
        }

        # Pending requests waiting for responses
        self.pending_requests: dict[str, A2AMessage] = {}

        # Message history for debugging
        self.message_history: list[A2AMessage] = []
        self.max_history_size = 1000

        logger.info(
            "A2A Protocol initialized",
            agent_id=agent_id,
            message_ttl_seconds=message_ttl_seconds,
        )

    def register_capability(self, capability: A2ACapability) -> None:
        """Register a capability this agent can perform.

        Args:
            capability: Capability to register.
        """
        self.capabilities[capability.name] = capability
        logger.info(
            "Capability registered",
            agent_id=self.agent_id,
            capability_name=capability.name,
            capability_version=capability.version,
        )

    def get_capabilities(self) -> dict[str, A2ACapability]:
        """Get all registered capabilities.

        Returns:
            Dictionary of capabilities by name.
        """
        return self.capabilities.copy()

    def register_handler(
        self,
        message_type: A2AMessageType,
        handler: Callable[[A2AMessage], Awaitable[None]],
    ) -> None:
        """Register an async handler for message type.

        Args:
            message_type: Type of message to handle.
            handler: Async callback function.
        """
        self.handlers[message_type].append(handler)
        logger.info(
            "Message handler registered",
            agent_id=self.agent_id,
            message_type=message_type,
        )

    async def send_message(
        self, message: A2AMessage
    ) -> A2AMessage:
        """Send a signed message to another agent.

        Args:
            message: Message to send (must have to_agent set).

        Returns:
            The signed message (with signature added).

        Raises:
            ValueError: If message validation fails.
        """
        if not message.from_agent:
            message.from_agent = self.agent_id

        if message.from_agent != self.agent_id:
            raise ValueError(
                f"Cannot send message from {message.from_agent}, "
                f"this agent is {self.agent_id}"
            )

        # Sign the message
        self._sign_message(message)

        # Store in history
        self._store_message(message)

        # Track if waiting for response
        if message.type in [
            A2AMessageType.TASK_REQUEST,
            A2AMessageType.CAPABILITY_QUERY,
        ]:
            self.pending_requests[message.message_id] = message

        logger.info(
            "Message sent",
            agent_id=self.agent_id,
            to_agent=message.to_agent,
            message_type=message.type,
            message_id=message.message_id,
        )

        return message

    async def receive_message(
        self, message: A2AMessage
    ) -> bool:
        """Receive and process a message from another agent.

        Args:
            message: Incoming message.

        Returns:
            True if message was valid and processed, False otherwise.
        """
        # Verify signature
        if not self._verify_signature(message):
            logger.error(
                "Message signature verification failed",
                agent_id=self.agent_id,
                from_agent=message.from_agent,
                message_id=message.message_id,
            )
            return False

        # Check TTL
        if not self._check_ttl(message):
            logger.warning(
                "Message expired",
                agent_id=self.agent_id,
                message_id=message.message_id,
            )
            return False

        # Store in history
        self._store_message(message)

        # Dispatch to handlers
        await self._dispatch_message(message)

        logger.info(
            "Message received and processed",
            agent_id=self.agent_id,
            from_agent=message.from_agent,
            message_type=message.type,
            message_id=message.message_id,
        )

        return True

    async def send_task_request(
        self,
        to_agent: str,
        task_id: str,
        task_type: str,
        parameters: dict[str, Any],
    ) -> A2AMessage:
        """Send a task request to another agent.

        Args:
            to_agent: Target agent ID.
            task_id: Unique task identifier.
            task_type: Type of task to perform.
            parameters: Task-specific parameters.

        Returns:
            The sent message.
        """
        message = A2AMessage(
            from_agent=self.agent_id,
            to_agent=to_agent,
            type=A2AMessageType.TASK_REQUEST,
            payload={
                "task_id": task_id,
                "task_type": task_type,
                "parameters": parameters,
            },
        )

        return await self.send_message(message)

    async def send_task_response(
        self,
        to_agent: str,
        correlation_id: str,
        status: str,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> A2AMessage:
        """Send a task response back to the requesting agent.

        Args:
            to_agent: Target agent ID (original requester).
            correlation_id: ID of the original task request.
            status: Task status (success, failure, in_progress).
            result: Result data if successful.
            error: Error message if failed.

        Returns:
            The sent message.
        """
        message = A2AMessage(
            from_agent=self.agent_id,
            to_agent=to_agent,
            type=A2AMessageType.TASK_RESPONSE,
            correlation_id=correlation_id,
            payload={
                "status": status,
                "result": result,
                "error": error,
            },
        )

        # Remove from pending
        if correlation_id in self.pending_requests:
            del self.pending_requests[correlation_id]

        return await self.send_message(message)

    async def query_capabilities(
        self, to_agent: str
    ) -> Optional[list[A2ACapability]]:
        """Query another agent's capabilities.

        Args:
            to_agent: Target agent ID.

        Returns:
            List of capabilities, or None if query fails.
        """
        message = A2AMessage(
            from_agent=self.agent_id,
            to_agent=to_agent,
            type=A2AMessageType.CAPABILITY_QUERY,
        )

        await self.send_message(message)

        # In a real system, this would wait for a response
        logger.info(
            "Capability query sent",
            to_agent=to_agent,
            message_id=message.message_id,
        )

        return None

    def get_message_history(
        self,
        from_agent: Optional[str] = None,
        to_agent: Optional[str] = None,
        message_type: Optional[A2AMessageType] = None,
        limit: Optional[int] = None,
    ) -> list[A2AMessage]:
        """Get message history (most recent first).

        Args:
            from_agent: Filter by sender.
            to_agent: Filter by recipient.
            message_type: Filter by message type.
            limit: Maximum number of messages to return.

        Returns:
            List of messages matching filters.
        """
        history = list(reversed(self.message_history))

        if from_agent:
            history = [m for m in history if m.from_agent == from_agent]

        if to_agent:
            history = [m for m in history if m.to_agent == to_agent]

        if message_type:
            history = [m for m in history if m.type == message_type]

        if limit:
            history = history[:limit]

        return history

    def clear_message_history(self) -> None:
        """Clear all stored message history."""
        self.message_history.clear()
        logger.info("Message history cleared", agent_id=self.agent_id)

    # Private methods

    def _sign_message(self, message: A2AMessage) -> None:
        """Sign a message with HMAC.

        Args:
            message: Message to sign (signature field will be updated).
        """
        # Create payload to sign (all fields except signature)
        payload = {
            "protocol_version": message.protocol_version,
            "message_id": message.message_id,
            "from_agent": message.from_agent,
            "to_agent": message.to_agent,
            "type": message.type.value,
            "payload": message.payload,
            "timestamp": message.timestamp.isoformat(),
            "correlation_id": message.correlation_id,
        }

        payload_str = json.dumps(payload, sort_keys=True)

        signature = (
            "sha256="
            + hmac.new(
                self.secret_key.encode(),
                payload_str.encode(),
                hashlib.sha256,
            ).hexdigest()
        )

        message.signature = signature

    def _verify_signature(self, message: A2AMessage) -> bool:
        """Verify a message's signature.

        Args:
            message: Message to verify.

        Returns:
            True if signature is valid, False otherwise.
        """
        if not message.signature:
            logger.warning(
                "Message has no signature",
                message_id=message.message_id,
            )
            return False

        # Recreate payload and calculate expected signature
        payload = {
            "protocol_version": message.protocol_version,
            "message_id": message.message_id,
            "from_agent": message.from_agent,
            "to_agent": message.to_agent,
            "type": message.type.value,
            "payload": message.payload,
            "timestamp": message.timestamp.isoformat(),
            "correlation_id": message.correlation_id,
        }

        payload_str = json.dumps(payload, sort_keys=True)

        expected_signature = (
            "sha256="
            + hmac.new(
                self.secret_key.encode(),
                payload_str.encode(),
                hashlib.sha256,
            ).hexdigest()
        )

        is_valid = hmac.compare_digest(message.signature, expected_signature)

        if not is_valid:
            logger.warning(
                "Signature mismatch",
                message_id=message.message_id,
                from_agent=message.from_agent,
            )

        return is_valid

    def _check_ttl(self, message: A2AMessage) -> bool:
        """Check if message is still within TTL.

        Args:
            message: Message to check.

        Returns:
            True if message is valid, False if expired.
        """
        now = datetime.utcnow()
        message_age = (now - message.timestamp).total_seconds()

        if message_age > self.message_ttl_seconds:
            logger.warning(
                "Message TTL exceeded",
                message_id=message.message_id,
                age_seconds=message_age,
                ttl_seconds=self.message_ttl_seconds,
            )
            return False

        return True

    async def _dispatch_message(self, message: A2AMessage) -> None:
        """Dispatch message to registered handlers.

        Args:
            message: Message to dispatch.
        """
        handlers = self.handlers.get(message.type, [])

        if not handlers:
            logger.debug(
                "No handlers for message type",
                agent_id=self.agent_id,
                message_type=message.type,
            )
            return

        for handler in handlers:
            try:
                await handler(message)
            except Exception as e:
                logger.error(
                    "Handler execution failed",
                    agent_id=self.agent_id,
                    message_id=message.message_id,
                    error=str(e),
                )

    def _store_message(self, message: A2AMessage) -> None:
        """Store message in history.

        Args:
            message: Message to store.
        """
        self.message_history.append(message)

        # Trim to max size
        if len(self.message_history) > self.max_history_size:
            self.message_history = self.message_history[-self.max_history_size :]
