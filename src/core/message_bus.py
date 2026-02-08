"""Unified message bus for agent communication using Redis and RabbitMQ."""

import hashlib
import hmac
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

import redis.asyncio as redis
import structlog
from aio_pika import Channel, Connection, Exchange, IncomingMessage, Queue
from aio_pika.patterns import RPC
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class MessageType(str, Enum):
    """Types of messages that can be exchanged.

    Attributes:
        COMMAND: Direct command to execute.
        EVENT: Broadcast event notification.
        REQUEST: Request/reply pattern call.
        RESPONSE: Response to a request.
        NOTIFICATION: General notification.
        ERROR: Error notification.
    """

    COMMAND = "command"
    EVENT = "event"
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class Message(BaseModel):
    """Message envelope for agent communication.

    Attributes:
        id: Unique message identifier.
        from_agent: Sender agent ID.
        to_agent: Recipient agent ID (optional, for broadcast).
        message_type: Type of message.
        payload: Message data.
        timestamp: Message creation time.
        correlation_id: For linking requests/responses.
        signature: HMAC-SHA256 signature for authentication.
        expiry_seconds: Message expiry time in seconds.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Message ID")
    from_agent: str = Field(description="Sender agent ID")
    to_agent: Optional[str] = Field(default=None, description="Recipient agent ID")
    message_type: MessageType = Field(description="Message type")
    payload: dict[str, Any] = Field(default_factory=dict, description="Message payload")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = Field(default=None, description="Correlation ID")
    signature: Optional[str] = Field(default=None, description="HMAC signature")
    expiry_seconds: int = Field(default=3600, gt=0, description="Message expiry")

    def serialize(self) -> str:
        """Serialize message to JSON string.

        Returns:
            JSON string representation.
        """
        return self.model_dump_json()

    @classmethod
    def deserialize(cls, data: str) -> "Message":
        """Deserialize message from JSON string.

        Args:
            data: JSON string to deserialize.

        Returns:
            Message object.

        Raises:
            ValueError: If deserialization fails.
        """
        try:
            return cls.model_validate_json(data)
        except Exception as exc:
            raise ValueError(f"Failed to deserialize message: {exc}") from exc

    def sign(self, secret: str) -> None:
        """Sign the message with HMAC-SHA256.

        Args:
            secret: Secret key for signing.
        """
        message_data = (
            f"{self.id}:{self.from_agent}:{self.to_agent}:"
            f"{self.message_type.value}:{self.timestamp.isoformat()}"
        )
        self.signature = hmac.new(
            secret.encode(),
            message_data.encode(),
            hashlib.sha256,
        ).hexdigest()

    def verify(self, secret: str) -> bool:
        """Verify the message signature.

        Args:
            secret: Secret key for verification.

        Returns:
            True if signature is valid, False otherwise.
        """
        if not self.signature:
            return False

        current_sig = self.signature
        self.signature = None
        self.sign(secret)
        calculated_sig = self.signature
        self.signature = current_sig

        return hmac.compare_digest(current_sig, calculated_sig)


class MessageBus:
    """Unified message bus for agent communication.

    Combines Redis (for fast ephemeral messages) and RabbitMQ (for durable queues).

    Attributes:
        redis_client: Redis async client.
        rabbitmq_connection: RabbitMQ connection.
        subscribers: Registered message subscribers.
    """

    def __init__(
        self,
        redis_url: str,
        rabbitmq_url: str,
    ) -> None:
        """Initialize the message bus.

        Args:
            redis_url: Redis connection URL.
            rabbitmq_url: RabbitMQ connection URL.
        """
        self.redis_url = redis_url
        self.rabbitmq_url = rabbitmq_url
        self.redis_client: Optional[redis.Redis[bytes]] = None
        self.rabbitmq_connection: Optional[Connection] = None
        self.rabbitmq_channel: Optional[Channel] = None
        self.rpc: Optional[RPC] = None
        self.subscribers: dict[str, list[Callable[[Message], Any]]] = {}
        self._logger = structlog.get_logger(__name__)

    async def startup(self) -> None:
        """Initialize connections to Redis and RabbitMQ.

        Raises:
            Exception: If connection fails.
        """
        try:
            # Connect to Redis
            self.redis_client = await redis.from_url(self.redis_url, decode_responses=False)
            await self.redis_client.ping()

            # Connect to RabbitMQ
            self.rabbitmq_connection = await Connection.connect(self.rabbitmq_url)
            self.rabbitmq_channel = await self.rabbitmq_connection.channel()
            self.rpc = RPC(self.rabbitmq_channel)

            await self._logger.ainfo(
                "message_bus_startup_success",
                redis_url=self.redis_url,
                rabbitmq_url=self.rabbitmq_url,
            )

        except Exception as exc:
            await self._logger.aerror(
                "message_bus_startup_failed",
                error=str(exc),
            )
            raise

    async def shutdown(self) -> None:
        """Close connections to Redis and RabbitMQ.

        Raises:
            Exception: If shutdown fails.
        """
        try:
            if self.redis_client:
                await self.redis_client.close()

            if self.rabbitmq_connection:
                await self.rabbitmq_connection.close()

            await self._logger.ainfo("message_bus_shutdown_success")

        except Exception as exc:
            await self._logger.aerror(
                "message_bus_shutdown_failed",
                error=str(exc),
            )
            raise

    async def publish(
        self,
        message: Message,
        durable: bool = False,
    ) -> None:
        """Publish a message to the bus.

        Uses Redis for fast ephemeral messages, RabbitMQ for durable messages.

        Args:
            message: Message to publish.
            durable: Whether to persist message (use RabbitMQ).

        Raises:
            RuntimeError: If message bus not initialized.
        """
        if not self.redis_client:
            raise RuntimeError("Message bus not initialized")

        channel = f"agent:{message.to_agent}" if message.to_agent else "broadcast"

        try:
            if durable and self.rabbitmq_channel:
                await self._publish_to_rabbitmq(message, channel)
            else:
                await self._publish_to_redis(message, channel)

            await self._logger.ainfo(
                "message_published",
                message_id=message.id,
                from_agent=message.from_agent,
                to_agent=message.to_agent,
                message_type=message.message_type.value,
                durable=durable,
            )

        except Exception as exc:
            await self._logger.aerror(
                "message_publish_failed",
                message_id=message.id,
                error=str(exc),
            )
            raise

    async def _publish_to_redis(self, message: Message, channel: str) -> None:
        """Publish message to Redis pub/sub.

        Args:
            message: Message to publish.
            channel: Redis channel name.
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")

        await self.redis_client.publish(channel, message.serialize())

    async def _publish_to_rabbitmq(self, message: Message, queue_name: str) -> None:
        """Publish message to RabbitMQ queue.

        Args:
            message: Message to publish.
            queue_name: Queue name.
        """
        if not self.rabbitmq_channel:
            raise RuntimeError("RabbitMQ channel not initialized")

        exchange = await self.rabbitmq_channel.declare_exchange(
            "agent_messages",
            auto_delete=False,
        )

        queue = await self.rabbitmq_channel.declare_queue(queue_name, auto_delete=False)
        await queue.bind(exchange, routing_key=queue_name)

        await exchange.publish(
            IncomingMessage(message.serialize().encode()),
            routing_key=queue_name,
        )

    async def subscribe(
        self,
        agent_id: str,
        callback: Callable[[Message], Any],
        message_type: Optional[MessageType] = None,
    ) -> None:
        """Subscribe to messages for an agent.

        Args:
            agent_id: Agent ID to subscribe for.
            callback: Callback function to handle messages.
            message_type: Optional message type filter.
        """
        key = f"{agent_id}:{message_type.value if message_type else 'all'}"

        if key not in self.subscribers:
            self.subscribers[key] = []

        self.subscribers[key].append(callback)

        await self._logger.ainfo(
            "subscriber_registered",
            agent_id=agent_id,
            message_type=message_type.value if message_type else "all",
        )

    async def unsubscribe(
        self,
        agent_id: str,
        callback: Callable[[Message], Any],
    ) -> None:
        """Unsubscribe from messages.

        Args:
            agent_id: Agent ID to unsubscribe from.
            callback: Callback function to remove.
        """
        keys_to_clean = []

        for key, callbacks in self.subscribers.items():
            if key.startswith(agent_id):
                if callback in callbacks:
                    callbacks.remove(callback)
                if not callbacks:
                    keys_to_clean.append(key)

        for key in keys_to_clean:
            del self.subscribers[key]

        await self._logger.ainfo(
            "subscriber_unregistered",
            agent_id=agent_id,
        )

    async def request(
        self,
        message: Message,
        timeout_seconds: float = 10.0,
    ) -> Optional[Message]:
        """Send a request and wait for a response.

        Args:
            message: Request message.
            timeout_seconds: How long to wait for response.

        Returns:
            Response message if received, None if timeout.

        Raises:
            RuntimeError: If message bus not initialized.
        """
        if not self.rpc:
            raise RuntimeError("RabbitMQ RPC not initialized")

        try:
            # Generate correlation ID if not present
            if not message.correlation_id:
                message.correlation_id = str(uuid.uuid4())

            # Set up response listener
            response_handler = RPC(self.rabbitmq_channel)

            # Send request
            await self.publish(message, durable=True)

            # Wait for response
            response_json = await asyncio.wait_for(
                response_handler.wait(message.correlation_id),
                timeout=timeout_seconds,
            )

            response = Message.deserialize(response_json)

            await self._logger.ainfo(
                "request_response_received",
                request_id=message.id,
                response_id=response.id,
            )

            return response

        except asyncio.TimeoutError:
            await self._logger.awarning(
                "request_timeout",
                request_id=message.id,
                timeout_seconds=timeout_seconds,
            )
            return None

    async def reply(
        self,
        request_message: Message,
        response_payload: dict[str, Any],
    ) -> None:
        """Send a response to a request.

        Args:
            request_message: The original request message.
            response_payload: Response data.

        Raises:
            ValueError: If request has no correlation ID.
        """
        if not request_message.correlation_id:
            raise ValueError("Request message has no correlation ID")

        response = Message(
            from_agent=request_message.to_agent or "orchestrator",
            to_agent=request_message.from_agent,
            message_type=MessageType.RESPONSE,
            payload=response_payload,
            correlation_id=request_message.correlation_id,
        )

        await self.publish(response, durable=True)

        await self._logger.ainfo(
            "response_sent",
            request_id=request_message.id,
            response_id=response.id,
        )

    async def send_to_dead_letter_queue(
        self,
        message: Message,
        reason: str,
    ) -> None:
        """Send a message to the dead letter queue.

        Args:
            message: Message that failed to process.
            reason: Reason for sending to DLQ.
        """
        if not self.rabbitmq_channel:
            raise RuntimeError("RabbitMQ channel not initialized")

        dlq_name = "dead_letter_queue"
        queue = await self.rabbitmq_channel.declare_queue(dlq_name, auto_delete=False)

        dlq_message = {
            "original_message": json.loads(message.serialize()),
            "dlq_reason": reason,
            "dlq_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        await queue.put(json.dumps(dlq_message).encode())

        await self._logger.awarning(
            "message_sent_to_dlq",
            message_id=message.id,
            reason=reason,
        )


import asyncio
