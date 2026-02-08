"""WhatsApp Cloud API bridge for sending and receiving messages."""

import time
import hmac
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Any
from enum import Enum
from collections import defaultdict

import structlog
import httpx
from pydantic import BaseModel, Field


logger = structlog.get_logger(__name__)


class WhatsAppMessageType(str, Enum):
    """Types of messages that can be sent via WhatsApp."""

    TEXT = "text"
    TEMPLATE = "template"
    INTERACTIVE = "interactive"
    IMAGE = "image"
    DOCUMENT = "document"


class WhatsAppMessage(BaseModel):
    """Schema for a WhatsApp message."""

    from_number: str = Field(
        ..., description="Phone number that sent the message (without + prefix)"
    )
    to_number: Optional[str] = Field(
        default=None, description="Recipient phone number"
    )
    text: str = Field(..., description="Message content")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the message was sent/received"
    )
    message_id: str = Field(
        default="", description="Unique message ID from WhatsApp"
    )
    media_url: Optional[str] = Field(
        default=None, description="URL to media attachment if present"
    )
    media_type: Optional[str] = Field(
        default=None, description="Type of media (image, document, etc.)"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "from_number": "14155552671",
                "to_number": "14155552672",
                "text": "Hello, how can I help?",
                "timestamp": "2024-01-15T10:30:00Z",
                "message_id": "wamid.ABC123XYZ",
            }
        }


class WhatsAppBridge:
    """Bridge for WhatsApp Cloud API integration.

    Handles sending/receiving messages, rate limiting, command parsing,
    and conversation threading.
    """

    # WhatsApp Cloud API endpoint
    BASE_URL = "https://graph.facebook.com/v21.0"

    # Message type field names
    MESSAGE_TYPES = {
        "text": "text",
        "template": "template",
        "interactive": "interactive",
        "image": "image",
        "document": "document",
    }

    def __init__(
        self,
        phone_number_id: str,
        access_token: str,
        business_account_id: str,
        rate_limit_per_minute: int = 60,
    ) -> None:
        """Initialize WhatsApp bridge.

        Args:
            phone_number_id: WhatsApp Business Account phone number ID.
            access_token: WhatsApp Cloud API access token.
            business_account_id: WhatsApp Business Account ID.
            rate_limit_per_minute: Maximum messages per minute per user.
        """
        self.phone_number_id = phone_number_id
        self.access_token = access_token
        self.business_account_id = business_account_id
        self.rate_limit_per_minute = rate_limit_per_minute

        self.client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

        # Rate limiting: track messages per user
        self._rate_limit_buckets: dict[str, list[float]] = defaultdict(list)

        # Conversation threading: store message history per user
        self._conversation_threads: dict[str, list[WhatsAppMessage]] = defaultdict(list)
        self._max_thread_length = 50

        logger.info(
            "WhatsAppBridge initialized",
            phone_number_id=phone_number_id,
            rate_limit_per_minute=rate_limit_per_minute,
        )

    async def send_text(
        self,
        to_number: str,
        text: str,
        conversation_id: Optional[str] = None,
    ) -> Optional[str]:
        """Send a text message to a user.

        Args:
            to_number: Recipient phone number (format: country_code + number).
            text: Message content.
            conversation_id: Optional conversation ID for threading.

        Returns:
            Message ID if successful, None otherwise.
        """
        if not await self._check_rate_limit(to_number):
            logger.warning(
                "Rate limit exceeded",
                to_number=to_number,
            )
            return None

        payload = {
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": "text",
            "text": {"preview_url": False, "body": text},
        }

        try:
            response = await self.client.post(
                f"/{self.phone_number_id}/messages",
                json=payload,
            )

            response.raise_for_status()
            data = response.json()
            message_id = data.get("messages", [{}])[0].get("id")

            logger.info(
                "Text message sent",
                to_number=to_number,
                message_id=message_id,
            )

            # Store in conversation thread
            if conversation_id:
                msg = WhatsAppMessage(
                    from_number=self.phone_number_id,
                    to_number=to_number,
                    text=text,
                    message_id=message_id or "",
                )
                self._store_message(conversation_id, msg)

            return message_id

        except httpx.HTTPError as e:
            logger.error(
                "Failed to send text message",
                to_number=to_number,
                error=str(e),
            )
            return None

    async def send_formatted_message(
        self,
        to_number: str,
        content: str,
        format_type: str = "plain",
        conversation_id: Optional[str] = None,
    ) -> Optional[str]:
        """Send a formatted message (bold, code, lists, etc.).

        Args:
            to_number: Recipient phone number.
            content: Message content with optional formatting markers.
            format_type: "plain", "bold", "code", "list".
            conversation_id: Optional conversation ID.

        Returns:
            Message ID if successful, None otherwise.
        """
        # WhatsApp supports markdown-style formatting
        formatted_text = self._apply_formatting(content, format_type)

        return await self.send_text(to_number, formatted_text, conversation_id)

    async def send_status_update(
        self,
        to_number: str,
        status: str,
        emoji: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[str]:
        """Send a status update message with optional emoji.

        Args:
            to_number: Recipient phone number.
            status: Status message.
            emoji: Optional emoji to prepend.
            conversation_id: Optional conversation ID.

        Returns:
            Message ID if successful, None otherwise.
        """
        if emoji:
            message = f"{emoji} {status}"
        else:
            message = status

        return await self.send_text(to_number, message, conversation_id)

    async def send_template(
        self,
        to_number: str,
        template_name: str,
        parameters: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Send a pre-approved template message.

        Args:
            to_number: Recipient phone number.
            template_name: Name of the template.
            parameters: Template parameters to substitute.

        Returns:
            Message ID if successful, None otherwise.
        """
        if not await self._check_rate_limit(to_number):
            logger.warning(
                "Rate limit exceeded",
                to_number=to_number,
            )
            return None

        payload = {
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {"code": "en_US"},
            },
        }

        if parameters:
            payload["template"]["parameters"] = {"body": {"parameters": parameters}}

        try:
            response = await self.client.post(
                f"/{self.phone_number_id}/messages",
                json=payload,
            )

            response.raise_for_status()
            data = response.json()
            message_id = data.get("messages", [{}])[0].get("id")

            logger.info(
                "Template message sent",
                to_number=to_number,
                template_name=template_name,
                message_id=message_id,
            )

            return message_id

        except httpx.HTTPError as e:
            logger.error(
                "Failed to send template message",
                to_number=to_number,
                template_name=template_name,
                error=str(e),
            )
            return None

    def parse_incoming_webhook(self, payload: dict[str, Any]) -> list[WhatsAppMessage]:
        """Parse incoming WhatsApp webhook payload.

        Args:
            payload: Webhook payload from WhatsApp.

        Returns:
            List of extracted messages.
        """
        messages = []

        try:
            changes = payload.get("entry", [{}])[0].get("changes", [])

            for change in changes:
                data = change.get("value", {})
                message_list = data.get("messages", [])

                for msg_data in message_list:
                    message = self._parse_message_data(msg_data)
                    if message:
                        messages.append(message)
                        logger.info(
                            "Incoming message parsed",
                            from_number=message.from_number,
                            message_id=message.message_id,
                        )

        except Exception as e:
            logger.error("Failed to parse webhook payload", error=str(e))

        return messages

    def parse_command(self, text: str) -> Optional[tuple[str, list[str]]]:
        """Parse a command from message text.

        Supports format: /command arg1 arg2 arg3

        Args:
            text: Message text to parse.

        Returns:
            Tuple of (command, args), or None if not a command.
        """
        text = text.strip()

        if not text.startswith("/"):
            return None

        parts = text.split(maxsplit=1)
        command = parts[0][1:]  # Remove leading /

        args = []
        if len(parts) > 1:
            # Simple arg parsing (space-separated)
            args = parts[1].split()

        return command, args

    def get_conversation(self, conversation_id: str) -> list[WhatsAppMessage]:
        """Get conversation history.

        Args:
            conversation_id: Unique conversation ID.

        Returns:
            List of messages in the conversation.
        """
        return self._conversation_threads.get(conversation_id, [])

    def clear_conversation(self, conversation_id: str) -> None:
        """Clear conversation history.

        Args:
            conversation_id: Conversation ID to clear.
        """
        if conversation_id in self._conversation_threads:
            del self._conversation_threads[conversation_id]
            logger.debug(
                "Conversation cleared",
                conversation_id=conversation_id,
            )

    async def mark_message_read(self, message_id: str) -> bool:
        """Mark a message as read.

        Args:
            message_id: ID of the message to mark as read.

        Returns:
            True if successful, False otherwise.
        """
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }

        try:
            response = await self.client.post(
                f"/{self.phone_number_id}/messages",
                json=payload,
            )

            response.raise_for_status()
            logger.debug("Message marked as read", message_id=message_id)
            return True

        except httpx.HTTPError as e:
            logger.error(
                "Failed to mark message as read",
                message_id=message_id,
                error=str(e),
            )
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
        logger.debug("WhatsAppBridge closed")

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        try:
            import asyncio

            asyncio.run(self.close())
        except Exception:
            pass

    # Private methods

    async def _check_rate_limit(self, to_number: str) -> bool:
        """Check if a number is within rate limits.

        Args:
            to_number: Phone number to check.

        Returns:
            True if within limits, False otherwise.
        """
        now = time.time()
        cutoff = now - 60  # Last 60 seconds

        # Clean old entries
        if to_number in self._rate_limit_buckets:
            self._rate_limit_buckets[to_number] = [
                ts for ts in self._rate_limit_buckets[to_number] if ts > cutoff
            ]

        # Check limit
        bucket = self._rate_limit_buckets[to_number]
        if len(bucket) >= self.rate_limit_per_minute:
            return False

        # Add timestamp
        bucket.append(now)
        return True

    def _apply_formatting(self, content: str, format_type: str) -> str:
        """Apply text formatting for WhatsApp.

        Args:
            content: Content to format.
            format_type: Type of formatting to apply.

        Returns:
            Formatted content.
        """
        if format_type == "bold":
            return f"*{content}*"
        elif format_type == "italic":
            return f"_{content}_"
        elif format_type == "code":
            return f"```{content}```"
        elif format_type == "strikethrough":
            return f"~{content}~"
        elif format_type == "list":
            # Simple list formatting
            lines = content.split("\n")
            return "\n".join(f"• {line}" for line in lines if line.strip())

        return content

    def _parse_message_data(self, msg_data: dict[str, Any]) -> Optional[WhatsAppMessage]:
        """Parse a single message from WhatsApp data.

        Args:
            msg_data: Message data from WhatsApp webhook.

        Returns:
            Parsed WhatsAppMessage, or None if parsing failed.
        """
        try:
            from_number = msg_data.get("from", "")
            message_id = msg_data.get("id", "")
            timestamp_str = msg_data.get("timestamp", str(int(time.time())))

            # Parse timestamp
            timestamp = datetime.fromtimestamp(int(timestamp_str))

            # Extract text or media
            text = ""
            media_url = None
            media_type = None

            if "text" in msg_data:
                text = msg_data["text"].get("body", "")
            elif "media" in msg_data:
                media_data = list(msg_data["media"].values())[0]
                media_url = media_data.get("link", "")
                media_type = msg_data["media"].get("type", "")
                text = f"[{media_type.upper()}] {media_url}"

            return WhatsAppMessage(
                from_number=from_number,
                text=text,
                message_id=message_id,
                timestamp=timestamp,
                media_url=media_url,
                media_type=media_type,
            )

        except Exception as e:
            logger.error("Failed to parse message data", error=str(e))
            return None

    def _store_message(
        self, conversation_id: str, message: WhatsAppMessage
    ) -> None:
        """Store message in conversation thread.

        Args:
            conversation_id: Conversation ID.
            message: Message to store.
        """
        thread = self._conversation_threads[conversation_id]
        thread.append(message)

        # Trim to max length
        if len(thread) > self._max_thread_length:
            self._conversation_threads[conversation_id] = thread[-self._max_thread_length :]
