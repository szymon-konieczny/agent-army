"""Webhook processing and verification for multiple platforms."""

import hmac
import hashlib
from enum import Enum
from typing import Optional, Any, Callable, Awaitable
from datetime import datetime

import structlog
from pydantic import BaseModel, Field


logger = structlog.get_logger(__name__)


class WebhookSource(str, Enum):
    """Supported webhook sources."""

    WHATSAPP = "whatsapp"
    GITHUB = "github"
    GENERIC = "generic"


class WebhookEvent(BaseModel):
    """Webhook event payload."""

    source: WebhookSource = Field(..., description="Origin of the webhook")
    event_type: str = Field(..., description="Type of event (e.g., 'message', 'push')")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the event occurred"
    )
    data: dict[str, Any] = Field(
        default_factory=dict, description="Event-specific data"
    )
    signature_valid: bool = Field(
        default=False, description="Whether signature was verified"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "source": "whatsapp",
                "event_type": "message",
                "timestamp": "2024-01-15T10:30:00Z",
                "data": {"message_id": "abc123", "from": "1234567890"},
                "signature_valid": True,
            }
        }


class WebhookHandler:
    """Handler for webhook processing, verification, and routing."""

    def __init__(
        self,
        whatsapp_verify_token: Optional[str] = None,
        github_secret: Optional[str] = None,
    ) -> None:
        """Initialize webhook handler.

        Args:
            whatsapp_verify_token: WhatsApp webhook verification token.
            github_secret: GitHub webhook secret for signature verification.
        """
        self.whatsapp_verify_token = whatsapp_verify_token
        self.github_secret = github_secret

        # Event handlers
        self.handlers: dict[WebhookSource, dict[str, list[Callable]]] = {
            WebhookSource.WHATSAPP: {},
            WebhookSource.GITHUB: {},
            WebhookSource.GENERIC: {},
        }

        # Event history for debugging
        self.event_history: list[WebhookEvent] = []
        self.max_history_size = 500

        logger.info(
            "WebhookHandler initialized",
            whatsapp_configured=bool(whatsapp_verify_token),
            github_configured=bool(github_secret),
        )

    def verify_whatsapp_webhook(
        self, token: str, challenge: str
    ) -> Optional[str]:
        """Verify WhatsApp webhook GET request (challenge-response).

        Called when WhatsApp platform needs to verify the webhook URL.

        Args:
            token: Verify token from WhatsApp platform.
            challenge: Challenge string to echo back.

        Returns:
            Challenge string if token is valid, None otherwise.
        """
        if token == self.whatsapp_verify_token:
            logger.info("WhatsApp webhook verified")
            return challenge

        logger.warning("WhatsApp webhook verification failed - invalid token")
        return None

    def verify_whatsapp_signature(
        self, payload: bytes, signature: str
    ) -> bool:
        """Verify WhatsApp webhook POST signature.

        Args:
            payload: Raw request body.
            signature: Signature from X-Hub-Signature header.

        Returns:
            True if signature is valid, False otherwise.
        """
        if not self.whatsapp_verify_token:
            logger.warning("WhatsApp verify token not configured")
            return False

        try:
            # WhatsApp signature format: sha256=<hash>
            expected_sig = (
                "sha256="
                + hmac.new(
                    self.whatsapp_verify_token.encode(),
                    payload,
                    hashlib.sha256,
                ).hexdigest()
            )

            is_valid = hmac.compare_digest(signature, expected_sig)

            if is_valid:
                logger.debug("WhatsApp signature verified")
            else:
                logger.warning("WhatsApp signature verification failed")

            return is_valid

        except Exception as e:
            logger.error("WhatsApp signature verification error", error=str(e))
            return False

    def verify_github_signature(
        self, payload: bytes, signature: str
    ) -> bool:
        """Verify GitHub webhook signature.

        Args:
            payload: Raw request body.
            signature: Signature from X-Hub-Signature-256 header.

        Returns:
            True if signature is valid, False otherwise.
        """
        if not self.github_secret:
            logger.warning("GitHub secret not configured")
            return False

        try:
            # GitHub signature format: sha256=<hash>
            expected_sig = (
                "sha256="
                + hmac.new(
                    self.github_secret.encode(),
                    payload,
                    hashlib.sha256,
                ).hexdigest()
            )

            is_valid = hmac.compare_digest(signature, expected_sig)

            if is_valid:
                logger.debug("GitHub signature verified")
            else:
                logger.warning("GitHub signature verification failed")

            return is_valid

        except Exception as e:
            logger.error("GitHub signature verification error", error=str(e))
            return False

    def parse_whatsapp_event(
        self, payload: dict[str, Any], signature_valid: bool = False
    ) -> Optional[WebhookEvent]:
        """Parse WhatsApp webhook payload into WebhookEvent.

        Args:
            payload: WhatsApp webhook payload.
            signature_valid: Whether signature was verified.

        Returns:
            Parsed WebhookEvent, or None if parsing failed.
        """
        try:
            changes = payload.get("entry", [{}])[0].get("changes", [])

            for change in changes:
                change_value = change.get("value", {})

                # Determine event type
                if "messages" in change_value:
                    event_type = "message"
                elif "statuses" in change_value:
                    event_type = "status"
                else:
                    event_type = "unknown"

                event = WebhookEvent(
                    source=WebhookSource.WHATSAPP,
                    event_type=event_type,
                    data=change_value,
                    signature_valid=signature_valid,
                )

                self._store_event(event)
                logger.info(
                    "WhatsApp event parsed",
                    event_type=event_type,
                )

                return event

        except Exception as e:
            logger.error("Failed to parse WhatsApp event", error=str(e))

        return None

    def parse_github_event(
        self, payload: dict[str, Any], event_type: str, signature_valid: bool = False
    ) -> Optional[WebhookEvent]:
        """Parse GitHub webhook payload into WebhookEvent.

        Args:
            payload: GitHub webhook payload.
            event_type: GitHub event type (from X-GitHub-Event header).
            signature_valid: Whether signature was verified.

        Returns:
            Parsed WebhookEvent, or None if parsing failed.
        """
        try:
            event = WebhookEvent(
                source=WebhookSource.GITHUB,
                event_type=event_type,
                data=payload,
                signature_valid=signature_valid,
            )

            self._store_event(event)
            logger.info(
                "GitHub event parsed",
                event_type=event_type,
            )

            return event

        except Exception as e:
            logger.error("Failed to parse GitHub event", error=str(e))

        return None

    def register_handler(
        self,
        source: WebhookSource,
        event_type: str,
        handler: Callable[[WebhookEvent], Awaitable[None]],
    ) -> None:
        """Register an async handler for a webhook event.

        Args:
            source: Webhook source to listen for.
            event_type: Specific event type (e.g., "message", "push").
            handler: Async callback function.
        """
        if event_type not in self.handlers[source]:
            self.handlers[source][event_type] = []

        self.handlers[source][event_type].append(handler)
        logger.info(
            "Handler registered",
            source=source,
            event_type=event_type,
        )

    async def dispatch(self, event: WebhookEvent) -> None:
        """Dispatch event to registered handlers.

        Args:
            event: Event to dispatch.
        """
        handlers = self.handlers.get(event.source, {}).get(event.event_type, [])

        if not handlers:
            logger.debug(
                "No handlers registered for event",
                source=event.source,
                event_type=event.event_type,
            )
            return

        logger.info(
            "Dispatching event to handlers",
            source=event.source,
            event_type=event.event_type,
            handler_count=len(handlers),
        )

        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(
                    "Handler execution failed",
                    source=event.source,
                    event_type=event.event_type,
                    error=str(e),
                )

    def get_event_history(
        self,
        source: Optional[WebhookSource] = None,
        event_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[WebhookEvent]:
        """Get event history (most recent first).

        Args:
            source: Filter by source (optional).
            event_type: Filter by event type (optional).
            limit: Maximum number of events to return.

        Returns:
            List of events matching filters.
        """
        history = list(reversed(self.event_history))

        if source:
            history = [e for e in history if e.source == source]

        if event_type:
            history = [e for e in history if e.event_type == event_type]

        if limit:
            history = history[:limit]

        return history

    def clear_event_history(self) -> None:
        """Clear all stored event history."""
        self.event_history.clear()
        logger.info("Event history cleared")

    # Private methods

    def _store_event(self, event: WebhookEvent) -> None:
        """Store event in history.

        Args:
            event: Event to store.
        """
        self.event_history.append(event)

        # Trim to max size
        if len(self.event_history) > self.max_history_size:
            self.event_history = self.event_history[-self.max_history_size :]
