"""Communication bridges package for external integrations."""

from src.bridges.whatsapp import WhatsAppBridge, WhatsAppMessage
from src.bridges.webhook_handler import WebhookHandler, WebhookEvent

__all__ = [
    "WhatsAppBridge",
    "WhatsAppMessage",
    "WebhookHandler",
    "WebhookEvent",
]
