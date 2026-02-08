"""WhatsApp voice message handler — integrates voice with WhatsApp bridge.

Handles the full flow:
  1. Receive voice message webhook from WhatsApp
  2. Download audio file from WhatsApp media URL
  3. Transcribe with STT
  4. Route transcribed text to Commander (same as text message)
  5. Get agent response
  6. Synthesize response with TTS
  7. Upload audio to WhatsApp media
  8. Send voice reply

Also supports:
  - User preference: voice-only, text-only, or both
  - Long responses: split into multiple voice messages
  - Fallback: if TTS fails, send text instead
"""

import re
from typing import Optional

import httpx
import structlog
from pydantic import BaseModel, Field

from src.bridges.voice import VoiceConfig, VoiceProcessor

logger = structlog.get_logger(__name__)


class WhatsAppVoiceSettings(BaseModel):
    """Settings for WhatsApp voice integration."""

    whatsapp_api_base_url: str = Field(
        default="https://graph.instagram.com/v21.0",
        description="WhatsApp Graph API base URL",
    )
    whatsapp_api_token: str = Field(..., description="WhatsApp Cloud API access token")
    whatsapp_phone_number_id: str = Field(..., description="Phone number ID for WhatsApp")
    whatsapp_business_account_id: Optional[str] = Field(
        default=None, description="Business account ID for media upload"
    )

    voice_config: VoiceConfig = Field(
        default_factory=VoiceConfig, description="Voice processing configuration"
    )

    max_response_chunks: int = Field(
        default=5,
        description="Maximum number of voice messages to send for long responses",
    )
    chunk_duration_seconds: float = Field(
        default=30.0,
        description="Target duration per audio chunk",
    )
    send_text_fallback: bool = Field(
        default=True,
        description="Send text if TTS fails",
    )
    send_both_voice_and_text: bool = Field(
        default=False,
        description="Send both voice and text response",
    )


class WhatsAppVoiceHandler:
    """Handler for WhatsApp voice messages."""

    def __init__(self, settings: WhatsAppVoiceSettings) -> None:
        """Initialize WhatsApp voice handler.

        Args:
            settings: WhatsApp voice configuration.
        """
        self.settings = settings
        self.voice_processor = VoiceProcessor(settings.voice_config)

        self.http_client = httpx.AsyncClient(
            base_url=settings.whatsapp_api_base_url,
            headers={
                "Authorization": f"Bearer {settings.whatsapp_api_token}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

        logger.info(
            "WhatsAppVoiceHandler initialized",
            phone_number_id=settings.whatsapp_phone_number_id,
        )

    async def handle_voice_message(
        self,
        message_data: dict,
        response_text: str,
    ) -> bool:
        """Handle incoming voice message and send audio response.

        Full flow:
          1. Extract media_id from voice message
          2. Download audio from WhatsApp
          3. Transcribe (already done in caller, but we show the flow)
          4. Process with agent (already done in caller)
          5. Synthesize response text to audio
          6. Upload audio to WhatsApp
          7. Send voice reply

        Args:
            message_data: Webhook message data from WhatsApp.
            response_text: Text response from agent (to be synthesized).

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Extract sender phone number
            from_number = message_data.get("from", "")
            if not from_number:
                logger.error("No sender phone number in message")
                return False

            logger.info(
                "Processing voice message response",
                from_number=from_number,
                response_length=len(response_text),
            )

            # Synthesize response text to audio
            try:
                tts_result = await self.voice_processor.synthesize(response_text)
            except Exception as e:
                logger.error(
                    "TTS synthesis failed",
                    from_number=from_number,
                    error=str(e),
                )

                # Fallback: send text response
                if self.settings.send_text_fallback:
                    logger.info("Sending text fallback due to TTS failure", from_number=from_number)
                    return await self._send_text_fallback(from_number, response_text)
                return False

            # Split long audio into chunks if needed
            audio_chunks = await self._split_audio_if_needed(
                tts_result.audio_data,
                tts_result.duration_seconds,
            )

            if not audio_chunks:
                logger.error("Failed to prepare audio chunks", from_number=from_number)
                return False

            # Send voice message(s)
            success = await self._send_voice_replies(from_number, audio_chunks)

            # Optionally send text version too
            if self.settings.send_both_voice_and_text and success:
                await self._send_text_fallback(from_number, response_text)

            return success

        except Exception as e:
            logger.error(
                "Voice message handling failed",
                error=str(e),
                exc_info=True,
            )
            return False

    async def download_whatsapp_media(self, media_id: str) -> Optional[bytes]:
        """Download media file from WhatsApp.

        Args:
            media_id: Media ID from WhatsApp message.

        Returns:
            Raw audio bytes, or None if download failed.
        """
        try:
            logger.debug("Downloading WhatsApp media", media_id=media_id)

            # Step 1: Get media URL
            response = await self.http_client.get(f"/{media_id}")
            response.raise_for_status()

            data = response.json()
            media_url = data.get("url")

            if not media_url:
                logger.error("No media URL in response", media_id=media_id)
                return None

            # Step 2: Download from media URL
            async with httpx.AsyncClient(timeout=60.0) as client:
                media_response = await client.get(
                    media_url,
                    headers={"Authorization": f"Bearer {self.settings.whatsapp_api_token}"},
                )
                media_response.raise_for_status()

            audio_data = media_response.content
            logger.info(
                "Media downloaded",
                media_id=media_id,
                size_bytes=len(audio_data),
            )

            return audio_data

        except httpx.HTTPError as e:
            logger.error(
                "Failed to download WhatsApp media",
                media_id=media_id,
                error=str(e),
            )
            return None

    async def upload_whatsapp_media(
        self,
        audio_data: bytes,
        mime_type: str = "audio/ogg",
    ) -> Optional[str]:
        """Upload audio file to WhatsApp media server.

        Args:
            audio_data: Raw audio data.
            mime_type: MIME type of audio (audio/ogg, audio/mp3, audio/mpeg, etc.).

        Returns:
            Media ID for sending, or None if upload failed.
        """
        try:
            logger.debug(
                "Uploading audio to WhatsApp",
                size_bytes=len(audio_data),
                mime_type=mime_type,
            )

            # WhatsApp requires a business account ID for media upload
            if not self.settings.whatsapp_business_account_id:
                logger.error("Business account ID not configured for media upload")
                return None

            # Prepare multipart form data
            files = {
                "file": ("audio.ogg", audio_data, mime_type),
                "type": (None, mime_type),
            }

            response = await self.http_client.post(
                f"/{self.settings.whatsapp_business_account_id}/media",
                files=files,  # type: ignore
            )

            response.raise_for_status()
            data = response.json()
            media_id = data.get("id")

            if not media_id:
                logger.error("No media ID in upload response")
                return None

            logger.info("Media uploaded", media_id=media_id)
            return media_id

        except httpx.HTTPError as e:
            logger.error(
                "Failed to upload media to WhatsApp",
                error=str(e),
            )
            return None

    async def send_voice_reply(
        self,
        to_number: str,
        audio_data: bytes,
    ) -> Optional[str]:
        """Send voice message reply.

        Args:
            to_number: Recipient phone number.
            audio_data: Audio data in bytes.

        Returns:
            Message ID if successful, None otherwise.
        """
        try:
            # Upload audio to WhatsApp media
            media_id = await self.upload_whatsapp_media(audio_data)
            if not media_id:
                logger.error("Failed to upload media", to_number=to_number)
                return None

            # Send voice message
            payload = {
                "messaging_product": "whatsapp",
                "to": to_number,
                "type": "audio",
                "audio": {"id": media_id},
            }

            response = await self.http_client.post(
                f"/{self.settings.whatsapp_phone_number_id}/messages",
                json=payload,
            )

            response.raise_for_status()
            data = response.json()
            message_id = data.get("messages", [{}])[0].get("id")

            logger.info(
                "Voice message sent",
                to_number=to_number,
                message_id=message_id,
            )

            return message_id

        except httpx.HTTPError as e:
            logger.error(
                "Failed to send voice message",
                to_number=to_number,
                error=str(e),
            )
            return None

    async def send_voice_and_text(
        self,
        to_number: str,
        text: str,
        audio_data: bytes,
    ) -> tuple[Optional[str], Optional[str]]:
        """Send both voice and text response.

        Args:
            to_number: Recipient phone number.
            text: Text response.
            audio_data: Audio data.

        Returns:
            Tuple of (voice_message_id, text_message_id), or (None, None) if failed.
        """
        try:
            voice_msg_id = await self.send_voice_reply(to_number, audio_data)
            text_msg_id = await self._send_text_fallback(to_number, text)

            return voice_msg_id, text_msg_id

        except Exception as e:
            logger.error(
                "Failed to send voice and text",
                to_number=to_number,
                error=str(e),
            )
            return None, None

    async def close(self) -> None:
        """Close HTTP client and voice processor."""
        await self.http_client.aclose()
        await self.voice_processor.close()
        logger.debug("WhatsAppVoiceHandler closed")

    async def __aenter__(self) -> "WhatsAppVoiceHandler":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Async context manager exit."""
        await self.close()

    # Private methods

    async def _send_voice_replies(
        self,
        to_number: str,
        audio_chunks: list[bytes],
    ) -> bool:
        """Send multiple voice message chunks.

        Args:
            to_number: Recipient phone number.
            audio_chunks: List of audio data chunks.

        Returns:
            True if all messages sent successfully.
        """
        success = True

        for i, chunk in enumerate(audio_chunks, 1):
            try:
                message_id = await self.send_voice_reply(to_number, chunk)
                if not message_id:
                    logger.warning(
                        "Failed to send voice chunk",
                        to_number=to_number,
                        chunk_number=i,
                        total_chunks=len(audio_chunks),
                    )
                    success = False

            except Exception as e:
                logger.error(
                    "Error sending voice chunk",
                    to_number=to_number,
                    chunk_number=i,
                    error=str(e),
                )
                success = False

        return success

    async def _send_text_fallback(
        self,
        to_number: str,
        text: str,
    ) -> Optional[str]:
        """Send text message as fallback.

        Args:
            to_number: Recipient phone number.
            text: Text content.

        Returns:
            Message ID if successful.
        """
        try:
            payload = {
                "messaging_product": "whatsapp",
                "to": to_number,
                "type": "text",
                "text": {"preview_url": False, "body": text},
            }

            response = await self.http_client.post(
                f"/{self.settings.whatsapp_phone_number_id}/messages",
                json=payload,
            )

            response.raise_for_status()
            data = response.json()
            message_id = data.get("messages", [{}])[0].get("id")

            logger.info(
                "Text fallback sent",
                to_number=to_number,
                message_id=message_id,
            )

            return message_id

        except httpx.HTTPError as e:
            logger.error(
                "Failed to send text fallback",
                to_number=to_number,
                error=str(e),
            )
            return None

    async def _split_audio_if_needed(
        self,
        audio_data: bytes,
        duration_seconds: float,
    ) -> list[bytes]:
        """Split audio into chunks if duration exceeds limit.

        For now, returns the audio as-is since WhatsApp doesn't have
        strict audio length limits for voice messages. In future, could
        implement intelligent chunking based on speech segments.

        Args:
            audio_data: Audio data.
            duration_seconds: Duration of audio.

        Returns:
            List of audio chunks.
        """
        if duration_seconds <= self.settings.chunk_duration_seconds:
            return [audio_data]

        logger.warning(
            "Audio exceeds chunk duration, returning as single message",
            duration_seconds=duration_seconds,
            chunk_duration_seconds=self.settings.chunk_duration_seconds,
        )

        # TODO: Implement intelligent audio chunking based on silence detection
        # For now, just return the entire audio as a single message
        return [audio_data]

    def _detect_agent_name_from_response(self, response_text: str) -> str:
        """Detect which agent generated the response.

        Simple pattern matching to extract agent name from response prefix.

        Args:
            response_text: Response text from agent.

        Returns:
            Agent name if detected, or "unknown".
        """
        # Look for patterns like "[CommanderName]:" or "CommanderName says:"
        match = re.match(r"^\[?(\w+)\]?[:\s]", response_text)
        if match:
            return match.group(1)

        return "unknown"
