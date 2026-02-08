"""Voice processing — STT and TTS for AgentArmy.

Multi-provider voice system:

STT (Speech-to-Text):
  1. OpenAI Whisper API (cloud, fast, excellent Polish support)
  2. Whisper local via faster-whisper (local, free, good for privacy)
  3. HuggingFace Inference API (cloud, free tier)

TTS (Text-to-Speech):
  1. OpenAI TTS API (cloud, natural voices: alloy/echo/fable/onyx/nova/shimmer)
  2. Edge-TTS (free, Microsoft voices, excellent Polish support)
  3. HuggingFace Inference API (cloud)

Voice flow in WhatsApp:
  User voice msg → download audio → STT → agent processes text →
  TTS → upload audio → send voice response

Also supports: audio format conversion (ogg/opus → wav → mp3),
language detection from audio, and configurable voice profiles per agent.
"""

import asyncio
import io
import time
from enum import Enum
from pathlib import Path
from typing import Optional

import httpx
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class VoiceProvider(str, Enum):
    """Available voice service providers."""

    OPENAI = "openai"
    LOCAL_WHISPER = "local_whisper"
    HUGGINGFACE = "huggingface"
    EDGE_TTS = "edge_tts"


class STTResult(BaseModel):
    """Result from speech-to-text transcription."""

    text: str = Field(..., description="Transcribed text")
    language: str = Field(default="en", description="Detected language code (ISO 639-1)")
    confidence: float = Field(
        default=0.95, description="Confidence score (0.0-1.0)", ge=0.0, le=1.0
    )
    duration_seconds: float = Field(..., description="Duration of audio in seconds", ge=0.0)
    provider: VoiceProvider = Field(..., description="Which provider performed transcription")
    processing_time_ms: float = Field(..., description="Time taken to process", ge=0.0)


class TTSResult(BaseModel):
    """Result from text-to-speech synthesis."""

    audio_data: bytes = Field(..., description="Raw audio data")
    format: str = Field(default="ogg", description="Audio format (mp3, ogg, wav)")
    duration_seconds: float = Field(..., description="Duration of audio in seconds", ge=0.0)
    voice: str = Field(..., description="Voice identifier used")
    provider: VoiceProvider = Field(..., description="Which provider performed synthesis")
    processing_time_ms: float = Field(..., description="Time taken to process", ge=0.0)


class VoiceConfig(BaseModel):
    """Voice processing configuration."""

    stt_provider: VoiceProvider = Field(
        default=VoiceProvider.OPENAI, description="Provider for speech-to-text"
    )
    tts_provider: VoiceProvider = Field(
        default=VoiceProvider.EDGE_TTS,
        description="Provider for text-to-speech (edge_tts is free + great Polish)",
    )
    default_language: str = Field(default="pl", description="Default language code")
    tts_voice: str = Field(
        default="pl-PL-MarekNeural",
        description="Default voice for edge-tts (Polish male)",
    )
    openai_tts_voice: str = Field(
        default="onyx", description="Default voice for OpenAI TTS"
    )
    audio_format: str = Field(
        default="ogg", description="Audio format for responses (ogg/opus/wav/mp3)"
    )
    max_audio_duration_seconds: int = Field(
        default=120, description="Maximum audio duration to process"
    )
    enable_language_detection: bool = Field(
        default=True, description="Auto-detect language from audio"
    )

    # API keys and configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    huggingface_api_key: Optional[str] = Field(default=None, description="HuggingFace API key")


class VoiceProcessor:
    """Main voice processing coordinator — transcription and synthesis."""

    def __init__(self, config: VoiceConfig) -> None:
        """Initialize voice processor.

        Args:
            config: Voice processing configuration.
        """
        self.config = config
        self._openai_client = (
            httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={"Authorization": f"Bearer {config.openai_api_key}"},
                timeout=60.0,
            )
            if config.openai_api_key
            else None
        )
        self._hf_client = (
            httpx.AsyncClient(
                base_url="https://api-inference.huggingface.co",
                headers={"Authorization": f"Bearer {config.huggingface_api_key}"},
                timeout=60.0,
            )
            if config.huggingface_api_key
            else None
        )

        # Try to import optional STT/TTS libraries
        self._faster_whisper_available = False
        self._edge_tts_available = False

        try:
            import faster_whisper  # noqa: F401

            self._faster_whisper_available = True
            logger.info("faster-whisper library available")
        except ImportError:
            logger.warning("faster-whisper not installed, local transcription disabled")

        try:
            import edge_tts  # noqa: F401

            self._edge_tts_available = True
            logger.info("edge-tts library available")
        except ImportError:
            logger.warning("edge-tts not installed, edge-tts synthesis disabled")

        logger.info(
            "VoiceProcessor initialized",
            stt_provider=self.config.stt_provider.value,
            tts_provider=self.config.tts_provider.value,
        )

    async def transcribe(
        self, audio_data: bytes, format: str = "ogg", language: Optional[str] = None
    ) -> STTResult:
        """Transcribe audio to text using configured provider.

        Args:
            audio_data: Raw audio data in bytes.
            format: Audio format (ogg, opus, wav, mp3, m4a, webm).
            language: Optional language code hint (ISO 639-1).

        Returns:
            STTResult with transcribed text and metadata.

        Raises:
            ValueError: If provider is unavailable or audio is too long.
        """
        start_time = time.time()

        # Validate audio duration
        try:
            duration = await self._estimate_audio_duration(audio_data, format)
            if duration > self.config.max_audio_duration_seconds:
                raise ValueError(
                    f"Audio duration {duration}s exceeds maximum "
                    f"{self.config.max_audio_duration_seconds}s"
                )
        except Exception as e:
            logger.warning("Could not estimate audio duration", error=str(e))

        # Route to appropriate provider
        if self.config.stt_provider == VoiceProvider.OPENAI:
            result = await self._transcribe_openai(audio_data, format, language)
        elif self.config.stt_provider == VoiceProvider.LOCAL_WHISPER:
            if not self._faster_whisper_available:
                raise ValueError("faster-whisper not available, install with: pip install faster-whisper")
            result = await self._transcribe_local(audio_data, format, language)
        elif self.config.stt_provider == VoiceProvider.HUGGINGFACE:
            result = await self._transcribe_huggingface(audio_data, format, language)
        else:
            raise ValueError(f"Unknown STT provider: {self.config.stt_provider}")

        # Update timing
        result.processing_time_ms = (time.time() - start_time) * 1000
        logger.info(
            "Audio transcribed",
            provider=result.provider.value,
            language=result.language,
            duration_ms=result.processing_time_ms,
        )

        return result

    async def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> TTSResult:
        """Synthesize text to speech using configured provider.

        Args:
            text: Text to synthesize.
            language: Language code (ISO 639-1). Defaults to config.default_language.
            voice: Voice identifier. Defaults to config tts_voice.

        Returns:
            TTSResult with audio data and metadata.

        Raises:
            ValueError: If provider is unavailable or text is too long.
        """
        start_time = time.time()

        if language is None:
            language = self.config.default_language
        if voice is None:
            voice = self.config.tts_voice

        # Validate text length (reasonable limit for synthesis)
        if len(text) > 5000:
            logger.warning("Text exceeds 5000 characters, will be truncated", text_len=len(text))
            text = text[:5000]

        # Route to appropriate provider
        if self.config.tts_provider == VoiceProvider.OPENAI:
            result = await self._synthesize_openai(text, voice)
        elif self.config.tts_provider == VoiceProvider.EDGE_TTS:
            if not self._edge_tts_available:
                raise ValueError("edge-tts not available, install with: pip install edge-tts")
            result = await self._synthesize_edge_tts(text, voice, language)
        elif self.config.tts_provider == VoiceProvider.HUGGINGFACE:
            result = await self._synthesize_huggingface(text, language)
        else:
            raise ValueError(f"Unknown TTS provider: {self.config.tts_provider}")

        # Update timing
        result.processing_time_ms = (time.time() - start_time) * 1000
        logger.info(
            "Text synthesized",
            provider=result.provider.value,
            voice=result.voice,
            duration_ms=result.processing_time_ms,
        )

        return result

    async def convert_audio(
        self, data: bytes, from_format: str, to_format: str
    ) -> bytes:
        """Convert audio between formats using ffmpeg.

        Args:
            data: Audio data in bytes.
            from_format: Source format (ogg, opus, wav, mp3, m4a, webm).
            to_format: Target format (ogg, opus, wav, mp3, m4a, webm).

        Returns:
            Converted audio data.

        Raises:
            RuntimeError: If ffmpeg is not available or conversion fails.
        """
        if from_format == to_format:
            return data

        logger.debug("Converting audio", from_format=from_format, to_format=to_format)

        # Use ffmpeg subprocess for format conversion
        try:
            import subprocess

            # Write input to temp file
            input_path = Path("/tmp") / f"audio_input_{int(time.time() * 1000)}.{from_format}"
            output_path = Path("/tmp") / f"audio_output_{int(time.time() * 1000)}.{to_format}"

            try:
                input_path.write_bytes(data)

                # Run ffmpeg
                cmd = [
                    "ffmpeg",
                    "-i",
                    str(input_path),
                    "-y",  # Overwrite output
                    str(output_path),
                ]

                result = await asyncio.to_thread(
                    subprocess.run, cmd, capture_output=True, timeout=30
                )

                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")

                # Read converted audio
                converted_data = output_path.read_bytes()
                logger.debug(
                    "Audio converted",
                    from_format=from_format,
                    to_format=to_format,
                    input_size=len(data),
                    output_size=len(converted_data),
                )

                return converted_data

            finally:
                # Cleanup temp files
                input_path.unlink(missing_ok=True)
                output_path.unlink(missing_ok=True)

        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg (macOS)")

    async def _transcribe_openai(
        self, audio_data: bytes, format: str, language: Optional[str]
    ) -> STTResult:
        """Transcribe using OpenAI Whisper API.

        Args:
            audio_data: Audio data in bytes.
            format: Audio format.
            language: Optional language hint.

        Returns:
            STTResult from OpenAI.
        """
        if not self._openai_client:
            raise ValueError("OpenAI API key not configured")

        # Prepare multipart form data
        files = {
            "file": (f"audio.{format}", audio_data, "application/octet-stream"),
            "model": (None, "whisper-1"),
        }

        if language:
            files["language"] = (None, language)

        try:
            response = await self._openai_client.post(
                "/audio/transcriptions",
                files=files,  # type: ignore
            )
            response.raise_for_status()
            data = response.json()

            detected_lang = language or data.get("language", "en")

            return STTResult(
                text=data.get("text", ""),
                language=detected_lang,
                duration_seconds=await self._estimate_audio_duration(audio_data, format),
                provider=VoiceProvider.OPENAI,
                confidence=0.95,
            )

        except httpx.HTTPError as e:
            logger.error("OpenAI transcription failed", error=str(e))
            raise

    async def _transcribe_local(
        self, audio_data: bytes, format: str, language: Optional[str]
    ) -> STTResult:
        """Transcribe using local faster-whisper model.

        Args:
            audio_data: Audio data in bytes.
            format: Audio format.
            language: Optional language hint.

        Returns:
            STTResult from local model.
        """
        from faster_whisper import WhisperModel

        # Convert to WAV if needed (faster-whisper prefers WAV)
        if format != "wav":
            audio_data = await self.convert_audio(audio_data, format, "wav")

        # Save to temp file for faster-whisper
        temp_path = Path("/tmp") / f"audio_{int(time.time() * 1000)}.wav"

        try:
            temp_path.write_bytes(audio_data)

            # Load model (large-v3-turbo: good balance of speed & quality, excellent Polish)
            model = WhisperModel(
                "large-v3-turbo",
                device="cpu",
                compute_type="default",
            )

            # Transcribe
            segments, info = await asyncio.to_thread(
                model.transcribe,
                str(temp_path),
                language=language,
            )

            # Combine segments
            text = " ".join(segment.text for segment in segments)
            detected_lang = info.language if info else (language or "en")

            return STTResult(
                text=text,
                language=detected_lang,
                duration_seconds=info.duration if info else 0.0,
                provider=VoiceProvider.LOCAL_WHISPER,
                confidence=0.90,  # Local model slightly lower confidence
            )

        except Exception as e:
            logger.error("Local transcription failed", error=str(e))
            raise
        finally:
            temp_path.unlink(missing_ok=True)

    async def _transcribe_huggingface(
        self, audio_data: bytes, format: str, language: Optional[str]
    ) -> STTResult:
        """Transcribe using HuggingFace Inference API.

        Args:
            audio_data: Audio data in bytes.
            format: Audio format.
            language: Optional language hint.

        Returns:
            STTResult from HuggingFace.
        """
        if not self._hf_client:
            raise ValueError("HuggingFace API key not configured")

        # Use openai/whisper-large-v3-turbo model
        model_id = "openai/whisper-large-v3-turbo"

        try:
            response = await self._hf_client.post(
                f"/models/{model_id}",
                content=audio_data,
                headers={"Content-Type": "audio/ogg"},
            )
            response.raise_for_status()
            data = response.json()

            # HuggingFace returns task_outputs with text
            text = data.get("text", "")
            detected_lang = data.get("language", language or "en")

            return STTResult(
                text=text,
                language=detected_lang,
                duration_seconds=await self._estimate_audio_duration(audio_data, format),
                provider=VoiceProvider.HUGGINGFACE,
                confidence=0.92,
            )

        except httpx.HTTPError as e:
            logger.error("HuggingFace transcription failed", error=str(e))
            raise

    async def _synthesize_openai(self, text: str, voice: str) -> TTSResult:
        """Synthesize using OpenAI TTS API.

        Args:
            text: Text to synthesize.
            voice: Voice identifier (alloy, echo, fable, onyx, nova, shimmer).

        Returns:
            TTSResult from OpenAI.
        """
        if not self._openai_client:
            raise ValueError("OpenAI API key not configured")

        try:
            response = await self._openai_client.post(
                "/audio/speech",
                json={
                    "model": "tts-1",
                    "input": text,
                    "voice": voice,
                    "response_format": "mp3",
                },
            )
            response.raise_for_status()
            audio_data = response.content

            # Estimate duration from text (rough: 150 words/min = 0.4 sec per word)
            words = len(text.split())
            duration = (words / 150) * 60

            return TTSResult(
                audio_data=audio_data,
                format="mp3",
                duration_seconds=duration,
                voice=voice,
                provider=VoiceProvider.OPENAI,
            )

        except httpx.HTTPError as e:
            logger.error("OpenAI synthesis failed", error=str(e))
            raise

    async def _synthesize_edge_tts(self, text: str, voice: str, language: str) -> TTSResult:
        """Synthesize using Edge-TTS (free Microsoft voices).

        Args:
            text: Text to synthesize.
            voice: Voice identifier (e.g., "pl-PL-MarekNeural").
            language: Language code.

        Returns:
            TTSResult from Edge-TTS.
        """
        from edge_tts import Communicate

        try:
            # Select voice based on language if not provided
            if not voice:
                voice = self._get_edge_tts_voice(language)

            communicate = Communicate(text, voice)
            audio_buffer = io.BytesIO()

            # Collect audio chunks
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])

            audio_data = audio_buffer.getvalue()

            # Estimate duration
            words = len(text.split())
            duration = (words / 150) * 60

            # Convert to OGG/Opus for WhatsApp compatibility if needed
            if self.config.audio_format in ("ogg", "opus"):
                audio_data = await self.convert_audio(audio_data, "webm", "ogg")

            return TTSResult(
                audio_data=audio_data,
                format=self.config.audio_format,
                duration_seconds=duration,
                voice=voice,
                provider=VoiceProvider.EDGE_TTS,
            )

        except Exception as e:
            logger.error("Edge-TTS synthesis failed", error=str(e))
            raise

    async def _synthesize_huggingface(self, text: str, language: str) -> TTSResult:
        """Synthesize using HuggingFace Inference API.

        Args:
            text: Text to synthesize.
            language: Language code.

        Returns:
            TTSResult from HuggingFace.
        """
        if not self._hf_client:
            raise ValueError("HuggingFace API key not configured")

        # Use a TTS model appropriate for the language
        model_id = "suno/bark"  # Multi-lingual TTS model

        try:
            response = await self._hf_client.post(
                f"/models/{model_id}",
                json={"inputs": text},
            )
            response.raise_for_status()

            # Response should contain audio
            audio_data = response.content

            # Estimate duration
            words = len(text.split())
            duration = (words / 150) * 60

            return TTSResult(
                audio_data=audio_data,
                format="ogg",
                duration_seconds=duration,
                voice=f"huggingface-{model_id}",
                provider=VoiceProvider.HUGGINGFACE,
            )

        except httpx.HTTPError as e:
            logger.error("HuggingFace synthesis failed", error=str(e))
            raise

    async def close(self) -> None:
        """Close HTTP clients."""
        if self._openai_client:
            await self._openai_client.aclose()
        if self._hf_client:
            await self._hf_client.aclose()
        logger.debug("VoiceProcessor closed")

    async def __aenter__(self) -> "VoiceProcessor":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Async context manager exit."""
        await self.close()

    # Private utility methods

    async def _estimate_audio_duration(self, audio_data: bytes, format: str) -> float:
        """Estimate audio duration from raw audio data.

        Uses ffprobe if available, otherwise falls back to rough estimate.

        Args:
            audio_data: Audio data in bytes.
            format: Audio format.

        Returns:
            Estimated duration in seconds.
        """
        try:
            import subprocess

            temp_path = Path("/tmp") / f"audio_{int(time.time() * 1000)}.{format}"

            try:
                temp_path.write_bytes(audio_data)

                cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1:noinput_type=1",
                    str(temp_path),
                ]

                result = await asyncio.to_thread(
                    subprocess.run, cmd, capture_output=True, timeout=5
                )

                if result.returncode == 0:
                    return float(result.stdout.decode().strip())

            finally:
                temp_path.unlink(missing_ok=True)

        except Exception as e:
            logger.debug("Could not estimate duration with ffprobe", error=str(e))

        # Fallback: rough estimate based on file size
        # Audio typically ~64-128 kbps, use 96 kbps average
        bytes_per_second = 12000  # 96 kbps / 8
        return len(audio_data) / bytes_per_second

    def _get_edge_tts_voice(self, language: str) -> str:
        """Get appropriate Edge-TTS voice for language.

        Args:
            language: Language code (ISO 639-1).

        Returns:
            Voice identifier for Edge-TTS.
        """
        # Map common languages to Edge-TTS voices
        voice_map = {
            "pl": "pl-PL-MarekNeural",  # Polish male
            "en": "en-US-GuyNeural",  # English male
            "es": "es-ES-AlvaroNeural",  # Spanish male
            "fr": "fr-FR-HenriNeural",  # French male
            "de": "de-DE-ConradNeural",  # German male
            "it": "it-IT-DiegoNeural",  # Italian male
            "pt": "pt-BR-AntonioNeural",  # Portuguese male
            "ru": "ru-RU-DmitryNeural",  # Russian male
        }

        return voice_map.get(language, "en-US-GuyNeural")
