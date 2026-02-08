"""Example: Voice message integration with AgentArmy.

Shows how to integrate WhatsApp voice messages with the voice processing system.

This example demonstrates:
  1. Handling incoming WhatsApp voice messages
  2. Transcribing audio with STT
  3. Processing text with agents
  4. Synthesizing response with TTS
  5. Sending voice reply back to user
"""

import asyncio
import os
from typing import Optional

import structlog

from src.bridges.voice import VoiceConfig, VoiceProcessor, VoiceProvider
from src.bridges.whatsapp_voice import (
    WhatsAppVoiceHandler,
    WhatsAppVoiceSettings,
)

logger = structlog.get_logger(__name__)


# Example 1: Simple STT + TTS pipeline
async def example_simple_voice_pipeline() -> None:
    """Example: Transcribe audio, then synthesize response."""
    logger.info("=== Example 1: Simple Voice Pipeline ===")

    # Configuration
    config = VoiceConfig(
        stt_provider=VoiceProvider.OPENAI,
        tts_provider=VoiceProvider.EDGE_TTS,
        default_language="pl",
        openai_api_key=os.getenv("AGENTARMY_OPENAI_API_KEY", "sk-test"),
    )

    async with VoiceProcessor(config) as processor:
        # Simulate receiving audio from WhatsApp
        # In production, this would be downloaded from WhatsApp media URL
        print("\n1. Simulating audio input (would come from WhatsApp)")

        # For testing, we'll skip actual audio and show the flow
        print("   → Audio received: 8 seconds, OGG format")

        # In real scenario:
        # audio_data = await handler.download_whatsapp_media(media_id)
        # stt_result = await processor.transcribe(audio_data, format="ogg")

        print("\n2. Transcribing audio...")
        print(f"   → Provider: {config.stt_provider.value}")
        print("   → (Would call OpenAI Whisper API)")

        # Simulated response
        transcribed_text = "Cześć! Czy możesz mi powiedzieć coś o AgentArmy?"

        print(f"\n3. Received transcription:")
        print(f"   → Text: {transcribed_text}")
        print(f"   → Language: Polish (detected)")
        print(f"   → Confidence: 0.95")

        print("\n4. Agent processes request...")
        response_text = (
            "AgentArmy to wieloagentowy system AI z orkiestracją, "
            "obsługą wiadomości i zarządzaniem zadaniami. "
            "Obsługuje integracje z WhatsApp, Ollama i wiele modeli LLM."
        )
        print(f"   → Response: {response_text[:50]}...")

        print("\n5. Synthesizing response to audio...")
        print(f"   → Provider: {config.tts_provider.value}")
        print(f"   → Voice: pl-PL-MarekNeural (Polish male)")
        print(f"   → Language: Polish")

        # In production:
        # tts_result = await processor.synthesize(
        #     text=response_text,
        #     language="pl",
        #     voice="pl-PL-MarekNeural",
        # )

        print("\n6. Would send voice reply back to user")
        print("   → Audio format: OGG (Opus codec)")
        print("   → Estimated duration: ~15 seconds")
        print("   → WhatsApp media upload complete")
        print("   → Message sent successfully!")


# Example 2: Multi-language support
async def example_multilanguage_support() -> None:
    """Example: Using voice in different languages."""
    logger.info("=== Example 2: Multi-Language Support ===")

    examples = [
        {
            "language": "pl",
            "language_name": "Polish",
            "text": "Cześć! Jak się masz?",
            "voice": "pl-PL-MarekNeural",
        },
        {
            "language": "en",
            "language_name": "English",
            "text": "Hello! How are you?",
            "voice": "en-US-GuyNeural",
        },
        {
            "language": "es",
            "language_name": "Spanish",
            "text": "¡Hola! ¿Cómo estás?",
            "voice": "es-ES-AlvaroNeural",
        },
    ]

    config = VoiceConfig(tts_provider=VoiceProvider.EDGE_TTS)

    print("\nEdge-TTS supports these languages via AgentArmy:\n")

    for example in examples:
        print(f"Language: {example['language_name']} ({example['language']})")
        print(f"  Voice: {example['voice']}")
        print(f"  Text:  {example['text']}")
        print(f"  → Synthesis: {len(example['text'].split())} words")
        print()


# Example 3: Provider comparison
async def example_provider_comparison() -> None:
    """Example: Choosing the right provider."""
    logger.info("=== Example 3: Provider Comparison ===")

    comparison = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                      PROVIDER COMPARISON & SELECTION                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

SPEECH-TO-TEXT (STT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. OpenAI Whisper (Default)
   ✓ Fastest (~2 seconds per minute of audio)
   ✓ Highest accuracy (95%+)
   ✓ Excellent Polish support
   ✗ Costs ~$0.02 per minute
   Best for: Production, high accuracy required

2. Local Whisper (faster-whisper)
   ✓ Free (no API costs)
   ✓ Privacy-first (no external calls)
   ✓ Good Polish support
   ✗ Slower (10-30 seconds per minute)
   ✗ CPU-intensive
   Best for: Privacy-sensitive, development, offline

3. HuggingFace Inference
   ✓ Free tier available
   ✓ Good accuracy
   ✗ Rate limited
   ✗ May be slower
   Best for: Low-volume, development

TEXT-TO-SPEECH (TTS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Edge-TTS (Default)
   ✓ Completely free
   ✓ Microsoft voices (natural sounding)
   ✓ Excellent Polish voices (Marek/Zofia)
   ✓ No authentication needed
   ✗ Cloud-dependent
   Best for: Cost-sensitive, production, Polish content

2. OpenAI TTS
   ✓ Natural sounding
   ✓ 6 different voices
   ✓ Consistent quality
   ✗ Costs ~$0.015 per 1000 characters
   ✗ English-optimized
   Best for: Premium English content, variety of voices

3. HuggingFace
   ✓ Open source models
   ✓ Customizable
   ✗ Lower quality
   ✗ Slower
   Best for: Research, experimentation

RECOMMENDED STRATEGIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Production Polish Content:
  STT: OpenAI Whisper → TTS: Edge-TTS
  (Fast + Polish, zero TTS cost)

Privacy-First:
  STT: Local Whisper → TTS: Edge-TTS
  (No external API calls for transcription)

Budget-Conscious:
  STT: HuggingFace → TTS: Edge-TTS
  (Both free, good quality)

Quality-First:
  STT: OpenAI → TTS: OpenAI
  (Best coherence, highest quality)

Mixed Language Support:
  STT: OpenAI → TTS: Edge-TTS
  (OpenAI multi-language, Edge-TTS Polish optimized)

    """

    print(comparison)


# Example 4: Error handling & fallbacks
async def example_error_handling() -> None:
    """Example: Handling voice errors gracefully."""
    logger.info("=== Example 4: Error Handling & Fallbacks ===")

    error_scenarios = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                      ERROR HANDLING STRATEGIES                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

SCENARIO 1: STT Fails (API timeout/error)
├─ Log error with structlog
├─ Check if text was provided in webhook (WhatsApp sometimes includes it)
└─ Send text-based response to user

SCENARIO 2: TTS Fails (API rate limit)
├─ Log error
├─ Check send_text_fallback setting
├─ If True: Send text response to user
└─ If False: Queue for retry with exponential backoff

SCENARIO 3: Audio Too Long (>120 seconds)
├─ Log warning
├─ Reject/truncate to max duration
└─ Send message to user: "Audio too long, please send shorter clips"

SCENARIO 4: Unsupported Format
├─ Log error
├─ Auto-convert using ffmpeg if available
└─ If conversion fails: Send error message

SCENARIO 5: Provider Unavailable
├─ Log critical error
├─ Fall back to alternate provider
├─ If all providers fail: Send text response
└─ Alert ops team

IMPLEMENTATION EXAMPLE:

config = VoiceConfig(
    stt_provider=VoiceProvider.OPENAI,
)

async with WhatsAppVoiceHandler(settings) as handler:
    try:
        # Attempt primary flow
        stt_result = await processor.transcribe(audio_data)
    except ValueError as e:
        # Audio validation error
        logger.error("Audio validation failed", error=str(e))
        await handler._send_text_fallback(
            to_number,
            "❌ Audio format not supported. Please send an audio message."
        )
        return

    except httpx.HTTPError as e:
        # API error
        logger.error("API call failed", error=str(e))

        # Try local fallback
        if config.stt_provider == VoiceProvider.OPENAI:
            logger.info("Attempting local fallback...")
            config.stt_provider = VoiceProvider.LOCAL_WHISPER
            stt_result = await processor.transcribe(audio_data)

    # Continue with processing...
    """

    print(error_scenarios)


# Example 5: WhatsApp webhook integration
async def example_whatsapp_webhook() -> None:
    """Example: Integrating with FastAPI webhook."""
    logger.info("=== Example 5: WhatsApp Webhook Integration ===")

    webhook_code = '''
# In your FastAPI webhook handler (src/main.py or similar)

from fastapi import FastAPI, Request
from src.bridges.whatsapp_voice import (
    WhatsAppVoiceHandler,
    WhatsAppVoiceSettings,
)
from src.bridges.voice import VoiceConfig, VoiceProvider

app = FastAPI()

# Initialize voice handler
voice_settings = WhatsAppVoiceSettings(
    whatsapp_api_token=os.getenv("AGENTARMY_WHATSAPP_API_TOKEN"),
    whatsapp_phone_number_id=os.getenv("AGENTARMY_WHATSAPP_PHONE_NUMBER_ID"),
    whatsapp_business_account_id=os.getenv("AGENTARMY_WHATSAPP_BUSINESS_ACCOUNT_ID"),
    voice_config=VoiceConfig(
        stt_provider=VoiceProvider.OPENAI,
        tts_provider=VoiceProvider.EDGE_TTS,
        openai_api_key=os.getenv("AGENTARMY_OPENAI_API_KEY"),
    ),
)

@app.post("/webhooks/whatsapp")
async def handle_whatsapp_webhook(request: Request):
    """Handle incoming WhatsApp messages (text and voice)."""
    payload = await request.json()
    messages = whatsapp_bridge.parse_incoming_webhook(payload)

    for msg in messages:
        # Check if message type is audio
        if msg.media_type == "audio":
            logger.info("Received voice message", from_number=msg.from_number)

            async with WhatsAppVoiceHandler(voice_settings) as handler:
                # Download audio from WhatsApp
                audio_data = await handler.download_whatsapp_media(msg.media_url)
                if not audio_data:
                    logger.error("Failed to download audio")
                    continue

                # Transcribe
                stt_result = await processor.transcribe(audio_data, format="ogg")
                logger.info("Transcribed", text=stt_result.text)

                # Process with agent
                response_text = await commander.process(stt_result.text)
                logger.info("Agent response", text=response_text)

                # Send voice response
                success = await handler.handle_voice_message(
                    message_data=msg.dict(),
                    response_text=response_text,
                )

                if not success:
                    # Fallback: send text
                    await whatsapp_bridge.send_text(msg.from_number, response_text)

        else:
            # Handle text messages normally
            response = await commander.process(msg.text)
            await whatsapp_bridge.send_text(msg.from_number, response)

    return {"status": "ok"}
    '''

    print(webhook_code)


# Main execution
async def main() -> None:
    """Run all examples."""
    try:
        await example_simple_voice_pipeline()
        print("\n" + "=" * 80 + "\n")

        await example_multilanguage_support()
        print("\n" + "=" * 80 + "\n")

        await example_provider_comparison()
        print("\n" + "=" * 80 + "\n")

        await example_error_handling()
        print("\n" + "=" * 80 + "\n")

        await example_whatsapp_webhook()

        logger.info("All examples completed!")

    except Exception as e:
        logger.error("Example failed", error=str(e), exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
