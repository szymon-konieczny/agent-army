# Voice (TTS/STT) Integration Guide

## Overview

Code Horde now includes production-quality voice modules for speech-to-text (STT) and text-to-speech (TTS) processing. This enables:

- WhatsApp voice messages: Users send voice notes → Code Horde transcribes, processes, and responds with audio
- Multi-provider support for redundancy and cost optimization
- Excellent Polish language support via Edge-TTS
- Local transcription options for privacy
- Async/await with full type hints and structured logging

## Architecture

### Components

1. **`src/bridges/voice.py`** (700 lines)
   - Core voice processing engine
   - Multi-provider STT and TTS support
   - Audio format conversion (ogg/opus → wav → mp3)
   - Language detection and voice profile selection

2. **`src/bridges/whatsapp_voice.py`** (507 lines)
   - WhatsApp-specific voice message handling
   - Full message flow: download → transcribe → process → synthesize → upload → send
   - Text fallback on TTS failure
   - Optional dual voice+text responses

### Supported Providers

#### Speech-to-Text (STT)

| Provider | Pros | Cons | Polish | Cost |
|----------|------|------|--------|------|
| OpenAI Whisper API | Fast, accurate, excellent Polish | Cloud-only, paid | ✅ Excellent | ~$0.02/min |
| Local Whisper (faster-whisper) | Free, private, CTranslate2 backend | CPU-only, slower | ✅ Good | Free |
| HuggingFace Inference | Free tier available, multi-language | Rate limited | ✅ Good | Free/Paid |

Default: **OpenAI** (fastest, most reliable)

#### Text-to-Speech (TTS)

| Provider | Pros | Cons | Polish | Cost |
|----------|------|------|--------|------|
| Edge-TTS | Free, Microsoft voices, great Polish | Cloud-dependent | ✅ Excellent | Free |
| OpenAI TTS API | Natural sounding, 6 voices | Paid, limited voices | ⚠️ English-optimized | ~$0.015/1k chars |
| HuggingFace Inference | Multi-language, open models | Lower quality | ⚠️ Decent | Free/Paid |

Default: **Edge-TTS** (free, excellent Polish: `pl-PL-MarekNeural` / `pl-PL-ZofiaNeural`)

## Configuration

### Environment Variables (.env.local)

```bash
# Speech-to-Text provider
AGENTARMY_VOICE_STT_PROVIDER=openai           # openai, local_whisper, huggingface
AGENTARMY_VOICE_TTS_PROVIDER=edge_tts          # openai, edge_tts, huggingface

# Languages & voices
AGENTARMY_VOICE_DEFAULT_LANGUAGE=pl            # ISO 639-1 code
AGENTARMY_VOICE_TTS_VOICE=pl-PL-MarekNeural   # edge-tts voice identifier
AGENTARMY_VOICE_OPENAI_TTS_VOICE=onyx         # OpenAI voice (alloy/echo/fable/onyx/nova/shimmer)

# Limits
AGENTARMY_VOICE_MAX_DURATION=120              # Max audio duration in seconds

# API keys (if using those providers)
AGENTARMY_OPENAI_API_KEY=sk-...
AGENTARMY_HF_API_KEY=hf_...
```

### Python Configuration

```python
from src.bridges.voice import VoiceConfig, VoiceProcessor, VoiceProvider

# Create config
config = VoiceConfig(
    stt_provider=VoiceProvider.OPENAI,
    tts_provider=VoiceProvider.EDGE_TTS,
    default_language="pl",
    tts_voice="pl-PL-MarekNeural",
    openai_api_key="sk-...",
)

# Initialize processor
async with VoiceProcessor(config) as processor:
    # Use processor...
```

## Usage Examples

### Basic Speech-to-Text (STT)

```python
from src.bridges.voice import VoiceConfig, VoiceProcessor, VoiceProvider

async def transcribe_voice_message():
    config = VoiceConfig(
        stt_provider=VoiceProvider.OPENAI,
        openai_api_key="sk-...",
    )

    async with VoiceProcessor(config) as processor:
        # Read audio file
        audio_data = open("voice_message.ogg", "rb").read()

        # Transcribe
        result = await processor.transcribe(
            audio_data,
            format="ogg",
            language="pl",  # Optional hint
        )

        print(f"Text: {result.text}")
        print(f"Language: {result.language}")
        print(f"Confidence: {result.confidence}")
        print(f"Duration: {result.duration_seconds}s")
        print(f"Processing time: {result.processing_time_ms}ms")
```

### Basic Text-to-Speech (TTS)

```python
from src.bridges.voice import VoiceConfig, VoiceProcessor, VoiceProvider

async def synthesize_response():
    config = VoiceConfig(
        tts_provider=VoiceProvider.EDGE_TTS,
    )

    async with VoiceProcessor(config) as processor:
        # Synthesize text
        result = await processor.synthesize(
            text="Cześć! Jak się masz?",
            language="pl",
            voice="pl-PL-MarekNeural",  # Male voice
        )

        # Save audio
        with open("response.ogg", "wb") as f:
            f.write(result.audio_data)

        print(f"Format: {result.format}")
        print(f"Duration: {result.duration_seconds}s")
        print(f"Voice: {result.voice}")
```

### WhatsApp Voice Message Integration

```python
from src.bridges.whatsapp_voice import (
    WhatsAppVoiceHandler,
    WhatsAppVoiceSettings,
)
from src.bridges.voice import VoiceConfig, VoiceProvider

# Setup
settings = WhatsAppVoiceSettings(
    whatsapp_api_token="EAA...",
    whatsapp_phone_number_id="123456789",
    whatsapp_business_account_id="987654321",
    voice_config=VoiceConfig(
        stt_provider=VoiceProvider.OPENAI,
        tts_provider=VoiceProvider.EDGE_TTS,
        default_language="pl",
        openai_api_key="sk-...",
    ),
    send_both_voice_and_text=True,
)

async def handle_incoming_voice():
    async with WhatsAppVoiceHandler(settings) as handler:
        # Webhook receives: {"messages": [{"from": "48...", "type": "audio", ...}]}

        # Download audio from WhatsApp
        audio_data = await handler.download_whatsapp_media(media_id="wamid.ABC123")

        # Transcribe
        from src.bridges.voice import VoiceProcessor
        config = settings.voice_config
        async with VoiceProcessor(config) as processor:
            stt_result = await processor.transcribe(audio_data, format="ogg")

        # Process with agent (your logic here)
        response_text = "Cześć! Otrzymałem twoją wiadomość."

        # Send voice response
        success = await handler.handle_voice_message(
            message_data={"from": "48123456789"},
            response_text=response_text,
        )

        print(f"Response sent: {success}")
```

### Full WhatsApp Webhook Integration

In your webhook handler (`src/bridges/webhook_handler.py` or similar):

```python
@app.post("/whatsapp/webhook")
async def handle_whatsapp_webhook(request: Request):
    payload = await request.json()
    messages = whatsapp_bridge.parse_incoming_webhook(payload)

    for msg in messages:
        if msg.media_type == "audio":
            # Route to voice handler
            from src.bridges.whatsapp_voice import WhatsAppVoiceHandler

            async with WhatsAppVoiceHandler(voice_settings) as handler:
                # Transcribe incoming voice
                audio = await handler.download_whatsapp_media(msg.media_url)
                stt_result = await processor.transcribe(audio, format="ogg")

                # Process text with agent
                response = await commander.process(stt_result.text)

                # Send voice response
                await handler.handle_voice_message(
                    message_data={"from": msg.from_number},
                    response_text=response,
                )
        else:
            # Handle text messages normally
            response = await commander.process(msg.text)
            await whatsapp_bridge.send_text(msg.from_number, response)
```

## Audio Format Support

### STT Input Formats

Automatically converted to WAV if needed:
- **ogg** (Opus codec) - WhatsApp default
- **opus** - Raw Opus
- **wav** - PCM/WAV
- **mp3** - MPEG-3
- **m4a** - MPEG-4 Audio
- **webm** - WebM audio

### TTS Output Formats

Configurable via `VoiceConfig.audio_format`:
- **ogg** (Opus) - Default, WhatsApp compatible
- **mp3** - Wide compatibility
- **wav** - Highest quality
- **m4a** - Apple devices

## Language Support

### Polish Voices (Edge-TTS)

```python
# Male voice (default)
voice = "pl-PL-MarekNeural"

# Female voice
voice = "pl-PL-ZofiaNeural"
```

### Multi-Language Support

Language code mapping for Edge-TTS auto-selection:

| Language | Code | Male Voice | Female Voice |
|----------|------|------------|--------------|
| Polish | `pl` | pl-PL-MarekNeural | pl-PL-ZofiaNeural |
| English (US) | `en` | en-US-GuyNeural | en-US-JennyNeural |
| Spanish | `es` | es-ES-AlvaroNeural | (varies) |
| French | `fr` | fr-FR-HenriNeural | (varies) |
| German | `de` | de-DE-ConradNeural | (varies) |
| Italian | `it` | it-IT-DiegoNeural | (varies) |
| Portuguese | `pt` | pt-BR-AntonioNeural | (varies) |
| Russian | `ru` | ru-RU-DmitryNeural | (varies) |

## Error Handling

All voice operations include graceful error handling:

```python
try:
    result = await processor.transcribe(audio_data)
except ValueError as e:
    # Audio too long, format unsupported, etc.
    logger.error("Transcription failed", error=str(e))
except httpx.HTTPError as e:
    # Network/API error
    logger.error("API call failed", error=str(e))
```

### Fallback Strategies

WhatsApp voice handler includes automatic fallbacks:

1. **TTS Fails → Send Text**: If `send_text_fallback=True` in settings
2. **Long Audio**: Automatically split if duration > `chunk_duration_seconds`
3. **Both Versions**: Send voice + text if `send_both_voice_and_text=True`

## Logging

All operations logged with structlog:

```
INFO VoiceProcessor initialized stt_provider=openai tts_provider=edge_tts
INFO Audio transcribed provider=openai language=pl duration_ms=1234.5
INFO Text synthesized provider=edge_tts voice=pl-PL-MarekNeural duration_ms=567.8
INFO Voice message sent to_number=48123456789 message_id=wamid.XYZ
```

## Performance Tuning

### Local Whisper Optimization

```python
# For CPU-bound local transcription, use threading pool
config = VoiceConfig(
    stt_provider=VoiceProvider.LOCAL_WHISPER,
)

processor = VoiceProcessor(config)
# Audio processing runs in thread pool via asyncio.to_thread()
result = await processor.transcribe(audio_data)
```

### Provider Selection Strategy

**For best latency**: OpenAI STT + Edge-TTS (cloud-based, parallelizable)
**For privacy**: Local Whisper + Edge-TTS (no external API calls for transcription)
**For cost**: Edge-TTS (free) + HuggingFace STT with free tier
**For quality**: OpenAI Whisper (STT) + OpenAI TTS (coherent voices)

## Troubleshooting

### Issue: "faster-whisper not installed"

```bash
pip install faster-whisper
# Or via pyproject.toml:
# pip install -e .
```

### Issue: "edge-tts not installed"

```bash
pip install edge-tts
```

### Issue: "ffmpeg not found"

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

### Issue: "Audio exceeds maximum duration"

Increase in config or chunk incoming audio:

```python
config = VoiceConfig(
    max_audio_duration_seconds=300,  # 5 minutes instead of 2
)
```

### Issue: "OpenAI rate limit exceeded"

Use fallback provider:

```python
config = VoiceConfig(
    stt_provider=VoiceProvider.LOCAL_WHISPER,  # Fallback to local
)
```

## Testing

### Unit Tests

```python
import pytest
from src.bridges.voice import VoiceConfig, VoiceProcessor, VoiceProvider

@pytest.mark.asyncio
async def test_voice_processor_initialization():
    config = VoiceConfig()
    async with VoiceProcessor(config) as processor:
        assert processor.config.stt_provider == VoiceProvider.OPENAI
        assert processor.config.tts_provider == VoiceProvider.EDGE_TTS

@pytest.mark.asyncio
async def test_transcribe_mock(mock_openai_response):
    config = VoiceConfig(openai_api_key="sk-test")
    async with VoiceProcessor(config) as processor:
        audio_data = b"fake_audio_data"
        result = await processor.transcribe(audio_data)
        assert result.text == "test transcription"
        assert result.provider == VoiceProvider.OPENAI
```

## Dependencies

Added to `pyproject.toml`:

```toml
edge-tts>=6.1.0          # Free Microsoft TTS, excellent Polish
faster-whisper>=1.0.0    # Local Whisper with CTranslate2 backend
```

Also required (already in pyproject.toml):
- `httpx>=0.25.0` - Async HTTP client
- `structlog>=24.1.0` - Structured logging
- `pydantic>=2.0` - Data validation

## Next Steps

1. **Install dependencies**: `pip install -e .`
2. **Configure** `.env.local` with API keys
3. **Test providers**: Try each provider with sample audio
4. **Integrate into webhooks**: Add voice handling to WhatsApp webhook
5. **Deploy**: Test in production with actual voice messages

## References

- OpenAI Whisper: https://platform.openai.com/docs/guides/speech-to-text
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- edge-tts: https://github.com/rany2/edge-tts
- WhatsApp API: https://developers.facebook.com/docs/whatsapp/cloud-api
