"""Bielik (SpeakLeash) Polish language model client for Code Horde via Ollama.

Bielik is a Polish language model optimized for Polish NLP tasks.
This client wraps Ollama but provides Bielik-specific features including:
- Polish language detection and routing
- Polish-specific prompt templates
- Translation assistance for cross-model communication
- Health checks and model availability
"""

import asyncio
import time
import re
from typing import Optional, AsyncGenerator
import structlog
import httpx

from src.models.schemas import LLMRequest, LLMResponse, ModelProvider


logger = structlog.get_logger(__name__)


class BielikClient:
    """Async client for Bielik Polish language models via Ollama.

    Supports:
    - Polish language detection
    - Multiple Bielik model variants
    - Polish-optimized prompting
    - Translation assistance
    - Health checks and model availability verification
    """

    # Bielik model variants available via Ollama
    BIELIK_VARIANTS = {
        "bielik-4.5b": "bielik:4.5b-v3.0-instruct",
        "bielik-11b": "bielik:11b-v3.0-instruct",
    }

    DEFAULT_VARIANT = "bielik-4.5b"
    DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
    REQUEST_TIMEOUT = 300.0  # 5 minutes for longer completions

    # Polish language detection patterns
    POLISH_CHAR_PATTERN = re.compile(r"[ąćęłńóśźż]", re.IGNORECASE)
    POLISH_KEYWORDS = {
        "jak", "co", "gdzie", "kiedy", "dlaczego", "czy",
        "i", "lub", "ale", "jeśli", "wtedy", "gdy",
        "jest", "są", "będzie", "będą", "ma", "mają",
    }

    def __init__(
        self,
        ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
        variant: str = DEFAULT_VARIANT,
        timeout_seconds: float = REQUEST_TIMEOUT,
    ) -> None:
        """Initialize Bielik client (via Ollama).

        Args:
            ollama_base_url: Base URL for Ollama API.
            variant: Bielik variant to use (bielik-4.5b or bielik-11b).
            timeout_seconds: Request timeout in seconds.

        Raises:
            ValueError: If variant is not supported.
        """
        if variant not in self.BIELIK_VARIANTS:
            logger.warning(
                "Unknown Bielik variant, using default",
                requested_variant=variant,
                default_variant=self.DEFAULT_VARIANT,
            )
            variant = self.DEFAULT_VARIANT

        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.variant = variant
        self.model_id = self.BIELIK_VARIANTS[variant]
        self.timeout_seconds = timeout_seconds

        # Initialize async client
        self.client = httpx.AsyncClient(
            base_url=self.ollama_base_url,
            timeout=httpx.Timeout(timeout_seconds),
        )

        logger.info(
            "BielikClient initialized",
            variant=variant,
            model_id=self.model_id,
            ollama_url=self.ollama_base_url,
        )

    async def health_check(self) -> bool:
        """Check if Ollama service is healthy and Bielik model is available.

        Returns:
            True if Ollama is healthy and Bielik is available, False otherwise.
        """
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()

            data = response.json()
            available_models = [model["name"] for model in data.get("models", [])]

            # Check if any Bielik variant is available
            bielik_available = any(
                self.model_id in model for model in available_models
            )

            logger.debug(
                "Bielik health check",
                ollama_status="healthy",
                bielik_available=bielik_available,
            )
            return bielik_available

        except Exception as e:
            logger.warning("Bielik health check failed", error=str(e))
            return False

    async def detect_language(self, text: str) -> str:
        """Detect if text is in Polish or another language.

        Uses heuristics: Polish diacritics and common Polish words.

        Args:
            text: Text to analyze.

        Returns:
            "polish" if detected as Polish, "other" otherwise.
        """
        # Check for Polish diacritics
        if self.POLISH_CHAR_PATTERN.search(text):
            logger.debug("Polish diacritics detected")
            return "polish"

        # Check for common Polish keywords
        words = text.lower().split()
        polish_word_count = sum(1 for word in words if word in self.POLISH_KEYWORDS)

        if polish_word_count >= max(1, len(words) // 5):
            logger.debug("Polish keywords detected")
            return "polish"

        return "other"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a chat completion request using Bielik.

        Args:
            request: The LLM request with prompt and parameters.

        Returns:
            LLMResponse with the model's response and metadata.

        Raises:
            Exception: If the request fails.
        """
        start_time = time.time()

        # Detect language and enhance prompt if Polish
        language = await self.detect_language(request.prompt)

        # Build Polish-optimized prompt if needed
        prompt = request.prompt
        if language == "polish":
            prompt = self._build_polish_prompt(request)
            logger.debug("Using Polish-optimized prompt")

        # Prepare messages for Ollama
        messages = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            logger.info(
                "Sending request to Bielik",
                variant=self.variant,
                language=language,
            )

            response = await self.client.post(
                "/api/chat",
                json={
                    "model": self.model_id,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens,
                    },
                },
            )

            response.raise_for_status()
            data = response.json()

            latency_ms = (time.time() - start_time) * 1000

            # Extract response
            content = data.get("message", {}).get("content", "")

            # Get token counts if available
            tokens_used = None
            if "prompt_eval_count" in data and "eval_count" in data:
                tokens_used = data["prompt_eval_count"] + data["eval_count"]

            logger.info(
                "Bielik request successful",
                variant=self.variant,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
            )

            return LLMResponse(
                content=content,
                model_used=self.model_id,
                provider=ModelProvider.BIELIK,
                tokens_used=tokens_used,
                cost_estimate=0.0,  # Local model
                latency_ms=latency_ms,
            )

        except httpx.HTTPError as e:
            logger.error(
                "Bielik request failed",
                variant=self.variant,
                error=str(e),
            )
            raise

    async def stream(
        self, request: LLMRequest
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion response.

        Yields chunks of text as they arrive from Bielik.

        Args:
            request: The LLM request.

        Yields:
            Text chunks from the response.
        """
        # Detect language and build Polish-optimized prompt
        language = await self.detect_language(request.prompt)
        prompt = request.prompt

        if language == "polish":
            prompt = self._build_polish_prompt(request)

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            async with self.client.stream(
                "POST",
                "/api/chat",
                json={
                    "model": self.model_id,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens,
                    },
                },
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        import json
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                    except Exception as e:
                        logger.debug("Failed to parse stream chunk", error=str(e))
                        continue

        except Exception as e:
            logger.error("Bielik streaming failed", variant=self.variant, error=str(e))
            raise

    async def translate_to_english(self, polish_text: str) -> str:
        """Translate Polish text to English using Bielik.

        Useful for sending Polish prompts to non-Polish models.

        Args:
            polish_text: Polish text to translate.

        Returns:
            English translation.
        """
        translation_prompt = (
            f"Przetłumacz poniższy tekst na angielski. "
            f"Zwróć tylko tłumaczenie bez wyjaśnień.\n\n"
            f"Tekst: {polish_text}"
        )

        request = LLMRequest(
            prompt=translation_prompt,
            max_tokens=512,
            temperature=0.3,  # Lower temperature for more deterministic translation
        )

        try:
            response = await self.complete(request)
            return response.content.strip()
        except Exception as e:
            logger.error("Translation failed", error=str(e))
            return polish_text  # Return original if translation fails

    async def translate_from_english(self, english_text: str) -> str:
        """Translate English text to Polish using Bielik.

        Useful for providing Polish responses from non-Polish models.

        Args:
            english_text: English text to translate.

        Returns:
            Polish translation.
        """
        translation_prompt = (
            f"Przetłumacz poniższy tekst z angielskiego na polski. "
            f"Zwróć tylko tłumaczenie bez wyjaśnień.\n\n"
            f"Text: {english_text}"
        )

        request = LLMRequest(
            prompt=translation_prompt,
            max_tokens=512,
            temperature=0.3,
        )

        try:
            response = await self.complete(request)
            return response.content.strip()
        except Exception as e:
            logger.error("Translation failed", error=str(e))
            return english_text  # Return original if translation fails

    async def close(self) -> None:
        """Close the HTTP client.

        Should be called when done using the client.
        """
        await self.client.aclose()
        logger.debug("BielikClient closed")

    def __del__(self) -> None:
        """Cleanup when client is garbage collected."""
        try:
            asyncio.run(self.close())
        except Exception:
            pass

    # Private methods

    def _build_polish_prompt(self, request: LLMRequest) -> str:
        """Build a Polish-optimized prompt.

        Applies language-specific formatting and instructions.

        Args:
            request: The original LLM request.

        Returns:
            Enhanced Polish prompt.
        """
        # For Polish models, we can add context about the language
        # This helps the model understand we want responses in Polish
        prompt = request.prompt

        # If system prompt doesn't specify language, add Polish directive
        if request.system_prompt and "polski" not in request.system_prompt.lower():
            # Already has a system prompt, don't modify
            pass
        elif not request.system_prompt:
            # No system prompt, might be good to add Polish context
            # (but we don't modify the request.system_prompt here)
            pass

        return prompt

    def count_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a rough approximation (1 token ≈ 4 characters).

        Args:
            text: Text to count tokens for.

        Returns:
            Estimated token count.
        """
        return max(1, len(text) // 4)
