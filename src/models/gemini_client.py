"""Google Gemini client for AgentArmy with async support and token counting."""

import asyncio
import time
import json
from typing import AsyncGenerator, Optional, Any
import structlog
import httpx

from src.models.schemas import LLMRequest, LLMResponse, ToolCall, ModelProvider


logger = structlog.get_logger(__name__)


class GeminiClient:
    """Async client for Google Gemini models with advanced features.

    Supports:
    - Async chat completion (httpx-based, minimal dependencies)
    - Streaming responses
    - Token counting and cost tracking
    - Exponential backoff retry logic
    - Model selection (gemini-2.0-flash, gemini-2.0-flash-lite, gemini-2.5-pro)
    """

    # Model pricing (per 1M tokens) as of knowledge cutoff
    MODEL_PRICING = {
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
        "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    }

    DEFAULT_MODEL = "gemini-2.0-flash"
    API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    REQUEST_TIMEOUT = 120.0  # 2 minutes

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_retries: int = 3,
        timeout_seconds: float = REQUEST_TIMEOUT,
    ) -> None:
        """Initialize Gemini client.

        Args:
            api_key: Google Gemini API key.
            model: Model to use for requests (gemini-2.0-flash, gemini-2.0-flash-lite, gemini-2.5-pro).
            max_retries: Maximum number of retries for failed requests.
            timeout_seconds: Request timeout in seconds.

        Raises:
            ValueError: If model is not supported.
        """
        if model not in self.MODEL_PRICING:
            logger.warning(
                "Unknown Gemini model, using default",
                requested_model=model,
                default_model=self.DEFAULT_MODEL,
            )
            model = self.DEFAULT_MODEL

        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

        # Initialize async client without auth header (uses query param instead)
        self.client = httpx.AsyncClient(
            base_url=self.API_BASE_URL,
            timeout=httpx.Timeout(timeout_seconds),
            headers={
                "Content-Type": "application/json",
            },
        )

        logger.info(
            "GeminiClient initialized",
            model=model,
            max_retries=max_retries,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute an async chat completion request.

        Args:
            request: The LLM request with prompt and parameters.

        Returns:
            LLMResponse with the model's response and metadata.

        Raises:
            Exception: If all retry attempts fail.
        """
        start_time = time.time()

        # Prepare contents array for Gemini
        contents = [
            {
                "parts": [
                    {"text": request.prompt}
                ]
            }
        ]

        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    "Sending request to Gemini",
                    model=self.model,
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                )

                # Build request payload
                payload = {
                    "contents": contents,
                    "generationConfig": {
                        "temperature": request.temperature,
                        "maxOutputTokens": request.max_tokens,
                    },
                }

                # Add system instruction if provided
                if request.system_prompt:
                    payload["systemInstruction"] = {
                        "parts": [
                            {"text": request.system_prompt}
                        ]
                    }

                # Send request with API key as query parameter
                response = await self.client.post(
                    f"/models/{self.model}:generateContent",
                    json=payload,
                    params={"key": self.api_key},
                )
                response.raise_for_status()

                data = response.json()
                latency_ms = (time.time() - start_time) * 1000

                # Process response
                content_text = ""

                # Extract text from response
                if "candidates" in data and len(data["candidates"]) > 0:
                    candidate = data["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        parts = candidate["content"]["parts"]
                        if parts and "text" in parts[0]:
                            content_text = parts[0]["text"]

                # Get token usage
                usage = data.get("usageMetadata", {})
                input_tokens = usage.get("promptTokenCount", 0)
                output_tokens = usage.get("candidatesTokenCount", 0)
                total_tokens = input_tokens + output_tokens

                # Calculate cost
                cost = self._calculate_cost(input_tokens, output_tokens)

                logger.info(
                    "Gemini request successful",
                    model=self.model,
                    tokens_used=total_tokens,
                    cost=cost,
                    latency_ms=latency_ms,
                )

                return LLMResponse(
                    content=content_text,
                    model_used=self.model,
                    provider="gemini",
                    tokens_used=total_tokens,
                    cost_estimate=cost,
                    latency_ms=latency_ms,
                    tool_calls=None,
                    stop_reason="stop",
                )

            except httpx.HTTPStatusError as e:
                last_error = e
                status_code = e.response.status_code

                # Check for rate limiting or temporary errors
                if status_code in (429, 500, 502, 503):
                    logger.warning(
                        "Gemini request failed (temporary error)",
                        model=self.model,
                        attempt=attempt + 1,
                        status_code=status_code,
                        error=str(e),
                    )

                    if attempt < self.max_retries - 1:
                        # Exponential backoff with jitter
                        wait_time = (2 ** attempt) + (time.time() % 1)
                        logger.info(
                            "Retrying after backoff",
                            wait_seconds=wait_time,
                        )
                        await asyncio.sleep(wait_time)
                        continue
                else:
                    # Non-retryable error
                    logger.error(
                        "Gemini request failed (non-retryable)",
                        model=self.model,
                        status_code=status_code,
                        error=str(e),
                    )
                    raise

            except Exception as e:
                last_error = e
                logger.warning(
                    "Gemini request failed",
                    model=self.model,
                    attempt=attempt + 1,
                    error=str(e),
                )

                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info("Retrying after backoff", wait_seconds=wait_time)
                    await asyncio.sleep(wait_time)

        # All retries exhausted
        logger.error(
            "Gemini request failed after all retries",
            model=self.model,
            max_retries=self.max_retries,
            error=str(last_error),
        )
        raise last_error

    async def stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream a chat completion response.

        Yields chunks of text as they arrive from Gemini.

        Args:
            request: The LLM request.

        Yields:
            Text chunks from the response.
        """
        # Prepare contents array for Gemini
        contents = [
            {
                "parts": [
                    {"text": request.prompt}
                ]
            }
        ]

        try:
            logger.info("Starting Gemini stream", model=self.model)

            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": request.temperature,
                    "maxOutputTokens": request.max_tokens,
                },
            }

            # Add system instruction if provided
            if request.system_prompt:
                payload["systemInstruction"] = {
                    "parts": [
                        {"text": request.system_prompt}
                    ]
                }

            # Use streaming endpoint
            async with self.client.stream(
                "POST",
                f"/models/{self.model}:streamGenerateContent",
                json=payload,
                params={"key": self.api_key},
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line or line.startswith(":"):
                        continue

                    try:
                        chunk = json.loads(line)
                        if "candidates" in chunk and len(chunk["candidates"]) > 0:
                            candidate = chunk["candidates"][0]
                            if "content" in candidate and "parts" in candidate["content"]:
                                parts = candidate["content"]["parts"]
                                if parts and "text" in parts[0]:
                                    yield parts[0]["text"]
                    except json.JSONDecodeError:
                        logger.debug("Failed to parse stream chunk", line=line[:50])
                        continue

        except Exception as e:
            logger.error("Gemini streaming failed", model=self.model, error=str(e))
            raise

    def count_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a rough approximation (1 token ≈ 4 characters).
        For exact counts, use Gemini's token counting API.

        Args:
            text: Text to count tokens for.

        Returns:
            Estimated token count.
        """
        return max(1, len(text) // 4)

    async def close(self) -> None:
        """Close the HTTP client.

        Should be called when done using the client.
        """
        await self.client.aclose()
        logger.debug("GeminiClient closed")

    def __del__(self) -> None:
        """Cleanup when client is garbage collected."""
        try:
            asyncio.run(self.close())
        except Exception:
            pass

    # Private methods

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost for a request.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        pricing = self.MODEL_PRICING.get(
            self.model,
            {"input": 0.10, "output": 0.40},  # Default to gemini-2.0-flash pricing
        )

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost
