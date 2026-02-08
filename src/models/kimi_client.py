"""Kimi (Moonshot AI) client for AgentArmy with async support and token counting."""

import asyncio
import time
import json
from typing import AsyncGenerator, Optional, Any
import structlog
import httpx

from src.models.schemas import LLMRequest, LLMResponse, ToolCall, ModelProvider


logger = structlog.get_logger(__name__)


class KimiClient:
    """Async client for Kimi (Moonshot AI) models with advanced features.

    Supports:
    - Async chat completion (httpx-based, minimal dependencies)
    - Streaming responses
    - Token counting and cost tracking
    - Exponential backoff retry logic
    - Model selection (moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k, kimi-latest)
    - OpenAI-compatible API format
    """

    # Model pricing (per 1M tokens) as of knowledge cutoff
    MODEL_PRICING = {
        "moonshot-v1-8k": {"input": 1.0, "output": 2.0},
        "moonshot-v1-32k": {"input": 2.0, "output": 4.0},
        "moonshot-v1-128k": {"input": 6.0, "output": 12.0},
        "kimi-latest": {"input": 2.0, "output": 4.0},
    }

    DEFAULT_MODEL = "kimi-latest"
    API_BASE_URL = "https://api.moonshot.cn/v1"
    REQUEST_TIMEOUT = 120.0  # 2 minutes

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_retries: int = 3,
        timeout_seconds: float = REQUEST_TIMEOUT,
    ) -> None:
        """Initialize Kimi client.

        Args:
            api_key: Kimi/Moonshot API key.
            model: Model to use for requests (moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k, kimi-latest).
            max_retries: Maximum number of retries for failed requests.
            timeout_seconds: Request timeout in seconds.

        Raises:
            ValueError: If model is not supported.
        """
        if model not in self.MODEL_PRICING:
            logger.warning(
                "Unknown Kimi model, using default",
                requested_model=model,
                default_model=self.DEFAULT_MODEL,
            )
            model = self.DEFAULT_MODEL

        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

        # Initialize async client with Bearer token auth
        self.client = httpx.AsyncClient(
            base_url=self.API_BASE_URL,
            timeout=httpx.Timeout(timeout_seconds),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        logger.info(
            "KimiClient initialized",
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

        # Prepare messages in OpenAI-compatible format
        messages = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        messages.append({"role": "user", "content": request.prompt})

        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    "Sending request to Kimi",
                    model=self.model,
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                )

                # Build request payload (OpenAI-compatible format)
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                }

                # Send request
                response = await self.client.post("/chat/completions", json=payload)
                response.raise_for_status()

                data = response.json()
                latency_ms = (time.time() - start_time) * 1000

                # Process response
                content_text = ""

                # Extract text from response
                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    message = choice.get("message", {})

                    # Text content
                    if "content" in message and message["content"]:
                        content_text = message["content"]

                # Get token usage
                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                total_tokens = input_tokens + output_tokens

                # Calculate cost
                cost = self._calculate_cost(input_tokens, output_tokens)

                logger.info(
                    "Kimi request successful",
                    model=self.model,
                    tokens_used=total_tokens,
                    cost=cost,
                    latency_ms=latency_ms,
                )

                return LLMResponse(
                    content=content_text,
                    model_used=self.model,
                    provider="kimi",
                    tokens_used=total_tokens,
                    cost_estimate=cost,
                    latency_ms=latency_ms,
                    tool_calls=None,
                    stop_reason=choice.get("finish_reason", "unknown"),
                )

            except httpx.HTTPStatusError as e:
                last_error = e
                status_code = e.response.status_code

                # Check for rate limiting or temporary errors
                if status_code in (429, 500, 502, 503):
                    logger.warning(
                        "Kimi request failed (temporary error)",
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
                        "Kimi request failed (non-retryable)",
                        model=self.model,
                        status_code=status_code,
                        error=str(e),
                    )
                    raise

            except Exception as e:
                last_error = e
                logger.warning(
                    "Kimi request failed",
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
            "Kimi request failed after all retries",
            model=self.model,
            max_retries=self.max_retries,
            error=str(last_error),
        )
        raise last_error

    async def stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream a chat completion response.

        Yields chunks of text as they arrive from Kimi.

        Args:
            request: The LLM request.

        Yields:
            Text chunks from the response.
        """
        # Prepare messages in OpenAI-compatible format
        messages = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        messages.append({"role": "user", "content": request.prompt})

        try:
            logger.info("Starting Kimi stream", model=self.model)

            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": True,
            }

            async with self.client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line or line.startswith(":"):
                        continue

                    if line.startswith("data: "):
                        line = line[6:]

                    if line == "[DONE]":
                        break

                    try:
                        chunk = json.loads(line)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if "content" in delta and delta["content"]:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        logger.debug("Failed to parse stream chunk", line=line[:50])
                        continue

        except Exception as e:
            logger.error("Kimi streaming failed", model=self.model, error=str(e))
            raise

    def count_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a rough approximation (1 token ≈ 4 characters).
        For exact counts, use Kimi's token counting API.

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
        logger.debug("KimiClient closed")

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
            Estimated cost in USD (if pricing is available).
        """
        pricing = self.MODEL_PRICING.get(
            self.model,
            {"input": 2.0, "output": 4.0},  # Default to kimi-latest pricing
        )

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost
