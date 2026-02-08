"""Hugging Face Inference API client for AgentArmy with async support."""

import asyncio
import time
import json
from typing import AsyncGenerator, Optional, Any
import structlog
import httpx

from src.models.schemas import LLMRequest, LLMResponse, ModelProvider


logger = structlog.get_logger(__name__)


class HuggingFaceClient:
    """Async client for Hugging Face Inference API models.

    Supports:
    - Async text generation via HF Inference API
    - Text embeddings
    - Model info retrieval
    - Dedicated endpoint support (user's own hosted endpoints)
    - Error handling with retries
    """

    # Popular open-source models available on HF Inference API
    AVAILABLE_MODELS = {
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.1",
        "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
        "llama2-13b": "meta-llama/Llama-2-13b-chat-hf",
        "falcon-7b": "tiiuae/falcon-7b-instruct",
        "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
        "openchat-3.5": "openchat/openchat_3.5",
    }

    # Estimated token cost (free tier has rate limits, paid tier has costs)
    # Free tier: 1 request per 2 seconds
    # Pricing: typically $0.001-0.005 per 1M tokens depending on model
    MODEL_PRICING = {
        "mistral-7b": {"input": 0.001, "output": 0.001},
        "llama2-7b": {"input": 0.001, "output": 0.001},
        "llama2-13b": {"input": 0.002, "output": 0.002},
        "falcon-7b": {"input": 0.001, "output": 0.001},
        "zephyr-7b": {"input": 0.001, "output": 0.001},
    }

    DEFAULT_BASE_URL = "https://api-inference.huggingface.co"
    DEFAULT_MODEL = "mistral-7b"
    REQUEST_TIMEOUT = 120.0  # 2 minutes

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: Optional[str] = None,
        dedicated_endpoint: Optional[str] = None,
        timeout_seconds: float = REQUEST_TIMEOUT,
        max_retries: int = 3,
    ) -> None:
        """Initialize Hugging Face client.

        Args:
            api_key: Hugging Face API token (from huggingface.co/settings/tokens).
            model: Model identifier (short name or full HF model ID).
            base_url: Base URL for HF Inference API (override default).
            dedicated_endpoint: URL of a dedicated/private endpoint (if using one).
            timeout_seconds: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
        """
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        # Resolve model ID
        self.model_short_name = model
        self.model_id = self.AVAILABLE_MODELS.get(model, model)

        # Use dedicated endpoint if provided, otherwise use standard API
        if dedicated_endpoint:
            self.base_url = dedicated_endpoint.rstrip("/")
            self.is_dedicated = True
            logger.info("Using dedicated HF endpoint", endpoint=self.base_url)
        else:
            self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
            self.is_dedicated = False

        # Initialize async client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_seconds),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        logger.info(
            "HuggingFaceClient initialized",
            model=model,
            model_id=self.model_id,
            is_dedicated=self.is_dedicated,
        )

    async def health_check(self) -> bool:
        """Check if HF Inference API is accessible.

        Returns:
            True if API is reachable, False otherwise.
        """
        try:
            # Try to get model info
            response = await self.client.get(f"/model-info/{self.model_id}")
            is_healthy = response.status_code == 200
            logger.debug("HF health check", status=response.status_code)
            return is_healthy
        except Exception as e:
            logger.warning("HF health check failed", error=str(e))
            return False

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a text generation request.

        Args:
            request: The LLM request with prompt and parameters.

        Returns:
            LLMResponse with the model's response and metadata.

        Raises:
            Exception: If all retry attempts fail.
        """
        start_time = time.time()

        # Build the full prompt (HF doesn't have system/user separation in basic API)
        if request.system_prompt:
            prompt = f"{request.system_prompt}\n\n{request.prompt}"
        else:
            prompt = request.prompt

        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    "Sending request to HuggingFace",
                    model=self.model_id,
                    attempt=attempt + 1,
                )

                # Prepare request payload
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": request.max_tokens,
                        "temperature": request.temperature,
                        "top_p": 0.9,
                        "do_sample": True,
                        "return_full_text": False,
                    },
                }

                # Send request to text-generation endpoint
                response = await self.client.post(
                    f"/models/{self.model_id}",
                    json=payload,
                )

                response.raise_for_status()
                data = response.json()

                latency_ms = (time.time() - start_time) * 1000

                # Extract text from response
                content_text = ""
                if isinstance(data, list) and len(data) > 0:
                    # Standard HF text-generation response
                    content_text = data[0].get("generated_text", "")
                elif isinstance(data, dict) and "generated_text" in data:
                    content_text = data["generated_text"]

                # Estimate token counts (HF doesn't always return these)
                # Rough estimation: 1 token ≈ 4 characters
                input_tokens = max(1, len(prompt) // 4)
                output_tokens = max(1, len(content_text) // 4)
                total_tokens = input_tokens + output_tokens

                # Calculate cost
                cost = self._calculate_cost(input_tokens, output_tokens)

                logger.info(
                    "HF request successful",
                    model=self.model_id,
                    tokens_used=total_tokens,
                    latency_ms=latency_ms,
                )

                return LLMResponse(
                    content=content_text,
                    model_used=self.model_id,
                    provider=ModelProvider.HUGGINGFACE,
                    tokens_used=total_tokens,
                    cost_estimate=cost,
                    latency_ms=latency_ms,
                )

            except httpx.HTTPStatusError as e:
                last_error = e
                status_code = e.response.status_code

                # Check for rate limiting or model loading
                if status_code == 429:
                    logger.warning(
                        "HF rate limited (free tier)",
                        attempt=attempt + 1,
                    )

                    if attempt < self.max_retries - 1:
                        # Exponential backoff for rate limiting
                        wait_time = (2 ** attempt) * 2  # Longer wait for rate limit
                        logger.info("Retrying after rate limit", wait_seconds=wait_time)
                        await asyncio.sleep(wait_time)
                        continue

                elif status_code == 503:
                    # Model is loading
                    logger.warning(
                        "HF model loading",
                        attempt=attempt + 1,
                    )

                    if attempt < self.max_retries - 1:
                        wait_time = 5 + (2 ** attempt)
                        logger.info("Waiting for model to load", wait_seconds=wait_time)
                        await asyncio.sleep(wait_time)
                        continue

                logger.error(
                    "HF request failed",
                    model=self.model_id,
                    status_code=status_code,
                    error=str(e),
                )
                raise

            except Exception as e:
                last_error = e
                logger.warning(
                    "HF request failed",
                    model=self.model_id,
                    attempt=attempt + 1,
                    error=str(e),
                )

                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info("Retrying after backoff", wait_seconds=wait_time)
                    await asyncio.sleep(wait_time)

        logger.error(
            "HF request failed after all retries",
            model=self.model_id,
            error=str(last_error),
        )
        raise last_error

    async def stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream a text generation response.

        Note: HF Inference API streaming requires special handling.
        This method implements a basic streaming via token-by-token generation.

        Args:
            request: The LLM request.

        Yields:
            Text chunks from the response.
        """
        # HF Inference API doesn't support true streaming in the same way
        # We can either:
        # 1. Use polling with max_new_tokens=1 (very slow)
        # 2. Just yield the complete response as one chunk
        # 3. Use the websocket approach (more complex)

        # For now, we'll generate the full response and yield it
        try:
            logger.info("Starting HF text generation (pseudo-stream)", model=self.model_id)

            # Build full prompt
            if request.system_prompt:
                prompt = f"{request.system_prompt}\n\n{request.prompt}"
            else:
                prompt = request.prompt

            response = await self.complete(
                LLMRequest(
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
            )

            # Yield the full response as one chunk
            # (In production, consider using HF's more advanced streaming)
            yield response.content

        except Exception as e:
            logger.error("HF streaming failed", model=self.model_id, error=str(e))
            raise

    async def embeddings(self, text: str) -> Optional[list[float]]:
        """Generate embeddings for text using feature-extraction task.

        Args:
            text: Text to generate embeddings for.

        Returns:
            List of embedding values, or None if generation failed.
        """
        try:
            logger.info("Generating embeddings", model=self.model_id)

            payload = {"inputs": text}

            response = await self.client.post(
                f"/models/{self.model_id}",
                json=payload,
                headers={"X-Task": "feature-extraction"},
            )

            response.raise_for_status()
            data = response.json()

            # HF feature-extraction returns embeddings as list of floats
            if isinstance(data, list) and len(data) > 0:
                embedding = data[0] if isinstance(data[0], list) else data
            elif isinstance(data, dict) and "embeddings" in data:
                embedding = data["embeddings"]
            else:
                embedding = data

            logger.debug("Embeddings generated", model=self.model_id)
            return embedding

        except Exception as e:
            logger.error("Embeddings generation failed", model=self.model_id, error=str(e))
            return None

    async def get_model_info(self) -> dict[str, Any]:
        """Get information about the model.

        Returns:
            Dictionary with model metadata.
        """
        try:
            response = await self.client.get(f"/model-info/{self.model_id}")
            response.raise_for_status()
            data = response.json()

            logger.debug("Model info retrieved", model=self.model_id)
            return data

        except Exception as e:
            logger.error("Failed to get model info", model=self.model_id, error=str(e))
            return {}

    def count_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a rough approximation (1 token ≈ 4 characters).

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
        logger.debug("HuggingFaceClient closed")

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
            Estimated cost in USD (approximate).
        """
        pricing = self.MODEL_PRICING.get(
            self.model_short_name,
            {"input": 0.001, "output": 0.001},  # Default pricing
        )

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost
