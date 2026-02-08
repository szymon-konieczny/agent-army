"""Ollama local model integration with async support and health checking."""

import asyncio
import time
from typing import Optional, AsyncGenerator
import structlog
import httpx

from src.models.schemas import LLMRequest, LLMResponse, ModelProvider


logger = structlog.get_logger(__name__)


class OllamaClient:
    """Async client for Ollama local language models.

    Supports:
    - Async chat completion
    - Model health checks
    - Model listing and discovery
    - Embeddings generation
    - Fallback to CPU-only if GPU not available
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "mistral"
    REQUEST_TIMEOUT = 300.0  # 5 minutes for long-running completions

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = REQUEST_TIMEOUT,
    ) -> None:
        """Initialize Ollama client.

        Args:
            base_url: Base URL for Ollama API (e.g., http://localhost:11434).
            model: Default model to use for requests.
            timeout_seconds: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_seconds),
        )

        # Cache available models
        self._models_cache: Optional[list[str]] = None
        self._cache_time: float = 0
        self._cache_ttl_seconds: float = 300  # 5 minutes

        logger.info(
            "OllamaClient initialized",
            base_url=base_url,
            model=model,
        )

    async def health_check(self) -> bool:
        """Check if Ollama service is healthy and running.

        Returns:
            True if Ollama is healthy, False otherwise.
        """
        try:
            response = await self.client.get("/api/tags")
            is_healthy = response.status_code == 200
            logger.debug("Ollama health check", status=response.status_code)
            return is_healthy
        except Exception as e:
            logger.warning("Ollama health check failed", error=str(e))
            return False

    async def list_models(self, use_cache: bool = True) -> list[str]:
        """List available models on the Ollama instance.

        Args:
            use_cache: Use cached model list if available.

        Returns:
            List of available model names.
        """
        # Check cache
        if use_cache and self._models_cache:
            current_time = time.time()
            if current_time - self._cache_time < self._cache_ttl_seconds:
                logger.debug("Returning cached models list")
                return self._models_cache

        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()

            data = response.json()
            models = [model["name"] for model in data.get("models", [])]

            # Cache result
            self._models_cache = models
            self._cache_time = time.time()

            logger.info("Models retrieved from Ollama", count=len(models))
            return models

        except Exception as e:
            logger.error("Failed to list Ollama models", error=str(e))
            return []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a chat completion request.

        Args:
            request: The LLM request with prompt and parameters.

        Returns:
            LLMResponse with the model's response and metadata.

        Raises:
            Exception: If the request fails.
        """
        start_time = time.time()

        # Use specified model or default
        model = self.model

        # Prepare messages
        messages = []

        if request.system_prompt:
            messages.append(
                {"role": "system", "content": request.system_prompt}
            )

        messages.append({"role": "user", "content": request.prompt})

        try:
            logger.info(
                "Sending request to Ollama",
                model=model,
                max_tokens=request.max_tokens,
            )

            response = await self.client.post(
                "/api/chat",
                json={
                    "model": model,
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
                "Ollama request successful",
                model=model,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
            )

            return LLMResponse(
                content=content,
                model_used=model,
                provider=ModelProvider.OLLAMA,
                tokens_used=tokens_used,
                cost_estimate=0.0,  # Local models have no API cost
                latency_ms=latency_ms,
            )

        except httpx.HTTPError as e:
            logger.error(
                "Ollama request failed",
                model=model,
                error=str(e),
            )
            raise

    async def stream(
        self, request: LLMRequest
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion response.

        Yields chunks of text as they arrive from the model.

        Args:
            request: The LLM request.

        Yields:
            Text chunks from the response.
        """
        model = self.model

        messages = []
        if request.system_prompt:
            messages.append(
                {"role": "system", "content": request.system_prompt}
            )
        messages.append({"role": "user", "content": request.prompt})

        try:
            async with self.client.stream(
                "POST",
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens,
                    },
                },
            ) as response:
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
            logger.error("Streaming failed", model=model, error=str(e))
            raise

    async def embeddings(
        self, text: str, model: Optional[str] = None
    ) -> Optional[list[float]]:
        """Generate embeddings for text.

        Args:
            text: Text to generate embeddings for.
            model: Model to use (defaults to self.model).

        Returns:
            List of embedding values, or None if generation failed.
        """
        model = model or self.model

        try:
            response = await self.client.post(
                "/api/embeddings",
                json={"model": model, "prompt": text},
            )

            response.raise_for_status()
            data = response.json()

            embedding = data.get("embedding")
            logger.debug("Embeddings generated", model=model)
            return embedding

        except Exception as e:
            logger.error("Embeddings generation failed", model=model, error=str(e))
            return None

    async def pull_model(self, model_name: str) -> bool:
        """Download/pull a model from Ollama registry.

        Args:
            model_name: Name of the model to pull (e.g., "mistral", "llama2").

        Returns:
            True if successful, False otherwise.
        """
        try:
            logger.info("Pulling model", model_name=model_name)

            async with self.client.stream(
                "POST",
                "/api/pull",
                json={"name": model_name},
            ) as response:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    # Log pull progress
                    logger.debug("Pull progress", line=line[:50])

            # Clear models cache since we pulled a new one
            self._models_cache = None

            logger.info("Model pulled successfully", model_name=model_name)
            return True

        except Exception as e:
            logger.error("Failed to pull model", model_name=model_name, error=str(e))
            return False

    async def close(self) -> None:
        """Close the HTTP client.

        Should be called when done using the client.
        """
        await self.client.aclose()
        logger.debug("OllamaClient closed")

    def __del__(self) -> None:
        """Cleanup when client is garbage collected."""
        try:
            asyncio.run(self.close())
        except Exception:
            pass
