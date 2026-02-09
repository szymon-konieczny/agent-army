"""Example usage of new model integrations for Code Horde.

This file demonstrates:
1. OpenAI client usage
2. HuggingFace client usage
3. Bielik Polish model usage
4. RLM engine for large documents
5. Enhanced router with multiple providers
"""

import asyncio
import os
from typing import Optional

from src.models.openai_client import OpenAIClient
from src.models.huggingface_client import HuggingFaceClient
from src.models.bielik_client import BielikClient
from src.models.rlm_engine import RLMEngine, RLMConfig
from src.models.router_extensions import EnhancedModelRouter
from src.models.schemas import (
    LLMRequest,
    LLMResponse,
    ToolDefinition,
    ModelTier,
    SensitivityLevel,
)


async def example_openai_basic():
    """Example: Basic OpenAI completion."""
    print("\n=== OpenAI Basic Completion ===")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set, skipping example")
        return

    client = OpenAIClient(api_key=api_key, model="gpt-4o-mini")

    try:
        request = LLMRequest(
            prompt="What is the capital of France? Answer in one word.",
            max_tokens=20,
            temperature=0.2,
        )

        response = await client.complete(request)
        print(f"Response: {response.content}")
        print(f"Tokens: {response.tokens_used}")
        print(f"Cost: ${response.cost_estimate:.6f}")
        print(f"Latency: {response.latency_ms:.1f}ms")

    finally:
        await client.close()


async def example_openai_tool_calling():
    """Example: OpenAI with tool/function calling."""
    print("\n=== OpenAI Function Calling ===")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set, skipping example")
        return

    client = OpenAIClient(api_key=api_key, model="gpt-4o")

    try:
        # Define tools
        tools = [
            ToolDefinition(
                name="get_current_weather",
                description="Get the current weather in a given location",
                input_schema={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            ),
            ToolDefinition(
                name="calculate",
                description="Perform arithmetic calculation",
                input_schema={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression like '2+2'",
                        },
                    },
                    "required": ["expression"],
                },
            ),
        ]

        request = LLMRequest(
            prompt="What's the weather in London and what is 15 * 12?",
            tools=tools,
            max_tokens=500,
        )

        response = await client.complete(request)
        print(f"Response: {response.content}")

        if response.tool_calls:
            print(f"Tool calls made: {len(response.tool_calls)}")
            for tool_call in response.tool_calls:
                print(f"  - {tool_call.tool_name}: {tool_call.arguments}")

    finally:
        await client.close()


async def example_openai_streaming():
    """Example: OpenAI with streaming."""
    print("\n=== OpenAI Streaming ===")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set, skipping example")
        return

    client = OpenAIClient(api_key=api_key, model="gpt-4o-mini")

    try:
        request = LLMRequest(
            prompt="Write a haiku about programming:",
            max_tokens=100,
        )

        print("Streaming response: ", end="", flush=True)
        async for chunk in client.stream(request):
            print(chunk, end="", flush=True)
        print()

    finally:
        await client.close()


async def example_huggingface():
    """Example: HuggingFace text generation."""
    print("\n=== HuggingFace Text Generation ===")

    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        print("HUGGINGFACE_API_KEY not set, skipping example")
        return

    # Note: Free tier has rate limits, so this may need retry handling
    client = HuggingFaceClient(
        api_key=api_key,
        model="mistral-7b",
        timeout_seconds=120,
        max_retries=3,
    )

    try:
        request = LLMRequest(
            prompt="What is machine learning?",
            max_tokens=150,
            temperature=0.7,
        )

        print("Calling HuggingFace API...")
        response = await client.complete(request)
        print(f"Response: {response.content[:200]}...")
        print(f"Tokens: {response.tokens_used}")
        print(f"Cost: ${response.cost_estimate:.6f}")

    except Exception as e:
        print(f"Error (expected on free tier): {e}")

    finally:
        await client.close()


async def example_huggingface_embeddings():
    """Example: HuggingFace embeddings."""
    print("\n=== HuggingFace Embeddings ===")

    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        print("HUGGINGFACE_API_KEY not set, skipping example")
        return

    client = HuggingFaceClient(api_key=api_key, model="mistral-7b")

    try:
        embeddings = await client.embeddings("Hello world")
        if embeddings:
            print(f"Embedding dimension: {len(embeddings)}")
            print(f"First 5 values: {embeddings[:5]}")
        else:
            print("Embeddings not available for this model")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        await client.close()


async def example_bielik_polish():
    """Example: Bielik Polish language support."""
    print("\n=== Bielik Polish Model ===")

    client = BielikClient(
        ollama_base_url="http://localhost:11434",
        variant="bielik-4.5b",
    )

    try:
        # Check if Bielik is available
        is_healthy = await client.health_check()
        if not is_healthy:
            print("Bielik not available (Ollama not running)")
            return

        # Polish language detection
        english_text = "What is the weather?"
        polish_text = "Jaka jest pogoda?"

        lang_en = await client.detect_language(english_text)
        lang_pl = await client.detect_language(polish_text)
        print(f"Detected: '{english_text}' -> {lang_en}")
        print(f"Detected: '{polish_text}' -> {lang_pl}")

        # Polish completion
        request = LLMRequest(
            prompt="Co to jest sztuczna inteligencja?",
            system_prompt="Jesteś pomocnym asystentem.",
            max_tokens=100,
        )

        response = await client.complete(request)
        print(f"Polish response: {response.content[:100]}...")

        # Translation examples
        english = "The weather is nice today"
        polish = await client.translate_from_english(english)
        print(f"Translated: '{english}' -> '{polish}'")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        await client.close()


async def example_rlm_large_document():
    """Example: RLM engine for processing large documents."""
    print("\n=== RLM Engine for Large Documents ===")

    # Create a mock large document
    large_doc = """
    ANNUAL REPORT 2024

    Executive Summary
    Our company achieved record revenue of $500M in 2024, a 25% increase from 2023.
    Net profit reached $75M despite challenges in Q3 related to supply chain disruptions
    and increased operational costs.

    Q1 Results
    - Revenue: $110M (up 20% YoY)
    - Profit: $18M
    - Major achievement: Launched new product line

    Q2 Results
    - Revenue: $125M (up 25% YoY)
    - Profit: $20M
    - Opened 5 new offices in Europe

    Q3 Challenges
    - Revenue: $120M (up 15% YoY, slower growth)
    - Profit: $16M (margin compression)
    - Supply chain disruptions cost us $8M
    - Increased freight costs
    - Staff turnover increased to 12%

    Q4 Recovery
    - Revenue: $145M (up 30% YoY)
    - Profit: $21M
    - Cleared supply chain backlog
    - Successfully implemented cost-reduction measures

    Outlook for 2025
    We expect continued growth with 20-25% revenue increase. Key focus areas:
    - Expansion in Asian markets
    - Product innovation and R&D investment
    - Operational efficiency improvement
    - Talent retention programs
    """ * 20  # Repeat to make it larger

    # Configure RLM
    config = RLMConfig(
        max_recursion_depth=2,
        chunk_size=2048,
        partition_strategy="semantic",
        temperature=0.3,
    )

    rlm = RLMEngine(config=config)

    try:
        # Load document
        context_id = await rlm.load_context(large_doc, metadata={"type": "annual_report"})
        print(f"Document loaded: {context_id}")
        print(f"Size: {len(large_doc)} characters")

        # Search example
        matches = await rlm.search_context(
            context_id=context_id,
            pattern="revenue|profit",
            search_type="regex",
        )
        print(f"Found {len(matches)} relevant sections")

        # For actual querying, we would need a model client:
        # rlm_query = await rlm.recursive_query(
        #     question="What were the main challenges in Q3?",
        #     context_id=context_id,
        #     model_client=openai_client,
        # )
        # print(f"Answer: {rlm_query.answer}")

        # Stats
        stats = rlm.get_stats()
        print(f"RLM Stats: {stats}")

    except Exception as e:
        print(f"Error: {e}")


async def example_enhanced_router():
    """Example: Enhanced router with multiple providers."""
    print("\n=== Enhanced Router ===")

    router = EnhancedModelRouter()

    # Example 1: Simple routing
    fast_request = LLMRequest(
        prompt="Quick question",
        model_preference=ModelTier.FAST,
    )
    provider, model = router.route(fast_request, agent_id="agent-001")
    print(f"FAST tier route: {provider.value}")

    # Example 2: Language detection
    polish_request = LLMRequest(
        prompt="Czym jest uczenie maszynowe?",
    )
    provider, model = router.route_with_language_detection(
        polish_request, agent_id="agent-002"
    )
    print(f"Polish detection route: {provider.value}")
    print(f"Should prefer Bielik: {provider.value == 'bielik'}")

    # Example 3: RLM awareness
    large_request = LLMRequest(
        prompt="Large prompt " * 500,  # ~3KB
    )
    provider, model, use_rlm = router.route_with_rlm_awareness(
        large_request, agent_id="agent-003"
    )
    print(f"Large prompt route: {provider.value}, Use RLM: {use_rlm}")


async def main():
    """Run all examples."""
    print("Code Horde Model Integration Examples")
    print("=" * 50)

    # Basic examples (don't require external services)
    await example_enhanced_router()
    await example_rlm_large_document()

    # Examples requiring API keys
    await example_openai_basic()
    await example_openai_tool_calling()
    await example_openai_streaming()
    await example_huggingface()
    await example_huggingface_embeddings()

    # Examples requiring Ollama
    await example_bielik_polish()

    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
