# Code Horde Multi-Model System Guide

This guide covers the new multi-model system extensions for Code Horde, including OpenAI, Hugging Face, Bielik (Polish), and the Recursive Language Model (RLM) engine.

## Overview

The models system has been extended with:

1. **OpenAI Integration** (`src/models/openai_client.py`) - GPT models with function calling
2. **Hugging Face Integration** (`src/models/huggingface_client.py`) - Open-source model inference
3. **Bielik Polish Model** (`src/models/bielik_client.py`) - Polish language support via Ollama
4. **RLM Engine** (`src/models/rlm_engine.py`) - Recursive Language Model for large contexts
5. **Enhanced Router** (`src/models/router_extensions.py`) - Intelligent routing across all providers

## Architecture

### Provider Hierarchy

```
ModelRouter (enhanced with new providers)
├── Claude (Anthropic)
├── OpenAI (GPT-4o, GPT-4o-mini, etc.)
├── HuggingFace (Open-source models)
├── Bielik (Polish language, via Ollama)
└── Ollama (Local models)
```

### Request Flow

```
User Request
    ↓
EnhancedModelRouter (language detection, RLM check)
    ↓
Provider Selection (routing rules applied)
    ↓
Model Client (async completion/streaming)
    ↓
LLMResponse (with cost, tokens, latency)
```

## Usage Examples

### 1. Using OpenAI Client

```python
from src.models.openai_client import OpenAIClient
from src.models.schemas import LLMRequest, ToolDefinition

# Initialize
client = OpenAIClient(api_key="sk-...", model="gpt-4o-mini")

# Simple completion
request = LLMRequest(
    prompt="What is the capital of France?",
    max_tokens=100,
    temperature=0.7,
)
response = await client.complete(request)
print(response.content)  # "Paris"
print(f"Cost: ${response.cost_estimate:.4f}")
print(f"Tokens: {response.tokens_used}")

# With tool use (function calling)
tools = [
    ToolDefinition(
        name="get_weather",
        description="Get weather for a city",
        input_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    )
]

request_with_tools = LLMRequest(
    prompt="What's the weather in London?",
    tools=tools,
    max_tokens=500,
)
response = await client.complete(request_with_tools)

# Check for tool calls
if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call.tool_name}")
        print(f"Args: {tool_call.arguments}")

# Streaming
async for chunk in client.stream(request):
    print(chunk, end="", flush=True)

await client.close()
```

### 2. Using Hugging Face Client

```python
from src.models.huggingface_client import HuggingFaceClient
from src.models.schemas import LLMRequest

# Initialize with public HF Inference API
client = HuggingFaceClient(
    api_key="hf_...",
    model="mistral-7b",  # or full model ID
)

# Text generation
request = LLMRequest(
    prompt="Write a short poem about Python:",
    max_tokens=256,
    temperature=0.8,
)
response = await client.complete(request)
print(response.content)

# Check model info
info = await client.get_model_info()
print(info)

# Generate embeddings
embeddings = await client.embeddings("Hello world")
print(f"Embedding dimension: {len(embeddings)}")

# Using a dedicated endpoint
client_dedicated = HuggingFaceClient(
    api_key="hf_...",
    dedicated_endpoint="https://my-custom-endpoint.example.com",
)

await client.close()
```

### 3. Using Bielik Polish Model

```python
from src.models.bielik_client import BielikClient
from src.models.schemas import LLMRequest

# Initialize (uses Ollama backend)
client = BielikClient(
    ollama_base_url="http://localhost:11434",
    variant="bielik-11b",  # or "bielik-4.5b"
)

# Polish language detection
lang = await client.detect_language("Jak się masz?")
print(lang)  # "polish"

# Complete in Polish
request = LLMRequest(
    prompt="Jakie jest 2+2?",
    system_prompt="Jesteś pomocnym asystentem.",
    max_tokens=100,
)
response = await client.complete(request)
print(response.content)  # Response in Polish

# Translation assistance
english_text = "Hello, how are you?"
polish = await client.translate_from_english(english_text)
print(polish)  # Polish translation

polish_text = "Cześć, jak się masz?"
english = await client.translate_to_english(polish_text)
print(english)  # English translation

# Health check
is_healthy = await client.health_check()
print(f"Bielik available: {is_healthy}")

await client.close()
```

### 4. Using RLM Engine for Large Contexts

```python
from src.models.rlm_engine import RLMEngine, RLMConfig
from src.models.openai_client import OpenAIClient
from src.models.schemas import LLMRequest

# Configure RLM
config = RLMConfig(
    max_recursion_depth=3,
    chunk_size=4096,
    chunk_overlap=512,
    partition_strategy="semantic",  # or "fixed"
    temperature=0.3,
)

# Initialize RLM engine
rlm = RLMEngine(config=config)

# Load a large document
large_document = open("huge_report.txt").read()  # 500KB+ document
context_id = await rlm.load_context(
    large_document,
    metadata={"source": "annual_report", "year": 2024},
)
print(f"Context loaded: {context_id}")

# Initialize a model client
model_client = OpenAIClient(api_key="sk-...")

# Query the large document recursively
query = await rlm.recursive_query(
    question="What were the top 3 challenges mentioned in Q3?",
    context_id=context_id,
    model_client=model_client,
)

print(f"Answer: {query.answer}")
print(f"Sub-queries: {len(query.subqueries)}")

# Search within the context
matches = await rlm.search_context(
    context_id=context_id,
    pattern="revenue|profit|earnings",
    search_type="regex",
)
print(f"Found {len(matches)} relevant sections")

# Submit final answer
await rlm.submit_answer(
    context_id=context_id,
    answer=query.answer,
    metadata={"answered_by": "rlm_engine", "confidence": 0.95},
)

# Get statistics
stats = rlm.get_stats()
print(stats)
```

### 5. Using Enhanced Router

```python
from src.models.router_extensions import EnhancedModelRouter, RouterExtensions
from src.models.schemas import LLMRequest, SensitivityLevel, ModelTier

# Create enhanced router (replaces standard ModelRouter)
router = EnhancedModelRouter(
    failure_threshold=5,
    circuit_breaker_timeout_seconds=300,
)

# Simple routing (with fallback chain extended)
request = LLMRequest(
    prompt="What is machine learning?",
    model_preference=ModelTier.BALANCED,
    sensitivity=SensitivityLevel.INTERNAL,
)
provider, model = router.route(request, agent_id="agent-001")
print(f"Route: {provider.value} / {model}")

# Routing with language detection
polish_request = LLMRequest(
    prompt="Czym jest uczenie maszynowe?",
    model_preference=ModelTier.POWERFUL,
)
provider, model = router.route_with_language_detection(
    polish_request,
    agent_id="agent-001",
)
# If Polish detected and Bielik available, will route to Bielik
print(f"Route: {provider.value} / {model}")

# Routing with RLM awareness
large_doc_request = LLMRequest(
    prompt="Big document here..." * 1000,  # Very large prompt
    max_tokens=2000,
)
provider, model, should_use_rlm = router.route_with_rlm_awareness(
    large_doc_request,
    agent_id="agent-001",
)
print(f"Route: {provider.value}, Use RLM: {should_use_rlm}")

# Record request results
response = await model_client.complete(request)
router.record_request(
    agent_id="agent-001",
    request=request,
    response=response,
    success=True,
)

# Get statistics
stats = router.get_stats(agent_id="agent-001")
print(stats)
```

### 6. Manual Provider Configuration

```python
from src.models.router_extensions import (
    add_openai_routing_rules,
    add_huggingface_routing_rules,
    extend_router_fallback_chain,
)
from src.models.router import ModelRouter

# Start with base router
router = ModelRouter()

# Add new providers to fallback chain
extend_router_fallback_chain(router)

# Add provider-specific routing rules
add_openai_routing_rules(router, openai_api_key="sk-...")
add_huggingface_routing_rules(router)

# Now router will prefer new providers for certain request types
request = LLMRequest(
    prompt="Fast response needed",
    model_preference=ModelTier.FAST,
)
provider, model = router.route(request)
# Will prefer HuggingFace for FAST tier
```

## Configuration & Environment Variables

### Required Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Hugging Face
export HUGGINGFACE_API_KEY="hf_..."

# Ollama (for Bielik and local models)
export OLLAMA_BASE_URL="http://localhost:11434"

# Anthropic Claude (existing)
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Model Selection

**OpenAI Models:**
- `gpt-4o` - Highest quality, most expensive
- `gpt-4o-mini` - Good quality, cheaper
- `o1` - Reasoning-focused
- `o3-mini` - Fast reasoning

**HuggingFace Models (via Inference API):**
- `mistral-7b` - Fast, capable
- `llama2-7b` / `llama2-13b` - Quality models
- `falcon-7b` - Efficient
- `zephyr-7b` - Instruction-tuned

**Bielik Variants:**
- `bielik-4.5b` - Smaller, faster
- `bielik-11b` - Larger, better quality

## Cost Tracking

All clients track costs based on token usage and model pricing:

```python
response = await client.complete(request)
print(f"Tokens used: {response.tokens_used}")
print(f"Cost estimate: ${response.cost_estimate:.4f}")
print(f"Latency: {response.latency_ms:.2f}ms")
```

### Pricing Tables

**OpenAI:**
- gpt-4o: $2.50 input / $10.00 output (per 1M tokens)
- gpt-4o-mini: $0.15 input / $0.60 output
- o1: $15.00 input / $60.00 output

**HuggingFace:**
- Most models: ~$0.001 input / $0.001 output (approximate)
- Free tier has rate limits (1 req/2 sec)

**Bielik & Ollama:**
- $0.00 (local inference)

## Error Handling

All clients implement exponential backoff retry logic:

```python
from src.models.openai_client import OpenAIClient

client = OpenAIClient(
    api_key="sk-...",
    max_retries=3,  # Try up to 3 times
    timeout_seconds=120,
)

try:
    response = await client.complete(request)
except Exception as e:
    # Handle final failure after retries exhausted
    logger.error(f"Request failed: {e}")
```

### Handled Scenarios

- **Rate limiting (429)**: Exponential backoff with longer wait
- **Server errors (5xx)**: Exponential backoff
- **Timeouts**: Automatic retry
- **Model loading (503)**: Wait for model to load, retry

## Streaming Support

All clients support streaming responses:

```python
# Using OpenAI
async for chunk in openai_client.stream(request):
    print(chunk, end="", flush=True)

# Using HuggingFace (pseudo-streaming)
async for chunk in hf_client.stream(request):
    print(chunk, end="", flush=True)

# Using Bielik
async for chunk in bielik_client.stream(request):
    print(chunk, end="", flush=True)
```

## Tool Calling (Function Calling)

OpenAI client supports parallel tool calls:

```python
tools = [
    ToolDefinition(
        name="search_db",
        description="Search database",
        input_schema={...},
    ),
    ToolDefinition(
        name="fetch_api",
        description="Fetch from API",
        input_schema={...},
    ),
]

request = LLMRequest(
    prompt="Find data and fetch details",
    tools=tools,
)

response = await openai_client.complete(request)

# Handle multiple tool calls
if response.tool_calls:
    for tool_call in response.tool_calls:
        # Execute tool
        result = await execute_tool(tool_call.tool_name, tool_call.arguments)
        # Feed result back to model...
```

## Performance Tuning

### RLM Configuration

```python
config = RLMConfig(
    max_recursion_depth=2,      # Shallow for speed
    chunk_size=2048,            # Smaller for faster processing
    chunk_overlap=256,
    partition_strategy="fixed",  # Faster than semantic
    temperature=0.2,            # Lower for consistency
)
```

### Router Circuit Breaker

```python
router = EnhancedModelRouter(
    failure_threshold=3,                    # Fail faster
    circuit_breaker_timeout_seconds=60,     # Quick recovery
)
```

## Testing

All modules are fully typed with Pydantic v2 models and support async/await:

```python
# Example test
async def test_openai_client():
    client = OpenAIClient(api_key="test-key")
    request = LLMRequest(prompt="Test")

    try:
        response = await client.complete(request)
        assert response.provider == ModelProvider.OPENAI
        assert response.tokens_used > 0
    finally:
        await client.close()

# Run with pytest
pytest -v tests/test_models.py
```

## Architecture Decision Records

### Why RLM?

Large documents (50KB+) exceed typical model context windows. RLM solves this by:
1. Partitioning documents intelligently
2. Creating focused sub-queries
3. Recursively aggregating answers
4. Reducing total tokens needed

### Why Multiple Providers?

- **Cost optimization**: Use cheaper models for simple tasks
- **Resilience**: Fallback when primary provider is down
- **Specialization**: Use best model for each task type
- **Language support**: Bielik for Polish, others for multilingual

### Why Not Just Modify router.py?

The extensions in `router_extensions.py`:
- Maintain backward compatibility
- Allow independent testing
- Support gradual migration
- Enable feature flags

## Troubleshooting

### OpenAI 429 (Rate Limited)

```
Solution: Increase max_retries, increase interval between requests
```

### HuggingFace Model Loading

```
Error: 503 Service Unavailable
Solution: Model is loading, RLM will retry automatically
         Or use dedicated endpoint for faster inference
```

### Bielik Not Available

```
Error: Health check failed
Solution: Ensure Ollama is running with: ollama pull bielik:11b-v3.0-instruct
```

### Large Context OOM

```
Error: Out of memory
Solution: Use RLM engine to partition context
         Or use smaller model variant
         Or increase chunk_size in RLMConfig
```

## Future Enhancements

- [ ] Streaming support for HuggingFace (true server-sent events)
- [ ] Vision model support (GPT-4 Vision, Llava)
- [ ] Audio capabilities (Whisper, text-to-speech)
- [ ] Fine-tuning support for OpenAI
- [ ] Local embeddings with Ollama
- [ ] Semantic routing based on prompt content
- [ ] Cost-aware batch processing
- [ ] Multi-language prompt translation

## References

- [OpenAI API Docs](https://platform.openai.com/docs)
- [HuggingFace Inference API](https://huggingface.co/docs/api-inference)
- [Bielik Model Card](https://huggingface.co/speakleash/Bielik-11B-v2.3)
- [RLM Paper](https://arxiv.org/abs/2512.24601)
- [Ollama Documentation](https://ollama.ai)
