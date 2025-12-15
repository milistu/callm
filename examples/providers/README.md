# Provider-Specific Examples

This directory contains examples specific to each LLM provider. Use these when you need:
- Provider-specific features or configurations
- Reference implementations for each provider
- Examples of different API endpoints per provider

For most use cases, start with the [use-case examples](../) (data_extraction, embeddings, etc.) which are provider-agnostic and focus on solving real problems.

## Providers

| Provider | Directory | Endpoints |
|----------|-----------|-----------|
| OpenAI | [`openai/`](openai/) | Chat Completions, Responses API, Embeddings |
| Anthropic | [`anthropic/`](anthropic/) | Messages API |
| Google Gemini | [`gemini/`](gemini/) | Generate Content, Embeddings |
| DeepSeek | [`deepseek/`](deepseek/) | Chat Completions |
| Cohere | [`cohere/`](cohere/) | Embed API |
| Voyage AI | [`voyageai/`](voyageai/) | Embeddings |

## When to Use Provider-Specific Examples

1. **Learning a new provider** - See how to configure and use each provider
2. **Specific API features** - Some providers have unique features not covered in use-case examples
3. **Structured outputs** - Provider-specific structured output configurations
4. **Debugging** - Simple examples to test provider connectivity

## Common Patterns

All provider examples follow the same structure:

```python
import asyncio
from callm import process_requests, RateLimitConfig
from callm.providers import [Provider]Provider

# 1. Configure provider with API key and endpoint
provider = [Provider]Provider(
    api_key=os.getenv("[PROVIDER]_API_KEY"),
    model="model-name",
    request_url="https://api.provider.com/v1/endpoint",
)

# 2. Create requests
requests = [...]

# 3. Process with rate limiting
results = await process_requests(
    provider=provider,
    requests=requests,
    rate_limit=RateLimitConfig(...),
)
```
