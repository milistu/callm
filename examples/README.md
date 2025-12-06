# callm Examples

This directory contains examples demonstrating how to use callm for real-world LLM processing tasks.

## Quick Navigation

| Directory | Description | Best For |
|-----------|-------------|----------|
| [`quickstart/`](quickstart/) | Minimal examples to get started | First-time users |
| [`data_extraction/`](data_extraction/) | Extract structured data from text | ETL pipelines, e-commerce |
| [`embeddings/`](embeddings/) | Generate embeddings for search/RAG | Vector databases, search systems |
| [`evaluation/`](evaluation/) | LLM-as-judge and quality scoring | ML pipelines, testing |
| [`synthetic_data/`](synthetic_data/) | Generate training and evaluation data | Dataset creation |
| [`classification/`](classification/) | Sentiment analysis, content moderation | Content analysis |
| [`translation/`](translation/) | Dataset and content translation | Multilingual datasets |
| [`providers/`](providers/) | Provider-specific advanced examples | Provider customization |

## Getting Started

1. **Install callm:**
   ```bash
   pip install callm
   ```

2. **Set up your API keys:**
   ```bash
   # Create a .env file in your project root
   echo "OPENAI_API_KEY=sk-..." >> .env
   echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
   # Add other provider keys as needed
   ```

3. **Run the quickstart:**
   ```bash
   python examples/quickstart/minimal.py
   ```

## Example Structure

Each example follows a consistent pattern:

```python
"""
Example: [Name]
Description: [What this example does]
Use case: [When you would use this]
Provider: [Which provider(s) this works with]
"""

import asyncio
from callm import process_requests, RateLimitConfig
from callm.providers import [Provider]

# 1. Configure the provider
provider = ...

# 2. Prepare your requests (realistic sample data)
requests = ...

# 3. Process with rate limiting
async def main():
    results = await process_requests(...)

if __name__ == "__main__":
    asyncio.run(main())
```

## Rate Limits by Provider

Use these as starting points—adjust based on your API tier:

| Provider | Default RPM | Default TPM | Notes |
|----------|-------------|-------------|-------|
| OpenAI | 5,000 | 2,000,000 | [Check your tier](https://platform.openai.com/docs/guides/rate-limits) |
| Anthropic | 1,000 | 450,000 | [Rate limits docs](https://docs.anthropic.com/en/api/rate-limits) |
| Gemini | 1,000 | 1,000,000 | Free tier: 15 RPM |
| DeepSeek | No limit* | No limit* | *Be reasonable, ~5,000 RPM suggested |
| Cohere | 2,000 | — | No TPM limit |
| Voyage AI | 2,000 | 3,000,000 | [Rate limits](https://docs.voyageai.com/docs/rate-limits) |

**Pro tip:** Use 80% of your limits to leave headroom:
```python
rate_limit = RateLimitConfig(
    max_requests_per_minute=YOUR_RPM * 0.8,
    max_tokens_per_minute=YOUR_TPM * 0.8,
)
```

## Processing Modes

### In-Memory (Small Batches)
Best for < 10,000 requests that fit in memory:
```python
results = await process_requests(
    provider=provider,
    requests=my_list,  # Python list
    rate_limit=rate_limit,
)
# Access results.successes, results.failures
```

### File-Based (Large Batches)
Best for large datasets—streams results to disk:
```python
results = await process_requests(
    provider=provider,
    requests="input.jsonl",      # Read from file
    rate_limit=rate_limit,
    output_path="output.jsonl",  # Write to file
)
# Results written incrementally, low memory usage
```

## Structured Outputs

callm has first-class support for structured outputs using Pydantic:

```python
from pydantic import BaseModel, Field
from callm.utils import pydantic_to_openai_response_format

class ProductInfo(BaseModel):
    brand: str = Field(description="Product brand name")
    category: str = Field(description="Product category")
    price: float = Field(description="Price in USD")

response_format = pydantic_to_openai_response_format(ProductInfo)

requests = [{
    "messages": [{"role": "user", "content": "Extract info from: Nike Air Max, $120"}],
    "response_format": response_format,
}]
```

## Need Help?

- [Main README](../README.md) — Overview and installation
- [Contributing Guide](../CONTRIBUTING.md) — How to contribute
- [Open an issue](https://github.com/milistu/callm/issues) — Report bugs or request features
