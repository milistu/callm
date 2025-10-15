# callm

**Keep CaLLM and make requests FAST 💨**

A Python library for high-throughput LLM API requests with automatic rate limiting, retries, and usage tracking. Process thousands of requests efficiently while respecting provider limits.

## Features

- **Rate Limiting** – Precise RPM (requests per minute) and TPM (tokens per minute) throttling
- **Automatic Retry** – Exponential backoff with jitter for failed requests
- **Usage Tracking** – Token consumption metrics for input, output, and total usage
- **Provider-Agnostic** – Clean abstraction layer for multiple LLM providers
- **JSONL Batch Processing** – Efficient processing of large request batches

## Installation

**Local Development:**
```bash
git clone <repository-url>
cd callm
pip install -e .
```

**PyPI:** Coming soon

## Quick Start

```python
import asyncio
from callm import process_api_requests_from_file, RateLimitConfig
from callm.providers import OpenAIProvider

# Configure provider
provider = OpenAIProvider(
    api_key="sk-...",
    model="gpt-4.1-nano",
    request_url="https://api.openai.com/v1/responses"
)

# Process requests with rate limiting
async def main():
    await process_api_requests_from_file(
        provider=provider,
        requests_file="requests.jsonl",
        rate_limit=RateLimitConfig(
            max_requests_per_minute=100,
            max_tokens_per_minute=50_000
        )
    )

asyncio.run(main())
```

## Usage

### JSONL Request Format

Each line in your `requests.jsonl` file should be a valid JSON object with your request parameters:

```jsonl
{"model": "gpt-4.1-nano-2025-04-14", "input": "Hello!", "metadata": {"row_id": 0}}
{"model": "gpt-4.1-nano-2025-04-14", "input": "Tell me a joke", "metadata": {"row_id": 1}}
```
```jsonl
{"model": "text-embedding-3-small", "input": "Text to embed 1", "metadata": {"row_id": 0}}
{"model": "text-embedding-3-small", "input": "Text to embed 2", "metadata": {"row_id": 1}}
```

### Configuration Options

**RateLimitConfig** – Control request throughput:
```python
RateLimitConfig(
    max_requests_per_minute=100,  # RPM limit
    max_tokens_per_minute=50_000   # TPM limit
)
```

**RetryConfig** – Customize retry behavior:
```python
RetryConfig(
    max_attempts=5,          # Maximum retry attempts
    initial_delay=1.0,       # Initial backoff delay (seconds)
    max_delay=60.0,          # Maximum backoff delay (seconds)
    backoff_factor=2.0,      # Exponential backoff multiplier
    jitter=True              # Add randomness to delays
)
```

**FilesConfig** – Specify output locations:
```python
FilesConfig(
    save_file="results.jsonl",  # Successful responses
    error_file="errors.jsonl"   # Failed requests
)
```

## Supported Providers

### OpenAI
- **Chat Completions**
- **Embeddings**
- **Responses**

**More providers coming soon** – Anthropic, Voyage AI, DeepSeek, and more

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! This library is designed to be extensible.
