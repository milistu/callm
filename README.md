<p align="center">
  <h1 align="center">callm</h1>
  <p align="center"><strong>Keep callm and process thousands of requests without (rate) limits</strong></p>
</p>

<p align="center">
  <a href="https://pypi.org/project/callm-py/"><img src="https://img.shields.io/pypi/v/callm-py?color=blue&label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/callm-py/"><img src="https://img.shields.io/pypi/pyversions/callm-py" alt="Python versions"></a>
  <a href="https://github.com/milistu/callm/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>


<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#supported-providers">Providers</a> •
  <a href="#examples">Examples</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## 😌 Why callm?

Building LLM-powered applications often means processing **thousands of API requests**. You've probably experienced:

| Problem | Without callm | With callm |
|---------|---------------|------------|
| **Rate limit errors** | Constant 429 errors, manual sleep/retry | Automatic RPM & TPM throttling |
| **Retry logic** | Write custom backoff for each project | Built-in exponential backoff with jitter |
| **Token tracking** | No visibility into usage | Real-time token consumption metrics |
| **Boilerplate code** | Copy-paste the same async code everywhere | One function call, any provider |
| **Waiting for batch APIs** | Provider batch APIs take up to 24 hours | Results in minutes, not hours |
| **Multiple SDKs** | Install openai, anthropic, cohere, ... | One library, all providers |

**Stop rewriting the same parallel processing code.** callm handles the infrastructure so you can focus on your application.

> *Testing multiple providers? Just swap the provider class—no new dependencies, no code changes. Find what works best for your use case.*

## Installation

```bash
pip install callm-py
```

**From source:**
```bash
git clone https://github.com/milistu/callm.git
cd callm
pip install -e .
```

## Quick Start

Process 1,000 product descriptions to extract structured data—in under a minute:

```python
import asyncio
from callm import process_requests, RateLimitConfig
from callm.providers import OpenAIProvider

# Configure your provider
provider = OpenAIProvider(
    api_key="sk-...",
    model="gpt-5-mini",
    request_url="https://api.openai.com/v1/responses",
)

# Your data processing requests
products = [
    {"id": 1, "description": "Nike Air Max 90 - Classic sneakers in white/black, size 10"},
    {"id": 2, "description": "Sony WH-1000XM5 Wireless Headphones - Noise cancelling, 30hr battery"},
    # ... thousands more
]

requests = [
    {
        "input": f"Extract brand, category, and key features from: {p['description']}",
        "metadata": {"product_id": p["id"]},
    }
    for p in products
]

async def main():
    results = await process_requests(
        provider=provider,
        requests=requests,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=5_000,    # Stay under your tier limit
            max_tokens_per_minute=2_000_000,
        ),
    )

    print(f"Processed {results.stats.successful} requests in {results.stats.duration_seconds:.1f}s")
    print(f"Tokens used: {results.stats.total_input_tokens + results.stats.total_output_tokens:,}")

    # Access results
    for result in results.successes:
        print(f"Product {result.metadata['product_id']}: {result.response}")

asyncio.run(main())
```

## Features

- **Precise Rate Limiting** — Token buckets for RPM and TPM, respects provider limits
- **Smart Retries** — Exponential backoff with jitter, automatic 429/5xx handling
- **Usage Tracking** — Metrics for input tokens and output tokens
- **Flexible I/O** — Process from Python lists or JSONL files, output to memory or disk
- **Structured Outputs** — Support for Pydantic models and JSON schemas
- **Provider Agnostic** — Same API across OpenAI, Anthropic, Gemini, DeepSeek, and more

## Supported Providers

<table>
  <tr>
    <td align="center" width="150">
      <img src="https://us1.discourse-cdn.com/openai1/original/4X/3/2/1/321a1ba297482d3d4060d114860de1aa5610f8a9.png" width="60" alt="OpenAI"><br>
      <strong>OpenAI</strong><br>
      <sub>Chat, Responses, Embeddings</sub>
    </td>
    <td align="center" width="150">
      <img src="https://cdn.worldvectorlogo.com/logos/anthropic-1.svg" width="60" alt="Anthropic"><br>
      <strong>Anthropic</strong><br>
      <sub>Messages API</sub>
    </td>
    <td align="center" width="150">
      <img src="https://registry.npmmirror.com/@lobehub/icons-static-png/1.75.0/files/light/gemini-color.png" width="60" alt="Google Gemini"><br>
      <strong>Gemini</strong><br>
      <sub>Generate, Embeddings</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="150">
      <img src="https://registry.npmmirror.com/@lobehub/icons-static-png/1.75.0/files/dark/deepseek-color.png" width="60" alt="DeepSeek"><br>
      <strong>DeepSeek</strong><br>
      <sub>Chat Completions</sub>
    </td>
    <td align="center" width="150">
      <img src="https://registry.npmmirror.com/@lobehub/icons-static-png/1.75.0/files/dark/cohere-color.png" width="60" alt="Cohere"><br>
      <strong>Cohere</strong><br>
      <sub>Embed API</sub>
    </td>
    <td align="center" width="150">
      <img src="https://i0.wp.com/blog.voyageai.com/wp-content/uploads/2023/10/logo.png?quality=80&ssl=1" width="60" alt="Voyage AI"><br>
      <strong>Voyage AI</strong><br>
      <sub>Embeddings</sub>
    </td>
  </tr>
</table>

## Examples

Explore real-world use cases in the [`examples/`](examples/) directory:

| Use Case | Description |
|----------|-------------|
| [**Data Extraction**](examples/data_extraction/) | Extract structured data from product listings, invoices |
| [**Embeddings**](examples/embeddings/) | Generate embeddings for RAG and semantic search |
| [**Evaluation**](examples/evaluation/) | Multi-judge consensus evaluation |
| [**Synthetic Data**](examples/synthetic_data/) | Generate training data and evaluation sets |
| [**Classification**](examples/classification/) | Sentiment analysis, content moderation |
| [**Translation**](examples/translation/) | Dataset translation for multilingual evaluation |

### Processing Modes

callm supports four processing modes depending on your input source and output destination:

| Input | Output | Best For |
|-------|--------|----------|
| Python list | In-memory | Small batches, interactive use |
| Python list | JSONL file | Medium batches, need persistence |
| JSONL file | JSONL file | Large batches, low memory |
| JSONL file | In-memory | Loading saved requests, testing |

```python
# 1. List → Memory (small batches)
results = await process_requests(
    provider=provider,
    requests=my_list,
    rate_limit=rate_limit,
)
# Access: results.successes, results.failures

# 2. List → File (persist results)
results = await process_requests(
    provider=provider,
    requests=my_list,
    rate_limit=rate_limit,
    output_path="results.jsonl",
)

# 3. File → File (large batches, low memory)
results = await process_requests(
    provider=provider,
    requests="input.jsonl",
    rate_limit=rate_limit,
    output_path="results.jsonl",
)

# 4. File → Memory (reload saved requests)
results = await process_requests(
    provider=provider,
    requests="input.jsonl",
    rate_limit=rate_limit,
)
```

### Configuration

```python
from callm import RateLimitConfig, RetryConfig

# Rate limiting (required)
rate_limit = RateLimitConfig(
    max_requests_per_minute=1000,
    max_tokens_per_minute=100_000,
)

# Retry behavior (optional, sensible defaults)
retry = RetryConfig(
    max_attempts=5,
    base_delay_seconds=0.5,
    max_delay_seconds=15.0,
    jitter=0.1,
)

results = await process_requests(
    provider=provider,
    requests=requests,
    rate_limit=rate_limit,
    retry=retry,
)
```

## API Reference

### `process_requests()`

Main function for parallel API request processing.

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider` | `BaseProvider` | Provider instance (OpenAI, Anthropic, etc.) |
| `requests` | `list[dict] \| str` | List of request dicts or path to JSONL file |
| `rate_limit` | `RateLimitConfig` | RPM and TPM limits |
| `retry` | `RetryConfig` | Optional retry configuration |
| `output_path` | `str` | Optional path for output JSONL (enables streaming) |
| `errors_path` | `str` | Optional path for error JSONL |
| `logging_level` | `int` | Logging verbosity (default: 20/INFO) |

**Returns:** `ProcessingResults` with `successes`, `failures`, and `stats`.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Setup development environment
git clone https://github.com/milistu/callm.git
cd callm
uv sync --dev
uv run pre-commit install

# Run tests
uv run nox
```

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with 🧡 for engineers who process data at scale</sub>
</p>
