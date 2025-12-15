"""
Provider: Google Gemini
Endpoint: Embed Content API
Description: Google's Gemini models for embedding text

This example shows:
- Gemini API configuration (different request format)
- Embedding text with Gemini
- Rate limits for free vs paid tiers

API Reference: https://ai.google.dev/gemini-api/docs/embeddings
Rate Limits: https://ai.google.dev/gemini-api/docs/rate-limits
"""

import asyncio
import os

from dotenv import load_dotenv

from callm import RateLimitConfig, process_requests
from callm.providers import GeminiProvider

load_dotenv()

# Gemini Embedding rate limits (Free tier)
# Tier 1:
RPM = 3_000
TPM = 1_000_000


async def main() -> None:
    provider = GeminiProvider(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-embedding-001",
        request_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent",
    )

    # Sample documents for embedding
    documents = [
        "Artificial intelligence is transforming industries worldwide.",
        "Machine learning models require large amounts of training data.",
        "Neural networks can recognize patterns in complex datasets.",
        "Deep learning has revolutionized computer vision applications.",
        "Natural language processing enables human-computer interaction.",
    ]

    # Cohere embed format
    requests = [
        {
            "content": {"parts": [{"text": doc}]},
            "task_type": "RETRIEVAL_DOCUMENT",
            "output_dimensionality": 2048,
            "metadata": {"doc_id": i},
        }
        for i, doc in enumerate(documents)
    ]

    results = await process_requests(
        provider=provider,
        requests=requests,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,
            max_tokens_per_minute=TPM * 0.8,
        ),
    )

    print("=" * 60)
    print("Google Gemini Embeddings")
    print("=" * 60)
    print(f"Model: {provider.model}")
    print(f"Processed: {results.stats.successful}/{results.stats.total_requests}")
    print()

    for result in results.successes:
        doc_id = result.metadata["doc_id"]
        embeddings = result.response.get("embedding", {}).get("values", [])
        print(f"Doc {doc_id}: {len(embeddings)} dimensions")


if __name__ == "__main__":
    asyncio.run(main())
