"""
Provider: Cohere
Endpoint: Embed API v2
Description: Cohere's embedding models for semantic search and classification

This example shows:
- Cohere Embed API v2 configuration
- Different input types (search_document vs search_query)
- Output dimension customization

API Reference: https://docs.cohere.com/reference/embed
"""

import asyncio
import os

from dotenv import load_dotenv

from callm import RateLimitConfig, process_requests
from callm.providers import CohereProvider

load_dotenv()

# Cohere rate limits
RPM = 2_000
TPM = None  # No TPM limit


async def main() -> None:
    provider = CohereProvider(
        api_key=os.getenv("COHERE_API_KEY"),
        model="embed-v4.0",
        request_url="https://api.cohere.com/v2/embed",
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
            "model": "embed-v4.0",
            "texts": [doc],  # Cohere expects list of texts
            "input_type": "search_document",  # or "search_query" for queries
            "output_dimension": 1024,  # Customize embedding size
            "embedding_types": ["float"],
            "metadata": {"doc_id": i},
        }
        for i, doc in enumerate(documents)
    ]

    results = await process_requests(
        provider=provider,
        requests=requests,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,
            max_tokens_per_minute=TPM,  # No TPM limit
        ),
    )

    print("=" * 60)
    print("Cohere Embed API v2")
    print("=" * 60)
    print(f"Model: {provider.model}")
    print(f"Processed: {results.stats.successful}/{results.stats.total_requests}")
    print()

    for result in results.successes:
        doc_id = result.metadata["doc_id"]
        embeddings = result.response.get("embeddings", {})
        float_emb = embeddings.get("float", [[]])[0]
        print(f"Doc {doc_id}: {len(float_emb)} dimensions")


if __name__ == "__main__":
    asyncio.run(main())
