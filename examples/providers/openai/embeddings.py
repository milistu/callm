"""
Provider: OpenAI
Endpoint: Embeddings API
Description: Generate text embeddings for semantic search, clustering, and RAG

This example shows:
- Embedding generation with text-embedding-3-small/large
- Batch processing for embedding generation
- Output formatting for vector databases

API Reference: https://platform.openai.com/docs/api-reference/embeddings
Rate Limits: https://platform.openai.com/docs/guides/rate-limits
"""

import asyncio
import os

from dotenv import load_dotenv

from callm import RateLimitConfig, process_requests
from callm.providers import OpenAIProvider

load_dotenv()

# Embedding model limits (Tier 1)
RPM = 3_000
TPM = 1_000_000


async def main() -> None:
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small",
        request_url="https://api.openai.com/v1/embeddings",
    )

    # Sample documents to embed
    documents = [
        {"id": "doc1", "text": "Machine learning is a subset of artificial intelligence."},
        {"id": "doc2", "text": "Python is a popular programming language for data science."},
        {"id": "doc3", "text": "Neural networks are inspired by biological brain structures."},
        {"id": "doc4", "text": "Deep learning enables complex pattern recognition in data."},
        {"id": "doc5", "text": "Natural language processing helps computers understand text."},
    ]

    requests = [
        {
            "model": "text-embedding-3-small",
            "input": doc["text"],
            "metadata": {"doc_id": doc["id"]},
        }
        for doc in documents
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
    print("OpenAI Embeddings API")
    print("=" * 60)
    print(f"Model: {provider.model}")
    print(f"Processed: {results.stats.successful}/{results.stats.total_requests}")
    print(f"Tokens used: {results.stats.total_input_tokens:,}")
    print()

    # Show embedding info
    for result in results.successes:
        doc_id = result.metadata["doc_id"]
        embedding = result.response["data"][0]["embedding"]
        print(f"{doc_id}: {len(embedding)} dimensions, first 3: {embedding[:3]}")

    # Export for vector DB
    print("\nVector DB format:")
    records = []
    for result in results.successes:
        records.append(
            {
                "id": result.metadata["doc_id"],
                "values": result.response["data"][0]["embedding"],
            }
        )
    print(f"  {len(records)} records ready for upsert")


if __name__ == "__main__":
    asyncio.run(main())
