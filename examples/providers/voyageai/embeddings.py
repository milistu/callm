"""
Provider: Voyage AI
Endpoint: Embeddings API
Description: Voyage AI embeddings optimized for retrieval and search

This example shows:
- Voyage AI API configuration
- Document vs query embedding types
- Retrieval-optimized embeddings

API Reference: https://docs.voyageai.com/docs/embeddings
Rate Limits: https://docs.voyageai.com/docs/rate-limits
"""

import asyncio
import os

from dotenv import load_dotenv

from callm import RateLimitConfig, process_requests
from callm.providers import VoyageAIProvider

load_dotenv()

# Voyage AI rate limits (Basic Tier 1)
RPM = 2_000
TPM = 16_000_000


async def main() -> None:
    provider = VoyageAIProvider(
        api_key=os.getenv("VOYAGEAI_API_KEY"),
        model="voyage-3.5-lite",
        request_url="https://api.voyageai.com/v1/embeddings",
    )

    # Sample documents and query
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning algorithms can identify patterns in data.",
        "Python is widely used for data science and AI development.",
        "Cloud computing enables scalable application deployment.",
    ]

    query = "What programming languages are used for artificial intelligence?"

    # Embed documents (use input_type: document)
    doc_requests = [
        {
            "model": "voyage-3",
            "input": doc,
            "input_type": "document",
            "metadata": {"type": "document", "doc_id": i},
        }
        for i, doc in enumerate(documents)
    ]

    # Embed query (use input_type: query)
    query_request = {
        "model": "voyage-3",
        "input": query,
        "input_type": "query",
        "metadata": {"type": "query"},
    }

    all_requests = doc_requests + [query_request]

    results = await process_requests(
        provider=provider,
        requests=all_requests,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,
            max_tokens_per_minute=TPM * 0.8,
        ),
    )

    print("=" * 60)
    print("Voyage AI Embeddings")
    print("=" * 60)
    print(f"Model: {provider.model}")
    print(f"Processed: {results.stats.successful}/{results.stats.total_requests}")
    print()

    doc_embeddings = []
    query_embedding = None

    for result in results.successes:
        embedding = result.response["data"][0]["embedding"]
        if result.metadata["type"] == "document":
            doc_embeddings.append((result.metadata["doc_id"], embedding))
            print(f"Document {result.metadata['doc_id']}: {len(embedding)} dims")
        else:
            query_embedding = embedding
            print(f"Query: {len(embedding)} dims")

    # Simple similarity demo (cosine similarity)
    if query_embedding and doc_embeddings:
        print("\nSimilarity to query:")
        for doc_id, doc_emb in doc_embeddings:
            # Dot product (embeddings are normalized)
            similarity = sum(a * b for a, b in zip(query_embedding, doc_emb, strict=True))
            print(f"  Doc {doc_id}: {similarity:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
