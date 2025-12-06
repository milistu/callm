"""
Example: Document Embeddings for RAG
Description: Generate embeddings for thousands of documents to build a RAG knowledge base
Use case: Building vector databases, semantic search, retrieval-augmented generation
Provider: Cohere embed-v4.0

This example demonstrates:
- Processing large document collections efficiently
- Generating embeddings at scale with rate limiting
- Preparing data for vector database ingestion
- File-based I/O for memory efficiency

Real-world application:
- You have 50,000 documentation pages, support tickets, or articles
- You need to build a searchable knowledge base for RAG
- Embedding APIs have rate limits (TPM), and you need to stay under them
- Results saved for later ingestion into Pinecone/Weaviate/Qdrant
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from callm import RateLimitConfig, process_requests
from callm.providers import CohereProvider

load_dotenv()


# Sample documents (replace with your actual document corpus)
SAMPLE_DOCUMENTS = [
    {
        "doc_id": "doc_001",
        "source": "user_guide",
        "title": "Getting Started with callm",
        "chunk_id": 0,
        "text": """
        callm is a Python library for processing thousands of LLM API requests
        efficiently. It handles rate limiting automatically, so you never hit 429 errors.
        The library supports multiple providers including OpenAI, Anthropic, Cohere, and Gemini.
        Installation is simple: pip install callm. The main function is process_requests()
        which accepts a list of requests or a JSONL file path.
        """,
    },
    {
        "doc_id": "doc_001",
        "source": "user_guide",
        "title": "Getting Started with callm",
        "chunk_id": 1,
        "text": """
        Rate limiting in callm uses a token bucket algorithm. You configure
        max_requests_per_minute (RPM) and max_tokens_per_minute (TPM). The library
        automatically throttles requests to stay within these limits. For best results,
        set these to 80% of your actual API tier limits to leave headroom.
        """,
    },
    {
        "doc_id": "doc_002",
        "source": "api_reference",
        "title": "RateLimitConfig",
        "chunk_id": 0,
        "text": """
        RateLimitConfig is a dataclass that configures rate limiting behavior.
        It has two required parameters: max_requests_per_minute (float) and
        max_tokens_per_minute (float or None). Set max_tokens_per_minute to None for
        APIs that don't have token limits, like some embedding endpoints.
        """,
    },
    {
        "doc_id": "doc_003",
        "source": "faq",
        "title": "Frequently Asked Questions",
        "chunk_id": 0,
        "text": """
        Q: How do I handle API errors? A: callm automatically retries failed
        requests with exponential backoff. By default, it retries up to 5 times with
        increasing delays. You can customize this with RetryConfig. Rate limit errors
        (429) trigger a 15-second pause before continuing.
        """,
    },
    {
        "doc_id": "doc_003",
        "source": "faq",
        "title": "Frequently Asked Questions",
        "chunk_id": 1,
        "text": """
        Q: Can I use callm with my own provider? A: Yes! callm has a
        BaseProvider class that you can extend. Implement estimate_input_tokens(),
        parse_error(), and extract_usage() methods. See the providers/ directory for
        examples of existing implementations.
        """,
    },
    {
        "doc_id": "doc_004",
        "source": "tutorial",
        "title": "Building a RAG System",
        "chunk_id": 0,
        "text": """
        This tutorial shows how to build a retrieval-augmented generation
        system using callm. First, prepare your documents by chunking them into
        500-1000 token segments. Then use callm to generate embeddings in parallel.
        Finally, store the embeddings in a vector database like Pinecone, Weaviate, or Qdrant.
        """,
    },
    {
        "doc_id": "doc_004",
        "source": "tutorial",
        "title": "Building a RAG System",
        "chunk_id": 1,
        "text": """
        For the retrieval step, embed the user's query using the same model.
        Search your vector database for the most similar documents. Pass these as context
        to your LLM along with the query. callm can help here too - use it to process
        many RAG queries in parallel while respecting rate limits.
        """,
    },
    {
        "doc_id": "doc_005",
        "source": "changelog",
        "title": "Version 0.1.0 Release Notes",
        "chunk_id": 0,
        "text": """
        Version 0.1.0 introduces support for six providers: OpenAI, Anthropic,
        Gemini, DeepSeek, Cohere, and Voyage AI. Key features include token bucket rate
        limiting, automatic retry with jitter, and both in-memory and file-based
        processing modes. The library requires Python 3.10 or higher.
        """,
    },
]

# Production rate limit
RPM = 2_000
# No TPM limit
TPM = None


async def main() -> None:
    # Configure Cohere embeddings provider
    provider = CohereProvider(
        api_key=os.getenv("COHERE_API_KEY"),
        model="embed-v4.0",
        request_url="https://api.cohere.com/v2/embed",
    )

    # Build embedding requests
    requests = []
    for doc in SAMPLE_DOCUMENTS:
        requests.append(
            {
                "model": "embed-v4.0",
                "texts": [doc["title"] + "\n" + doc["text"]],
                "input_type": "search_document",
                "output_dimension": 1536,
                "embedding_types": ["float"],
                "metadata": {
                    "doc_id": doc["doc_id"],
                    "chunk_id": doc["chunk_id"],
                    "source": doc["source"],
                },
            }
        )

    output_path = Path("data/cohere_embeddings.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process embedding requests
    results = await process_requests(
        provider=provider,
        requests=requests,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,  # 80% of 2000
            max_tokens_per_minute=TPM,  # None TPM
        ),
        output_path=str(output_path),
        logging_level=20,
    )

    print(f"Number of successful requests: {results.successes}")
    print(
        "Embeddings are saved to file and memory is not used. "
        "That is why we do not return embeddings when output_path is provided"
    )


if __name__ == "__main__":
    asyncio.run(main())
