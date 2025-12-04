import asyncio
import os

from dotenv import load_dotenv

from callm import (
    RateLimitConfig,
    process_requests,
)
from callm.providers import GeminiProvider

load_dotenv()

# Gemini Embedding rate limits (Free tier)
# Source: https://ai.google.dev/gemini-api/docs/rate-limits
# Tier 1:
RPM = 3_000
TPM = 1_000_000

provider = GeminiProvider(
    api_key=os.getenv("GEMINI_API_KEY"),
    model="gemini-embedding-001",
    request_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent",
)

# Create a file with embedding requests
num_requests = 5_000
requests = [
    {
        "content": {
            "parts": [
                {
                    "text": (
                        f"This is document number {i}. "
                        "And it keeps increasing! "
                        "It contains important information."
                    )
                }
            ]
        },
        "metadata": {"row_id": i},
    }
    for i in range(num_requests)
]


async def main() -> None:
    results = await process_requests(
        provider=provider,
        requests=requests,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,
            max_tokens_per_minute=TPM * 0.8,
        ),
        output_path="data/gemini_embed_results.jsonl",
    )

    print(f"Finished in {results.stats.duration_seconds:.2f}s")
    print(f"Success: {results.stats.successful}, Failed: {results.stats.failed}")
    print(f"Total input tokens: {results.stats.total_input_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
