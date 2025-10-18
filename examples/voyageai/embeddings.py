import asyncio
import json
import os

from dotenv import load_dotenv

from callm import (
    RateLimitConfig,
    process_api_requests_from_file,
)
from callm.providers import VoyageAIProvider

load_dotenv()

# Voyage AI rate limits (Basic Tier 1)
# Source: https://docs.voyageai.com/docs/rate-limits
#
# Model-specific limits:
# - voyage-3.5:      8M TPM, 2000 RPM
# - voyage-3.5-lite: 16M TPM, 2000 RPM
# - voyage-3-large:  3M TPM, 2000 RPM
# - voyage-code-3:   3M TPM, 2000 RPM
#
# Higher tiers multiply these limits:
# - Tier 2 (≥$100 paid): 2x
# - Tier 3 (≥$1000 paid): 3x

# Using voyage-3-large limits (adjust based on your model and tier)
TPM = 3_000_000  # 3M tokens per minute for voyage-3-large (Basic Tier 1)
RPM = 2_000  # 2000 requests per minute (Basic Tier 1)

provider = VoyageAIProvider(
    api_key=os.getenv("VOYAGEAI_API_KEY"),
    model="voyage-3-large",
    request_url="https://api.voyageai.com/v1/embeddings",
)

# Create a file with 1000 requests
with open("data/voyageai_embed_requests.jsonl", "w") as f:
    for i in range(1000):
        f.write(
            json.dumps(
                {
                    "input": f"Document {i}: This is a sample text for embeddings generation.",
                    "input_type": "document",  # or "query" for search queries
                    "metadata": {"row_id": i},
                }
            )
            + "\n"
        )


async def main() -> None:
    await process_api_requests_from_file(
        provider=provider,
        requests_file="data/voyageai_embed_requests.jsonl",
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,  # 80% of your limit
            max_tokens_per_minute=TPM * 0.8,  # 80% of your limit
        ),
    )


if __name__ == "__main__":
    asyncio.run(main())
