import asyncio
import json
import os

from dotenv import load_dotenv

from callm import (
    RateLimitConfig,
    process_api_requests_from_file,
)
from callm.providers import CohereProvider

load_dotenv()

RPM = 2_000
# No TPM limit
TPM = None

provider = CohereProvider(
    api_key=os.getenv("COHERE_API_KEY"),
    model="embed-v4.0",
    request_url="https://api.cohere.com/v2/embed",
)

# Create a file with 1000 requests
with open("data/cohere_embed_requests.jsonl", "w") as f:
    for i in range(1000):
        f.write(
            json.dumps(
                {
                    "model": "embed-v4.0",
                    "texts": [f"Number is currently: {i}. And it keeps increasing!"],
                    "input_type": "search_document",
                    "output_dimension": 1536,
                    "embedding_types": ["float"],
                    "metadata": {"row_id": i},
                }
            )
            + "\n"
        )


async def main() -> None:
    await process_api_requests_from_file(
        provider=provider,
        requests_file="data/cohere_embed_requests.jsonl",
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,
            max_tokens_per_minute=TPM,
        ),
    )


if __name__ == "__main__":
    asyncio.run(main())
