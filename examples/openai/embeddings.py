import asyncio
import json
import os

from dotenv import load_dotenv

from callm import (
    FilesConfig,
    RateLimitConfig,
    RetryConfig,
    process_api_requests_from_file,
)
from callm.providers import OpenAIProvider

load_dotenv()

# Find what is your Tier and copy the values from model page: https://platform.openai.com/docs/models/text-embedding-3-small
# Tier 1:
TPM = 1_000_000
RPM = 3_000

provider = OpenAIProvider(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small",
    request_url="https://api.openai.com/v1/embeddings",
)

# Create a file with 1000 requests
with open("data/openai_embeddings_requests.jsonl", "w") as f:
    for i in range(1000):
        f.write(
            json.dumps(
                {
                    "model": "text-embedding-3-small",
                    "input": f"Number is currently: {i}. And it keeps increasing!",
                    "metadata": {"row_id": i},
                }
            )
            + "\n"
        )


async def main() -> None:
    await process_api_requests_from_file(
        provider=provider,
        requests_file="data/openai_embeddings_requests.jsonl",
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,  # 80% of your limit
            max_tokens_per_minute=TPM * 0.8,  # 80% of your limit
        ),
        retry=RetryConfig(),
        files=FilesConfig(
            save_file="data/openai_embeddings_results.jsonl",
            error_file="data/openai_embeddings_errors.jsonl",
        ),
        # logging_level=10,
    )


if __name__ == "__main__":
    asyncio.run(main())
