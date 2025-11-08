import asyncio
import os

from dotenv import load_dotenv

from callm import (
    RateLimitConfig,
    process_api_requests,
)
from callm.providers import OpenAIProvider

load_dotenv()

# Find what is your Tier and copy the values from model page: https://platform.openai.com/docs/models/gpt-4.1-nano
# Tier 2:
TPM = 2_000_000
RPM = 5_000

provider = OpenAIProvider(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4.1-nano-2025-04-14",
    request_url="https://api.openai.com/v1/responses",
)

# Create a list with 1000 requests
requests = [
    {
        "model": "gpt-4.1-nano-2025-04-14",
        "input": (
            f"This is an example number {i}: "
            f"Multiple this number {i} by 3 and return the result"
        ),
        "metadata": {"row_id": i},
    }
    for i in range(1000)
]


async def main() -> None:
    results = await process_api_requests(
        provider=provider,
        requests=requests,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,  # 80% of your limit
            max_tokens_per_minute=TPM * 0.8,  # 80% of your limit
        ),
    )

    # Print summary
    print("\nProcessing complete!")
    print(f"Successful: {results.stats.successful}")
    print(f"Failed: {results.stats.failed}")


if __name__ == "__main__":
    asyncio.run(main())
