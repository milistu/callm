"""
Example: Minimal Quickstart
Description: The simplest possible callm example - process a few requests in memory
Use case: Learning the basics, quick testing
Provider: OpenAI (easily adaptable to others)

This example demonstrates:
- Basic provider setup
- In-memory request processing
- Accessing results
"""

import asyncio
import os

from dotenv import load_dotenv

from callm import RateLimitConfig, process_requests
from callm.providers import OpenAIProvider

load_dotenv()


# Rate limits by tier (tier 2 for selected model https://platform.openai.com/docs/models/gpt-5-nano)
# Note: adjust rate limits based on your tier
RPM = 5_000
TPM = 2_000_000


async def main() -> None:
    # 1. Configure the provider
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-5-nano",
        request_url="https://api.openai.com/v1/responses",
    )

    # 2. Create your requests as a simple list
    requests = [
        {
            "input": "What is the capital of France?",
            "metadata": {"question_id": 1},
        },
        {
            "input": "What is the capital of Germany?",
            "metadata": {"question_id": 2},
        },
        {
            "input": "What is the capital of Italy?",
            "metadata": {"question_id": 3},
        },
    ]

    # 3. Process requests with rate limiting
    results = await process_requests(
        provider=provider,
        requests=requests,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,  # 80% of the limit
            max_tokens_per_minute=TPM * 0.8,  # 80% of the limit
        ),
    )

    # 4. Access your results

    for result in results.successes:
        question_id = result.metadata["question_id"]
        # Response structure depends on the API endpoint
        answer = result.response.get("output", [{}])[1].get("content", [{}])[0].get("text", "")
        print(f"Q{question_id}: {answer[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
