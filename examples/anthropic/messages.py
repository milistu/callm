import asyncio
import json
import os

from dotenv import load_dotenv

from callm import (
    RateLimitConfig,
    process_requests,
)
from callm.providers import AnthropicProvider

load_dotenv()

# Anthropic rate limits (Tier 2)
# Source: https://docs.anthropic.com/en/api/rate-limits
# Tier 2:
RPM = 1_000
TPM = 450_000

provider = AnthropicProvider(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-sonnet-4-5",
)

# Create requests
num_requests = 50
requests_path = "data/anthropic_messages_requests.jsonl"
with open(requests_path, mode="w", encoding="utf-8") as f:
    for i in range(num_requests):
        request = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": f"What is {i} multiplied by 5? Reply with just the number.",
                }
            ],
            "metadata": {"row_id": i},
        }
        f.write(json.dumps(request) + "\n")


async def main() -> None:
    results = await process_requests(
        provider=provider,
        requests=requests_path,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,
            max_tokens_per_minute=TPM * 0.8,
        ),
        output_path="data/anthropic_messages_results.jsonl",
    )

    print(f"Finished in {results.stats.duration_seconds:.2f}s")
    print(f"Success: {results.stats.successful}, Failed: {results.stats.failed}")
    print(f"Total input tokens: {results.stats.total_input_tokens}")
    print(f"Total output tokens: {results.stats.total_output_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
