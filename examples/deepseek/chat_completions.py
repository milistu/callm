import asyncio
import json
import os

from dotenv import load_dotenv

from callm import (
    RateLimitConfig,
    process_requests,
)
from callm.providers import DeepSeekProvider

load_dotenv()

# DeepSeek rate limits
# Source: https://api-docs.deepseek.com/quick_start/rate_limit
#
# DeepSeek does NOT constrain rate limits!
# Quote: "DeepSeek API does NOT constrain user's rate limit.
#         We will try our best to serve every request."
#
# However, be reasonable with RPM to avoid overloading their servers.
# My suggestion is to use OpenAI's rate limits as a reference.
# GPT-5 Tier 2 rates:
RPM = 5_000  # Conservative estimate
TPM = 1_000_000  # 1M tokens per minute

provider = DeepSeekProvider(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="DeepSeek-V3.2-Exp",  # or "deepseek-reasoner" for thinking mode
    request_url="https://api.deepseek.com/chat/completions",
)

# Create a file with 1000 requests
num_requests = 1_000
requests_path = "data/deepseek_completions_requests.jsonl"
with open(requests_path, mode="w", encoding="utf-8") as f:
    for i in range(num_requests):
        f.write(
            json.dumps(
                {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": f"What is {i} multiplied by 3? Reply with just the number.",
                        },
                    ],
                    "temperature": 0.0,
                    "max_tokens": 50,
                    "metadata": {"row_id": i},
                }
            )
            + "\n"
        )


async def main() -> None:
    results = await process_requests(
        provider=provider,
        requests="data/deepseek_completions_requests.jsonl",
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,  # 80% of your limit
            max_tokens_per_minute=TPM * 0.8,  # 80% of your limit
        ),
        output_path="data/deepseek_completions_results.jsonl",
    )

    print(f"Finished in {results.stats.duration_seconds:.2f}s")
    print(f"Success: {results.stats.successful}, Failed: {results.stats.failed}")


if __name__ == "__main__":
    asyncio.run(main())
