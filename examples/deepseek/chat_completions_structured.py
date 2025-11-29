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
    model="DeepSeek-V3.2-Exp",
    request_url="https://api.deepseek.com/chat/completions",
)


prompt = """
You are a math assistant that provides structured calculation results.

Return results in the following JSON format:
{
    "number": int,          # The number to be multiplied
    "multiplied_by": int,   # The number to multiply the number by
    "result": int,          # The result of the multiplication
    "explanation": str      # The explanation of the multiplication
}
"""

# Create requests with structured output format
num_requests = 1_000
requests_path = "data/deepseek_completions_structured_requests.jsonl"
with open(requests_path, mode="w", encoding="utf-8") as f:
    for i in range(num_requests):
        f.write(
            json.dumps(
                {
                    "model": "deepseek-chat",
                    "messages": [
                        {
                            "role": "system",
                            "content": prompt,
                        },
                        {"role": "user", "content": f"What is {i} multiplied by 3?"},
                    ],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                    "metadata": {"row_id": i},
                }
            )
            + "\n"
        )


async def main() -> None:
    results = await process_requests(
        provider=provider,
        requests=requests_path,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,  # 80% of your limit
            max_tokens_per_minute=TPM * 0.8,  # 80% of your limit
        ),
        output_path="data/deepseek_completions_structured_results.jsonl",
    )

    print(f"Finished in {results.stats.duration_seconds:.2f}s")
    print(f"Success: {results.stats.successful}, Failed: {results.stats.failed}")


if __name__ == "__main__":
    asyncio.run(main())
