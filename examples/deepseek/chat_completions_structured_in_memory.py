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
requests = []
for i in range(num_requests):
    requests.append(
        {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"What is {i} multiplied by 3?"},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "metadata": {"row_id": i},
        }
    )


async def main() -> None:
    results = await process_requests(
        provider=provider,
        requests=requests,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,  # 80% of your limit
            max_tokens_per_minute=TPM * 0.8,  # 80% of your limit
        ),
    )

    # Check if responses are JSON serializable
    num_failures = 0
    for sample in results.successes:
        try:
            json.loads(sample.response["choices"][0]["message"]["content"])
        except json.JSONDecodeError:
            num_failures += 1
            print(f"Response {sample.metadata['row_id']} is not JSON serializable")

    if num_failures > 0:
        print(f"Failed to parse {num_failures} responses")
    else:
        print("All responses parsed successfully")


if __name__ == "__main__":
    asyncio.run(main())
