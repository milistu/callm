"""
Provider: DeepSeek
Endpoint: Chat Completions API (OpenAI-compatible)
Description: DeepSeek's cost-effective LLMs with OpenAI-compatible API

This example shows:
- DeepSeek API configuration
- OpenAI-compatible request format
- Cost-effective high-volume processing

API Reference: https://api-docs.deepseek.com/
Rate Limits: https://api-docs.deepseek.com/quick_start/rate_limit
"""

import asyncio
import os

from dotenv import load_dotenv

from callm import RateLimitConfig, process_requests
from callm.providers import DeepSeekProvider

load_dotenv()

# DeepSeek doesn't enforce strict rate limits
# Use reasonable values based on OpenAI tier 2 as reference
RPM = 5_000
TPM = 1_000_000


async def main() -> None:
    provider = DeepSeekProvider(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model="DeepSeek-V3.2",
        request_url="https://api.deepseek.com/chat/completions",
    )

    # Standard OpenAI-compatible format
    requests = [
        {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": f"Write a Python function to {task}"},
            ],
            "temperature": 0.7,
            "max_tokens": 500,
            "metadata": {"task_id": i},
        }
        for i, task in enumerate(
            [
                "calculate the factorial of a number recursively",
                "reverse a linked list",
                "find the longest common substring of two strings",
                "implement binary search",
                "validate a JSON string",
            ]
        )
    ]

    results = await process_requests(
        provider=provider,
        requests=requests,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,
            max_tokens_per_minute=TPM * 0.8,
        ),
    )

    print("=" * 60)
    print("DeepSeek Chat Completions")
    print("=" * 60)
    print(f"Model: {provider.model}")
    print(f"Processed: {results.stats.successful}/{results.stats.total_requests}")
    print(f"Tokens: {results.stats.total_input_tokens + results.stats.total_output_tokens:,}")
    print()

    for result in results.successes:
        task_id = result.metadata["task_id"]
        content = result.response["choices"][0]["message"]["content"]
        # Show first line of code
        first_line = content.split("\n")[0] if "\n" in content else content[:80]
        print(f"Task {task_id}: {first_line}...")


if __name__ == "__main__":
    asyncio.run(main())
