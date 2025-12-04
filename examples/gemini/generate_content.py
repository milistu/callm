import asyncio
import json
import os

from dotenv import load_dotenv

from callm import (
    RateLimitConfig,
    process_requests,
)
from callm.providers import GeminiProvider

load_dotenv()

# Gemini API rate limits (Free tier)
# Source: https://ai.google.dev/gemini-api/docs/rate-limits
# Free tier for gemini-2.0-flash:
#   - 15 RPM (requests per minute)
#   - 1,000,000 TPM (tokens per minute)
#   - 1,500 RPD (requests per day)
# Pay-as-you-go tier has higher limits.
RPM = 1_000
TPM = 1_000_000

provider = GeminiProvider(
    api_key=os.getenv("GEMINI_API_KEY"),
    model="gemini-flash-latest",
    request_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent",
)

# Create a file with requests
num_requests = 100  # Keep small for free tier (1,500 RPD limit)
requests_path = "data/gemini_generate_requests.jsonl"
with open(requests_path, mode="w", encoding="utf-8") as f:
    for i in range(num_requests):
        f.write(
            json.dumps(
                {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": (
                                        f"What is {i} multiplied by 3? "
                                        "Reply with just the number."
                                    )
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.0,
                        "maxOutputTokens": 50,
                    },
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
        output_path="data/gemini_generate_results.jsonl",
    )

    print(f"Finished in {results.stats.duration_seconds:.2f}s")
    print(f"Success: {results.stats.successful}, Failed: {results.stats.failed}")
    print(f"Total tokens: {results.stats.total_input_tokens + results.stats.total_output_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
