import asyncio
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
RPM = 1_000
TPM = 1_000_000

provider = GeminiProvider(
    api_key=os.getenv("GEMINI_API_KEY"),
    model="gemini-flash-latest",
    request_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent",
)

# Create a list with requests (in-memory)
requests = [
    {
        "contents": [
            {"parts": [{"text": f"What is {i} multiplied by 3? Reply with just the number."}]}
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 50,
        },
        "metadata": {"row_id": i},
    }
    for i in range(50)  # Small batch for testing
]


async def main() -> None:
    results = await process_requests(
        provider=provider,
        requests=requests,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,
            max_tokens_per_minute=TPM * 0.8,
        ),
    )

    print(f"Finished in {results.stats.duration_seconds:.2f}s")
    print(f"Success: {results.stats.successful}, Failed: {results.stats.failed}")
    print(f"Input tokens: {results.stats.total_input_tokens}")
    print(f"Output tokens: {results.stats.total_output_tokens}")

    # Print first few results
    for i, result in enumerate(results.successes[:3]):
        text = result.response["candidates"][0]["content"]["parts"][0]["text"]
        print(f"  [{i}] {text.strip()}")


if __name__ == "__main__":
    asyncio.run(main())
