"""
Provider: Google Gemini
Endpoint: Generate Content API
Description: Google's Gemini models for text generation and multimodal tasks

This example shows:
- Gemini API configuration (different request format)
- Content generation with Gemini
- Rate limits for free vs paid tiers

API Reference: https://ai.google.dev/gemini-api/docs/text-generation
Rate Limits: https://ai.google.dev/gemini-api/docs/rate-limits
"""

import asyncio
import os

from dotenv import load_dotenv

from callm import RateLimitConfig, process_requests
from callm.providers import GeminiProvider

load_dotenv()

# Gemini rate limits
# Free tier: 15 RPM, 1M TPM, 1500 RPD
# Pay-as-you-go: 1000 RPM, 4M TPM
RPM = 1_000
TPM = 1_000_000


async def main() -> None:
    provider = GeminiProvider(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-flash-latest",
        request_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent",
    )

    # Gemini uses 'contents' with 'parts' structure
    prompts = [
        "What are the main differences between Python and JavaScript?",
        "Explain how a neural network learns from data.",
        "What is the CAP theorem in distributed systems?",
    ]

    requests = [
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 500,
            },
            "metadata": {"prompt_id": i},
        }
        for i, prompt in enumerate(prompts)
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
    print("Google Gemini Generate Content")
    print("=" * 60)
    print(f"Model: {provider.model}")
    print(f"Processed: {results.stats.successful}/{results.stats.total_requests}")
    print()

    for result in results.successes:
        prompt_id = result.metadata["prompt_id"]
        # Gemini response structure
        text = result.response["candidates"][0]["content"]["parts"][0]["text"]
        print(f"Prompt {prompt_id}:")
        print(f"  {text[:150]}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
