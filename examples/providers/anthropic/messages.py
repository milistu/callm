"""
Provider: Anthropic
Endpoint: Messages API
Description: Claude models for conversational AI and complex reasoning

This example shows:
- Anthropic Messages API configuration
- Claude-specific features (system prompts, max_tokens requirement)
- Structured output patterns

API Reference: https://docs.anthropic.com/en/api/messages
Rate Limits: https://docs.anthropic.com/en/api/rate-limits
"""

import asyncio
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from callm import RateLimitConfig, process_requests
from callm.providers import AnthropicProvider

load_dotenv()

# Anthropic rate limits (Tier 2)
RPM = 1_000
TPM = 450_000


async def basic_example() -> None:
    """Basic Messages API usage."""
    provider = AnthropicProvider(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-haiku-4-5",
    )

    requests = [
        {
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Explain the difference between "
                        "supervised and unsupervised learning in 2-3 sentences."
                    ),
                }
            ],
            "metadata": {"topic": "ml_basics"},
        },
        {
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": "What are the key principles of RESTful API design?",
                }
            ],
            "metadata": {"topic": "api_design"},
        },
    ]

    results = await process_requests(
        provider=provider,
        requests=requests,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,
            max_tokens_per_minute=TPM * 0.8,
        ),
    )

    print("Basic Messages API:")
    for result in results.successes:
        topic = result.metadata["topic"]
        content = result.response["content"][0]["text"]
        print(f"\n{topic}:")
        print(f"  {content[:200]}...")


async def structured_example() -> None:
    """Structured output with Claude using output_format."""
    provider = AnthropicProvider(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-haiku-4-5",
    )

    class StructuredOutput(BaseModel):

        model_config = {"extra": "forbid"}
        summary: str = Field(description="Summary of the Kubernetes container orchestration")
        key_points: list[str] = Field(
            description="Key points of the Kubernetes container orchestration"
        )
        difficulty: str = Field(
            description="Difficulty level of the Kubernetes container orchestration"
        )

    requests = [
        {
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Analyze Kubernetes container orchestration. "
                        "Provide a summary, key points, and difficulty level."
                    ),
                }
            ],
            "output_format": {
                "type": "json_schema",
                "schema": StructuredOutput.model_json_schema(mode="serialization"),
            },
            "metadata": {"topic": "kubernetes"},
        }
    ]

    results = await process_requests(
        provider=provider,
        requests=requests,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,
            max_tokens_per_minute=TPM * 0.8,
        ),
    )

    print("\nStructured Output:")
    for result in results.successes:
        content = result.response["content"][0]["text"]
        parsed = StructuredOutput.model_validate_json(content)
        print(f"  Summary: {parsed.summary[:100]}...")
        print(f"  Key points: {len(parsed.key_points)} items")
        print(f"  Difficulty: {parsed.difficulty}")


async def main() -> None:
    print("=" * 60)
    print("Anthropic Messages API Examples")
    print("=" * 60)

    await basic_example()
    await structured_example()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
