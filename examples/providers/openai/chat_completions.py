"""
Provider: OpenAI
Endpoint: Chat Completions API
Description: Standard chat completions endpoint for conversational AI

This example shows:
- OpenAI Chat Completions API configuration
- Structured outputs with Pydantic
- Both file-based and in-memory processing options

API Reference: https://platform.openai.com/docs/api-reference/chat/create
Rate Limits: https://platform.openai.com/docs/guides/rate-limits
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from callm import RateLimitConfig, process_requests
from callm.providers import OpenAIProvider
from callm.utils import pydantic_to_openai_response_format

load_dotenv()

# Tier 2:
RPM = 5_000
TPM = 2_000_000


async def basic_example() -> None:
    """Basic chat completions without structured output."""
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-5-nano",
        request_url="https://api.openai.com/v1/chat/completions",
    )

    requests = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Explain the concept of API rate limiting in one paragraph.",
                },
            ],
            "metadata": {"request_id": 1},
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What are the benefits of asynchronous programming?"},
            ],
            "metadata": {"request_id": 2},
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

    print("Basic Chat Completions Results:")
    for result in results.successes:
        content = result.response["choices"][0]["message"]["content"]
        print(f"  Request {result.metadata['request_id']}: {content[:100]}...")


class SummaryResult(BaseModel):
    """Structured output schema for text summarization."""

    main_points: list[str] = Field(description="Key points from the text")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    word_count: int = Field(description="Approximate word count of summary")


async def structured_example() -> None:
    """Chat completions with structured (JSON) output."""
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-5-nano",
        request_url="https://api.openai.com/v1/chat/completions",
    )

    response_format = pydantic_to_openai_response_format(SummaryResult, endpoint="completions")

    sample_text = """
    The quarterly earnings report exceeded analyst expectations, with revenue up 15%
    year-over-year. The company announced expansion into three new markets and plans
    to hire 500 additional employees. However, supply chain challenges continue to
    impact margins, and the CEO cautioned that Q4 may see slower growth due to
    seasonal factors. Investors responded positively, with shares rising 3% in
    after-hours trading.
    """

    requests = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Summarize the following text and extract key information.",
                },
                {"role": "user", "content": sample_text},
            ],
            "response_format": response_format,
            "metadata": {"doc_id": "earnings_q3"},
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

    print("\nStructured Output Results:")
    for result in results.successes:
        content = result.response["choices"][0]["message"]["content"]
        parsed = SummaryResult.model_validate_json(content)
        print(f"  Main points: {parsed.main_points}")
        print(f"  Sentiment: {parsed.sentiment}")
        print(f"  Word count: {parsed.word_count}")


async def file_based_example() -> None:
    """Process requests from JSONL file, save results to file."""
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-5-nano",
        request_url="https://api.openai.com/v1/chat/completions",
    )

    # Create sample requests file
    os.makedirs("data", exist_ok=True)
    requests_file = "data/openai_chat_requests.jsonl"

    with open(requests_file, "w") as f:
        for i in range(10):
            request = {
                "messages": [
                    {"role": "user", "content": f"What is {i + 1} × 7? Reply with just the number."}
                ],
                "metadata": {"index": i},
            }
            f.write(json.dumps(request) + "\n")

    # Process from file to file
    results = await process_requests(
        provider=provider,
        requests=requests_file,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,
            max_tokens_per_minute=TPM * 0.8,
        ),
        output_path="data/openai_chat_results.jsonl",
    )

    print("\nFile-based Processing:")
    print(f"  Processed: {results.stats.successful}/{results.stats.total_requests}")
    print("  Results saved to: data/openai_chat_results.jsonl")


async def main() -> None:
    print("=" * 60)
    print("OpenAI Chat Completions Examples")
    print("=" * 60)

    await basic_example()
    await structured_example()
    await file_based_example()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
