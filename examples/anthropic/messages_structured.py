import asyncio
import json
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

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


class CalculationResult(BaseModel):
    """Schema for math calculation responses."""

    model_config = {"extra": "forbid"}  # Mandatory to prevent extra fields
    number: int = Field(description="The number to be multiplied")
    multiplied_by: int = Field(description="The number to multiply by")
    result: int = Field(description="The result of the multiplication")
    explanation: str = Field(description="A brief explanation")


provider = AnthropicProvider(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-sonnet-4-5",
)

num_requests = 50
requests_path = "data/anthropic_structured_requests.jsonl"
with open(requests_path, mode="w", encoding="utf-8") as f:
    for i in range(num_requests):
        request = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 500,
            "system": "You are a math assistant. Provide structured calculation results.",
            "messages": [{"role": "user", "content": f"What is {i} multiplied by 7?"}],
            "output_format": {
                "type": "json_schema",
                "schema": CalculationResult.model_json_schema(mode="serialization"),
            },
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
        output_path="data/anthropic_structured_results.jsonl",
    )

    print(f"Finished in {results.stats.duration_seconds:.2f}s")
    print(f"Success: {results.stats.successful}, Failed: {results.stats.failed}")

    # Show sample
    print("\n--- Sample Response ---")
    with open("data/anthropic_structured_results.jsonl") as f:
        result = json.loads(f.readline())
        # Result is [request, response, metadata] - get the response (index 1)
        response = result[1] if isinstance(result, list) else result
        if "content" in response:
            # JSON mode returns text in content[0].text
            for block in response["content"]:
                if block.get("type") == "text":
                    parsed = json.loads(block.get("text", "{}"))
                    print(json.dumps(parsed, indent=2))
                    break


if __name__ == "__main__":
    asyncio.run(main())
