import asyncio
import json
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from callm import (
    RateLimitConfig,
    process_requests,
)
from callm.providers import GeminiProvider

load_dotenv()

# Gemini API rate limits
# Source: https://ai.google.dev/gemini-api/docs/rate-limits
RPM = 1_000
TPM = 1_000_000


# Define the structured output schema using Pydantic
class CalculationResult(BaseModel):
    """Schema for math calculation responses."""

    number: int = Field(description="The number to be multiplied")
    multiplied_by: int = Field(description="The number to multiply by")
    result: int = Field(description="The result of the multiplication")
    explanation: str = Field(description="A brief explanation of the calculation")


provider = GeminiProvider(
    api_key=os.getenv("GEMINI_API_KEY"),
    model="gemini-flash-latest",
    request_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent",
)

# Create requests with structured output format
num_requests = 50  # Keep small for free tier
requests_path = "data/gemini_structured_requests.jsonl"
with open(requests_path, mode="w", encoding="utf-8") as f:
    for i in range(num_requests):
        request = {
            "contents": [
                {
                    "parts": [{"text": f"What is {i} multiplied by 7?"}],
                    "role": "user",
                }
            ],
            "systemInstruction": {
                "parts": [
                    {"text": "You are a math assistant. Provide structured calculation results."}
                ]
            },
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 200,
                "responseMimeType": "application/json",
                "responseJsonSchema": CalculationResult.model_json_schema(),
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
        output_path="data/gemini_structured_results.jsonl",
    )

    print(f"Finished in {results.stats.duration_seconds:.2f}s")
    print(f"Success: {results.stats.successful}, Failed: {results.stats.failed}")
    print(f"Total input tokens: {results.stats.total_input_tokens}")
    print(f"Total output tokens: {results.stats.total_output_tokens}")

    # Show a sample result
    print("\n--- Sample Response ---")
    with open("data/gemini_structured_results.jsonl") as f:
        first_line = f.readline()
        result = json.loads(first_line)
        if "candidates" in result:
            content = result["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(content)
            print(json.dumps(parsed, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
