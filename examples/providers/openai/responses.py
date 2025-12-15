"""
Provider: OpenAI
Endpoint: Responses API
Description: OpenAI's newer Responses API for simpler request/response patterns

This example shows:
- Responses API configuration (simpler than Chat Completions)
- Structured outputs with the Responses API
- In-memory processing for quick iterations

API Reference: https://platform.openai.com/docs/api-reference/responses
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


class EntityExtraction(BaseModel):
    """Schema for named entity extraction."""

    people: list[str] = Field(description="Names of people mentioned")
    organizations: list[str] = Field(description="Organization names")
    locations: list[str] = Field(description="Place names")
    dates: list[str] = Field(description="Dates or time references")


async def main() -> None:
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-5-nano",
        request_url="https://api.openai.com/v1/responses",
    )

    # Responses API uses 'input' instead of 'messages'
    sample_texts = [
        (
            "Apple CEO Tim Cook announced the new iPhone "
            "at the Steve Jobs Theater in Cupertino on September 12, 2024."
        ),
        (
            "Microsoft and Google are competing in the AI space. "
            "Satya Nadella spoke at the Build conference in Seattle."
        ),
        (
            "The European Union passed new regulations in Brussels. "
            "Ursula von der Leyen praised the initiative."
        ),
    ]

    response_format = pydantic_to_openai_response_format(EntityExtraction, "responses")

    requests = [
        {
            "input": f"Extract named entities from: {text}",
            "text": {"format": response_format},
            "metadata": {"text_id": i},
        }
        for i, text in enumerate(sample_texts)
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
    print("OpenAI Responses API - Entity Extraction")
    print("=" * 60)
    print(f"Processed: {results.stats.successful}/{results.stats.total_requests}")
    print()

    for result in results.successes:
        text_id = result.metadata["text_id"]
        output = result.response.get("output", [{}])[1].get("content", [{}])[0].get("text", "{}")
        entities = json.loads(output)

        print(f"Text {text_id}:")
        print(f"  People: {entities.get('people', [])}")
        print(f"  Organizations: {entities.get('organizations', [])}")
        print(f"  Locations: {entities.get('locations', [])}")
        print(f"  Dates: {entities.get('dates', [])}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
