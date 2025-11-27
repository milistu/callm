import asyncio
import json
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from callm import FilesConfig, RateLimitConfig, process_api_requests_from_file
from callm.providers import OpenAIProvider
from callm.utils import pydantic_to_openai_response_format

load_dotenv()

# Find what is your Tier and copy the values from model page: https://platform.openai.com/docs/models/gpt-4.1-nano
# Tier 2:
TPM = 2_000_000
RPM = 5_000

provider = OpenAIProvider(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4.1-nano-2025-04-14",
    request_url="https://api.openai.com/v1/responses",
)


class CalculationResult(BaseModel):
    number: int = Field(description="The number to be multiplied")
    multiplied_by: int = Field(description="The number to multiply the number by")
    result: int = Field(description="The result of the multiplication")
    explanation: str = Field(description="The explanation of the multiplication")


response_format = pydantic_to_openai_response_format(CalculationResult, method="responses")


# Create requests with structured output format
num_requests = 1_000
requests_path = "data/openai_responses_structured_requests.jsonl"
with open(requests_path, mode="w", encoding="utf-8") as f:
    for i in range(num_requests):
        f.write(
            json.dumps(
                {
                    "model": "gpt-4.1-nano-2025-04-14",
                    "input": f"Calculate {i} multiplied by 3",
                    "text": {
                        "format": response_format,
                    },
                    "metadata": {"row_id": i},
                }
            )
            + "\n"
        )


async def main() -> None:
    await process_api_requests_from_file(
        provider=provider,
        requests_file=requests_path,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,
            max_tokens_per_minute=TPM * 0.8,
        ),
        files=FilesConfig(
            save_file="data/openai_responses_structured_results.jsonl",
            error_file="data/openai_responses_structured_errors.jsonl",
        ),
    )


if __name__ == "__main__":
    asyncio.run(main())
