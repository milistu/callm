"""
Example: Multi-Judge Consensus Evaluation
Description: Use multiple LLMs as judges and combine their assessments
Use case: High-stakes evaluation, reducing bias, ensemble judging
Provider: Multiple (OpenAI, Anthropic, Gemini)

This example demonstrates:
- Running the same evaluation across multiple LLM judges
- Aggregating scores to get consensus
- Identifying disagreements for human review

Real-world application:
- Single-judge evaluations can be biased or inconsistent
- Research shows multi-judge consensus improves reliability
- Critical decisions (content moderation, hiring) benefit from multiple perspectives
- Disagreements between judges flag edge cases for human review

Note: Requires API keys for multiple providers (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY)
"""

import asyncio
import json
import os
from statistics import mean, stdev

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from callm import RateLimitConfig, process_requests
from callm.providers import AnthropicProvider, GeminiProvider, OpenAIProvider
from callm.utils import pydantic_to_openai_response_format

load_dotenv()


CODE_REVIEWS = [
    {
        "id": "review_001",
        "code": """
def calculate_discount(price, discount_percent):
    return price - (price * discount_percent / 100)
""",
        "review": "This function looks fine. It calculates the discount correctly.",
    },
    {
        "id": "review_002",
        "code": """
def calculate_discount(price, discount_percent):
    if not isinstance(price, (int, float)) or price < 0:
        raise ValueError("Price must be a non-negative number")
    if not isinstance(discount_percent, (int, float)) or not 0 <= discount_percent <= 100:
        raise ValueError("Discount must be between 0 and 100")
    return round(price * (1 - discount_percent / 100), 2)
""",
        "review": "Good implementation with input validation and rounding for currency precision.",
    },
    {
        "id": "review_003",
        "code": """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
""",
        "review": "The function works but could use some optimization for larger datasets.",
    },
    {
        "id": "review_004",
        "code": """
import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
""",
        "review": (
            "Standard email validation using regex. The pattern covers most common "
            "cases but may miss some valid edge cases like + addressing."
        ),
    },
]


class CodeReviewResult(BaseModel):

    model_config = {"extra": "forbid"}  # Mandatory to prevent extra fields
    accuracy: int = Field(description="How accurate is the review? (1-5)")
    completeness: int = Field(description="How complete is the review? (1-5)")
    helpfulness: int = Field(description="How helpful is the review? (1-5)")
    missed_issues: list[str] = Field(description="The issues that were missed in the review")
    reasoning: str = Field(
        description="The reasoning for the scores, brief explanation or 1 - 2 sentences max."
    )


JUDGE_PROMPT = """
You are evaluating the quality of a code review.

## The Code
```python
{code}
```

## The Review
{review}

## Evaluation Criteria
Rate the review on a scale of 1-5:
- **Accuracy** (1-5): Does the review correctly identify issues/strengths in the code?
- **Completeness** (1-5): Does it cover all important aspects (bugs, security, style)?
- **Helpfulness** (1-5): Would this review help the developer improve?

## Important Issues to Check
- SQL injection vulnerabilities
- Missing input validation
- Error handling
- Security concerns
- Code clarity

Respond with ONLY a JSON object.
"""


async def run_judge(provider, provider_name: str, reviews: list[dict]) -> list[dict]:
    """Run evaluation with a single judge."""
    # Build requests based on provider type
    requests = []
    for item in reviews:
        prompt = JUDGE_PROMPT.format(code=item["code"], review=item["review"])

        if provider_name == "anthropic":
            requests.append(
                {
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": prompt}],
                    "output_format": {
                        "type": "json_schema",
                        "schema": CodeReviewResult.model_json_schema(mode="serialization"),
                    },
                    "metadata": {"review_id": item["id"]},
                }
            )
        elif provider_name == "gemini":
            requests.append(
                {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "responseJsonSchema": CodeReviewResult.model_json_schema(),
                    },
                    "metadata": {"review_id": item["id"]},
                }
            )
        else:  # openai
            requests.append(
                {
                    "input": prompt,
                    "text": {
                        "format": pydantic_to_openai_response_format(CodeReviewResult, "responses")
                    },
                    "metadata": {"review_id": item["id"]},
                }
            )

    results = await process_requests(
        provider=provider,
        requests=requests,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=100,
            max_tokens_per_minute=50_000,
        ),
        logging_level=30,  # WARNING - reduce noise
    )

    # Parse results
    judgments = []
    for result in results.successes:
        try:
            # Extract text based on provider response format
            if provider_name == "anthropic":
                text = result.response.get("content", [{}])[0].get("text", "{}")
            elif provider_name == "gemini":
                text = (
                    result.response.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "{}")
                )
            else:  # openai
                text = (
                    result.response.get("output", [{}])[1].get("content", [{}])[0].get("text", "{}")
                )

            # Clean JSON from markdown if present
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            scores = json.loads(text.strip())
            scores["review_id"] = result.metadata["review_id"]
            scores["judge"] = provider_name
            judgments.append(scores)

        except (json.JSONDecodeError, KeyError, IndexError):
            continue

    return judgments


async def main() -> None:
    """Run multi-judge evaluation and compute consensus."""
    # Initialize judges
    judges = {}

    if os.getenv("OPENAI_API_KEY"):
        judges["openai"] = OpenAIProvider(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-5-nano",
            request_url="https://api.openai.com/v1/responses",
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        judges["anthropic"] = AnthropicProvider(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-haiku-4-5",
        )

    if os.getenv("GEMINI_API_KEY"):
        judges["gemini"] = GeminiProvider(
            api_key=os.getenv("GEMINI_API_KEY"),
            model="gemini-flash-latest",
            request_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent",
        )

    if len(judges) < 2:
        print("⚠️  Multi-judge evaluation requires at least 2 providers configured.")
        print("   Set API keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY")
        return

    print(f"Running evaluation with {len(judges)} judges: {', '.join(judges.keys())}")
    print(f"Evaluating {len(CODE_REVIEWS)} code reviews\n")

    # Run all judges in parallel
    tasks = [run_judge(provider, name, CODE_REVIEWS) for name, provider in judges.items()]
    all_judgments = await asyncio.gather(*tasks)

    # Flatten and group by review
    judgments_by_review: dict[str, list[dict]] = {}
    for judge_results in all_judgments:
        for j in judge_results:
            review_id = j["review_id"]
            judgments_by_review.setdefault(review_id, []).append(j)

    # Compute consensus
    print(f"{'='*60}")
    print("MULTI-JUDGE CONSENSUS RESULTS")
    print(f"{'='*60}\n")

    consensus_results = []

    for review_id, judgments in judgments_by_review.items():
        if len(judgments) < 2:
            continue

        # Average scores across judges
        accuracy_scores = [j["accuracy"] for j in judgments]
        completeness_scores = [j["completeness"] for j in judgments]
        helpfulness_scores = [j["helpfulness"] for j in judgments]

        avg_accuracy = mean(accuracy_scores)
        avg_completeness = mean(completeness_scores)
        avg_helpfulness = mean(helpfulness_scores)

        # Check for disagreement (high variance)
        all_scores = accuracy_scores + completeness_scores + helpfulness_scores
        score_variance = stdev(all_scores) if len(all_scores) > 1 else 0
        has_disagreement = score_variance > 1.0

        # Collect missed issues from all judges
        all_missed = set()
        for j in judgments:
            all_missed.update(j.get("missed_issues", []))

        result = {
            "review_id": review_id,
            "consensus_accuracy": round(avg_accuracy, 1),
            "consensus_completeness": round(avg_completeness, 1),
            "consensus_helpfulness": round(avg_helpfulness, 1),
            "judge_count": len(judgments),
            "has_disagreement": has_disagreement,
            "missed_issues": list(all_missed),
        }
        consensus_results.append(result)

        # Display
        flag = "⚠️ DISAGREEMENT" if has_disagreement else "✓"
        print(f"{review_id} {flag}")
        print(f"  Accuracy: {avg_accuracy:.1f}")
        print(f"  Completeness: {avg_completeness:.1f}")
        print(f"  Helpfulness: {avg_helpfulness:.1f}")
        if all_missed:
            print(f"  Missed issues: {', '.join(list(all_missed)[:3])}")
        print()

    # Save results
    output_file = "data/multi_judge_results.jsonl"
    os.makedirs("data", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for r in consensus_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    disagreements = sum(1 for r in consensus_results if r["has_disagreement"])
    print(f"Results saved to: {output_file}")
    print(f"Disagreements flagged for human review: {disagreements}/{len(consensus_results)}")


if __name__ == "__main__":
    asyncio.run(main())
