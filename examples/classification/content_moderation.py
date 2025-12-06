"""
Example: Content Moderation at Scale
Description: Classify user-generated content for policy violations
Use case: Social media moderation, comment filtering, UGC safety
Provider: OpenAI

This example demonstrates:
- Classifying content against multiple policy categories
- Structured moderation decisions with reasoning
- High-throughput processing for real-time feeds
- Confidence scores for human review escalation

Real-world application:
- Your platform receives 100,000 user comments daily
- You need to automatically flag harmful content
- Human moderators can only review a fraction
- LLM classification pre-filters for human review
"""

import asyncio
import json
import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from callm import RateLimitConfig, process_requests
from callm.providers import OpenAIProvider
from callm.utils import pydantic_to_openai_response_format

load_dotenv()


# Moderation schema
class ModerationResult(BaseModel):
    """Content moderation classification result."""

    action: Literal["approve", "flag_for_review", "remove"] = Field(description="Action to take")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in decision (0-1)")
    categories_flagged: list[
        Literal[
            "hate_speech",
            "harassment",
            "violence",
            "sexual_content",
            "spam",
            "misinformation",
            "personal_information",
            "illegal_activity",
            "self_harm",
        ]
    ] = Field(description="List of policy categories violated (empty if clean)")
    severity: Literal["none", "low", "medium", "high", "critical"] = Field(
        description="Severity of the violation"
    )
    reasoning: str = Field(description="Brief explanation of the decision, 1-2 sentences max.")
    needs_human_review: bool = Field(description="Should this go to human moderator?")


# Policy categories
POLICY_CATEGORIES = [
    "hate_speech",
    "harassment",
    "violence",
    "sexual_content",
    "spam",
    "misinformation",
    "personal_information",
    "illegal_activity",
    "self_harm",
]


# Sample content to moderate (replace with your actual content queue)
SAMPLE_CONTENT = [
    {
        "content_id": "C001",
        "user_id": "U123",
        "content_type": "comment",
        "text": "Great product! Really helped me with my project. Definitely recommend to anyone "
        "looking for a reliable solution. Five stars! ⭐⭐⭐⭐⭐",
    },
    {
        "content_id": "C002",
        "user_id": "U456",
        "content_type": "comment",
        "text": "This is absolutely terrible. The worst purchase I've ever made. Complete waste "
        "of money and the company should be ashamed. Never buying again.",
    },
    {
        "content_id": "C003",
        "user_id": "U789",
        "content_type": "reply",
        "text": "You're an idiot if you think this works. People like you shouldn't be allowed "
        "to post reviews. Go back to school.",
    },
    {
        "content_id": "C004",
        "user_id": "U234",
        "content_type": "comment",
        "text": "🔥🔥🔥 FLASH SALE! Click here for FREE iPhone! Limited time only! "
        "www.totallynotascam.com 💰💰💰 DM me for details!",
    },
    {
        "content_id": "C005",
        "user_id": "U567",
        "content_type": "post",
        "text": "Just finished building my first PC! Took about 4 hours but the tutorials on "
        "YouTube really helped. Happy to answer questions if anyone else is thinking about it.",
    },
    {
        "content_id": "C006",
        "user_id": "U890",
        "content_type": "comment",
        "text": "The CEO's home address is 123 Main Street and their phone number is 555-0123. "
        "Let's all show up and tell them what we think of their policies.",
    },
    {
        "content_id": "C007",
        "user_id": "U345",
        "content_type": "reply",
        "text": "Thanks for the detailed review! Really helpful to see real-world performance "
        "numbers. Did you test it with the latest firmware update?",
    },
    {
        "content_id": "C008",
        "user_id": "U678",
        "content_type": "comment",
        "text": "I've been feeling really down lately and don't know what to do anymore. "
        "Sometimes I wonder if things will ever get better.",
    },
    {
        "content_id": "C009",
        "user_id": "U901",
        "content_type": "post",
        "text": "Here's a simple recipe for chocolate chip cookies: 2 cups flour, 1 cup butter, "
        "3/4 cup sugar... Mix ingredients and bake at 350°F for 12 minutes. Enjoy!",
    },
    {
        "content_id": "C010",
        "user_id": "U012",
        "content_type": "reply",
        "text": "Everyone who bought this product is a [slur] and deserves what they get. "
        "Your entire [group] should be eliminated.",
    },
]

MODERATION_PROMPT = """
You are a content moderation system.
Analyze the following user-generated content for policy violations.

## Policy Categories to Check
{categories}

## Severity Levels
- none: No policy violations
- low: Minor issues, borderline content
- medium: Clear violation but not severe
- high: Serious violation requiring immediate action
- critical: Severe violation (threats, illegal content)

## Actions
- approve: Content is acceptable
- flag_for_review: Needs human moderator review
- remove: Should be removed immediately

## Content to Analyze
Type: {content_type}
Text: {text}

Analyze this content and provide your moderation decision as a JSON object.
"""

# Rate limits by tier (tier 2 for selected model https://platform.openai.com/docs/models/gpt-5-nano)
# Note: adjust rate limits based on your tier
RPM = 5_000
TPM = 2_000_000


# Main processing logic
async def main() -> None:
    # Configure provider
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-5-nano",
        request_url="https://api.openai.com/v1/responses",
    )

    # Create structured output format
    response_format = pydantic_to_openai_response_format(ModerationResult, "responses")

    # Build moderation requests
    requests = []
    for content in SAMPLE_CONTENT:
        prompt = MODERATION_PROMPT.format(
            categories="\n".join(f"- {cat}" for cat in POLICY_CATEGORIES),
            content_type=content["content_type"],
            text=content["text"],
        )

        requests.append(
            {
                "input": prompt,
                "text": {"format": response_format},
                "reasoning": {"effort": "minimal"},
                "metadata": {
                    "content_id": content["content_id"],
                    "user_id": content["user_id"],
                    "content_type": content["content_type"],
                },
            }
        )

    # Process moderation requests
    # Using high throughput for real-time moderation
    results = await process_requests(
        provider=provider,
        requests=requests,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,  # 80% of the limit
            max_tokens_per_minute=TPM * 0.8,  # 80% of the limit
        ),
        logging_level=20,
    )
    # Process results
    print(f"\n{'='*70}")
    print("CONTENT MODERATION RESULTS")
    print(f"{'='*70}")
    print(f"Processed: {results.stats.successful}/{results.stats.total_requests} items")
    print(f"Duration: {results.stats.duration_seconds:.2f}s")
    print(f"Throughput: {results.stats.successful / results.stats.duration_seconds:.1f} items/sec")
    print(f"{'='*70}\n")

    moderation_results = []
    for result in results.successes:
        metadata = result.metadata

        try:
            output_text = (
                result.response.get("output", [{}])[1].get("content", [{}])[0].get("text", "{}")
            )
            mod_result = json.loads(output_text)
            mod_result["content_id"] = metadata["content_id"]
            mod_result["user_id"] = metadata["user_id"]
            moderation_results.append(mod_result)

            # Display with action-based formatting
            action = mod_result.get("action", "unknown")
            severity = mod_result.get("severity", "unknown")

            if action == "remove":
                icon = "🚫"
            elif action == "flag_for_review":
                icon = "⚠️"
            else:
                icon = "✅"

            print(f"{icon} {metadata['content_id']}: {action.upper()} (severity: {severity})")
            if mod_result.get("categories_flagged"):
                print(f"   Categories: {', '.join(mod_result['categories_flagged'])}")
            print(f"   Confidence: {mod_result.get('confidence', 0):.0%}")
            print(f"   Reasoning: {mod_result.get('reasoning', 'N/A')[:80]}...\n")

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Parse error for {metadata['content_id']}: {e}")

    # Save results
    output_file = "data/moderation_results.jsonl"
    os.makedirs("data", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for r in moderation_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
