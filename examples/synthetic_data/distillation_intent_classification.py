"""
Example: Knowledge Distillation for Intent Classification
Description: Use a large model to generate labeled training data for a smaller model
Use case: Creating fine-tuning datasets, model distillation, bootstrapping classifiers
Provider: Anthropic Claude

This example demonstrates:
- Using a powerful model to generate training examples with labels
- Creating diverse examples across multiple intent categories
- Building a dataset suitable for fine-tuning a smaller/faster model

Real-world application:
- You want to deploy a fast, cheap intent classifier in production
- LLM is too slow/expensive for real-time classification
- Solution: Use LLM to label examples, fine-tune a smaller model
- The smaller model learns from LLM's "knowledge" (distillation)
"""

import asyncio
import json
import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from callm import RateLimitConfig, process_requests
from callm.providers import AnthropicProvider

load_dotenv()


INTENT_CATEGORIES = {
    "order_status": "Questions about order tracking, delivery status, shipping updates",
    "returns_refunds": "Requests to return items, get refunds, or exchange products",
    "product_info": "Questions about product features, specifications, availability",
    "account_issues": "Login problems, password reset, account settings",
    "payment_billing": "Payment failures, billing questions, charges, invoices",
    "technical_support": "Product not working, troubleshooting, setup help",
    "complaint": "Expressing dissatisfaction, escalation requests, negative feedback",
    "general_inquiry": "General questions, store hours, policies, misc info",
}


class GeneratedExample(BaseModel):
    """A training example with message and intent label."""

    model_config = {"extra": "forbid"}  # Mandatory to prevent extra fields
    user_message: str = Field(description="A realistic customer message (1-3 sentences)")
    intent: Literal[
        "order_status",
        "returns_refunds",
        "product_info",
        "account_issues",
        "payment_billing",
        "technical_support",
        "complaint",
        "general_inquiry",
    ] = Field(description="The intent category this message belongs to")


class ExampleBatch(BaseModel):
    """Batch of generated training examples."""

    model_config = {"extra": "forbid"}  # Mandatory to prevent extra fields
    examples: list[GeneratedExample] = Field(description="List of 5 diverse training examples")


GENERATION_PROMPT = """You are generating training data for an intent classification model.

## Intent Categories
{categories}

## Your Task
Generate 5 diverse, realistic customer messages for the intent: **{target_intent}**

Requirements:
- Messages should sound like real customers (varied vocabulary, tones, lengths)
- Include different phrasings of the same intent
- Mix formal and informal styles
- Some should be straightforward, others more ambiguous
- Each message should be 1-3 sentences

Generate examples that would help a model learn to recognize this intent.
"""

# Anthropic rate limits (Tier 2)
# Source: https://docs.anthropic.com/en/api/rate-limits
# Tier 2:
RPM = 1_000
TPM = 450_000


async def main() -> None:
    """Generate distillation dataset for intent classification."""
    provider = AnthropicProvider(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-sonnet-4-5",
    )

    # Format categories for prompt
    categories_text = "\n".join(
        f"- **{intent}**: {desc}" for intent, desc in INTENT_CATEGORIES.items()
    )

    # Generate multiple batches per intent for diversity
    batches_per_intent = 3
    requests = []

    for intent in INTENT_CATEGORIES:
        for batch_num in range(batches_per_intent):
            requests.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": GENERATION_PROMPT.format(
                                categories=categories_text, target_intent=intent
                            ),
                        }
                    ],
                    "max_tokens": 5000,
                    "temperature": 0.9,  # Higher for diversity
                    "output_format": {
                        "type": "json_schema",
                        "schema": ExampleBatch.model_json_schema(mode="serialization"),
                    },
                    "metadata": {"intent": intent, "batch": batch_num},
                }
            )

    total_expected = len(INTENT_CATEGORIES) * batches_per_intent * 5
    print(f"Generating ~{total_expected} training examples...")
    print(f"Intent categories: {len(INTENT_CATEGORIES)}")
    print(f"Batches per intent: {batches_per_intent}\n")

    results = await process_requests(
        provider=provider,
        requests=requests,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,
            max_tokens_per_minute=TPM * 0.8,
        ),
    )

    # Collect all examples
    training_data = []

    for result in results.successes:
        try:
            output_text = result.response.get("content", [{}])[0].get("text", "")
            batch = json.loads(output_text)

            for example in batch.get("examples", []):
                training_data.append(
                    {
                        "text": example["user_message"],
                        "label": example["intent"],
                    }
                )

        except (json.JSONDecodeError, KeyError, IndexError):
            continue

    # Show samples
    print(f"\n{'='*60}")
    print("SAMPLE GENERATED EXAMPLES")
    print(f"{'='*60}\n")

    for intent in list(INTENT_CATEGORIES.keys())[:3]:
        samples = [ex for ex in training_data if ex["label"] == intent][:2]
        print(f"[{intent}]")
        for s in samples:
            print(f"  → {s['text'][:80]}...")
        print()

    # Save training dataset
    output_file = "data/intent_classification_train.jsonl"
    os.makedirs("data", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for example in training_data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    # Summary
    print(f"Generated {len(training_data)} training examples")
    print(f"Saved to: {output_file}")
    print("\nNext steps:")
    print("  1. Review and clean the dataset")
    print("  2. Fine-tune a smaller model (BERT, DistilBERT, etc.)")
    print("  3. Deploy the fast classifier in production")


if __name__ == "__main__":
    asyncio.run(main())
