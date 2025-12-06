"""
Example: Batch Sentiment Analysis
Description: Analyze sentiment of customer reviews at scale
Use case: E-commerce analytics, brand monitoring, customer feedback analysis
Provider: Gemini

This example demonstrates:
- Processing thousands of reviews efficiently
- Structured sentiment classification
- Handling diverse text inputs
- Aggregating results for analytics

Real-world application:
- You have 50,000 product reviews to analyze
- You need sentiment (positive/negative/neutral) + specific aspects
- Traditional ML models don't capture nuance like LLMs
- Results feed into dashboards and alerts
"""

import asyncio
import json
import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from callm import RateLimitConfig, process_requests
from callm.providers import GeminiProvider

load_dotenv()


# Analysis schema
class AnalysisResult(BaseModel):
    """Analysis result for a product review."""

    overall_sentiment: Literal["positive", "negative", "neutral"]
    sentiment_score: float = Field(
        ge=-1.0,
        le=1.0,
        description=(
            "Sentiment score of the review from -1.0 (very negative) to 1.0 (very positive)"
        ),
    )
    key_positives: list[str] = Field(
        description="List of specific things the reviewer liked (empty if none)"
    )
    key_negatives: list[str] = Field(
        description="List of specific complaints or issues (empty if none)"
    )
    mentioned_aspects: list[str] = Field(
        description="List of product aspects mentioned (e.g., 'battery', 'price', 'quality')"
    )
    purchase_intent: Literal["would_recommend", "neutral", "would_not_recommend"] = Field(
        description="Purchase intent of the reviewer"
    )
    emotional_tone: Literal["enthusiastic", "satisfied", "neutral", "disappointed", "angry"] = (
        Field(description="Emotional tone of the review")
    )


# Sample review data (replace with your actual reviews)
SAMPLE_REVIEWS = [
    {
        "review_id": "R001",
        "product_id": "P123",
        "product_name": "Wireless Bluetooth Headphones",
        "rating": 5,
        "text": "Amazing sound quality! Battery lasts forever and they're super comfortable. "
        "Best headphones I've ever owned. The noise cancellation is incredible for the price.",
    },
    {
        "review_id": "R002",
        "product_id": "P123",
        "product_name": "Wireless Bluetooth Headphones",
        "rating": 2,
        "text": "Disappointed. The Bluetooth keeps disconnecting and the ear cups are too small. "
        "Returned after 3 days. Customer service was helpful at least.",
    },
    {
        "review_id": "R003",
        "product_id": "P456",
        "product_name": "Coffee Maker Pro",
        "rating": 4,
        "text": "Makes great coffee but took a while to figure out the settings. Once I got it "
        "dialed in, it's been perfect every morning. Wish the water tank was bigger.",
    },
    {
        "review_id": "R004",
        "product_id": "P456",
        "product_name": "Coffee Maker Pro",
        "rating": 1,
        "text": "Broke after 2 weeks. Total waste of money. The company won't honor the warranty "
        "because I don't have the original receipt. Never buying from them again.",
    },
    {
        "review_id": "R005",
        "product_id": "P789",
        "product_name": "Running Shoes X500",
        "rating": 5,
        "text": "Perfect for long runs! Great arch support and they're lightweight. "
        "Ran a half marathon in these with zero discomfort. Highly recommend for serious runners.",
    },
    {
        "review_id": "R006",
        "product_id": "P789",
        "product_name": "Running Shoes X500",
        "rating": 3,
        "text": "Decent shoes but run small. Had to return for a larger size. Once I got the "
        "right fit they're comfortable enough. Nothing special for the price though.",
    },
    {
        "review_id": "R007",
        "product_id": "P234",
        "product_name": "Smart Watch Ultra",
        "rating": 4,
        "text": "Love the fitness tracking features! Heart rate monitor is accurate. "
        "Battery life could be better - only lasts 2 days with GPS on. Screen is gorgeous.",
    },
    {
        "review_id": "R008",
        "product_id": "P234",
        "product_name": "Smart Watch Ultra",
        "rating": 2,
        "text": "The watch itself is fine but the app is terrible. Constantly crashes, "
        "loses sync with the watch, and the UI is confusing. Fix the software!",
    },
    {
        "review_id": "R009",
        "product_id": "P567",
        "product_name": "Laptop Stand Deluxe",
        "rating": 5,
        "text": "Simple but effective. Sturdy build, adjustable height, and looks sleek on my "
        "desk. My neck pain is gone after a week of using this. Worth every penny.",
    },
    {
        "review_id": "R010",
        "product_id": "P567",
        "product_name": "Laptop Stand Deluxe",
        "rating": 3,
        "text": "It works but feels cheaper than expected. The adjustment mechanism is a bit "
        "wobbly. Does the job but I've seen better stands for the same price.",
    },
]

ANALYSIS_PROMPT = """
Analyze the sentiment of this product review.

Product: {product_name}
Star Rating: {rating}/5
Review Text: {text}

Provide your analysis as a JSON object only, no other text.
"""


# Rate limits by tier (tier 1 for selected model https://ai.google.dev/gemini-api/docs/rate-limits#current-rate-limits)
# Note: adjust rate limits based on your tier
RPM = 1_000
TPM = 1_000_000


# Main processing logic
async def main() -> None:
    # Configure Gemini provider
    provider = GeminiProvider(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-flash-latest",
        request_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent",
    )

    # Build analysis requests
    requests = []
    for review in SAMPLE_REVIEWS:
        prompt = ANALYSIS_PROMPT.format(
            product_name=review["product_name"], rating=review["rating"], text=review["text"]
        )

        requests.append(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "responseMimeType": "application/json",
                    "responseJsonSchema": AnalysisResult.model_json_schema(),
                },
                "metadata": {
                    "review_id": review["review_id"],
                    "product_id": review["product_id"],
                    "product_name": review["product_name"],
                    "star_rating": review["rating"],
                },
            }
        )

    # Process with Gemini
    results = await process_requests(
        provider=provider,
        requests=requests,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,  # 80% of the limit
            max_tokens_per_minute=TPM * 0.8,  # 80% of the limit
        ),
        logging_level=20,
    )

    # Parse and analyze results
    print(f"\n{'='*70}")
    print("SENTIMENT ANALYSIS RESULTS")
    print(f"{'='*70}")
    print(f"Analyzed: {results.stats.successful}/{results.stats.total_requests} reviews")
    print(f"Duration: {results.stats.duration_seconds:.2f}s")
    print(f"{'='*70}\n")

    analyses = []
    for result in results.successes:
        metadata = result.metadata

        try:
            # Gemini response structure
            response_text = (
                result.response.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "{}")
            )

            analysis = json.loads(response_text)
            analysis["review_id"] = metadata["review_id"]
            analysis["product_id"] = metadata["product_id"]
            analysis["product_name"] = metadata["product_name"]
            analysis["star_rating"] = metadata["star_rating"]
            analyses.append(analysis)

            # Display individual result
            sentiment = analysis.get("overall_sentiment", "unknown")
            score = analysis.get("sentiment_score", 0)
            emoji = "😊" if sentiment == "positive" else "😐" if sentiment == "neutral" else "😞"

            print(f"{emoji} {metadata['review_id']}: {sentiment.upper()} (score: {score:+.2f})")
            print(f"   Product: {metadata['product_name']} ({metadata['star_rating']}★)")
            if analysis.get("key_positives"):
                print(f"   + {', '.join(analysis['key_positives'][:2])}")
            if analysis.get("key_negatives"):
                print(f"   - {', '.join(analysis['key_negatives'][:2])}")
            print()

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Parse error for {metadata['review_id']}: {e}")

    # Save results
    output_file = "data/sentiment_analysis_results.jsonl"
    os.makedirs("data", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for analysis in analyses:
            f.write(json.dumps(analysis, ensure_ascii=False) + "\n")

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
