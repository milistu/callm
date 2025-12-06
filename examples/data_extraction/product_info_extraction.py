"""
Example: Product Information Extraction
Description: Extract structured product data from unstructured e-commerce listings
Use case: E-commerce data pipelines, catalog enrichment, product analytics
Provider: OpenAI (with structured outputs)

This example demonstrates:
- Processing e-commerce product descriptions at scale
- Using Pydantic models for structured outputs
- Handling real product data from listings

Real-world application:
- You have 100,000 product listings with messy descriptions
- You need to extract brand, category, price, and attributes
- Traditional regex/NLP fails on the variety of formats
- LLMs handle the variability but you need to process them efficiently
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


# Define the structured output schema
class ProductInfo(BaseModel):
    """Structured product information extracted from listing description."""

    brand: str = Field(description="Brand or manufacturer name")
    category: Literal[
        "Electronics",
        "Clothing",
        "Sports",
        "Kitchen",
        "Home & Garden",
        "Toys & Games",
        "Health & Beauty",
        "Books & Media",
        "Pet Supplies",
        "Food & Beverage",
        "Office Supplies",
        "Tools & Hardware",
        "Baby & Toddler",
        "Other",
    ] = Field(description="Product category")
    subcategory: str = Field(description="More specific product type")
    price_usd: float | None = Field(description="Price in USD if mentioned, null otherwise")
    key_features: list[str] = Field(description="List of 3-5 key product features")
    target_audience: str = Field(description="Who this product is for")


# Sample product data (replace with your actual data source)
SAMPLE_PRODUCTS = [
    {
        "id": "B08N5WRWNW",
        "title": "Sony WH-1000XM4 Wireless Premium Noise Canceling Headphones",
        "description": """
        Sony WH-1000XM4 Wireless Premium Noise Canceling Overhead Headphones
        with Mic for Phone-Call and Alexa Voice Control, Black. Industry-leading noise canceling
        with Dual Noise Sensor technology. Next-level music with Edge-AI, co-developed with
        Sony Music Studios Tokyo. Up to 30-hour battery life with quick charging (10 min charge
        for 5 hours of playback). Touch Sensor controls to pause play skip tracks and control
        volume. Speak-to-chat technology automatically reduces volume during conversations.
        Superior call quality with precise voice pickup. $348.00
        """,
    },
    {
        "id": "B07FZ8S74R",
        "title": "Echo Dot (3rd Gen) - Smart speaker with Alexa",
        "description": """
        Meet Echo Dot - Our most popular smart speaker with a fabric design.
        It is our most compact smart speaker that fits perfectly into small spaces. Improved
        speaker quality - Better speaker quality than Echo Dot Gen 2 for richer and louder sound.
        Voice control your music - Stream songs from Amazon Music, Apple Music, Spotify, Sirius XM,
        and others. Just ask Alexa to play music, answer questions, read the news, set alarms,
        check the weather, control compatible smart home devices, and more. Price: $29.99
        """,
    },
    {
        "id": "B09V3KXJPB",
        "title": "Apple AirPods Pro (2nd Generation)",
        "description": """
        RICHER AUDIO EXPERIENCE — The Apple-designed H2 chip pushes advanced audio
        performance even further. Low distortion and custom amplifier delivers crisp, clear high
        notes and deep, rich bass in stunning definition.
        NEXT-LEVEL ACTIVE NOISE CANCELLATION — Up to 2x more Active Noise Cancellation
        than the previous AirPods Pro. Adaptive Transparency lets you comfortably hear
        the world around you. CUSTOMIZABLE FIT — Includes four pairs of silicone tips
        (XS, S, M, L) to fit a wide range of ears. MSRP $249
        """,
    },
    {
        "id": "B0BSHF7WHW",
        "title": "Nike Men's Air Max 90 Shoes",
        "description": """
        Nothing as iconic satisfies like the original. The Men's Nike Air Max 90 stays true to its
        OG running roots with the iconic Waffle sole, stitched overlays and classic TPU details.
        Classic colorways like white/black and sport red celebrate your fresh style while Max Air
        cushioning adds comfort to your journey. SIZING: Men's US 8-13. Features: Leather and
        synthetic upper, Foam midsole, Rubber Waffle outsole, Max Air unit in heel.
        Athletic footwear for running, casual wear, streetwear. Great for everyday comfort.
        """,
    },
    {
        "id": "B0C1H26C46",
        "title": "Instant Pot Duo 7-in-1 Electric Pressure Cooker",
        "description": """
        7-IN-1 FUNCTIONALITY: Pressure cook, slow cook, rice cooker, yogurt maker,
        steamer, sauté pan and food warmer. QUICK ONE-TOUCH COOKING: 13 customizable Smart Programs
        for pressure cooking ribs, soups, beans, rice, poultry, yogurt, desserts and more.
        QUICK COOK TIMES: Healthy, flavorful dishes in minutes, not hours. Cooks up to 70% faster
        than traditional cooking methods.
        6 QT capacity - feeds up to 6 people, perfect for families.
        Kitchen appliance, home cooking, meal prep essential. $89.95 retail.
        """,
    },
]

# Rate limits by tier (tier 2 for selected model https://platform.openai.com/docs/models/gpt-5-nano)
# Note: adjust rate limits based on your tier
RPM = 5_000
TPM = 2_000_000


async def main() -> None:
    # Configure provider
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-5-nano",
        request_url="https://api.openai.com/v1/responses",
    )

    # Create the structured output format
    response_format = pydantic_to_openai_response_format(ProductInfo, "responses")

    # Build extraction requests
    requests = []
    for product in SAMPLE_PRODUCTS:
        requests.append(
            {
                "input": (
                    f"Extract structured product information from this e-commerce listing."
                    f"Title: {product['title']}"
                    f"Description: {product['description']}"
                ),
                "text": {"format": response_format},
                "reasoning": {"effort": "minimal"},
                "metadata": {
                    "product_id": product["id"],
                    "original_title": product["title"],
                },
            }
        )

    # Process extraction requests
    results = await process_requests(
        provider=provider,
        requests=requests,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,  # 80% of the limit
            max_tokens_per_minute=TPM * 0.8,  # 80% of the limit
        ),
        logging_level=20,
    )

    # Display results
    print(f"\n{'='*70}")
    print("EXTRACTION RESULTS")
    print(f"{'='*70}")
    print(f"Processed: {results.stats.successful}/{results.stats.total_requests} products")
    print(f"Duration: {results.stats.duration_seconds:.2f}s")
    print(f"Total tokens: {results.stats.total_input_tokens + results.stats.total_output_tokens:,}")
    print(f"{'='*70}\n")

    for result in results.successes:
        product_id = result.metadata["product_id"]
        original_title = result.metadata["original_title"]

        # Parse the structured response
        try:
            # Navigate to the text content in the response
            output_text = (
                result.response.get("output", [{}])[1].get("content", [{}])[0].get("text", "{}")
            )
            extracted = json.loads(output_text)

            print(f"Product: {product_id}")
            print(f"  Original Title: {original_title[:50]}...")
            print(f"  Brand: {extracted.get('brand', 'N/A')}")
            print(f"  Category: {extracted.get('category', 'N/A')}")
            print(f"  Subcategory: {extracted.get('subcategory', 'N/A')}")
            print(f"  Price: ${extracted.get('price_usd', 'N/A')}")
            print(f"  Features: {', '.join(extracted.get('key_features', [])[:3])}...")
            print(f"  Target: {extracted.get('target_audience', 'N/A')}")
            print()
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Product {product_id}: Error parsing response - {e}")

    # Handle any failures
    if results.failures:
        print(f"Failures: {len(results.failures)}")
        for failure in results.failures:
            print(f"  Product {failure.metadata.get('product_id')}: {failure.error}")


if __name__ == "__main__":
    asyncio.run(main())
