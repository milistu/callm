"""
Example: Semantic Search Index Preparation
Description: Prepare a semantic search index using Voyage AI embeddings
Use case: Building search systems, e-commerce product search, content discovery
Provider: Voyage AI

This example demonstrates:
- Using Voyage AI embeddings
- Processing product catalog for semantic search
- File-based output for large datasets

Real-world application:
- You're building product search for an e-commerce site
- Keyword search misses relevant products ("cozy blanket" vs "warm throw")
- Semantic search understands intent and finds related products
- You need to embed your entire 100K - 1M product catalog efficiently
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from callm import RateLimitConfig, process_requests
from callm.providers import VoyageAIProvider

load_dotenv()

# Sample product catalog (replace with your actual product data)
SAMPLE_PRODUCTS = [
    {
        "id": "SKU-001",
        "name": "Cozy Fleece Throw Blanket",
        "category": "Home & Living",
        "description": "Super soft microfiber fleece blanket, perfect for movie nights. "
        "50x60 inches, machine washable. Available in 12 colors.",
        "price": 29.99,
        "tags": ["blanket", "fleece", "cozy", "soft", "home decor"],
    },
    {
        "id": "SKU-002",
        "name": "Ergonomic Office Chair",
        "category": "Furniture",
        "description": "Adjustable lumbar support, breathable mesh back, 4D armrests. "
        "Supports up to 300 lbs. Perfect for long work sessions.",
        "price": 349.99,
        "tags": ["office", "chair", "ergonomic", "work from home"],
    },
    {
        "id": "SKU-003",
        "name": "Wireless Noise-Canceling Earbuds",
        "category": "Electronics",
        "description": "Active noise cancellation, 8-hour battery life, IPX5 water resistant. "
        "Touch controls, Bluetooth 5.2, includes wireless charging case.",
        "price": 149.99,
        "tags": ["earbuds", "wireless", "noise canceling", "audio"],
    },
    {
        "id": "SKU-004",
        "name": "Stainless Steel Water Bottle",
        "category": "Sports & Outdoors",
        "description": "Double-wall vacuum insulated, keeps drinks cold 24 hours or hot 12 hours. "
        "32 oz capacity, BPA-free, leak-proof lid.",
        "price": 34.99,
        "tags": ["water bottle", "insulated", "eco-friendly", "hydration"],
    },
    {
        "id": "SKU-005",
        "name": "Natural Bamboo Cutting Board Set",
        "category": "Kitchen",
        "description": "Set of 3 boards in different sizes. Antimicrobial bamboo, "
        "juice groove, easy grip handles. Gentle on knife edges.",
        "price": 39.99,
        "tags": ["cutting board", "bamboo", "kitchen", "eco-friendly"],
    },
    {
        "id": "SKU-006",
        "name": "LED Desk Lamp with Wireless Charger",
        "category": "Home Office",
        "description": "5 brightness levels, 3 color temperatures. Built-in 10W wireless charger "
        "for phones. USB port, touch controls, foldable design.",
        "price": 59.99,
        "tags": ["desk lamp", "LED", "wireless charger", "office"],
    },
    {
        "id": "SKU-007",
        "name": "Yoga Mat with Alignment Lines",
        "category": "Fitness",
        "description": "6mm thick TPE material, non-slip surface. Alignment markers for proper "
        "positioning. 72x26 inches, includes carrying strap.",
        "price": 44.99,
        "tags": ["yoga mat", "fitness", "exercise", "wellness"],
    },
    {
        "id": "SKU-008",
        "name": "Smart WiFi Power Strip",
        "category": "Smart Home",
        "description": "4 smart outlets + 4 USB ports. Voice control via Alexa and Google. "
        "Individual outlet control, surge protection, energy monitoring.",
        "price": 34.99,
        "tags": ["smart home", "power strip", "wifi", "voice control"],
    },
    {
        "id": "SKU-009",
        "name": "Ceramic Coffee Mug Set",
        "category": "Kitchen",
        "description": "Set of 4 handcrafted ceramic mugs, 16 oz each. Microwave and dishwasher "
        "safe. Modern minimalist design in earth tones.",
        "price": 42.99,
        "tags": ["coffee mug", "ceramic", "kitchen", "gift set"],
    },
    {
        "id": "SKU-010",
        "name": "Portable Bluetooth Speaker",
        "category": "Electronics",
        "description": "360° sound, 20-hour playtime, IPX7 waterproof. Built-in microphone "
        "for calls. True wireless stereo pairing with second speaker.",
        "price": 79.99,
        "tags": ["bluetooth speaker", "portable", "waterproof", "outdoor"],
    },
]


def create_searchable_text(product: dict) -> str:
    """
    Combine product fields into a single searchable text.

    This is a common pattern for semantic search - combine relevant fields
    so the embedding captures all product information.
    """
    parts = [
        product["name"],
        product["category"],
        product["description"],
        ", ".join(product["tags"]),
    ]
    return " | ".join(parts)


# Voyage AI rate limits (Basic Tier 1)
# Source: https://docs.voyageai.com/docs/rate-limits
# Using voyage-3-lite limits (adjust based on your model and tier)
TPM = 16_000_000
RPM = 2_000


async def main() -> None:
    # Configure Voyage AI provider
    provider = VoyageAIProvider(
        api_key=os.getenv("VOYAGEAI_API_KEY"),
        model="voyage-3.5-lite",
        request_url="https://api.voyageai.com/v1/embeddings",
    )

    # Build embedding requests with combined searchable text
    requests = []
    for product in SAMPLE_PRODUCTS:
        searchable_text = create_searchable_text(product)
        requests.append(
            {
                "model": "voyage-3.5-lite",
                "input": searchable_text,
                "input_type": "document",  # Optimize for document embedding
                "metadata": {"product_id": product["id"]},
            }
        )

    output_path = Path("data/voyage_embeddings.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process embedding requests
    results = await process_requests(
        provider=provider,
        requests=requests,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,  # 80% of 2000
            max_tokens_per_minute=TPM * 0.8,  # 80% of 16M
        ),
        output_path=str(output_path),
        logging_level=20,
    )

    print(f"Number of successful requests: {results.successes}")
    print(
        "Embeddings are saved to file and memory is not used. "
        "That is why we do not return embeddings when output_path is provided"
    )


if __name__ == "__main__":
    asyncio.run(main())
