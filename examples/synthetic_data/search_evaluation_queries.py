"""
Example: Search Evaluation Query Generation
Description: Generate keyword and semantic queries for evaluating search systems
Use case: Creating search evaluation datasets, testing retrieval systems
Provider: OpenAI

This example demonstrates:
- Generating both keyword-style and natural language queries
- Creating ground-truth query-document pairs for evaluation
- Building test sets for search/retrieval system benchmarking

Real-world application:
- You're building a product search system
- You need queries to evaluate retrieval quality (NDCG, MAP, MRR)
- Manual query creation or real user queries are hard to get
- LLMs can generate realistic user queries at scale
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


class SearchQueries(BaseModel):
    """Generated search queries for a product."""

    keyword_query: str = Field(description="Keyword-style query (how users type in search boxes)")
    semantic_query: str = Field(description="Natural language query (how users ask questions)")
    edge_case_query: str = Field(description="Tricky query (misspellings, synonyms, partial info)")


PRODUCTS = [
    {
        "id": "prod_001",
        "name": "Sony WH-1000XM5 Wireless Headphones",
        "category": "Electronics > Audio > Headphones",
        "description": (
            "Premium noise-cancelling wireless headphones with 30-hour battery, "
            "touch controls, and speak-to-chat technology."
        ),
        "attributes": {"brand": "Sony", "color": "Black", "type": "Over-ear", "wireless": True},
    },
    {
        "id": "prod_002",
        "name": "Atomic Habits by James Clear",
        "category": "Books > Self-Help > Personal Development",
        "description": (
            "An Easy & Proven Way to Build Good Habits & Break Bad Ones. "
            "#1 New York Times bestseller with over 10 million copies sold."
        ),
        "attributes": {"author": "James Clear", "format": "Paperback", "pages": 320},
    },
    {
        "id": "prod_003",
        "name": "Instant Pot Duo 7-in-1 Pressure Cooker",
        "category": "Home & Kitchen > Appliances > Cookers",
        "description": (
            "7-in-1 functionality: pressure cooker, slow cooker, rice cooker, steamer, "
            "sauté, yogurt maker, and warmer. 6 quart capacity."
        ),
        "attributes": {"brand": "Instant Pot", "capacity": "6 qt", "functions": 7},
    },
    {
        "id": "prod_004",
        "name": "Patagonia Better Sweater Fleece Jacket",
        "category": "Clothing > Outerwear > Fleece",
        "description": (
            "Classic fleece jacket made from recycled polyester. "
            "Full-zip with stand-up collar. Fair Trade Certified."
        ),
        "attributes": {"brand": "Patagonia", "material": "Recycled Polyester", "gender": "Unisex"},
    },
    {
        "id": "prod_005",
        "name": "Apple AirPods Pro (2nd Generation)",
        "category": "Electronics > Audio > Earbuds",
        "description": (
            "Active noise cancellation, adaptive transparency, personalized spatial audio. "
            "Up to 6 hours listening time."
        ),
        "attributes": {"brand": "Apple", "type": "In-ear", "noise_cancelling": True},
    },
]


GENERATION_PROMPT = """You are helping create a search evaluation dataset.

Given this product, generate realistic search queries that users would type to find it.
Generate one query per type.

## Product Information
Name: {name}
Category: {category}
Description: {description}
Attributes: {attributes}

## Query Types to Generate

**Keyword query**: Short, typical search box input
- Example: "sony wireless headphones", "noise cancelling earbuds"

**Semantic query**: Natural language query, keyword query transformed into a more natural query
- Example: "headphones that block out airplane noise", "best book about building habits"

**Edge case query**: Challenging but realistic queries
- Common misspellings, synonyms, partial product names
- Example: "airpod pros", "instant pot pressure cooker thing"

"""

# Rate limits by tier (tier 2 for selected model https://platform.openai.com/docs/models/gpt-5-nano)
# Note: adjust rate limits based on your tier
RPM = 5_000
TPM = 2_000_000


async def main() -> None:
    """Generate search evaluation queries for products."""
    provider = OpenAIProvider(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-5-nano",
        request_url="https://api.openai.com/v1/responses",
    )

    response_format = pydantic_to_openai_response_format(SearchQueries, "responses")

    # Build requests
    requests = [
        {
            "input": GENERATION_PROMPT.format(
                name=p["name"],
                category=p["category"],
                description=p["description"],
                attributes=json.dumps(p["attributes"]),
            ),
            "text": {"format": response_format},
            "metadata": {"product_id": p["id"], "product_name": p["name"]},
        }
        for p in PRODUCTS
    ]

    print(f"Generating search queries for {len(PRODUCTS)} products...")

    results = await process_requests(
        provider=provider,
        requests=requests,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,  # 80% of the limit
            max_tokens_per_minute=TPM * 0.8,  # 80% of the limit
        ),
    )

    # Build evaluation dataset
    evaluation_set = []

    print(f"\n{'='*60}")
    print("GENERATED SEARCH QUERIES")
    print(f"{'='*60}\n")

    for result in results.successes:
        try:
            output_text = (
                result.response.get("output", [{}])[1].get("content", [{}])[0].get("text", "{}")
            )
            queries = json.loads(output_text)

            product_id = result.metadata["product_id"]
            product_name = result.metadata["product_name"]

            print(f"Product: {product_name}")
            print(f"  Keyword: {queries['keyword_query']}")
            print(f"  Semantic: {queries['semantic_query']}")
            print(f"  Edge case: {queries['edge_case_query']}")
            print()

            # Create query-document pairs for evaluation
            evaluation_set.append(
                {"query": queries["keyword_query"], "relevant_doc": product_id, "type": "keyword"}
            )
            evaluation_set.append(
                {"query": queries["semantic_query"], "relevant_doc": product_id, "type": "semantic"}
            )
            evaluation_set.append(
                {
                    "query": queries["edge_case_query"],
                    "relevant_doc": product_id,
                    "type": "edge_case",
                }
            )

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error parsing {result.metadata['product_id']}: {e}")

    # Save evaluation dataset
    output_file = "data/search_evaluation_queries.jsonl"
    os.makedirs("data", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in evaluation_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(evaluation_set)} query-document pairs to: {output_file}")
    print("\nUse these pairs to evaluate your search system:")
    print("  - For each query, check if relevant_doc appears in top-K results")
    print("  - Calculate precision@K, recall@K, MRR, etc.")


if __name__ == "__main__":
    asyncio.run(main())
