"""
Example: Dataset Translation (NanoBEIR)
Description: Translate benchmark datasets to other languages for multilingual evaluation
Use case: Creating multilingual datasets, localization, cross-lingual IR research
Provider: DeepSeek (best value for high-volume translation)

This example demonstrates:
- Translating HuggingFace datasets at scale
- Separate handling of queries vs corpus (different prompt strategies)
- Test mode for prompt iteration before full runs
- Resumable processing with status checks

Real-world application:
- You want to evaluate retrieval models in your language
- NanoBEIR is in English
- Professional translation of 50K+ texts is expensive
- LLM translation provides fast, affordable dataset creation

IMPORTANT: Translation quality depends heavily on your prompts!
The prompts below are templates - you MUST customize them for your target language.
Test with small samples first (TEST_MODE=True) before running full translation.
"""

import asyncio
import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from callm import RateLimitConfig, process_requests
from callm.providers import DeepSeekProvider

load_dotenv()


# CONFIGURATION - Customize these for your use case

# Set to True to only process 10 examples (for testing prompts)
TEST_MODE = True
TEST_SAMPLE_SIZE = 10

# Target language - change this to your language
TARGET_LANGUAGE = "Serbian"
TARGET_LANGUAGE_CODE = "sr"

# Output directory
OUTPUT_DIR = Path("data/nanobeir_translated")

# DeepSeek rate limits
# Source: https://api-docs.deepseek.com/quick_start/rate_limit
#
# DeepSeek does NOT constrain rate limits!
# Quote: "DeepSeek API does NOT constrain user's rate limit.
#         We will try our best to serve every request."
#
# However, be reasonable with RPM to avoid overloading their servers.
# My suggestion is to use OpenAI's rate limits as a reference.
# GPT-5 Tier 2 rates:
RPM = 5_000  # Conservative estimate
TPM = 1_000_000  # 1M tokens per minute


# =============================================================================
# TRANSLATION PROMPTS - Customize these for your target language!
#
# These prompts are CRITICAL for translation quality. Tips:
# 1. Be specific about the domain (information retrieval, search)
# 2. Specify how to handle technical terms, names, acronyms
# 3. Test extensively with TEST_MODE=True before full runs
# 4. Consider using few-shot examples for better quality
# =============================================================================

QUERY_TRANSLATION_PROMPT = f"""
You are a professional translator specializing in search queries.
Translate the following search query from English to {TARGET_LANGUAGE}.

Guidelines:
- Keep the query natural and how a native speaker would search
- Preserve the search intent exactly
- Keep named entities (people, places, organizations) in their commonly used form
- Keep technical terms that are commonly used in English
- Do NOT add explanations or alternatives - just the translation

Translate this query:
"""

CORPUS_TRANSLATION_PROMPT = f"""
You are a professional translator for technical and scientific documents.
Translate the following text from English to {TARGET_LANGUAGE}.

Guidelines:
- Maintain the original meaning precisely
- Keep the same level of formality and technicality
- Preserve named entities in their commonly used form in {TARGET_LANGUAGE}
- Keep acronyms and translate their expansions if needed
- Maintain paragraph structure
- Do NOT add explanations - just the translation

Translate this text:
"""


# NanoBEIR Dataset Configuration

# Available NanoBEIR datasets (subset of BEIR for faster experimentation)
# Collection: https://huggingface.co/collections/zeta-alpha-ai/nanobeir
NANOBEIR_DATASETS = [
    "zeta-alpha-ai/NanoClimateFEVER",
    "zeta-alpha-ai/NanoDBPedia",
    "zeta-alpha-ai/NanoFEVER",
    "zeta-alpha-ai/NanoFiQA2018",
    "zeta-alpha-ai/NanoHotpotQA",
    "zeta-alpha-ai/NanoMSMARCO",
    "zeta-alpha-ai/NanoNFCorpus",
    "zeta-alpha-ai/NanoNQ",
    "zeta-alpha-ai/NanoQuoraRetrieval",
    "zeta-alpha-ai/NanoSCIDOCS",
    "zeta-alpha-ai/NanoArguAna",
    "zeta-alpha-ai/NanoSciFact",
    "zeta-alpha-ai/NanoToupsecupe2020",
]


def get_dataset_name(dataset_id: str) -> str:
    """Extract short name from dataset ID."""
    # "zeta-alpha-ai/NanoClimateFEVER-corpus" -> "climatefever"
    return dataset_id.split("/")[-1].replace("Nano", "").lower()


def check_translation_status(dataset_name: str) -> dict[str, bool]:
    """Check if dataset subsets are already translated."""
    data_dir = OUTPUT_DIR / dataset_name
    return {
        "queries": (data_dir / "queries_translated.jsonl").exists(),
        "corpus": (data_dir / "corpus_translated.jsonl").exists(),
    }


async def translate_subset(
    dataset_id: str,
    subset: str,  # "queries" or "corpus"
    provider: DeepSeekProvider,
) -> None:
    """Translate a single dataset subset (queries or corpus)."""
    dataset_name = get_dataset_name(dataset_id)
    data_dir = OUTPUT_DIR / dataset_name
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset from HuggingFace
    print(f"  Loading {subset} from HuggingFace...")
    dataset = load_dataset(dataset_id, name=subset, split="train")

    # Apply test mode limit
    if TEST_MODE:
        dataset = dataset.select(range(min(TEST_SAMPLE_SIZE, len(dataset))))
        print(f"  TEST MODE: Using only {len(dataset)} samples")

    # Select prompt based on subset type
    prompt = QUERY_TRANSLATION_PROMPT if subset == "queries" else CORPUS_TRANSLATION_PROMPT

    # Build translation requests
    requests = [
        {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": row["text"]},
            ],
            "temperature": 1.3,  # Suggested by DeepSeek
            "metadata": {"id": row["_id"]},
        }
        for row in dataset
    ]

    print(f"  Translating {len(requests)} {subset}...")

    # Process translations
    results = await process_requests(
        provider=provider,
        requests=requests,
        rate_limit=RateLimitConfig(
            max_requests_per_minute=RPM * 0.8,
            max_tokens_per_minute=TPM * 0.8,
        ),
        output_path=str(data_dir / f"{subset}_translated.jsonl"),
        logging_level=20,
    )

    print(f"Number of successful requests: {results.successes}")
    print(f"Number of failed requests: {results.failures}")


async def translate_dataset(dataset_id: str, provider: DeepSeekProvider) -> None:
    """Translate both queries and corpus for a dataset."""
    dataset_name = get_dataset_name(dataset_id)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    status = check_translation_status(dataset_name)

    # Translate queries
    if status["queries"] and not TEST_MODE:
        print("  ✓ Queries already translated")
    else:
        await translate_subset(dataset_id, "queries", provider)

    # Translate corpus
    if status["corpus"] and not TEST_MODE:
        print("  ✓ Corpus already translated")
    else:
        await translate_subset(dataset_id, "corpus", provider)


async def main() -> None:
    """Main translation pipeline."""
    print("=" * 60)
    print(f"NanoBEIR Translation to {TARGET_LANGUAGE}")
    print("=" * 60)

    if TEST_MODE:
        print(f"\n⚠️  TEST MODE ENABLED - Only {TEST_SAMPLE_SIZE} samples per subset")
        print("   Set TEST_MODE = False for full translation\n")

    # Initialize provider
    provider = DeepSeekProvider(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model="DeepSeek-V3.2",  # Naming from DeepSeek HuggingFace
        request_url="https://api.deepseek.com/chat/completions",
    )

    # Show translation status
    print("\nTranslation Status:")
    print("-" * 60)
    to_translate = []
    for dataset_id in NANOBEIR_DATASETS:
        name = get_dataset_name(dataset_id)
        status = check_translation_status(name)
        q = "✓" if status["queries"] else "○"
        c = "✓" if status["corpus"] else "○"
        if not status["queries"] or not status["corpus"]:
            to_translate.append(dataset_id)
        print(f"  {name:30} Queries: {q}  Corpus: {c}")
    print("-" * 60)

    # Process each dataset
    for i, dataset_id in tqdm(enumerate(to_translate, 1), desc="Datasets", total=len(to_translate)):
        print(f"\n[{i}/{len(to_translate)}]")
        await translate_dataset(dataset_id, provider)

        # In test mode, only do one dataset
        # if TEST_MODE:
        #     print("\n⚠️  TEST MODE: Stopping after first dataset")
        #     print("   Review the output, adjust prompts, then set TEST_MODE = False")
        #     break

    print("\n" + "=" * 60)
    print("Translation complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
