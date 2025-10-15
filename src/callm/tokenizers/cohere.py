"""Note: Cohere does not have TPM limits, so we don't need to worry about that here. However, here is an implementation of the tokenizer for reference if they add it later."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests
from tokenizers import Tokenizer


def get_cohere_tokenizer(model: str) -> Tokenizer:
    """
    Download and cache the Cohere tokenizer for a specific model.

    Args:
        model (str): The Cohere model name (e.g., "embed-v4.0")

    Returns:
        Tokenizer: HuggingFace tokenizer instance

    Raises:
        ValueError: If tokenizer cannot be downloaded for the model
    """
    # Cache dir for tokenizers
    cache_dir = Path("~/.cache/cohere_tokenizers").expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f"{model}.json"

    if cache_file.exists():
        with open(cache_file, mode="r", encoding="utf-8") as f:
            tokenizer_json = f.read()
        return Tokenizer.from_str(tokenizer_json)

    # Download tokenizer from Cohere's public storage
    tokenizer_url = (
        f"https://storage.googleapis.com/cohere-public/tokenizers/{model}.json"
    )

    try:
        response = requests.get(tokenizer_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(
            f"Failed to download tokenizer for model '{model}'. "
            f"Make sure the model name is correct. Error: {e}"
        ) from e

    with open(cache_file, mode="w", encoding="utf-8") as f:
        f.write(response.text)

    return Tokenizer.from_str(response.text)


def num_tokens_from_cohere_request(
    request_json: dict[str, Any],
    api_endpoint: str,
    tokenizer: Tokenizer,
) -> int:
    """
    Count the number of tokens in a Cohere API request.

    Currently supports embeddings endpoint only.

    Args:
        request_json (dict[str, Any]): The request payload
        api_endpoint (str): The API endpoint (e.g., "embed")
        tokenizer (Tokenizer): The Cohere tokenizer instance

    Returns:
        int: Estimated number of input tokens

    Raises:
        NotImplementedError: For unsupported endpoints
        TypeError: For invalid input types
    """
    if api_endpoint == "embed":
        inputs = request_json.get("texts", [])

        if not inputs:
            return 0

        total_tokens = 0

        for input_item in inputs:
            # Handle different input formats
            if isinstance(input_item, str):
                # Simple string input
                encoding = tokenizer.encode(input_item, add_special_tokens=False)
                total_tokens += len(encoding.tokens)
            else:
                raise TypeError(
                    f"Unexpected input type: {type(input_item)}. "
                    "Expected string or dict with content array."
                )

        return total_tokens

    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not yet implemented for Cohere provider. '
            "Please submit an issue at https://github.com/milistu/callm/issues."
        )
