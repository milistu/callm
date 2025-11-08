from __future__ import annotations

from typing import Any

from tokenizers import Tokenizer


def get_voyageai_tokenizer(model: str, namespace: str = "voyageai") -> Tokenizer:
    """
    Download and cache the Voyage AI tokenizer for a specific model.

    Args:
        model (str): The Voyage AI model name (e.g., "voyage-3.5", "voyage-3-large")
        namespace (str): HuggingFace organization/namespace (e.g., "voyageai")

    Returns:
        Tokenizer: HuggingFace tokenizer instance

    Raises:
        ValueError: If tokenizer cannot be downloaded for the model
    """
    try:
        tokenizer = Tokenizer.from_pretrained(f"{namespace}/{model}")
        return tokenizer
    except Exception as e:
        raise ValueError(f"Failed to initialize tokenizer for model '{model}': {e}") from e


def num_tokens_from_voyageai_request(
    request_json: dict[str, Any], api_endpoint: str, tokenizer: Tokenizer
) -> int:
    """
    Count the number of tokens in a Voyage AI API request.

    Currently supports embeddings endpoint only.

    Args:
        request_json (dict[str, Any]): The request payload
        api_endpoint (str): The API endpoint (e.g., "embeddings")
        tokenizer (Tokenizer): The Voyage AI tokenizer instance

    Returns:
        int: Estimated number of input tokens

    Raises:
        NotImplementedError: For unsupported endpoints
        TypeError: For invalid input types
    """
    if api_endpoint == "embeddings":
        # Voyage AI uses "input" field (can be string or list)
        input_data = request_json.get("input", [])

        if not input_data:
            return 0

        if isinstance(input_data, str):
            # Single string input
            num_tokens = len(tokenizer.encode(input_data, add_special_tokens=False))
        elif isinstance(input_data, list):
            # List of strings
            num_tokens = sum(
                [len(tokenizer.encode(i, add_special_tokens=False)) for i in input_data]
            )
        else:
            raise TypeError(
                f"Unexpected input type: {type(input_data)}. " "Expected string or list of strings."
            )

        return num_tokens

    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not yet implemented for Voyage AI provider. '
            "Please submit an issue at https://github.com/milistu/callm/issues."
        )
