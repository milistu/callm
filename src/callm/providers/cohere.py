from __future__ import annotations

from typing import Any

from callm.providers.base import BaseProvider
from callm.providers.models import Usage
from callm.tokenizers.cohere import get_cohere_tokenizer, num_tokens_from_cohere_request


class CohereProvider(BaseProvider):
    """
    Provider implementation for Cohere API.

    Currently supports:
    - Embeddings (v2/embed endpoint)

    Future support planned for:
    - Chat completions
    - Reranking

    Attributes:
        name (str): Always "cohere"
        api_key (str): Cohere API key
        model (str): Model identifier (e.g., "embed-v4.0", "embed-english-v3.0")
        request_url (str): Full API endpoint URL
        tokenizer (Tokenizer): Cohere tokenizer for the specified model

    Example:
        >>> provider = CohereProvider(
        ...     api_key="your-cohere-api-key",
        ...     model="embed-v4.0",
        ...     request_url="https://api.cohere.com/v2/embed"
        ... )
    """

    name = "cohere"

    def __init__(self, api_key: str, model: str, request_url: str) -> None:
        """
        Initialize Cohere provider.

        Args:
            api_key (str): Cohere API key
            model (str): Model name for tokenization (e.g., "embed-v4.0")
            request_url (str): Full API endpoint URL

        Raises:
            ValueError: If tokenizer cannot be loaded for the model
        """
        self.api_key = api_key
        self.model = model
        self.request_url = request_url

        # Download and cache tokenizer
        try:
            self.tokenizer = get_cohere_tokenizer(model)
        except Exception as e:
            raise ValueError(f"Failed to initialize tokenizer for model '{model}': {e}") from e

    def estimate_input_tokens(self, request_json: dict[str, Any]) -> int:
        """
        Estimate input tokens using Cohere's tokenizer.

        Supports the embed endpoint with various input formats:
        - Simple string array
        - Structured content with text/image components

        Args:
            request_json (dict[str, Any]): The request payload

        Returns:
            int: Estimated number of input tokens
        """
        endpoint = self._extract_endpoint()
        return num_tokens_from_cohere_request(request_json, endpoint, self.tokenizer)

    def parse_error(self, payload: dict[str, Any]) -> str | None:
        """
        Parse error from Cohere API response.

        Cohere error format (per official docs):
        {
            "id": "string",
            "message": "string"
        }

        All error status codes (400, 401, 402, 404, 429, 499, 500) use this format.

        See: https://docs.cohere.com/reference/errors

        Args:
            payload (dict[str, Any]): API response payload

        Returns:
            Optional[str]: Error message if present, None otherwise
        """
        # Check for standard error message field (per Cohere docs)
        if "message" in payload:
            return str(payload["message"])

        # Fallback: check for error field (defensive, not in official docs)
        error = payload.get("error")
        if error:
            if isinstance(error, dict):
                return str(error.get("message") or error)
            return str(error)

        return None

    def extract_usage(self, payload: dict[str, Any]) -> Usage | None:
        """
        Extract token usage from Cohere API response.

        Cohere embed v2 API returns usage in meta.tokens:
        {
            "meta": {
                "tokens": {
                    "input_tokens": 123,
                    "output_tokens": 0
                }
            }
        }

        For embeddings, output_tokens is typically 0 or not present.

        Args:
            payload (dict[str, Any]): API response payload

        Returns:
            Optional[Usage]: Usage object with token counts, or None if unavailable
        """
        # Check for error first
        if self.parse_error(payload):
            return None

        # Extract from meta.tokens
        meta = payload.get("meta", {})
        billed_units = meta.get("billed_units", {})

        if not billed_units:
            return None

        input_tokens = int(billed_units.get("input_tokens", 0))
        output_tokens = int(billed_units.get("output_tokens", 0))
        total_tokens = input_tokens + output_tokens

        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
