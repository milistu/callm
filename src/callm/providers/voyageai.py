from __future__ import annotations

from typing import Any

from aiohttp import ClientSession

from callm.providers.base import BaseProvider
from callm.providers.models import Usage
from callm.tokenizers.voyageai import (
    get_voyageai_tokenizer,
    num_tokens_from_voyageai_request,
)


class VoyageAIProvider(BaseProvider):
    """
    Provider implementation for Voyage AI API.

    Currently supports:
    - Embeddings (v1/embeddings endpoint)

    Future support planned for:
    - Reranking

    Attributes:
        name (str): Always "voyageai"
        api_key (str): Voyage AI API key
        model (str): Model identifier (e.g., "voyage-3.5", "voyage-3-large")
        request_url (str): Full API endpoint URL
        tokenizer (Tokenizer): Voyage AI tokenizer for the specified model

    Example:
        >>> provider = VoyageAIProvider(
        ...     api_key="your-voyageai-api-key",
        ...     model="voyage-3.5",
        ...     request_url="https://api.voyageai.com/v1/embeddings"
        ... )
    """

    name = "voyageai"

    def __init__(self, api_key: str, model: str, request_url: str) -> None:
        """
        Initialize Voyage AI provider.

        Args:
            api_key (str): Voyage AI API key
            model (str): Model name for tokenization (e.g., "voyage-3.5")
            request_url (str): Full API endpoint URL

        Raises:
            ValueError: If tokenizer cannot be loaded for the model
        """
        self.api_key = api_key
        self.model = model
        self.request_url = request_url

        # Download tokenizer from HuggingFace
        try:
            self.tokenizer = get_voyageai_tokenizer(model)
        except Exception as e:
            raise ValueError(f"Failed to initialize tokenizer for model '{model}': {e}") from e

    async def estimate_input_tokens(
        self, request_json: dict[str, Any], session: ClientSession | None = None
    ) -> int:
        """
        Estimate input tokens using Voyage AI's tokenizer.
        Note: session parameter is unused - Voyage AI uses local tokenizer.

        Supports the embeddings endpoint with string or list inputs.

        Args:
            request_json (dict[str, Any]): The request payload
            session (Optional[ClientSession]): Aiohttp session for API-based counting.

        Returns:
            int: Estimated number of input tokens
        """
        endpoint = self._extract_endpoint()
        return num_tokens_from_voyageai_request(request_json, endpoint, self.tokenizer)

    def parse_error(self, payload: dict[str, Any]) -> str | None:
        """
        Parse error from Voyage AI API response.

        Voyage AI returns standard HTTP error codes with structured error information.
        See: https://docs.voyageai.com/docs/error-codes

        Common error codes:
        - 400: Invalid Request (invalid JSON, wrong parameter types, batch too large)
        - 401: Unauthorized (invalid API key)
        - 403: Forbidden (IP address blocked)
        - 429: Rate Limit Exceeded
        - 500: Server Error (unexpected server issue)
        - 502/503/504: Service Unavailable (high traffic)

        Error response format (from API docs):
        {
            "detail": "error description string"
        }

        Source: https://docs.voyageai.com/reference/embeddings-api

        Args:
            payload (dict[str, Any]): API response payload

        Returns:
            Optional[str]: Error message if present, None otherwise
        """
        # Check for detail field (actual VoyageAI format per their docs)
        detail = payload.get("detail")
        if detail:
            return str(detail)

        # Fallback: check for error field (in case they have multiple formats)
        error = payload.get("error")
        if error:
            if isinstance(error, dict):
                message = error.get("message", "")
                error_type = error.get("type", "")
                code = error.get("code", "")

                # Build comprehensive error message
                parts = []
                if message:
                    parts.append(str(message))
                if error_type:
                    parts.append(f"Type: {error_type}")
                if code:
                    parts.append(f"Code: {code}")

                return " | ".join(parts) if parts else str(error)
            return str(error)

        return None

    def extract_usage(
        self, payload: dict[str, Any], estimated_input_tokens: int | None = None
    ) -> Usage | None:
        """
        Extract token usage from Voyage AI API response.

        Voyage AI embeddings API returns usage in standard format:
        {
            "data": [...],
            "model": "voyage-3.5",
            "usage": {
                "total_tokens": 123
            }
        }

        For embeddings, only total_tokens is provided (no input/output split).

        Args:
            payload (dict[str, Any]): API response payload

        Returns:
            Optional[Usage]: Usage object with token counts, or None if unavailable
        """
        # Check for error first
        if self.parse_error(payload):
            return None

        # Extract usage from response
        usage = payload.get("usage", {})
        if not usage:
            return None

        total_tokens = int(usage.get("total_tokens", 0))

        # For embeddings, all tokens are "input" tokens
        return Usage(
            input_tokens=total_tokens,
            output_tokens=0,
            total_tokens=total_tokens,
        )
