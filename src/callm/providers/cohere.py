from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from aiohttp import ClientSession

from callm.providers.base import Provider
from callm.providers.models import Usage
from callm.tokenizers.cohere import get_cohere_tokenizer, num_tokens_from_cohere_request
from callm.utils import api_endpoint_from_url


class CohereProvider(Provider):
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
            raise ValueError(
                f"Failed to initialize tokenizer for model '{model}': {e}"
            ) from e

    def build_headers(self) -> dict[str, str]:
        """
        Build authentication headers for Cohere API.

        Cohere uses Bearer token authentication.

        Returns:
            dict[str, str]: Headers with Authorization and Content-Type
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

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
        # Extract endpoint from URL (e.g., "v2/embed" -> "embed")
        try:
            endpoint = api_endpoint_from_url(self.request_url)
        except ValueError:
            # Fallback: try to extract just the last part
            endpoint = self.request_url.rstrip("/").split("/")[-1]

        return num_tokens_from_cohere_request(request_json, endpoint, self.tokenizer)

    async def send(
        self,
        session: ClientSession,
        headers: Mapping[str, str],
        request_json: dict[str, Any],
    ) -> Tuple[dict[str, Any], Optional[Mapping[str, str]]]:
        """
        Send request to Cohere API.

        Automatically adds model to payload if not present.

        Args:
            session (ClientSession): Aiohttp session
            headers (Mapping[str, str]): Request headers
            request_json (dict[str, Any]): Request payload

        Returns:
            Tuple[dict[str, Any], Optional[Mapping[str, str]]]: Response data and headers
        """
        payload = dict(request_json)

        # Add model to payload if not present
        if "model" not in payload:
            payload["model"] = self.model

        async with session.post(
            self.request_url, headers=headers, json=payload
        ) as response:
            data = await response.json()
            return data, response.headers

    def parse_error(self, payload: dict[str, Any]) -> Optional[str]:
        """
        Parse error from Cohere API response.

        Cohere error format: {"message": "error description"}

        Args:
            payload (dict[str, Any]): API response payload

        Returns:
            Optional[str]: Error message if present, None otherwise
        """
        # Check for standard error message field
        if "message" in payload:
            return str(payload["message"])

        # Check for explicit error field
        error = payload.get("error")
        if error:
            if isinstance(error, dict):
                return str(error.get("message") or error)
            return str(error)

        return None

    def is_rate_limited(
        self,
        payload: dict[str, Any],
        headers: Optional[Mapping[str, str]] = None,
    ) -> bool:
        """
        Detect rate limiting from Cohere API response.

        Checks for:
        - "rate limit" or "too many requests" in error message
        - HTTP 429 status code from headers

        Args:
            payload (dict[str, Any]): API response payload
            headers (Optional[Mapping[str, str]]): Response headers

        Returns:
            bool: True if rate limited, False otherwise
        """
        # Check status code from headers
        if headers:
            status = headers.get("status")
            if status == "429":
                return True

        # Check error message
        error_msg = (self.parse_error(payload) or "").lower()
        return "rate limit" in error_msg or "too many requests" in error_msg

    def extract_usage(self, payload: dict[str, Any]) -> Optional[Usage]:
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
