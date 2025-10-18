from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from aiohttp import ClientSession

from callm.providers.base import Provider
from callm.providers.models import Usage
from callm.tokenizers.voyageai import (
    get_voyageai_tokenizer,
    num_tokens_from_voyageai_request,
)
from callm.utils import api_endpoint_from_url


class VoyageAIProvider(Provider):
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
            raise ValueError(
                f"Failed to initialize tokenizer for model '{model}': {e}"
            ) from e

    def build_headers(self) -> dict[str, str]:
        """
        Build authentication headers for Voyage AI API.

        Voyage AI uses Bearer token authentication.

        Returns:
            dict[str, str]: Headers with Authorization and Content-Type
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def estimate_input_tokens(self, request_json: dict[str, Any]) -> int:
        """
        Estimate input tokens using Voyage AI's tokenizer.

        Supports the embeddings endpoint with string or list inputs.

        Args:
            request_json (dict[str, Any]): The request payload

        Returns:
            int: Estimated number of input tokens
        """
        # Extract endpoint from URL (e.g., "v1/embeddings" -> "embeddings")
        try:
            endpoint = api_endpoint_from_url(self.request_url)
        except ValueError:
            # Fallback: try to extract just the last part
            endpoint = self.request_url.rstrip("/").split("/")[-1]

        return num_tokens_from_voyageai_request(request_json, endpoint, self.tokenizer)

    async def send(
        self,
        session: ClientSession,
        headers: Mapping[str, str],
        request_json: dict[str, Any],
    ) -> Tuple[dict[str, Any], Optional[Mapping[str, str]]]:
        """
        Send request to Voyage AI API.

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

        Error response format:
        {
            "error": {
                "message": "error description",
                "type": "error_type",
                "code": "error_code"
            }
        }

        Args:
            payload (dict[str, Any]): API response payload

        Returns:
            Optional[str]: Error message if present, None otherwise
        """
        # Check for error field
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

    def is_rate_limited(
        self,
        payload: dict[str, Any],
        headers: Optional[Mapping[str, str]] = None,
    ) -> bool:
        """
        Detect rate limiting from Voyage AI API response.

        Checks for:
        - "rate_limit_exceeded" error type
        - "rate limit" in error message
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

        # Check error type and message
        error = payload.get("error", {})
        if isinstance(error, dict):
            # Check error type
            if error.get("type") == "rate_limit_exceeded":
                return True

            # Check error message
            message = str(error.get("message", "")).lower()
            if "rate limit" in message or "too many requests" in message:
                return True

        return False

    def extract_usage(self, payload: dict[str, Any]) -> Optional[Usage]:
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
