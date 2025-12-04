from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from aiohttp import ClientSession

from callm.providers.models import Usage
from callm.utils import api_endpoint_from_url


class BaseProvider(ABC):
    """
    Abstract base class for LLM provider implementations.

    This class provides default implementations for common provider patterns
    while allowing providers to override when they need custom behavior.

    Default implementations provided:
    - Bearer token authentication (build_headers)
    - Standard request sending with model injection (send)
    - Common rate limit detection (is_rate_limited)
    - Endpoint extraction helper (_extract_endpoint)

    Subclasses must implement:
    - estimate_input_tokens: Provider-specific token counting
    - parse_error: Provider-specific error message extraction
    - extract_usage: Provider-specific usage metric extraction

    Attributes:
        name (str): Human-readable provider identifier (e.g., "openai", "deepseek")
        api_key (str): API authentication key
        model (str): Model identifier
        request_url (str): Full API endpoint URL

    Example Implementation:
        >>> class MyProvider(BaseProvider):
        ...     name = "myprovider"
        ...
        ...     def __init__(self, api_key: str, model: str, request_url: str):
        ...         self.api_key = api_key
        ...         self.model = model
        ...         self.request_url = request_url
        ...
        ...     def estimate_input_tokens(self, request_json):
        ...         # Custom token counting logic
        ...         return len(str(request_json))
        ...
        ...     def parse_error(self, payload):
        ...         return payload.get("error", {}).get("message")
        ...
        ...     def extract_usage(self, payload):
        ...         usage = payload.get("usage", {})
        ...         return Usage(
        ...             input_tokens=usage.get("input_tokens", 0),
        ...             output_tokens=usage.get("output_tokens", 0),
        ...             total_tokens=usage.get("total_tokens", 0)
        ...         )
    """

    name: str
    api_key: str
    model: str
    request_url: str

    def build_headers(self) -> dict[str, str]:
        """
        Build authentication headers for API requests.

        Default implementation uses Bearer token authentication, which is
        standard for most LLM APIs (DeepSeek, Cohere, VoyageAI, etc.).

        Providers with different auth schemes (e.g., Azure OpenAI) should
        override this method.

        Returns:
            dict[str, str]: HTTP headers including authentication

        Example:
            >>> provider.build_headers()
            {'Authorization': 'Bearer sk-...', 'Content-Type': 'application/json'}
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def send(
        self,
        session: ClientSession,
        headers: Mapping[str, str],
        request_json: dict[str, Any],
    ) -> tuple[dict[str, Any], Mapping[str, str] | None]:
        """
        Perform the HTTP request to the provider's API.

        Default implementation:
        1. Copies the request payload
        2. Adds model to payload if not present
        3. POSTs to request_url
        4. Returns JSON response and headers

        Most providers can use this default. Providers with special payload
        handling (e.g., OpenAI non-Azure) can override.

        Args:
            session (ClientSession): Aiohttp client session
            headers (Mapping[str, str]): HTTP headers from build_headers()
            request_json (dict[str, Any]): Request payload

        Returns:
            Tuple of (response_payload, response_headers):
            - response_payload (dict[str, Any]): Parsed JSON response
            - response_headers (Optional[Mapping[str, str]]): Response headers

        Raises:
            aiohttp.ClientError: For network/connection errors
            asyncio.TimeoutError: For request timeouts
        """
        payload = dict(request_json)

        # Add model to payload if not present (common pattern)
        if "model" not in payload:
            payload["model"] = self.model

        async with session.post(self.request_url, headers=headers, json=payload) as response:
            data = await response.json()
            return data, response.headers

    def is_rate_limited(
        self,
        payload: dict[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> bool:
        """
        Determine if the response indicates rate limiting.

        Default implementation checks:
        1. HTTP 429 status code in response headers
        2. "rate limit" or "too many requests" in error message

        This works for most providers. Providers with special rate limit
        detection (e.g., checking error types) can override.

        Args:
            payload (dict[str, Any]): API response payload
            headers (Optional[Mapping[str, str]]): Response headers

        Returns:
            bool: True if rate limited, False otherwise

        Example:
            >>> provider.is_rate_limited(
            ...     {"error": {"message": "Rate limit exceeded"}},
            ...     {"status": "429"}
            ... )
            True
        """
        # Check HTTP status code from headers
        if headers:
            status = headers.get("status")
            if status == "429":
                return True

        # Check error message content
        error_msg = (self.parse_error(payload) or "").lower()
        return "rate limit" in error_msg or "too many requests" in error_msg

    def _extract_endpoint(self) -> str:
        """
        Extract the API endpoint type from the request URL.

        This is a helper method for providers that need to determine the
        endpoint type (e.g., "chat/completions", "embeddings") for
        token counting logic.

        Returns:
            str: The endpoint path (e.g., "chat/completions", "embeddings")

        Raises:
            ValueError: If endpoint cannot be extracted from URL

        Example:
            >>> provider.request_url = "https://api.openai.com/v1/chat/completions"
            >>> provider._extract_endpoint()
            'chat/completions'
        """
        try:
            return api_endpoint_from_url(self.request_url)
        except ValueError:
            # Fallback: extract last segment of path
            return self.request_url.rstrip("/").split("/")[-1]

    @abstractmethod
    async def estimate_input_tokens(
        self, request_json: dict[str, Any], session: ClientSession | None = None
    ) -> int:
        """
        Estimate the number of input tokens for rate limit budgeting.

        This method must be implemented by each provider as token counting
        is provider-specific (different tokenizers, different logic).

        For providers with local tokenizers (OpenAI, DeepSeek, etc.), the session
        parameter is ignored and token counting is done locally.

        For providers requiring API-based token counting (Gemini, Claude, etc.), the session
        is used to make an async HTTP call to the token counting endpoint.

        Args:
            request_json (dict[str, Any]): The request payload
            session (Optional[ClientSession]): Aiohttp session for API-based counting.
                                              Required for providers like Gemini.

        Returns:
            int: Estimated number of input tokens (always >= 0)

        Note:
            This only estimates INPUT tokens. Output tokens are not known
            until the response is received.
        """
        ...

    @abstractmethod
    def parse_error(self, payload: dict[str, Any]) -> str | None:
        """
        Extract error message from API response payload.

        This method must be implemented by each provider as error formats
        differ across providers.

        Args:
            payload (dict[str, Any]): The API response payload

        Returns:
            Optional[str]: Error message if present, None if successful response

        Example:
            >>> provider.parse_error({"error": {"message": "Invalid API key"}})
            'Invalid API key'
            >>> provider.parse_error({"id": "chatcmpl-123", "choices": [...]})
            None
        """
        ...

    @abstractmethod
    def extract_usage(
        self, payload: dict[str, Any], estimated_input_tokens: int | None = None
    ) -> Usage | None:
        """
        Extract token usage metrics from a successful API response.

        This method must be implemented by each provider as usage formats
        differ across providers.

        Args:
            payload (dict[str, Any]): The API response payload
            estimated_input_tokens (Optional[int]): Pre-calculated input tokens from
                estimate_input_tokens(). Useful for providers/endpoints that don't
                return usage in the response (e.g., Gemini embeddings).

        Returns:
            Optional[Usage]: Usage object with token counts, or None if unavailable
        """
        ...
