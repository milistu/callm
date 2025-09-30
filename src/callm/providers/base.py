from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Tuple

from aiohttp import ClientSession

from callm.providers.models import Usage


class Provider(Protocol):
    """
    Protocol defining the interface for LLM provider adapters.

    This protocol allows the core engine to work uniformly across different
    LLM providers (OpenAI, Anthropic, Gemini, etc.) without provider-specific
    logic. Each provider implementation handles the details of authentication,
    token counting, request formatting, and response parsing.

    The protocol enforces a contract that ensures:
    - Consistent authentication via headers
    - Token estimation for rate limit budgeting
    - Standardized error handling and rate limit detection
    - Usage metrics extraction when available

    Attributes:
        name (str): Human-readable provider identifier (e.g., "openai", "anthropic")
        request_url (str): Full API endpoint URL for making requests

    Example Implementation:
        >>> class MyProvider:
        ...     name = "myprovider"
        ...
        ...     def __init__(self, api_key: str, request_url: str):
        ...         self.api_key = api_key
        ...         self.request_url = request_url
        ...
        ...     def build_headers(self) -> dict[str, str]:
        ...         return {"Authorization": f"Bearer {self.api_key}"}
        ...
        ...     # ... implement other methods
    """

    name: str
    request_url: str

    def build_headers(self) -> dict[str, str]:
        """
        Build authentication and required headers for API requests.

        This method constructs the HTTP headers needed for authenticating
        with the provider's API. Different providers use different auth
        schemes (Bearer tokens, API keys, custom headers).

        Returns:
            dict[str, str]: Dictionary of HTTP headers including authentication credentials

        Example:
            >>> provider.build_headers()
            {'Authorization': 'Bearer sk-...'}
        """
        ...

    def estimate_input_tokens(self, request_json: dict[str, Any]) -> int:
        """
        Estimate the number of input tokens for rate limit budgeting.

        This estimate is used to enforce TPM (tokens per minute) limits
        before sending requests. It should be conservative (slightly
        over-estimate) to avoid exceeding rate limits.

        For different providers you may use different methods to estimate the number of input tokens:
        - Dedicated token counting API
        - Tiktoken for OpenAI-compatible endpoints
        - Provider-specific tokenizer libraries

        Args:
            request_json (dict[str, Any]): The request payload dictionary

        Returns:
            int: Estimated number of input tokens (always >= 0)

        Note:
            This only estimates INPUT tokens. Output tokens are not known
            until the response is received and don't count toward rate limits.
        """
        ...

    async def send(
        self,
        session: ClientSession,
        headers: Mapping[str, str],
        request_json: dict[str, Any],
    ) -> Tuple[dict[str, Any], Optional[Mapping[str, str]]]:
        """
        Perform the HTTP request to the provider's API.

        This method executes the actual API call and returns both the
        response payload and headers. Response headers are optional but
        useful for header-based rate limit detection.

        Args:
            session (ClientSession): Aiohttp client session for making the request
            headers (Mapping[str, str]): HTTP headers including authentication from build_headers()
            request_json (dict[str, Any]): Request payload (will be sent as JSON body)

        Returns:
            Tuple of (response_payload, response_headers):
            - response_payload (dict[str, Any]): Parsed JSON response as dictionary
            - response_headers (Optional[Mapping[str, str]]): Optional response headers for rate limit detection

        Raises:
            aiohttp.ClientError: For network/connection errors
            asyncio.TimeoutError: For request timeouts

        Example:
            >>> async with session.post(url, headers=headers, json=payload) as resp:
            ...     data = await resp.json()
            ...     return data, resp.headers
        """
        ...

    def parse_error(self, payload: dict[str, Any]) -> Optional[str]:
        """
        Extract error message from API response payload.

        Checks if the response contains an error and returns a
        human-readable error message. Returns None if no error.

        Different providers format errors differently:
        - OpenAI: {"error": {"message": "...", "type": "..."}}
        - Anthropic: {"error": {"type": "...", "message": "..."}}
        - Others may vary

        Args:
            payload (dict[str, Any]): The API response payload dictionary

        Returns:
            Optional[str]: Error message string if error present, None if successful response

        Example:
            >>> provider.parse_error({"error": {"message": "Invalid API key"}})
            'Invalid API key'
            >>> provider.parse_error({"id": "chatcmpl-123", "choices": [...]})
            None
        """
        ...

    def is_rate_limited(
        self,
        payload: dict[str, Any],
        headers: Optional[Mapping[str, str]] = None,
    ) -> bool:
        """
        Determine if the response indicates rate limiting.

        This method classifies errors to distinguish rate limit errors
        (HTTP 429, "rate limit exceeded" messages) from other errors.
        Rate-limited requests trigger special handling with longer pauses.

        Detection methods:
        - Check for "rate limit" in error message
        - Check HTTP status from headers
        - Check provider-specific rate limit indicators

        Args:
            payload (dict[str, Any]): The API response payload dictionary
            headers (Optional[Mapping[str, str]]): Optional response headers for status code checking

        Returns:
            bool: True if the response indicates rate limiting, False otherwise

        Example:
            >>> payload = {"error": {"message": "Rate limit exceeded"}}
            >>> provider.is_rate_limited(payload)
            True
        """
        ...

    def extract_usage(self, payload: dict[str, Any]) -> Optional[Usage]:
        """
        Extract token usage metrics from a successful API response.

        Parses the response to extract input/output token counts for
        tracking and billing purposes. Returns None if usage info is
        not available or the response is an error.

        Different providers use different field names:
        - OpenAI chat: prompt_tokens, completion_tokens, total_tokens
        - OpenAI responses: input_tokens, output_tokens, total_tokens
        - Anthropic: input_tokens, output_tokens

        Args:
            payload (dict[str, Any]): The API response payload dictionary

        Returns:
            Optional[Usage]: Usage object with token counts, or None if unavailable

        Example:
            >>> payload = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}
            >>> usage = provider.extract_usage(payload)
            >>> usage.input_tokens
            10
        """
        ...
