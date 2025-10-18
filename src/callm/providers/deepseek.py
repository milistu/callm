from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from aiohttp import ClientSession

from callm.providers.base import Provider
from callm.providers.models import Usage
from callm.tokenizers.deepseek import (
    get_deepseek_tokenizer,
    num_tokens_from_deepseek_request,
)
from callm.utils import api_endpoint_from_url


class DeepSeekProvider(Provider):
    """
    Provider implementation for DeepSeek API.

    DeepSeek API is OpenAI-compatible and supports chat completions.
    See: https://api-docs.deepseek.com/

    Currently supports:
    - Chat completions (chat/completions endpoint)

    Key Features:
    - NO RATE LIMITS: DeepSeek does not constrain rate limits
    - OpenAI-compatible API format
    - Reasoning model support (deepseek-reasoner)

    Attributes:
        name (str): Always "deepseek"
        api_key (str): DeepSeek API key
        model (str): Model identifier (e.g., "deepseek-chat", "deepseek-reasoner")
        request_url (str): Full API endpoint URL
        tokenizer (Tokenizer): DeepSeek tokenizer for token estimation

    Example:
        >>> provider = DeepSeekProvider(
        ...     api_key="your-deepseek-api-key",
        ...     model="deepseek-chat",
        ...     request_url="https://api.deepseek.com/chat/completions"
        ... )
    """

    name = "deepseek"

    def __init__(self, api_key: str, model: str, request_url: str) -> None:
        """
        Initialize DeepSeek provider.

        Args:
            api_key (str): DeepSeek API key
            model (str): Model name (e.g., "deepseek-chat", "deepseek-reasoner")
            request_url (str): Full API endpoint URL

        Raises:
            ValueError: If tokenizer cannot be loaded
        """
        self.api_key = api_key
        self.model = model
        self.request_url = request_url

        # Download and cache tokenizer from HuggingFace
        try:
            self.tokenizer = get_deepseek_tokenizer(model)
        except Exception as e:
            raise ValueError(
                f"Failed to initialize tokenizer for model '{model}': {e}"
            ) from e

    def build_headers(self) -> dict[str, str]:
        """
        Build authentication headers for DeepSeek API.

        DeepSeek uses Bearer token authentication (OpenAI-compatible).

        Returns:
            dict[str, str]: Headers with Authorization and Content-Type
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def estimate_input_tokens(self, request_json: dict[str, Any]) -> int:
        """
        Estimate input tokens using DeepSeek's tokenizer.

        Supports chat completions with messages format.

        Args:
            request_json (dict[str, Any]): The request payload

        Returns:
            int: Estimated number of input tokens
        """
        # Extract endpoint from URL (e.g., "chat/completions")
        try:
            endpoint = api_endpoint_from_url(self.request_url)
        except ValueError:
            # Fallback: try to extract last part
            endpoint = self.request_url.rstrip("/").split("/")[-1]
            # If just "completions", prepend "chat/"
            if endpoint == "completions" and "chat" in self.request_url:
                endpoint = "chat/completions"

        return num_tokens_from_deepseek_request(request_json, endpoint, self.tokenizer)

    async def send(
        self,
        session: ClientSession,
        headers: Mapping[str, str],
        request_json: dict[str, Any],
    ) -> Tuple[dict[str, Any], Optional[Mapping[str, str]]]:
        """
        Send request to DeepSeek API.

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
        Parse error from DeepSeek API response.

        DeepSeek uses OpenAI-compatible error format.
        See: https://api-docs.deepseek.com/quick_start/error_codes

        Error codes:
        - 400: Invalid Format (invalid request body format)
        - 401: Authentication Fails (wrong API key)
        - 402: Insufficient Balance (run out of balance)
        - 422: Invalid Parameters (invalid request parameters)
        - 429: Rate Limit Reached (sending requests too quickly)
        - 500: Server Error (server issue)
        - 503: Server Overloaded (high traffic)

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
        error = payload.get("error") or payload.get("error_msg")
        if not error:
            return None
        if isinstance(error, dict):
            return str(error.get("message") or error)
        return str(error)

    def is_rate_limited(
        self,
        payload: dict[str, Any],
        headers: Optional[Mapping[str, str]] = None,
    ) -> bool:
        """
        Detect rate limiting from DeepSeek API response.

        Note: DeepSeek does NOT constrain rate limits officially, but may
        return 429 errors during high traffic. Under high load, requests
        may take longer but won't fail immediately.

        See: https://api-docs.deepseek.com/quick_start/rate_limit

        Checks for:
        - HTTP 429 status code
        - "rate limit" in error message

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
        error = payload.get("error", {})
        if isinstance(error, dict):
            message = str(error.get("message", "")).lower()
            if "rate limit" in message or "too many requests" in message:
                return True

        return False

    def extract_usage(self, payload: dict[str, Any]) -> Optional[Usage]:
        """
        Extract token usage from DeepSeek API response.

        DeepSeek uses OpenAI-compatible usage format:
        {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }

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

        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))

        return Usage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
