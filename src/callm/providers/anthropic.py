from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aiohttp import ClientSession

from callm.providers.base import BaseProvider
from callm.providers.models import Usage
from callm.tokenizers.anthropic import num_tokens_from_anthropic_request

"""
Anthropic Claude API provider implementation.

Supports Anthropic Claude API including:
- Messages API (text generation, vision, PDFs)
- Tool use / Function calling
- Structured outputs

Uses countTokens API endpoint for precise token counting.
"""


class AnthropicProvider(BaseProvider):
    """
    Provider implementation for Anthropic Claude API.

    Key differences from other providers:
    - Uses x-api-key header for authentication (not Bearer token)
    - Requires anthropic-version header
    - Token counting requires API call to /v1/messages/count_tokens
    - Token counting has separate rate limits from messages API

    API Reference: https://docs.anthropic.com/en/api/overview

    Note on Token Counting:
        Anthropic's countTokens endpoint has separate rate limits (2x the messages RPM).
        Since we call countTokens once per message request, we'll always hit messages
        rate limits before token counting limits. No separate rate limiting needed.

    Attributes:
        name (str): Always "anthropic"
        api_key (str): Anthropic API key
        model (str): Model identifier (e.g., "claude-sonnet-4-5")
        request_url (str): Full API endpoint URL
        anthropic_version (str): API version string

    Example:
        >>> provider = AnthropicProvider(
        ...     api_key="sk-ant-...",
        ...     model="claude-sonnet-4-5",
        ...     request_url="https://api.anthropic.com/v1/messages"
        ... )
    """

    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str,
        request_url: str = "https://api.anthropic.com/v1/messages",
        anthropic_version: str = "2023-06-01",
    ) -> None:
        """
        Initialize Anthropic Claude provider.

        Args:
            api_key: Anthropic API key (from Anthropic Console)
            model: Model name (e.g., "claude-sonnet-4-20250514", "claude-opus-4-20250514")
            request_url: Full API endpoint URL (default: messages endpoint)
            anthropic_version: API version (default: "2023-06-01")
        """
        self.api_key = api_key
        self.model = model
        self.request_url = request_url
        self.anthropic_version = anthropic_version

    def build_headers(self) -> dict[str, str]:
        """
        Build authentication headers for Anthropic API.

        Anthropic requires:
        - x-api-key: API key
        - anthropic-version: API version string
        - content-type: application/json

        Returns:
            dict[str, str]: HTTP headers
        """
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }

    async def estimate_input_tokens(
        self, request_json: dict[str, Any], session: ClientSession | None = None
    ) -> int:
        """
        Count input tokens using Anthropic's countTokens API endpoint.

        Unlike other providers that use local tokenizers, Anthropic requires
        an API call for precise token counting.

        Note: Token counting is FREE but has separate RPM limits:
        - Tier 1: 100 RPM
        - Tier 2: 2,000 RPM
        - Tier 3: 4,000 RPM
        - Tier 4: 8,000 RPM

        Args:
            request_json: The request payload (Anthropic messages format)
            session: Aiohttp session for making the API call (required)

        Returns:
            int: Number of input tokens

        Raises:
            ValueError: If session is not provided
        """
        if session is None:
            raise ValueError(
                "AnthropicProvider.estimate_input_tokens requires an aiohttp ClientSession. "
                "This is needed to call the countTokens API endpoint."
            )

        request_with_model = dict(request_json)
        if "model" not in request_with_model:
            request_with_model["model"] = self.model

        headers = self.build_headers()

        count_tokens_url = self.request_url.replace("/messages", "/messages/count_tokens")

        return await num_tokens_from_anthropic_request(
            request_json=request_with_model,
            count_tokens_url=count_tokens_url,
            headers=headers,
            session=session,
        )

    async def send(
        self,
        session: ClientSession,
        headers: Mapping[str, str],
        request_json: dict[str, Any],
    ) -> tuple[dict[str, Any], Mapping[str, str] | None]:
        """
        Send request to Anthropic Messages API.

        Automatically adds model to payload if not present.
        Automatically adds beta header for structured outputs.

        Args:
            session: Aiohttp client session
            headers: HTTP headers from build_headers()
            request_json: Request payload (Anthropic messages format)

        Returns:
            Tuple of (response_payload, response_headers)
        """
        payload = dict(request_json)
        if "model" not in payload:
            payload["model"] = self.model

        # Convert headers to mutable dict
        request_headers = dict(headers)

        # Auto-add beta header for structured outputs
        if "output_format" in payload:
            request_headers["anthropic-beta"] = "structured-outputs-2025-11-13"

        async with session.post(
            self.request_url, headers=request_headers, json=payload
        ) as response:
            data = await response.json()
            return data, response.headers

    def parse_error(self, payload: dict[str, Any]) -> str | None:
        """
        Parse error from Anthropic API response.

        Anthropic error format:
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "..."
            }
        }
        """
        # Check for error type at top level
        if payload.get("type") == "error" or payload.get("error"):
            error = payload.get("error", {})
            if isinstance(error, dict):
                return str(error.get("message") or error)
            return str(error)

        return None

    def is_rate_limited(
        self,
        payload: dict[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> bool:
        """
        Determine if the response indicates rate limiting.

        Anthropic rate limit indicators:
        - HTTP 429 status
        - Error type "rate_limit_error"
        - Error type "overloaded_error"
        """
        if headers:
            status = headers.get("status")
            if status == "429":
                return True

        # Check error type
        if payload.get("type") == "error":
            error = payload.get("error", {})
            if isinstance(error, dict):
                error_type = error.get("type", "")
                if error_type in ("rate_limit_error", "overloaded_error"):
                    return True

                message = (error.get("message") or "").lower()
                if "rate limit" in message or "too many requests" in message:
                    return True

        return False

    def extract_usage(
        self, payload: dict[str, Any], estimated_input_tokens: int | None = None
    ) -> Usage | None:
        """
        Extract token usage from Anthropic API response.

        Anthropic usage format:
        {
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20
            }
        }

        For cached prompts, also includes:
        - cache_creation_input_tokens
        - cache_read_input_tokens
        """
        if self.parse_error(payload):
            return None

        usage = payload.get("usage", {})
        if usage:
            input_tokens = int(usage.get("input_tokens", 0))
            output_tokens = int(usage.get("output_tokens", 0))
            total_tokens = input_tokens + output_tokens
            return Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
        elif estimated_input_tokens is not None:
            return Usage(
                input_tokens=estimated_input_tokens,
                output_tokens=0,
                total_tokens=estimated_input_tokens,
            )

        return None
