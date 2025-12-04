from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aiohttp import ClientSession

from callm.providers.base import BaseProvider
from callm.providers.models import Usage
from callm.tokenizers.gemini import num_tokens_from_gemini_request

"""
Google Gemini API provider implementation.

Supports Google Gemini API including:
- Generate Content (generateContent)
- Embed Content (embedContent)

Uses countTokens API endpoint for precise token counting.
"""


class GeminiProvider(BaseProvider):
    """
    Provider implementation for Google Gemini API.

    Key differences from other providers:
    - Uses x-goog-api-key header for authentication (not Bearer token)
    - Model is embedded in the URL, not the payload
    - Token counting requires API call to countTokens endpoint
    - Different request/response format from OpenAI

    API Reference: https://ai.google.dev/gemini-api/docs

    Attributes:
        name (str): Always "gemini"
        api_key (str): Google AI API key
        model (str): Model identifier (e.g., "gemini-flash-latest", "gemini-embedding-001")
        request_url (str): Full API endpoint URL

    Example:
        >>> provider = GeminiProvider(
        ...     api_key="your-gemini-api-key",
        ...     model="gemini-flash-latest",
        ...     request_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"
        ... )
    """

    name = "gemini"

    def __init__(self, api_key: str, model: str, request_url: str) -> None:
        """
        Initialize Google Gemini provider.

        Args:
            api_key: Google AI API key (from Google AI Studio)
            model: Model name (e.g., "gemini-flash-latest", "gemini-3-pro-preview",
                   "gemini-embedding-001")
            request_url: Full API endpoint URL.
                Example URLs:
                - Text: https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent
                - Embed: https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent
        """
        self.api_key = api_key
        self.model = model
        self.request_url = request_url

    def build_headers(self) -> dict[str, str]:
        """
        Build authentication headers for Gemini API.

        Gemini uses x-goog-api-key header instead of Bearer token.

        Returns:
            dict[str, str]: HTTP headers including API key
        """
        return {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _get_count_tokens_url(self) -> str:
        """
        Derive the countTokens endpoint URL from request_url.

        Transforms:
        - .../gemini-flash-latest:generateContent → .../gemini-flash-latest:countTokens
        - .../gemini-embedding-001:embedContent → .../gemini-embedding-001:countTokens
        """
        if ":" in self.request_url:
            base_url = self.request_url.rsplit(":", 1)[0]
            return f"{base_url}:countTokens"
        raise ValueError(f"Invalid request URL: {self.request_url}")

    async def estimate_input_tokens(
        self, request_json: dict[str, Any], session: ClientSession | None = None
    ) -> int:
        """
        Count input tokens using Gemini's countTokens API endpoint.

        Unlike other providers that use local tokenizers, Gemini requires
        an API call for precise token counting.

        Args:
            request_json: The request payload (Gemini format)
            session: Aiohttp session for making the API call (required)

        Returns:
            int: Number of input tokens

        Raises:
            ValueError: If session is not provided
        """
        if session is None:
            raise ValueError(
                "GeminiProvider.estimate_input_tokens requires an aiohttp ClientSession. "
                "This is needed to call the countTokens API endpoint."
            )

        headers = self.build_headers()
        count_tokens_url = self._get_count_tokens_url()

        return await num_tokens_from_gemini_request(
            request_json=request_json,
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
        Send request to Gemini API.

        Unlike other providers, we don't inject model into payload because
        the model is already specified in the URL.

        Args:
            session: Aiohttp client session
            headers: HTTP headers from build_headers()
            request_json: Request payload (Gemini format)

        Returns:
            Tuple of (response_payload, response_headers)
        """
        async with session.post(self.request_url, headers=headers, json=request_json) as response:
            data = await response.json()
            return data, response.headers

    def parse_error(self, payload: dict[str, Any]) -> str | None:
        """
        Parse error from Gemini API response.

        Gemini error format:
        {
            "error": {
                "code": 400,
                "message": "Invalid argument...",
                "status": "INVALID_ARGUMENT"
            }
        }
        """
        error = payload.get("error")
        if not error:
            return None
        if isinstance(error, dict):
            return str(error.get("message") or error)
        return str(error)

    def is_rate_limited(
        self,
        payload: dict[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> bool:
        """
        Determine if the response indicates rate limiting.

        Gemini rate limit indicators:
        - HTTP 429 status
        - Error status "RESOURCE_EXHAUSTED"
        """
        if headers:
            status = headers.get("status")
            if status == "429":
                return True

        error = payload.get("error", {})
        if isinstance(error, dict):
            status = error.get("status", "")
            if status == "RESOURCE_EXHAUSTED":
                return True

            message = (error.get("message") or "").lower()
            if "rate limit" in message or "quota" in message or "too many requests" in message:
                return True

        return False

    def extract_usage(
        self, payload: dict[str, Any], estimated_input_tokens: int | None = None
    ) -> Usage | None:
        """
        Extract token usage from Gemini API response.

        Gemini usage format:
        {
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20,
                "totalTokenCount": 30
            }
        }
        """
        if self.parse_error(payload):
            return None

        usage = payload.get("usageMetadata", {})
        if usage:
            prompt_tokens = int(usage.get("promptTokenCount", 0))
            candidates_tokens = int(usage.get("candidatesTokenCount", 0))
            total_tokens = int(usage.get("totalTokenCount", prompt_tokens + candidates_tokens))
            return Usage(
                input_tokens=prompt_tokens,
                output_tokens=candidates_tokens,
                total_tokens=total_tokens,
            )
        elif estimated_input_tokens is not None:
            return Usage(
                input_tokens=estimated_input_tokens,
                output_tokens=0,
                total_tokens=estimated_input_tokens,
            )

        return None
