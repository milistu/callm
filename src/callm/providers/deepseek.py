from __future__ import annotations

from typing import Any, Optional

from callm.providers.base import BaseProvider
from callm.providers.models import Usage
from callm.tokenizers.deepseek import (
    get_deepseek_tokenizer,
    num_tokens_from_deepseek_request,
)


class DeepSeekProvider(BaseProvider):
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

    def estimate_input_tokens(self, request_json: dict[str, Any]) -> int:
        """
        Estimate input tokens using DeepSeek's tokenizer.

        Supports chat completions with messages format.

        Args:
            request_json (dict[str, Any]): The request payload

        Returns:
            int: Estimated number of input tokens
        """
        endpoint = self._extract_endpoint()
        return num_tokens_from_deepseek_request(request_json, endpoint, self.tokenizer)

    def parse_error(self, payload: dict[str, Any]) -> Optional[str]:
        """
        Parse error from DeepSeek API response.

        DeepSeek uses OpenAI-compatible error format (primary format).
        See: https://api-docs.deepseek.com/quick_start/error_codes

        Error codes:
        - 400: Invalid Format (invalid request body format)
        - 401: Authentication Fails (wrong API key)
        - 402: Insufficient Balance (run out of balance)
        - 422: Invalid Parameters (invalid request parameters)
        - 429: Rate Limit Reached (sending requests too quickly)
        - 500: Server Error (server issue)
        - 503: Server Overloaded (high traffic)

        Primary error response format (official):
        {
            "error": {
                "message": "error description",
                "type": "error_type",
                "param": null,
                "code": "error_code"
            }
        }

        Alternative error format (observed in testing):
        {
            "error_msg": "error description string"
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
