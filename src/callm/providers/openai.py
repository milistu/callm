from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from aiohttp import ClientSession
from tiktoken import encoding_for_model

from callm.providers.base import Provider
from callm.providers.models import Usage
from callm.tokenizers.openai import num_tokens_from_openai_request
from callm.utils import api_endpoint_from_url

"""
OpenAI API provider implementation.

Supports OpenAI and Azure OpenAI endpoints including:
- Chat completions
- Text completions
- Embeddings
- Responses API

Uses tiktoken for accurate token counting.
"""


class OpenAIProvider(Provider):
    """
    Provider implementation for OpenAI and Azure OpenAI APIs.

    This provider handles:
    - OpenAI-style authentication (Bearer token or Azure API key)
    - Token estimation using tiktoken
    - Multiple endpoint types (chat, completions, embeddings, responses)
    - Usage extraction from responses

    Attributes:
        name (str): Always "openai"
        api_key (str): OpenAI or Azure API key
        model (str): Model identifier (e.g., "gpt-4o", "text-embedding-3-small")
        request_url (str): Full API endpoint URL
        use_azure (bool): Whether using Azure OpenAI (different auth header)
        tokenizer (Encoding): Tiktoken encoder for the specified model

    Example:
        >>> provider = OpenAIProvider(
        ...     api_key="sk-...",
        ...     model="gpt-4o",
        ...     request_url="https://api.openai.com/v1/chat/completions"
        ... )
    """

    name = "openai"

    def __init__(
        self, api_key: str, model: str, request_url: str, use_azure: bool = False
    ) -> None:
        """
        Initialize OpenAI provider.

        Args:
            api_key (str): OpenAI or Azure API key
            model (str): Model name for tiktoken encoding
            request_url (str): Full API endpoint URL
            use_azure (bool): If True, use Azure OpenAI authentication

        Raises:
            ValueError: If model is not recognized by tiktoken
        """
        self.api_key = api_key
        self.model = model
        self.request_url = request_url
        self.use_azure = use_azure
        try:
            self.tokenizer = encoding_for_model(model)
        except Exception as e:
            raise ValueError(f"Invalid model: {model}") from e

    def build_headers(self) -> dict[str, str]:
        """Build authentication headers for OpenAI or Azure OpenAI."""
        if self.use_azure:
            return {"api-key": self.api_key}
        return {"Authorization": f"Bearer {self.api_key}"}

    def estimate_input_tokens(self, request_json: dict[str, Any]) -> int:
        """
        Estimate input tokens using tiktoken.

        Supports multiple endpoint types with different counting logic:
        - chat/completions: Counts message tokens + formatting overhead
        - completions: Counts prompt tokens
        - embeddings: Counts input text tokens
        - responses: Counts input content tokens
        """
        endpoint = api_endpoint_from_url(self.request_url)
        return num_tokens_from_openai_request(request_json, endpoint, self.tokenizer)

    async def send(
        self,
        session: ClientSession,
        headers: Mapping[str, str],
        request_json: dict[str, Any],
    ) -> Tuple[dict[str, Any], Optional[Mapping[str, str]]]:
        """
        Send request to OpenAI API.

        Automatically adds model to payload if not present (for non-Azure).
        """
        payload = dict(request_json)
        if not self.use_azure and "model" not in payload:
            payload["model"] = self.model

        async with session.post(
            self.request_url, headers=headers, json=payload
        ) as response:
            data = await response.json()
            return data, response.headers

    def parse_error(self, payload: dict[str, Any]) -> Optional[str]:
        """
        Parse error from OpenAI response.

        OpenAI errors have format: {"error": {"message": "...", "type": "..."}}
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
        headers: Optional[Mapping[str, str]] = None,
    ) -> bool:
        """
        Detect rate limiting from error message.

        Checks for "rate limit" in error message text.
        """
        msg = (self.parse_error(payload) or "").lower()
        return "rate limit" in msg

    def extract_usage(self, payload: dict[str, Any]) -> Optional[Usage]:
        """
        Extract token usage from OpenAI response.

        Handles different usage formats:
        - Responses endpoint: input_tokens, output_tokens, total_tokens
        - Chat/Completions: prompt_tokens, completion_tokens, total_tokens
        - Embeddings: prompt_tokens, total_tokens (no completions)
        """
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        if "input_tokens" in usage or "output_tokens" in usage:
            # Responses endpoint https://platform.openai.com/docs/api-reference/responses/object#responses/object-usage
            input_tokens = int(usage.get("input_tokens", 0))
            output_tokens = int(usage.get("output_tokens", 0))
            total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens))
            return Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )

        # Embeddings endpoint https://platform.openai.com/docs/guides/embeddings#how-to-get-embeddings
        # Chat completions endpoint https://platform.openai.com/docs/api-reference/chat/object#chat/object-usage
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
        return Usage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
