from __future__ import annotations

from typing import Any

from aiohttp import ClientSession


async def num_tokens_from_anthropic_request(
    request_json: dict[str, Any],
    count_tokens_url: str,
    headers: dict[str, str],
    session: ClientSession,
) -> int:
    """
    Count tokens using Anthropic's count_tokens API endpoint.

    Unlike other providers that use local tokenizers, Anthropic requires
    an API call for precise token counting. Token counting is FREE but
    has separate RPM limits based on usage tier.

    API Reference: https://platform.claude.com/docs/en/build-with-claude/token-counting

    Args:
        request_json: The request payload (Anthropic messages format)
        headers: HTTP headers including API key and version
        session: Aiohttp session for making the API call

    Returns:
        int: Number of input tokens

    Raises:
        Exception: If the API call fails
    """

    count_request = _build_count_tokens_request(request_json)

    async with session.post(count_tokens_url, headers=headers, json=count_request) as response:
        data = await response.json()

        if "error" in data:
            error_info = data["error"]
            if isinstance(error_info, dict):
                error_msg = error_info.get("message", str(error_info))
            else:
                error_msg = str(error_info)
            raise Exception(f"Anthropic count_tokens API error: {error_msg}")

        input_tokens = data.get("input_tokens", 0)
        return int(input_tokens)


def _build_count_tokens_request(request_json: dict[str, Any]) -> dict[str, Any]:
    """
    Build the request payload for the count_tokens endpoint.

    The count_tokens API accepts the same structured inputs as the messages API:
    - model (required)
    - messages (required)
    - system (optional)
    - tools (optional)

    Args:
        request_json: The original request payload

    Returns:
        dict: The countTokens request payload
    """
    count_request: dict[str, Any] = {
        "model": request_json["model"],
        "messages": request_json["messages"],
    }

    if "system" in request_json:
        count_request["system"] = request_json["system"]

    if "tools" in request_json:
        count_request["tools"] = request_json["tools"]

    if "thinking" in request_json:
        count_request["thinking"] = request_json["thinking"]

    return count_request
