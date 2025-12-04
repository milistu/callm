from __future__ import annotations

import json
from typing import Any

from aiohttp import ClientSession


async def num_tokens_from_gemini_request(
    request_json: dict[str, Any],
    count_tokens_url: str,
    headers: dict[str, str],
    session: ClientSession,
) -> int:
    """
    Count tokens using Gemini's countTokens API endpoint.

    Unlike other providers that use local tokenizers, Gemini requires
    an API call for precise token counting.

    API Reference: https://ai.google.dev/gemini-api/docs/tokens

    Args:
        request_json: The request payload (Gemini format)
        count_tokens_url: The countTokens endpoint URL
        headers: HTTP headers including API key
        session: Aiohttp session for making the API call

    Returns:
        int: Number of input tokens

    Raises:
        Exception: If the API call fails
    """
    model = count_tokens_url.split("/")[-1].split(":")[0]
    count_request = _build_count_tokens_request(request_json, model)

    async with session.post(count_tokens_url, headers=headers, json=count_request) as response:
        data = await response.json()

        if "error" in data:
            error_msg = data["error"].get("message", str(data["error"]))
            raise Exception(f"Gemini countTokens API error: {error_msg}")

        total_tokens = data.get("totalTokens", 0)
        return int(total_tokens)


def _build_count_tokens_request(request_json: dict[str, Any], model: str) -> dict[str, Any]:
    """
    Build the request payload for the countTokens endpoint.

    The countTokens API accepts:
    - "contents": Same format as generateContent
    - "generateContentRequest": Full generateContent request (optional)

    Args:
        request_json: The original request payload
        model: The model to use for counting tokens
    Returns:
        dict: The countTokens request payload
    """
    # For generateContent requests
    if "contents" in request_json:
        # Check if we have additional config that affects token count
        has_system_instruction = "systemInstruction" in request_json
        has_generation_config = "generationConfig" in request_json
        has_tools = "tools" in request_json

        # Check for responseSchema that needs special handling
        response_schema = None
        if has_generation_config:
            response_schema = request_json["generationConfig"].get("responseJsonSchema")

        if has_system_instruction or has_generation_config or has_tools:

            # If there's a responseSchema, add it as text content for accurate counting
            # The API doesn't count it from generationConfig, so we pass it as content
            if response_schema:
                schema_text = json.dumps(response_schema, separators=(",", ":"))
                # Append schema as additional content part
                if request_json["contents"] and isinstance(request_json["contents"], list):
                    # Add to last content's parts
                    if "parts" in request_json["contents"][-1]:
                        request_json["contents"][-1]["parts"].append({"text": schema_text})

            generate_content_request: dict[str, Any] = {"contents": request_json["contents"]}
            generate_content_request["model"] = f"models/{model}"

            if has_system_instruction:
                generate_content_request["systemInstruction"] = request_json["systemInstruction"]

            if has_generation_config:
                generate_content_request["generationConfig"] = request_json["generationConfig"]

            if has_tools:
                generate_content_request["tools"] = request_json["tools"]

            return {"generateContentRequest": generate_content_request}

        # Simple case: just contents
        return {"contents": request_json["contents"]}

    # For embedContent requests
    elif "content" in request_json:
        return {"contents": [request_json["content"]]}

    else:
        raise ValueError(
            f"Unsupported Gemini request format. Expected 'contents' (generateContent) "
            f"or 'content' (embedContent), got keys: {list(request_json.keys())}"
        )
