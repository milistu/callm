import json
from typing import Any

from tiktoken import Encoding


def num_tokens_from_openai_request(
    request_json: dict[str, Any],
    api_endpoint: str,
    tokenizer: Encoding,
) -> int:
    """
    Count the number of tokens in the request.
    Only supports `responses`, `completion` and `embedding` requests.

    Args:
        request_json (dict[str, Any]): the request json
        api_endpoint (str): the API endpoint
        tokenizer (Encoding): the tokenizer

    Returns:
        int: the number of tokens consumed
    """
    if api_endpoint.endswith("completions"):
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json.get("messages", []):
                num_tokens += 3  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(tokenizer.encode(value))
                    if key == "name":  # If there's a name, the role is omitted
                        num_tokens += 1  # Role is always required and always 1 token
            num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

            # Add tokens from response_format (Structured Outputs)
            response_format = request_json.get("response_format")
            if response_format and isinstance(response_format, dict):
                if response_format.get("type") == "json_schema":
                    json_schema = response_format.get("json_schema")
                    if json_schema:
                        schema_str = json.dumps(json_schema["schema"], separators=(",", ":"))
                        num_tokens += len(tokenizer.encode(schema_str))

            return num_tokens
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # Single prompt
                prompt_tokens = len(tokenizer.encode(prompt))
                return prompt_tokens
            elif isinstance(prompt, list):  # Multiple prompts
                prompt_tokens = sum([len(tokenizer.encode(p)) for p in prompt])
                return prompt_tokens
            else:
                raise TypeError(
                    'Expecting either string or list of strings for "prompt" field'
                    "in completion request."
                )
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # Single input
            num_tokens = len(tokenizer.encode(input))
            return num_tokens
        elif isinstance(input, list):  # Multiple inputs
            num_tokens = sum([len(tokenizer.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "input" field in embedding request.'
            )
    elif api_endpoint == "responses":
        input = request_json["input"]
        if isinstance(input, str):  # single input string
            num_tokens = len(tokenizer.encode(input))
            return num_tokens
        elif isinstance(input, list):  # array of message objects (similar to chat completions)
            num_tokens = 0
            for item in input:
                if isinstance(item, dict):
                    # Handle message objects
                    if "content" in item:
                        content = item["content"]
                        if isinstance(content, str):
                            num_tokens += len(tokenizer.encode(content))
                        elif isinstance(content, list):
                            # Handle content array with different types (text, images, etc.)
                            for content_item in content:
                                if isinstance(content_item, dict) and "text" in content_item:
                                    num_tokens += len(tokenizer.encode(content_item["text"]))
                                elif (
                                    isinstance(content_item, dict)
                                    and content_item.get("type") == "input_text"
                                ):
                                    num_tokens += len(tokenizer.encode(content_item["text"]))
                    # Add tokens for role and message structure overhead
                    num_tokens += 4  # every message follows similar structure
                    for key, value in item.items():
                        if key != "content" and isinstance(value, str):
                            num_tokens += len(tokenizer.encode(value))
                elif isinstance(item, str):
                    # Handle simple string items in the array
                    num_tokens += len(tokenizer.encode(item))
        else:
            raise TypeError(
                'Expecting either string or list for "input" field in responses request.'
            )

        # Add tokens from text.format (Structured Outputs)
        text_options = request_json.get("text")
        if text_options and isinstance(text_options, dict):
            format = text_options.get("format")
            if format and isinstance(format, dict):
                if format.get("type") == "json_schema":
                    format_str = json.dumps(format["schema"], separators=(",", ":"))
                    num_tokens += len(tokenizer.encode(format_str))

        return num_tokens
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not yet implemented in this library, '
            f"please submit an issue at https://github.com/milistu/callm/issues."
        )
