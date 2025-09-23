from typing import Any

from tiktoken import Encoding

DEFAULT_MAX_TOKENS = 16
DEFAULT_N = 1


def num_tokens_consumed_from_request(
    request_json: dict[str, Any],
    api_endpoint: str,
    tokenizer: Encoding,
) -> int:
    """
    Count the number of tokens in the request.
    Only supports `completion` and `embedding` requests.

    Args:
        request_json (dict[str, Any]): the request json
        api_endpoint (str): the API endpoint
        tokenizer (Encoding): the tokenizer

    Returns:
        int: the number of tokens consumed
    """
    if api_endpoint.endswith("completions"):
        max_tokens: int = request_json.get("max_tokens", DEFAULT_MAX_TOKENS)
        n: int = request_json.get("n", DEFAULT_N)
        completion_tokens = n * max_tokens

        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(tokenizer.encode(value))
                    if key == "name":  # If there's a name, the role is omitted
                        num_tokens -= 1  # Role is always required and always 1 token
            num_tokens += 2  # Every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # Single prompt
                prompt_tokens = len(tokenizer.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # Multiple prompts
                prompt_tokens = sum([len(tokenizer.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError(
                    'Expecting either string or list of strings for "prompt" field in completion request.'
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
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not yet implemented in this library, please submit an issue at https://github.com/milistu/callm/issues.'
        )
