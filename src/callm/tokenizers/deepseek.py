from typing import Any

from tokenizers import Tokenizer

DEFAULT_MAX_TOKENS = 16
DEFAULT_N = 1


def get_deepseek_tokenizer(model: str, namespace: str = "deepseek-ai") -> Tokenizer:
    """
    Download and cache the DeepSeek tokenizer for a specific model.

    DeepSeek tokenizers are available on HuggingFace.
    See: https://api-docs.deepseek.com/quick_start/token_usage

    Args:
        model (str): The DeepSeek model name (e.g., "deepseek-chat", "deepseek-reasoner")
                     Maps to "deepseek-ai/DeepSeek-V3.2-Exp" on HuggingFace
        namespace (str): HuggingFace organization/namespace (e.g., "deepseek-ai")

    Returns:
        AutoTokenizer: HuggingFace AutoTokenizer instance

    Raises:
        ValueError: If tokenizer cannot be downloaded for the model
    """
    try:
        # DeepSeek models map to the same tokenizer on HuggingFace
        tokenizer = Tokenizer.from_pretrained(f"{namespace}/{model}")
    except Exception as e:
        raise ValueError(
            f"Failed to initialize tokenizer for model '{model}'. Error: {e}"
        ) from e
    return tokenizer


def num_tokens_from_deepseek_request(
    request_json: dict[str, Any],
    api_endpoint: str,
    tokenizer: Tokenizer,
) -> int:
    """
    Count the number of tokens in a DeepSeek API request.

    Supports chat/completions endpoint.

    Args:
        request_json (dict[str, Any]): The request payload
        api_endpoint (str): The API endpoint (e.g., "chat/completions")
        tokenizer (AutoTokenizer): The DeepSeek tokenizer instance

    Returns:
        int: Estimated number of input tokens

    Raises:
        NotImplementedError: For unsupported endpoints
        TypeError: For invalid input types
    """
    if api_endpoint.endswith("completions"):
        max_tokens: int = request_json.get("max_tokens", DEFAULT_MAX_TOKENS)
        n: int = request_json.get("n", DEFAULT_N)
        completion_tokens = n * max_tokens

        # Chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json.get("messages", []):
                # Every message follows <im_start>{role/name}\n{content}<im_end>\n
                num_tokens += 4
                for key, value in message.items():
                    if isinstance(value, str):
                        num_tokens += len(tokenizer.encode(value))
                        if key == "name":
                            # If there's a name, the role is omitted
                            num_tokens -= 1

            # Every reply is primed with <im_start>assistant
            num_tokens += 2
            return num_tokens + completion_tokens
        else:
            # Standard completions
            prompt = request_json.get("prompt", "")
            if isinstance(prompt, str):
                prompt_tokens = len(tokenizer.encode(prompt))
                return prompt_tokens + completion_tokens
            elif isinstance(prompt, list):
                prompt_tokens = sum([len(tokenizer.encode(p)) for p in prompt])
                return prompt_tokens + completion_tokens * len(prompt)
            else:
                raise TypeError(
                    "Expecting either string or list of strings for 'prompt' field."
                )
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not yet implemented for DeepSeek provider. '
            "Please submit an issue at https://github.com/milistu/callm/issues."
        )
