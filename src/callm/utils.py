from __future__ import annotations

import json
import re
from typing import Any, Generator


def api_endpoint_from_url(url: str) -> str:
    """
    Extract the API endpoint from a URL.

    Handles various URL patterns:
    - With version: https://api.provider.com/v1/chat/completions → chat/completions
    - Without version: https://api.provider.com/chat/completions → chat/completions
    - Azure: https://xxx.openai.azure.com/openai/deployments/xxx/chat/completions → chat/completions

    Args:
        url (str): The URL to extract the API endpoint from.

    Returns:
        str: The API endpoint path.

    Raises:
        ValueError: If endpoint cannot be extracted from URL.
    """
    # Pattern 1: Standard versioned URLs (e.g., OpenAI, Cohere, VoyageAI)
    # https://api.openai.com/v1/chat/completions → chat/completions
    # https://api.cohere.com/v2/embed → embed
    match = re.search(r"^https://[^/]+/v\d+/(.+)$", url)
    if match:
        return match[1]

    # Pattern 2: Azure OpenAI deployment URLs
    # https://xxx.openai.azure.com/openai/deployments/gpt-4/chat/completions → chat/completions
    match = re.search(r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", url)
    if match:
        return match[1]

    # Pattern 3: Non-versioned URLs (e.g., DeepSeek)
    # https://api.deepseek.com/chat/completions → chat/completions
    # Extract everything after the domain, removing leading slash
    match = re.search(r"^https://[^/]+/(.+)$", url)
    if match:
        return match[1]

    # If none of the patterns match, raise error
    raise ValueError(f"Could not extract API endpoint from URL: {url}")


def task_id_generator_function() -> Generator[int, None, None]:
    """
    Generate integers 0, 1, 2, and so on.

    Returns:
        Generator[int, None, None]: A generator that yields integers 0, 1, 2, and so on.
    """
    task_id = 0
    while True:
        yield task_id
        task_id += 1


def append_to_jsonl(data: list[Any], file: str) -> None:
    """
    Append a json payload to the end of a jsonl file.

    Args:
        data (list[Any]): the data to append to the file
        file (str): the file to append the data to

    Returns:
        None
    """
    json_string = json.dumps(data, ensure_ascii=False)
    with open(file, mode="a", encoding="utf-8") as f:
        f.write(json_string + "\n")


def validate_jsonl_file(filepath: str, file_type: str = "File") -> None:
    """
    Validate that a filepath ends with .jsonl extension.

    Args:
        filepath: Path to validate
        file_type: Description of file type for error message

    Raises:
        ValueError: If filepath doesn't end with .jsonl
    """
    if not filepath.endswith(".jsonl"):
        raise ValueError(f"{file_type} must be a JSONL file")
