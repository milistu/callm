from __future__ import annotations

import json
import re
from typing import Generator, TypeAlias

JSONValue: TypeAlias = (
    str | int | float | bool | None | dict[str, "JSONValue"] | list["JSONValue"]
)


def api_endpoint_from_url(url: str) -> str:
    """
    Extract the API endpoint from a URL.

    Args:
        url (str): The URL to extract the API endpoint from.

    Returns:
        str: The API endpoint.
    """
    match = re.search(r"^https://[^/]+/v\d+/(.+)$", url)
    if match is None:
        # Try for Azure OpenAI deployment urls
        match = re.search(r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", url)
    if match is None:
        raise ValueError(f"Could not extract API endpoint from URL: {url}")
    return match[1]


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


def append_to_jsonl(
    data: list[dict[str, JSONValue] | list[JSONValue]], file: str
) -> None:
    """
    Append a json payload to the end of a jsonl file.

    Args:
        data (list[dict[str, JSONValue] | list[JSONValue]]): the data to append to the file
        file (str): the file to append the data to

    Returns:
        None
    """
    json_string = json.dumps(data)
    with open(file, mode="a", encoding="utf-8") as f:
        f.write(json_string + "\n")
