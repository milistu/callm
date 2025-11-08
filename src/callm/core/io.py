from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

from callm.utils import append_to_jsonl

"""
JSONL file I/O utilities.

Provides functions for streaming JSONL files and writing
results/errors in a thread-safe manner.
"""


def stream_jsonl(filepath: str) -> Generator[dict[str, Any], None, None]:
    """
    Stream JSON objects from a JSONL file line by line.

    Skips empty lines and yields parsed JSON objects.
    Memory efficient for large files.

    Args:
        filepath (str): Path to JSONL file

    Yields:
        Generator[dict[str, Any], None, None]: Parsed JSON objects (dictionaries)

    Raises:
        json.JSONDecodeError: If a line contains invalid JSON
        FileNotFoundError: If file doesn't exist
    """
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def write_result(entry: list[Any], save_file: str) -> None:
    """
    Write a successful result to the output file.

    Args:
        entry (list[Any]): List containing [request, response, optional_metadata]
        save_file (str): Path to output JSONL file
    """
    append_to_jsonl(data=entry, file=save_file)


def write_error(entry: list[Any], error_file: str) -> None:
    """
    Write a failed request to the error file.

    Args:
        entry (list[Any]): List containing [request, errors, optional_metadata]
        error_file (str): Path to error JSONL file
    """
    append_to_jsonl(data=entry, file=error_file)
