from __future__ import annotations

import json
from typing import Any, Generator

from callm.utils import append_to_jsonl


def stream_jsonl(filepath: str) -> Generator[dict[str, Any], None, None]:
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def write_result(entry: list[dict[str, Any] | list[Any]], save_file: str) -> None:
    append_to_jsonl(data=entry, file=save_file)


def write_error(entry: list[dict[str, Any] | list[Any]], error_file: str) -> None:
    append_to_jsonl(data=entry, file=error_file)
