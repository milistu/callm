from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Tuple

from aiohttp import ClientSession

from callm.providers.models import Usage


class Provider(Protocol):
    # Human readable identifier
    name: str
    request_url: str

    # Build auth and any required headers
    def build_headers(self) -> dict[str, str]: ...

    # Estimate input tokens for rate-limit budgeting
    def estimate_input_tokens(self, request_json: dict[str, Any]) -> int: ...

    # Perform the HTTP request; return payload and response headers (if needed by is_rate_limited)
    async def send(
        self,
        session: ClientSession,
        headers: Mapping[str, str],
        request_json: dict[str, Any],
    ) -> Tuple[dict[str, Any], Optional[Mapping[str, str]]]: ...

    # Return None if no error; otherwise a concise error message string
    def parse_error(self, payload: dict[str, Any]) -> Optional[str]: ...

    # Decide if result indicates throttling (optionally using headers)
    def is_rate_limited(
        self,
        payload: dict[str, Any],
        headers: Optional[Mapping[str, str]] = None,
    ) -> bool: ...

    # Extract token usage if available
    def extract_usage(self, payload: dict[str, Any]) -> Optional[Usage]: ...
