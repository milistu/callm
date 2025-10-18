"""
callm: Calmly call LLMs in parallel, within rate limits.

A Python library for parallel processing of LLM API requests with:
- Rate limiting (requests per minute and tokens per minute)
- Automatic retry with exponential backoff
- Provider-agnostic architecture
- JSONL batch processing
- Usage tracking and metrics

Example:
    >>> from callm import process_api_requests_from_file, RateLimitConfig
    >>> from callm.providers import OpenAIProvider
    >>>
    >>> provider = OpenAIProvider(
    ...     api_key="sk-...",
    ...     model="gpt-4o",
    ...     request_url="https://api.openai.com/v1/chat/completions"
    ... )
    >>>
    >>> await process_api_requests_from_file(
    ...     provider=provider,
    ...     requests_file="requests.jsonl",
    ...     rate_limit=RateLimitConfig(
    ...         max_requests_per_minute=100,
    ...         max_tokens_per_minute=50000
    ...     )
    ... )
"""

from callm.core.engine import process_api_requests, process_api_requests_from_file
from callm.core.models import (
    FilesConfig,
    ProcessingResult,
    ProcessingStats,
    RateLimitConfig,
    RequestResult,
    RetryConfig,
)
from callm.providers import BaseProvider, get_provider, register_provider
from callm.providers.models import Usage

__version__ = "0.1.0"

__all__ = [
    # Main processing function
    "process_api_requests_from_file",
    "process_api_requests",
    # Configuration models
    "RateLimitConfig",
    "RetryConfig",
    "FilesConfig",
    # Result models
    "ProcessingResult",
    "ProcessingStats",
    "RequestResult",
    "Usage",
    # Provider interface
    "BaseProvider",
    "get_provider",
    "register_provider",
    # Version
    "__version__",
]
