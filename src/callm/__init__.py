"""
callm: Calmly call LLMs in parallel, within rate limits.

A Python library for parallel processing of LLM API requests with:
- Rate limiting (requests per minute and tokens per minute)
- Automatic retry with exponential backoff
- Provider-agnostic architecture
- JSONL batch processing
- Usage tracking and metrics
"""

from callm.core.engine import process_requests
from callm.core.models import (
    ProcessingResults,
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
    "process_requests",
    # Configuration models
    "RateLimitConfig",
    "RetryConfig",
    # Result models
    "ProcessingResults",
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
