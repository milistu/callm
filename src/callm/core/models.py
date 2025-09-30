from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RateLimitConfig:
    """
    Configuration for API rate limiting.

    Attributes:
        max_requests_per_minute (float): Maximum number of requests allowed per minute
        max_tokens_per_minute (float): Maximum number of tokens allowed per minute
    """

    max_requests_per_minute: float
    max_tokens_per_minute: float


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior with exponential backoff.

    Attributes:
        max_attempts (int): Maximum number of retry attempts for failed requests
        base_delay_seconds (float): Initial delay before first retry
        max_delay_seconds (float): Maximum delay between retries (caps exponential growth)
        jitter (float): Random variation factor (0.0-1.0) to prevent thundering herd
    """

    max_attempts: int = 5
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 15.0
    jitter: float = 0.1


@dataclass
class FilesConfig:
    """
    Configuration for input/output file paths.

    Attributes:
        save_file (str): Path to save successful API responses (JSONL format)
        error_file (str): Path to save failed requests and errors (JSONL format)
    """

    save_file: str
    error_file: str
