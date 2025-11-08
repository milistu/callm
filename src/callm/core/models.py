from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RateLimitConfig:
    """
    Configuration for API rate limiting.

    Attributes:
        max_requests_per_minute (float): Maximum number of requests allowed per minute
        max_tokens_per_minute (float): Maximum number of tokens allowed per minute
    """

    max_requests_per_minute: float
    max_tokens_per_minute: float | None


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


@dataclass
class RequestResult:
    """
    Result of a single API request.

    Attributes:
        request (dict[str, Any]): The original request payload
        response (Optional[dict[str, Any]]): The API response if successful
        error (Optional[str]): Error message if request failed
        metadata (Optional[dict[str, Any]]): Optional metadata from request
        attempts (int): Number of attempts made for this request, defaults to 1
    """

    request: dict[str, Any]
    response: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None
    attempts: int = 1


@dataclass
class ProcessingStats:
    """
    Statistics from a processing run.

    Attributes:
        total_requests (int): Total number of requests processed
        successful (int): Number of successful requests
        failed (int): Number of failed requests
        total_input_tokens (int): Total input tokens consumed
        total_output_tokens (int): Total output tokens generated
        rate_limit_errors (int): Number of rate limit errors encountered
        api_errors (int): Number of API errors encountered
        other_errors (int): Number of other errors encountered
        duration_seconds (float): Total processing time in seconds
    """

    total_requests: int
    successful: int
    failed: int
    total_input_tokens: int
    total_output_tokens: int
    rate_limit_errors: int
    api_errors: int
    other_errors: int
    duration_seconds: float


@dataclass
class ProcessingResult:
    """
    Results from parallel API processing.

    Contains all successful results, failed results, and processing statistics.

    Attributes:
        successes (list[RequestResult]): List of successful request results
        failures (list[RequestResult]): List of failed request results
        stats (ProcessingStats): Processing statistics

    Example:
        >>> result = await process_api_requests(provider, requests, rate_limit)
        >>> print(f"Success rate: {result.stats.successful}/{result.stats.total_requests}")
        >>> for success in result.successes:
        ...     print(success.response)
    """

    successes: list[RequestResult]
    failures: list[RequestResult]
    stats: ProcessingStats
