from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from aiohttp import ClientSession
from loguru import logger
from tqdm import tqdm

from callm.core.io import stream_jsonl, write_error, write_result
from callm.core.models import (
    FilesConfig,
    ProcessingResult,
    ProcessingStats,
    RateLimitConfig,
    RequestResult,
    RetryConfig,
)
from callm.core.rate_limit import TokenBucket
from callm.core.retry import Backoff
from callm.providers.base import BaseProvider
from callm.utils import task_id_generator, validate_jsonl_file

"""
Core async engine for parallel API request processing.

This module implements the main processing loop that:
- Reads requests from JSONL files
- Enforces rate limits (RPM and TPM) using token buckets
- Handles retries with exponential backoff
- Writes results and errors to JSONL files

The engine is provider-agnostic and works with any Provider implementation.
"""

SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR = 15
SECONDS_TO_SLEEP_EACH_LOOP = 0.001


@dataclass
class StatusTracker:
    """
    Tracks metrics and status during parallel API processing.

    This tracker maintains counters for different types of outcomes
    and is shared across all async tasks to provide global statistics.

    Attributes:
        num_tasks_started (int): Total number of tasks initiated
        num_tasks_in_progress (int): Current number of active tasks
        num_tasks_succeeded (int): Number of successfully completed tasks
        num_tasks_failed (int): Number of tasks that failed after all retries
        num_rate_limit_errors (int): Count of rate limit (429) errors encountered
        num_api_errors (int): Count of other API errors (4xx, 5xx)
        num_other_errors (int): Count of network/parsing/unexpected errors
        time_of_last_rate_limit_error (float): Timestamp of most recent rate limit error
        total_input_tokens (int): Sum of input tokens across all successful requests
        total_output_tokens (int): Sum of output tokens across all successful requests
    """

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0
    time_of_last_rate_limit_error: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


@dataclass
class APIRequest:
    """
    Represents a single API request with retry logic.

    Each request tracks its own state including retry attempts,
    token consumption for rate limiting, and results/errors.

    Attributes:
        task_id (int): Unique identifier for this request
        request_json (dict[str, Any]): The API request payload
        token_consumption (int): Estimated tokens for rate limit budgeting
        attempts_left (int): Remaining retry attempts
        metadata (Optional[dict[str, Any]]): Optional metadata to include in output
        result (list[object]): List of errors encountered across all attempts
    """

    task_id: int
    request_json: dict[str, Any]
    token_consumption: int
    attempts_left: int
    metadata: dict[str, Any] | None = None
    result: list[object] = field(default_factory=list)

    def _format_error_data(self) -> list[Any]:
        """Format error data for JSONL output."""
        if self.metadata is not None:
            return [
                self.request_json,
                [str(e) for e in self.result],
                self.metadata,
            ]
        return [self.request_json, [str(e) for e in self.result]]

    def _format_success_data(self, payload: dict[str, Any]) -> list[Any]:
        """Format success data for JSONL output."""
        if self.metadata is not None:
            return [self.request_json, payload, self.metadata]
        return [self.request_json, payload]

    async def call_api(
        self,
        session: ClientSession,
        provider: BaseProvider,
        headers: dict[str, str],
        retry_queue: asyncio.Queue[APIRequest],
        status: StatusTracker,
        backoff: Backoff,
        max_attempts: int,
        pbar: Any,  # tqdm progress bar (no type stubs available)
        files: FilesConfig | None = None,
        on_success: Callable[[list[Any]], None] | None = None,
        on_failure: Callable[[list[Any]], None] | None = None,
    ) -> None:
        """
        Execute the API request with error handling and retry logic.

        This method:
        1. Sends the request via the provider
        2. Checks for errors and classifies them
        3. Either retries (if attempts remain) or logs failure
        4. Writes results to appropriate output files

        Args:
            session (ClientSession): Aiohttp client session for HTTP requests
            provider (Provider): Provider implementation for API calls
            headers (dict[str, str]): HTTP headers including authentication
            retry_queue (asyncio.Queue["APIRequest"]): Queue for scheduling retries
            status (StatusTracker): Shared status tracker for metrics
            backoff (Backoff): Backoff calculator for retry delays
            max_attempts (int): Maximum number of retry attempts
            pbar (tqdm): Progress bar for tracking requests
            files (FilesConfig): Configuration for output files (optional)
            on_success (Callable[[list[Any]], None]): Callback for successful requests (optional)
            on_failure (Callable[[list[Any]], None]): Callback for failed requests (optional)
        """
        error: Any | None = None
        payload: dict[str, Any] | None = None
        try:
            payload, response_headers = await provider.send(
                session=session, headers=headers, request_json=self.request_json
            )
            parsed_error = provider.parse_error(payload)
            if parsed_error:
                error = payload
                # rate limit detection
                if provider.is_rate_limited(payload, response_headers):
                    status.time_of_last_rate_limit_error = time.time()
                    status.num_rate_limit_errors += 1
                    logger.debug(f"Task {self.task_id}: Rate limited by {provider.name}")
                else:
                    status.num_api_errors += 1
                    logger.debug(
                        f"Task {self.task_id}: API error from {provider.name}: {parsed_error}"
                    )
        except Exception as e:
            error = e
            status.num_other_errors += 1
            logger.warning(
                f"Task {self.task_id}: Error for {provider.name}: {type(e).__name__}: {e}"
            )

        if error is not None:
            self.result.append(error)
            if self.attempts_left:
                attempt_index = max_attempts - self.attempts_left - 1
                delay = backoff.compute_delay(attempt_index)
                logger.debug(
                    f"Task {self.task_id}: Retrying in {delay:.2f}s "
                    f"(attempt {attempt_index + 2}/{max_attempts})"
                )
                asyncio.create_task(_requeue_after(retry_queue, self, delay))
            else:
                logger.info(f"Task {self.task_id}: Failed after {max_attempts} attempts")
                error_data = self._format_error_data()
                if on_failure:
                    on_failure(error_data)
                elif files:
                    write_error(error_data, files.error_file)

                status.num_tasks_in_progress -= 1
                status.num_tasks_failed += 1
                pbar.update(1)
        else:
            if payload is None:
                logger.error(f"Task {self.task_id}: No payload received despite no error")
                status.num_tasks_failed += 1
                status.num_tasks_in_progress -= 1
                return

            # Extract usage metrics if available
            usage = provider.extract_usage(payload)
            if usage:
                status.total_input_tokens += usage.input_tokens
                status.total_output_tokens += usage.output_tokens
                logger.debug(
                    f"Task {self.task_id}: Completed successfully "
                    f"(input: {usage.input_tokens}, output: {usage.output_tokens} tokens)"
                )
            else:
                logger.debug(f"Task {self.task_id}: Completed successfully")

            success_data = self._format_success_data(payload)

            if on_success:
                on_success(success_data)
            elif files:
                write_result(success_data, files.save_file)

            status.num_tasks_in_progress -= 1
            status.num_tasks_succeeded += 1
            pbar.update(1)


async def _requeue_after(q: asyncio.Queue[APIRequest], req: APIRequest, seconds: float) -> None:
    """
    Schedule a request to be retried after a delay.

    Args:
        q (asyncio.Queue["APIRequest"]): Retry queue to add the request to
        req (APIRequest): The request to retry
        seconds (float): Delay in seconds before retrying
    """
    await asyncio.sleep(seconds)
    q.put_nowait(req)


def _setup_logger(logging_level: int) -> None:
    """
    Configure logger with clean format.

    Args:
        logging_level (int): Loguru logging level (20=INFO, 10=DEBUG)
    """
    logger.remove()

    # Show module info only at DEBUG level (10 or lower)
    if logging_level <= 10:
        # DEBUG: Include module and line number for debugging
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    else:
        # INFO/WARNING/ERROR: Clean format for end users
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        )

    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format=log_format,
        colorize=True,
        level=logging_level,
    )


def _setup_rate_limiting(
    rate_limit: RateLimitConfig, retry: RetryConfig
) -> tuple[TokenBucket, TokenBucket | None, Backoff]:
    """
    Initialize rate limiting token buckets and retry backoff calculator.

    Creates two independent token buckets for request and token rate limiting,
    plus a backoff calculator for retry delays.

    Args:
        rate_limit: Rate limit configuration
        retry: Retry configuration

    Returns:
        Tuple of (requests_bucket, tokens_bucket, backoff).
        tokens_bucket will be None if max_tokens_per_minute is None.
    """
    requests_bucket = TokenBucket.start(capacity_per_minute=rate_limit.max_requests_per_minute)
    # Only create tokens bucket if TPM limit exists
    tokens_bucket = None
    if rate_limit.max_tokens_per_minute is not None:
        tokens_bucket = TokenBucket.start(capacity_per_minute=rate_limit.max_tokens_per_minute)

    backoff = Backoff(
        base_delay_seconds=retry.base_delay_seconds,
        max_delay_seconds=retry.max_delay_seconds,
        jitter=retry.jitter,
    )
    return requests_bucket, tokens_bucket, backoff


def _log_summary(
    status: StatusTracker,
    duration: float,
    output_description: str = "Results saved",
) -> None:
    """
    Log processing summary and statistics.

    Args:
        status: Status tracker with metrics
        duration: Processing duration in seconds
        output_description: Description of where results went
    """
    logger.info(f"Parallel processing complete. {output_description}")
    logger.info(
        f"Successfully completed {status.num_tasks_succeeded:,} / "
        f"{status.num_tasks_started:,} requests "
        f"in {duration:.1f}s"
    )

    # Log usage statistics if available
    if status.total_input_tokens > 0 or status.total_output_tokens > 0:
        total_tokens = status.total_input_tokens + status.total_output_tokens
        logger.info(
            f"Token usage - Input: {status.total_input_tokens:,}, "
            f"Output: {status.total_output_tokens:,}, "
            f"Total: {total_tokens:,}"
        )

    # Log warnings for failures
    if status.num_tasks_failed > 0:
        logger.warning(
            f"{status.num_tasks_failed:,} / {status.num_tasks_started:,} requests failed"
        )
    if status.num_rate_limit_errors > 0:
        logger.warning(
            f"{status.num_rate_limit_errors:,} rate limit errors received. "
            f"Consider running at lower rate."
        )


async def _process_requests_internal(
    provider: BaseProvider,
    request_iterator: Iterator[dict[str, Any]],
    total_requests: int,
    rate_limit: RateLimitConfig,
    retry: RetryConfig,
    status: StatusTracker,
    files: FilesConfig | None = None,
    on_success: Callable[[list[Any]], None] | None = None,
    on_failure: Callable[[list[Any]], None] | None = None,
) -> None:
    """
    Internal function for processing requests (shared between file and memory modes).

    Args:
        provider (Provider): Provider implementation
        request_iterator (Iterator[dict[str, Any]]): Iterator over requests
        total_requests (int): Total number of requests for progress bar
        rate_limit (RateLimitConfig): Rate limit configuration
        retry: Retry configuration
        status (StatusTracker): Status tracker to update
        files (Optional[FilesConfig]): Files configuration (optional)
        on_success (Optional[Callable[[list[Any]], None]]): Callback for successful
            requests (optional)
        on_failure (Optional[Callable[[list[Any]], None]]): Callback for failed requests (optional)
    """

    headers = provider.build_headers()
    requests_bucket, tokens_bucket, backoff = _setup_rate_limiting(rate_limit, retry)

    queue_of_requests_to_retry: asyncio.Queue[APIRequest] = asyncio.Queue()
    generator = task_id_generator()
    next_request: APIRequest | None = None

    pbar = tqdm(total=total_requests or None, desc="Completed requests", unit="req")

    async with ClientSession() as session:
        while True:
            if next_request is None:
                if not queue_of_requests_to_retry.empty():
                    next_request = queue_of_requests_to_retry.get_nowait()
                else:
                    try:
                        request_json = next(request_iterator)
                        next_request = APIRequest(
                            task_id=next(generator),
                            request_json=request_json,
                            token_consumption=provider.estimate_input_tokens(request_json),
                            attempts_left=retry.max_attempts,
                            metadata=request_json.pop("metadata", None),
                        )
                        status.num_tasks_started += 1
                        status.num_tasks_in_progress += 1
                    except StopIteration:
                        pass

            if next_request is not None:
                enough_requests = requests_bucket.try_consume(1)

                # Only check token limit if tokens bucket exists
                enough_tokens = True
                if tokens_bucket is not None:
                    enough_tokens = tokens_bucket.try_consume(next_request.token_consumption)

                if enough_requests and enough_tokens:
                    next_request.attempts_left -= 1
                    asyncio.create_task(
                        next_request.call_api(
                            session=session,
                            provider=provider,
                            headers=headers,
                            retry_queue=queue_of_requests_to_retry,
                            files=files,
                            status=status,
                            backoff=backoff,
                            max_attempts=retry.max_attempts,
                            pbar=pbar,
                            on_success=on_success,
                            on_failure=on_failure,
                        )
                    )
                    next_request = None

            if status.num_tasks_in_progress == 0:
                break

            await asyncio.sleep(SECONDS_TO_SLEEP_EACH_LOOP)

            since_rate_limit_error = time.time() - status.time_of_last_rate_limit_error
            if since_rate_limit_error < SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR:
                await asyncio.sleep(
                    SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR - since_rate_limit_error
                )

    pbar.close()


async def process_api_requests_from_file(
    provider: BaseProvider,
    requests_file: str,
    rate_limit: RateLimitConfig,
    retry: RetryConfig | None = None,
    files: FilesConfig | None = None,
    logging_level: int = 20,
) -> None:
    """
    Process API requests from a JSONL file in parallel with rate limiting.

    This is the main entry point for the library. It reads requests from a
    JSONL file, processes them in parallel while respecting rate limits,
    and writes results and errors to output files.

    Features:
    - Parallel processing with configurable concurrency
    - Rate limiting for both requests per minute (RPM) and tokens per minute (TPM)
    - Automatic retry with exponential backoff for failed requests
    - Separate output files for successes and failures
    - Progress bar and structured logging

    Args:
        provider (Provider): Provider implementation for the target API
        requests_file (str): Path to input JSONL file with requests
        rate_limit (RateLimitConfig): Rate limit configuration (RPM and TPM)
        retry (Optional[RetryConfig]): Optional retry configuration (uses defaults if None)
        files (Optional[FilesConfig]): Optional file configuration (auto-generated if None)
        logging_level (int): Loguru logging level (20=INFO, 10=DEBUG)

    Raises:
        ValueError: If file paths don't end with .jsonl
        FileNotFoundError: If requests_file doesn't exist

    Example:
        >>> provider = OpenAIProvider(
        ...     api_key="sk-...",
        ...     model="gpt-4o",
        ...     request_url="https://api.openai.com/v1/responses"
        ... )
        >>> await process_api_requests_from_file(
        ...     provider=provider,
        ...     requests_file="requests.jsonl",
        ...     rate_limit=RateLimitConfig(
        ...         max_requests_per_minute=100,
        ...         max_tokens_per_minute=50000
        ...     )
        ... )
    """
    _setup_logger(logging_level)

    validate_jsonl_file(requests_file, "Requests file")
    if not os.path.exists(requests_file):
        raise FileNotFoundError(f"Requests file not found: {requests_file}")

    if files is None:
        files = FilesConfig(
            save_file=requests_file.replace(".jsonl", "_results.jsonl"),
            error_file=requests_file.replace(".jsonl", "_errors.jsonl"),
        )
    else:
        validate_jsonl_file(files.save_file, "Save file")
        validate_jsonl_file(files.error_file, "Error file")

    retry = retry or RetryConfig()

    # Get number of requests in file
    try:
        with open(requests_file, encoding="utf-8") as _f:
            total_requests = sum(1 for _ in _f)
    except Exception:
        total_requests = 0

    status = StatusTracker()
    start_time = time.time()

    await _process_requests_internal(
        provider=provider,
        request_iterator=stream_jsonl(requests_file),
        total_requests=total_requests,
        rate_limit=rate_limit,
        retry=retry,
        status=status,
        files=files,
    )

    duration = time.time() - start_time
    description = f"Results saved to {files.save_file}"
    description += f" and {files.error_file}" if status.num_tasks_failed > 0 else ""
    _log_summary(status, duration, output_description=description)


async def process_api_requests(
    provider: BaseProvider,
    requests: list[dict[str, Any]],
    rate_limit: RateLimitConfig,
    retry: RetryConfig | None = None,
    logging_level: int = 20,
    max_batch_size: int = 10_000,
) -> ProcessingResult:
    """
    Process API requests from a list with in-memory results.

    This function processes requests in parallel while respecting rate limits and
    returns results in memory. Perfect for small to medium batches and programmatic use.

    For large batches (> 10,000 requests), use process_api_requests_from_file()
    instead to avoid memory issues.

    Args:
        provider (Provider): Provider implementation for the target API
        requests (list[dict[str, Any]]): List of request dictionaries
        rate_limit (RateLimitConfig): Rate limit configuration (RPM and TPM)
        retry (Optional[RetryConfig]): Optional retry configuration (uses defaults if None)
        logging_level (int): Loguru logging level (20=INFO, 10=DEBUG)
        max_batch_size (int): Maximum allowed batch size (default: 10,000)

    Returns:
        ProcessingResult: Object containing successes, failures, and statistics

    Raises:
        ValueError: If batch size exceeds max_batch_size

    Example:
        >>> requests = [
        ...     {"messages": [{"role": "user", "content": "Hello"}]},
        ...     {"messages": [{"role": "user", "content": "Hi there"}]},
        ... ]
        >>> result = await process_api_requests(
        ...     provider=provider,
        ...     requests=requests,
        ...     rate_limit=RateLimitConfig(
        ...         max_requests_per_minute=100,
        ...         max_tokens_per_minute=50000
        ...     )
        ... )
        >>> print(f"Processed: {result.stats.successful}/{result.stats.total_requests}")
        >>> for success in result.successes:
        ...     print(success.response)
    """
    _setup_logger(logging_level)

    if len(requests) > max_batch_size:
        raise ValueError(
            f"Batch size {len(requests)} exceeds maximum {max_batch_size}. "
            f"For large batches, use process_api_requests_from_file() instead."
        )

    retry = retry or RetryConfig()

    # In-memory result collectors
    successes: list[RequestResult] = []
    failures: list[RequestResult] = []

    # Memory output handlers
    def on_success(entry: list[Any]) -> None:
        """Capture successful result in memory instead of writing to file."""
        request_data = entry[0]
        response_data = entry[1]
        metadata = entry[2] if len(entry) > 2 else None

        successes.append(
            RequestResult(
                request=request_data,
                response=response_data,
                error=None,
                metadata=metadata,
                attempts=1,
            )
        )

    def on_failure(entry: list[Any]) -> None:
        """Capture failed result in memory instead of writing to file."""
        request_data = entry[0]
        errors = entry[1]
        metadata = entry[2] if len(entry) > 2 else None

        error_msg = "; ".join(str(e) for e in errors) if isinstance(errors, list) else str(errors)

        failures.append(
            RequestResult(
                request=request_data,
                response=None,
                error=error_msg,
                metadata=metadata,
                attempts=1,
            )
        )

    status = StatusTracker()
    start_time = time.time()

    await _process_requests_internal(
        provider=provider,
        request_iterator=iter(requests),
        total_requests=len(requests),
        rate_limit=rate_limit,
        retry=retry,
        status=status,
        on_success=on_success,
        on_failure=on_failure,
    )

    duration = time.time() - start_time

    # Log summary
    _log_summary(status, duration, "Processing complete (in-memory)")

    return ProcessingResult(
        successes=successes,
        failures=failures,
        stats=ProcessingStats(
            total_requests=status.num_tasks_started,
            successful=status.num_tasks_succeeded,
            failed=status.num_tasks_failed,
            total_input_tokens=status.total_input_tokens,
            total_output_tokens=status.total_output_tokens,
            rate_limit_errors=status.num_rate_limit_errors,
            api_errors=status.num_api_errors,
            other_errors=status.num_other_errors,
            duration_seconds=duration,
        ),
    )
