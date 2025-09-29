from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from aiohttp import ClientSession
from loguru import logger
from tqdm import tqdm

from callm.core.io import stream_jsonl, write_error, write_result
from callm.core.rate_limit import TokenBucket
from callm.core.retry import Backoff
from callm.providers.base import Provider
from callm.providers.models import FilesConfig, RateLimitConfig, RetryConfig
from callm.utils import task_id_generator_function

SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR = 15
SECONDS_TO_SLEEP_EACH_LOOP = 0.001


@dataclass
class StatusTracker:
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0
    time_of_last_rate_limit_error: float = 0.0


@dataclass
class APIRequest:
    task_id: int
    request_json: dict[str, Any]
    token_consumption: int
    attempts_left: int
    metadata: Optional[dict[str, Any]] = None
    result: list[object] = field(default_factory=list)

    async def call_api(
        self,
        session: ClientSession,
        provider: Provider,
        headers: dict[str, str],
        retry_queue: asyncio.Queue["APIRequest"],
        files: FilesConfig,
        status: StatusTracker,
        backoff: Backoff,
        max_attempts: int,
    ) -> None:
        error: Optional[Any] = None
        payload: Optional[dict[str, Any]] = None
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
                else:
                    status.num_api_errors += 1
        except Exception as e:
            error = e
            status.num_other_errors += 1

        if error is not None:
            self.result.append(error)
            if self.attempts_left:
                attempt_index = max_attempts - self.attempts_left - 1
                delay = backoff.compute_delay(attempt_index)
                asyncio.create_task(_requeue_after(retry_queue, self, delay))
            else:
                if self.metadata is not None:
                    error_data: list[dict[str, Any] | list[Any]] = [
                        self.request_json,
                        [str(e) for e in self.result],
                        self.metadata,
                    ]
                else:
                    error_data = [self.request_json, [str(e) for e in self.result]]
                write_error(error_data, files.error_file)
                status.num_tasks_in_progress -= 1
                status.num_tasks_failed += 1
        else:
            assert payload is not None
            if self.metadata is not None:
                success_data: list[dict[str, Any] | list[Any]] = [
                    self.request_json,
                    payload,
                    self.metadata,
                ]
            else:
                success_data = [self.request_json, payload]
            write_result(success_data, files.save_file)
            status.num_tasks_in_progress -= 1
            status.num_tasks_succeeded += 1


async def _requeue_after(
    q: asyncio.Queue["APIRequest"], req: "APIRequest", seconds: float
) -> None:
    await asyncio.sleep(seconds)
    q.put_nowait(req)


async def process_api_requests_from_file(
    provider: Provider,
    requests_file: str,
    rate_limit: RateLimitConfig,
    retry: RetryConfig | None = None,
    files: FilesConfig | None = None,
    logging_level: int = 20,
) -> None:

    if retry is None:
        retry = RetryConfig()

    if not requests_file.endswith(".jsonl"):
        raise ValueError("Requests file must be a JSONL file")

    if files is None:
        files = FilesConfig(
            save_file=requests_file.replace(".jsonl", "_results.jsonl"),
            error_file=requests_file.replace(".jsonl", "_errors.jsonl"),
        )
    else:
        if not files.save_file.endswith(".jsonl"):
            raise ValueError("Save file must be a JSONL file")
        if not files.error_file.endswith(".jsonl"):
            raise ValueError("Error file must be a JSONL file")

    # Initialize logging
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level=logging_level)
    logger.debug(f"Logging initialized at level {logging_level}")

    headers = provider.build_headers()

    requests_bucket = TokenBucket.start(
        capacity_per_minute=rate_limit.max_requests_per_minute
    )
    tokens_bucket = TokenBucket.start(
        capacity_per_minute=rate_limit.max_tokens_per_minute
    )
    backoff = Backoff(
        base_delay_seconds=retry.base_delay_seconds,
        max_delay_seconds=retry.max_delay_seconds,
        jitter=retry.jitter,
    )

    queue_of_requests_to_retry: asyncio.Queue[APIRequest] = asyncio.Queue()
    task_id_generator = task_id_generator_function()
    status = StatusTracker()
    next_request: Optional[APIRequest] = None

    # Get number of requests in file
    try:
        with open(requests_file, mode="r", encoding="utf-8") as _f:
            total_requests = sum(1 for _ in _f)
    except Exception:
        total_requests = 0

    pbar = tqdm(total=total_requests or None, desc="Starting requests", unit="req")

    async with ClientSession() as session:
        req_iter = stream_jsonl(requests_file)
        while True:
            if next_request is None:
                if not queue_of_requests_to_retry.empty():
                    next_request = queue_of_requests_to_retry.get_nowait()
                else:
                    try:
                        request_json = next(req_iter)
                        next_request = APIRequest(
                            task_id=next(task_id_generator),
                            request_json=request_json,
                            token_consumption=provider.estimate_input_tokens(
                                request_json
                            ),
                            attempts_left=retry.max_attempts,
                            metadata=request_json.pop("metadata", None),
                        )
                        status.num_tasks_started += 1
                        status.num_tasks_in_progress += 1
                        pbar.update(1)
                    except StopIteration:
                        pass

            if next_request is not None:
                enough_requests = requests_bucket.try_consume(1)
                enough_tokens = tokens_bucket.try_consume(
                    next_request.token_consumption
                )
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
    logger.info(f"Parallel processing complete. Results saved to {files.save_file}")
    if status.num_tasks_failed > 0:
        logger.warning(
            f"{status.num_tasks_failed} / {status.num_tasks_started} requests failed. Errors logged to {files.error_file}"
        )
    if status.num_rate_limit_errors > 0:
        logger.warning(
            f"{status.num_rate_limit_errors} rate limit errors received. Consider running at lower rate."
        )
