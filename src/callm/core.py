import asyncio
import json
import time

from aiohttp import ClientSession
from asyncio import Queue
from loguru import logger
from tqdm import tqdm
from tiktoken import get_encoding

from callm.modules import APIRequest, StatusTracker
from callm.tokenizers.openai import num_tokens_consumed_from_request
from callm.utils import api_endpoint_from_url, task_id_generator_function

SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR = 15
SECONDS_TO_SLEEP_EACH_LOOP = 0.001


async def process_api_requests_from_file(
    requests_file: str,
    save_file: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
) -> None:
    """
    Process API requests in parallel, throttling to stay under rate limits.

    Args:
        requests_file (str): path to the file containing the requests to be processed
        save_file (str): path to the file where the results will be saved
        request_url (str): URL of the API endpoint to call
        api_key (str): API key to use
        max_requests_per_minute (float): target number of requests to make per minute (will make less if limited by tokens)
        max_tokens_per_minute (float): target number of tokens to use per minute (will use less if limited by requests)
        token_encoding_name (str): name of the token encoding used, as defined in the `tiktoken` package
        max_attempts (int): number of times to retry a failed request before giving up
        logging_level (int): level of logging to use; higher numbers will log fewer messages

    Returns:
        None
    """
    # Initialize logging
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level=logging_level)
    logger.debug(f"Logging initialized at level {logging_level}")

    # Initialize tokenizer
    tokenizer = get_encoding(token_encoding_name)

    api_endpoint = api_endpoint_from_url(request_url)

    if "/deployments" in request_url:
        request_header = {"api-key": f"{api_key}"}
    else:
        request_header = {"Authorization": f"Bearer {api_key}"}

    # Initialize trackers
    queue_of_requests_to_retry: Queue[APIRequest] = asyncio.Queue()
    task_id_generator = task_id_generator_function()

    status_tracker = StatusTracker()
    next_request: APIRequest | None = None

    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()
    file_not_finished = True

    logger.debug("Initialization complete.")

    # Count total for progress bar
    try:
        with open(requests_file, "r", encoding="utf-8") as _f:
            total_requests = sum(1 for _ in _f)
    except Exception:
        total_requests = 0

    with open(requests_file) as file:
        requests = file.__iter__()
        logger.debug("File opened. Entering main loop.")

        # Create a single tqdm progress bar at the bottom
        pbar = tqdm(total=total_requests or None, desc="Starting requests", unit="req")

        async with ClientSession() as session:
            while True:
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logger.debug(
                            f"Retrying request {next_request.task_id}: {next_request}"
                        )
                    elif file_not_finished:
                        try:
                            request_json = json.loads(next(requests))
                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=request_json,
                                token_consumption=num_tokens_consumed_from_request(
                                    request_json, api_endpoint, tokenizer
                                ),
                                attempts_left=max_attempts,
                                metadata=request_json.pop("metadata", None),
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            pbar.update(1)
                            logger.debug(
                                f"Reading request {next_request.task_id}: {next_request}"
                            )
                        except StopIteration:
                            logger.debug("Read file exhausted")
                            file_not_finished = False

                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity
                    + max_requests_per_minute * seconds_since_update / 60.0,
                    max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity
                    + max_tokens_per_minute * seconds_since_update / 60.0,
                    max_tokens_per_minute,
                )
                last_update_time = current_time

                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if (
                        available_request_capacity >= 1
                        and available_token_capacity >= next_request_tokens
                    ):
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        # Call API
                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                save_file=save_file,
                                status_tracker=status_tracker,
                            )
                        )
                        next_request = None

                if status_tracker.num_tasks_in_progress == 0:
                    break

                await asyncio.sleep(SECONDS_TO_SLEEP_EACH_LOOP)

                seconds_since_rate_limit_error = (
                    time.time() - status_tracker.time_of_last_rate_limit_error
                )
                if (
                    seconds_since_rate_limit_error
                    < SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR
                ):
                    remaining_seconds_to_pause = (
                        SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR
                        - seconds_since_rate_limit_error
                    )
                    await asyncio.sleep(remaining_seconds_to_pause)

                    logger.warning(
                        f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR)}"
                    )

        pbar.close()
        logger.info(f"Parallel processing complete. Results saved to {save_file}")
        if status_tracker.num_tasks_failed > 0:
            logger.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_file}"
            )
        if status_tracker.num_rate_limit_errors > 0:
            logger.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at lower rate."
            )
