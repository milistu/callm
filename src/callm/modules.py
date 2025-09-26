import time
from asyncio import Queue
from dataclasses import dataclass, field
from typing import Optional

from aiohttp import ClientSession
from loguru import logger

from callm.utils import JSONValue, append_to_jsonl


@dataclass
class StatusTracker:
    """
    Stores metadata about the script's progress. Only one instance is created.
    """

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: float = (
        0  # used to cool off after hitting rate limits
    )


@dataclass
class APIRequest:
    """
    Stores an API request's inputs, outputs and other metadata.
    Contains a method to call the API.
    """

    task_id: int
    request_json: dict[str, JSONValue]
    token_consumption: int
    attempts_left: int
    metadata: Optional[dict[str, JSONValue]] = None
    result: list[object] = field(default_factory=list)

    async def call_api(
        self,
        session: ClientSession,
        request_url: str,
        request_header: dict[str, str],
        retry_queue: Queue["APIRequest"],
        save_file: str,
        error_file: str,
        status_tracker: StatusTracker,
    ) -> None:
        """
        Calls the API and saves results.

        Args:
            session (ClientSession): the session to use to make the API call
            request_url (str): the URL of the API endpoint to call
            request_header: (dict) - the header to use to make the API call
            retry_queue (Queue): the queue to use to retry failed requests
            save_file (str): the file to save the results to
            error_file (str): the file to save the errors to
            status_tracker (StatusTracker): the tracker to use to track the status of the request

        Returns:
            None
        """
        error: Optional[object] = None
        payload: Optional[dict[str, JSONValue]] = None

        try:
            async with session.post(
                url=request_url, headers=request_header, json=self.request_json
            ) as response:
                payload = await response.json()
            if isinstance(payload, dict) and "error" in payload:
                logger.warning(
                    f"Request {self.task_id} failed with error: {payload['error']}"
                )
                error = payload
                err_val = payload.get("error")
                if isinstance(err_val, dict):
                    message = str(err_val.get("message", ""))
                else:
                    message = str(err_val)
                if "rate limit" in message.lower():
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                else:
                    status_tracker.num_api_errors += 1

        except Exception as e:
            logger.warning(f"Request {self.task_id} failed with Exception: {e}")
            status_tracker.num_other_errors += 1
            error = e

        if error is not None:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logger.error(
                    f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}"
                )
                if self.metadata is not None:
                    error_data: list[dict[str, JSONValue] | list[JSONValue]] = [
                        self.request_json,
                        [str(e) for e in self.result],
                        self.metadata,
                    ]
                else:
                    error_data = [self.request_json, [str(e) for e in self.result]]
                append_to_jsonl(error_data, error_file)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            assert payload is not None
            if self.metadata is not None:
                success_data: list[dict[str, JSONValue] | list[JSONValue]] = [
                    self.request_json,
                    payload,
                    self.metadata,
                ]
            else:
                success_data = [self.request_json, payload]
            append_to_jsonl(success_data, save_file)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logger.debug(f"Request {self.task_id} saved to {save_file}")
