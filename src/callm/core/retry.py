from __future__ import annotations

import random
import time
from dataclasses import dataclass

"""
Exponential backoff retry logic with jitter.

Provides configurable retry delays that grow exponentially
with random jitter to avoid thundering herd problems.
"""


@dataclass
class Backoff:
    """
    Exponential backoff calculator with jitter.

    Computes retry delays that:
    - Grow exponentially with each attempt (2^attempt)
    - Are capped at a maximum delay
    - Include random jitter to prevent synchronized retries

    Attributes:
        base_delay_seconds (float): Initial delay for first retry
        max_delay_seconds (float): Maximum delay (caps exponential growth)
        jitter (float): Random variation factor (0.0-1.0)
    """

    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 15.0
    jitter: float = 0.1

    def sleep(self, attempt_index: int) -> None:
        """
        Sleep for the computed backoff delay (synchronous).

        Note: This method is synchronous and not used in async code.
        Use compute_delay() for async contexts.

        Args:
            attempt_index (int): Zero-based attempt number
        """
        # attempt_index: 0-based; grows exponentially
        delay = min(
            self.max_delay_seconds, self.base_delay_seconds * (2**attempt_index)
        )
        # jitter in range [-jitter, +jitter] proportionally
        noise = delay * self.jitter * (2 * random.random() - 1)
        time.sleep(max(0.0, delay + noise))

    def compute_delay(self, attempt_index: int) -> float:
        """
        Calculate backoff delay for the given attempt.

        Delay = min(max_delay, base_delay * 2^attempt_index) + jitter

        Args:
            attempt_index (int): Zero-based attempt number

        Returns:
            float: Delay in seconds (always >= 0)
        """
        delay = min(
            self.max_delay_seconds, self.base_delay_seconds * (2**attempt_index)
        )
        noise = delay * self.jitter * (2 * random.random() - 1)
        result: float = max(0.0, delay + noise)
        return result
