from __future__ import annotations

import time
from dataclasses import dataclass

"""
Token bucket rate limiter implementation.

Provides smooth rate limiting based on tokens that refill continuously
over time, rather than fixed time windows.
"""


@dataclass
class TokenBucket:
    """
    Token bucket rate limiter with continuous refill.

    Implements a token bucket algorithm that:
    - Starts with full capacity
    - Refills at a constant rate per second
    - Allows consumption only if enough tokens available
    - Provides smooth rate limiting without burst spikes

    Attributes:
        capacity_per_minute (float): Maximum tokens available
        available (float): Current number of available tokens
        last_update_time (float): Timestamp of last refill calculation
    """

    capacity_per_minute: float
    available: float
    last_update_time: float

    @classmethod
    def start(cls, capacity_per_minute: float) -> "TokenBucket":
        """
        Create a new token bucket starting at full capacity.

        Args:
            capacity_per_minute (float): Maximum capacity and refill rate

        Returns:
            TokenBucket: New TokenBucket instance
        """
        now = time.time()
        return cls(capacity_per_minute, capacity_per_minute, now)

    def refill(self) -> None:
        """
        Refill tokens based on time elapsed since last refill.

        Tokens refill at a constant rate of capacity_per_minute/60 per second.
        Available tokens are capped at capacity_per_minute.
        """
        now = time.time()
        elapsed = now - self.last_update_time
        self.available = min(
            self.capacity_per_minute,
            self.available + self.capacity_per_minute * elapsed / 60.0,
        )
        self.last_update_time = now

    def try_consume(self, amount: float) -> bool:
        """
        Attempt to consume tokens from the bucket.

        First refills the bucket, then checks if enough tokens are available.
        If yes, consumes the tokens and returns True. Otherwise returns False.

        Args:
            amount (float): Number of tokens to consume

        Returns:
            bool: True if tokens were consumed, False if insufficient tokens
        """
        self.refill()
        if self.available >= amount:
            self.available -= amount
            return True
        return False
