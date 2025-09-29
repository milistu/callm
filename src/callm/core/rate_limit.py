from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class TokenBucket:
    capacity_per_minute: float
    available: float
    last_update_time: float

    @classmethod
    def start(cls, capacity_per_minute: float) -> "TokenBucket":
        now = time.time()
        return cls(capacity_per_minute, capacity_per_minute, now)

    def refill(self) -> None:
        now = time.time()
        elapsed = now - self.last_update_time
        self.available = min(
            self.capacity_per_minute,
            self.available + self.capacity_per_minute * elapsed / 60.0,
        )
        self.last_update_time = now

    def try_consume(self, amount: float) -> bool:
        self.refill()
        if self.available >= amount:
            self.available -= amount
            return True
        return False
