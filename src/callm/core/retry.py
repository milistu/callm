from __future__ import annotations

import random
import time
from dataclasses import dataclass


@dataclass
class Backoff:
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 15.0
    jitter: float = 0.1

    def sleep(self, attempt_index: int) -> None:
        # attempt_index: 0-based; grows exponentially
        delay = min(
            self.max_delay_seconds, self.base_delay_seconds * (2**attempt_index)
        )
        # jitter in range [-jitter, +jitter] proportionally
        noise = delay * self.jitter * (2 * random.random() - 1)
        time.sleep(max(0.0, delay + noise))

    def compute_delay(self, attempt_index: int) -> float:
        delay = min(
            self.max_delay_seconds, self.base_delay_seconds * (2**attempt_index)
        )
        noise = delay * self.jitter * (2 * random.random() - 1)
        result: float = max(0.0, delay + noise)
        return result
