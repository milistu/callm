from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RateLimitConfig:
    max_requests_per_minute: float
    max_tokens_per_minute: float


@dataclass
class RetryConfig:
    max_attempts: int = 5
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 15.0
    jitter: float = 0.1


@dataclass
class FilesConfig:
    save_file: str
    error_file: str


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
