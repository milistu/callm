import pytest
from pytest import approx

from callm.core.rate_limit import TokenBucket


class TestTokenBucket:
    """Tests for the TokenBucket rate limiter."""

    def test_start_creates_full_bucket(self) -> None:
        """Test that a new bucket starts at full capacity."""
        capacity = 100.0

        bucket = TokenBucket.start(capacity_per_minute=capacity)

        assert bucket.capacity_per_minute == capacity
        assert bucket.available == capacity
        assert bucket.last_update_time > 0

    @pytest.mark.parametrize(
        argnames="capacity,consume,expected_remaining",
        argvalues=[
            (100.0, 30.0, 70.0),
            (100.0, 100.0, 0.0),
        ],
    )
    def test_consume_tokens_successfully(
        self, capacity: float, consume: float, expected_remaining: float
    ) -> None:
        """Test consuming tokens when enough are available."""
        bucket = TokenBucket.start(capacity_per_minute=capacity)

        result = bucket.try_consume(amount=consume)

        assert result is True
        assert bucket.available == approx(expected_remaining, abs=0.01)

    @pytest.mark.parametrize(
        argnames="capacity,consume",
        argvalues=[
            (100.0, 101.0),
            (100.0, 150.0),
        ],
    )
    def test_consume_fails_when_insufficient_tokens(self, capacity: float, consume: float) -> None:
        """Test that consumption fails when not enough tokens available."""
        bucket = TokenBucket.start(capacity_per_minute=capacity)

        result = bucket.try_consume(amount=consume)

        assert result is False
        assert bucket.available == approx(capacity, abs=0.01)

    def test_multiple_consumptions(self) -> None:
        """Test multiple sequential consumptions."""
        bucket = TokenBucket.start(capacity_per_minute=100.0)

        assert bucket.try_consume(amount=30.0) is True
        assert bucket.available == approx(70.0, abs=0.01)

        assert bucket.try_consume(amount=40.0) is True
        assert bucket.available == approx(30.0, abs=0.01)

        assert bucket.try_consume(amount=30.0) is True
        assert bucket.available == approx(0.0, abs=0.01)

        assert bucket.try_consume(amount=1.0) is False

    def test_tokens_refill_over_time(self) -> None:
        """Test that tokens refill based on elapsed time."""
        bucket = TokenBucket.start(capacity_per_minute=60.0)

        bucket.try_consume(amount=60.0)
        assert bucket.available == approx(0.0, abs=0.01)

        bucket.last_update_time -= 2.0  # 2 seconds ago
        bucket.refill()
        assert bucket.available >= 1.5  # 60 tokens per minute / 60 seconds = 1 token per second
        assert bucket.available <= 2.5  # but not more than 2 tokens

    def test_refill_does_not_exceed_capacity(self) -> None:
        """Test that refilling stops at maximum capacity."""
        bucket = TokenBucket.start(capacity_per_minute=60.0)

        bucket.try_consume(amount=30.0)
        assert bucket.available == approx(30.0, abs=0.01)

        # Simulate time passing
        bucket.last_update_time -= 100.0
        bucket.refill()

        assert bucket.available == approx(60.0, abs=0.01)
