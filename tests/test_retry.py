from pytest import approx

from callm.core.retry import Backoff


class TestBackoff:
    """Test for exponential backoff calculator."""

    def test_exponential_growth_without_jitter(self) -> None:
        """Test that delays grow exponentially when jitter is disabled."""
        backoff = Backoff(base_delay_seconds=1.0, max_delay_seconds=10.0, jitter=0.0)

        # Exponential pattern: base * 2^attempt
        assert backoff.compute_delay(0) == approx(1.0)  # 1 * 2^0 = 1
        assert backoff.compute_delay(1) == approx(2.0)  # 1 * 2^1 = 2
        assert backoff.compute_delay(2) == approx(4.0)  # 1 * 2^2 = 4
        assert backoff.compute_delay(3) == approx(8.0)  # 1 * 2^3 = 8

    def test_delay_capped_at_maximum(self) -> None:
        """Test that delays never exceed max_delay_seconds."""
        backoff = Backoff(base_delay_seconds=1.0, max_delay_seconds=10.0, jitter=0.0)

        assert backoff.compute_delay(10) == approx(
            10.0
        )  # 1 * 2^10 = 1024, but capped at 10

    def test_jitter_adds_randomness_with_bounds(self) -> None:
        """Test that jitter keeps delays within expected range."""
        backoff = Backoff(base_delay_seconds=10.0, max_delay_seconds=100.0, jitter=0.1)

        # For attempt 0: base delay = 10.0
        # Jitter range: +/-1.0 (10% of 10.0)
        # Expected range: 9.0 to 11.0
        delays = [backoff.compute_delay(0) for _ in range(20)]

        assert all(
            9.0 <= delay <= 11.0 for delay in delays
        ), "Delays should be within 9-11 seconds"

    def test_delay_never_negative(self) -> None:
        """Test that deplays are never negative, even with extreme jitter."""
        backoff = Backoff(base_delay_seconds=0.1, max_delay_seconds=1.0, jitter=1.0)

        delays = [backoff.compute_delay(0) for _ in range(50)]

        assert all(delay >= 0.0 for delay in delays), "Delay should never be negative"
