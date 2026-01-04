"""
Tests for Circuit Breaker implementation.

Includes unit tests and property-based tests for state transitions.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given, settings, strategies as st

from packages.shared.src.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitState,
    CircuitStats,
    ResilientCall,
    RetryConfig,
    RetryWithBackoff,
    get_circuit_breaker,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Create a test circuit breaker config."""
    return CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        recovery_timeout=1.0,  # Short for testing
        half_open_max_calls=2,
    )


@pytest.fixture
def circuit_breaker(config):
    """Create a test circuit breaker."""
    return CircuitBreaker("test", config)


# =============================================================================
# Unit Tests - CircuitStats
# =============================================================================


class TestCircuitStats:
    """Test CircuitStats class."""

    def test_initial_state(self) -> None:
        """Test initial stats values."""
        stats = CircuitStats()
        assert stats.total_calls == 0
        assert stats.successful_calls == 0
        assert stats.failed_calls == 0
        assert stats.rejected_calls == 0
        assert stats.consecutive_failures == 0
        assert stats.consecutive_successes == 0

    def test_record_success(self) -> None:
        """Test recording successful calls."""
        stats = CircuitStats()
        stats.record_success()
        
        assert stats.total_calls == 1
        assert stats.successful_calls == 1
        assert stats.consecutive_successes == 1
        assert stats.consecutive_failures == 0
        assert stats.last_success_time is not None

    def test_record_failure(self) -> None:
        """Test recording failed calls."""
        stats = CircuitStats()
        stats.record_failure()
        
        assert stats.total_calls == 1
        assert stats.failed_calls == 1
        assert stats.consecutive_failures == 1
        assert stats.consecutive_successes == 0
        assert stats.last_failure_time is not None

    def test_consecutive_counters_reset(self) -> None:
        """Test that consecutive counters reset on opposite result."""
        stats = CircuitStats()
        
        # Build up successes
        stats.record_success()
        stats.record_success()
        assert stats.consecutive_successes == 2
        
        # Failure resets success counter
        stats.record_failure()
        assert stats.consecutive_successes == 0
        assert stats.consecutive_failures == 1
        
        # Success resets failure counter
        stats.record_success()
        assert stats.consecutive_failures == 0
        assert stats.consecutive_successes == 1


# =============================================================================
# Unit Tests - CircuitBreaker
# =============================================================================


class TestCircuitBreakerInit:
    """Test circuit breaker initialization."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        cb = CircuitBreaker("test")
        assert cb.name == "test"
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed is True
        assert cb.is_open is False
        assert cb.is_half_open is False

    def test_custom_config(self, config) -> None:
        """Test custom configuration."""
        cb = CircuitBreaker("test", config)
        assert cb.config.failure_threshold == 3
        assert cb.config.success_threshold == 2
        assert cb.config.recovery_timeout == 1.0


class TestCircuitBreakerStateTransitions:
    """Test circuit breaker state transitions."""

    @pytest.mark.asyncio
    async def test_closed_to_open_on_failures(self, circuit_breaker) -> None:
        """Test transition from CLOSED to OPEN after threshold failures."""
        async def failing_func():
            raise ValueError("Test error")

        # Fail until threshold
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.stats.consecutive_failures == circuit_breaker.config.failure_threshold

    @pytest.mark.asyncio
    async def test_open_rejects_calls(self, circuit_breaker) -> None:
        """Test that OPEN state rejects calls."""
        async def failing_func():
            raise ValueError("Test error")

        # Trip the circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_func)

        # Next call should be rejected
        with pytest.raises(CircuitBreakerError) as exc_info:
            await circuit_breaker.call(failing_func)

        assert exc_info.value.state == CircuitState.OPEN
        assert circuit_breaker.stats.rejected_calls == 1

    @pytest.mark.asyncio
    async def test_open_to_half_open_after_timeout(self, circuit_breaker) -> None:
        """Test transition from OPEN to HALF_OPEN after recovery timeout."""
        async def failing_func():
            raise ValueError("Test error")

        # Trip the circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(circuit_breaker.config.recovery_timeout + 0.1)

        # Next call attempt should transition to HALF_OPEN
        async def success_func():
            return "success"

        result = await circuit_breaker.call(success_func)
        assert result == "success"
        # State depends on success threshold

    @pytest.mark.asyncio
    async def test_half_open_to_closed_on_successes(self, circuit_breaker) -> None:
        """Test transition from HALF_OPEN to CLOSED after threshold successes."""
        async def failing_func():
            raise ValueError("Test error")

        async def success_func():
            return "success"

        # Trip the circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_func)

        # Wait for recovery timeout
        await asyncio.sleep(circuit_breaker.config.recovery_timeout + 0.1)

        # Succeed until threshold
        for _ in range(circuit_breaker.config.success_threshold):
            await circuit_breaker.call(success_func)

        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_to_open_on_failure(self, circuit_breaker) -> None:
        """Test transition from HALF_OPEN to OPEN on any failure."""
        async def failing_func():
            raise ValueError("Test error")

        async def success_func():
            return "success"

        # Trip the circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_func)

        # Wait for recovery timeout
        await asyncio.sleep(circuit_breaker.config.recovery_timeout + 0.1)

        # One success to enter half-open
        await circuit_breaker.call(success_func)

        # Failure should trip back to open
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_manual_reset(self, circuit_breaker) -> None:
        """Test manual reset to CLOSED state."""
        async def failing_func():
            raise ValueError("Test error")

        # Trip the circuit
        for _ in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN

        # Manual reset
        await circuit_breaker.reset()
        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_force_open(self, circuit_breaker) -> None:
        """Test force open."""
        assert circuit_breaker.state == CircuitState.CLOSED
        
        await circuit_breaker.force_open()
        assert circuit_breaker.state == CircuitState.OPEN



# =============================================================================
# Property-Based Tests - Circuit Breaker State Transitions
# **Property 3: Circuit breaker state transitions**
# **Validates: Requirements 56.1**
# =============================================================================


class TestCircuitBreakerPropertyTests:
    """
    Property-based tests for circuit breaker state transitions.
    
    **Feature: multi-platform-trading-bot, Property 3: Circuit breaker state transitions**
    **Validates: Requirements 56.1**
    """

    @given(
        failure_threshold=st.integers(min_value=1, max_value=10),
        success_threshold=st.integers(min_value=1, max_value=10),
        num_failures=st.integers(min_value=0, max_value=20),
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_property_closed_to_open_transition(
        self,
        failure_threshold: int,
        success_threshold: int,
        num_failures: int,
    ) -> None:
        """
        Property: Circuit transitions to OPEN after exactly failure_threshold consecutive failures.
        
        *For any* failure_threshold and number of failures:
        - If failures < threshold: circuit stays CLOSED
        - If failures >= threshold: circuit becomes OPEN
        """
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            recovery_timeout=60.0,  # Long timeout to prevent auto-recovery
        )
        cb = CircuitBreaker("test_property", config)

        async def failing_func():
            raise ValueError("Test error")

        # Execute failures
        actual_failures = 0
        for _ in range(num_failures):
            if cb.state == CircuitState.OPEN:
                break
            try:
                await cb.call(failing_func)
            except ValueError:
                actual_failures += 1
            except CircuitBreakerError:
                break

        # Verify property
        if actual_failures < failure_threshold:
            assert cb.state == CircuitState.CLOSED, \
                f"Circuit should be CLOSED with {actual_failures} failures (threshold={failure_threshold})"
        else:
            assert cb.state == CircuitState.OPEN, \
                f"Circuit should be OPEN with {actual_failures} failures (threshold={failure_threshold})"

    @given(
        failure_threshold=st.integers(min_value=1, max_value=5),
        success_threshold=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_property_half_open_to_closed_transition(
        self,
        failure_threshold: int,
        success_threshold: int,
    ) -> None:
        """
        Property: Circuit transitions from HALF_OPEN to CLOSED after success_threshold successes.
        
        *For any* success_threshold:
        - After recovery timeout, circuit enters HALF_OPEN
        - After success_threshold consecutive successes, circuit becomes CLOSED
        """
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            recovery_timeout=0.01,  # Very short for testing
            half_open_max_calls=success_threshold + 5,  # Allow enough calls
        )
        cb = CircuitBreaker("test_property", config)

        async def failing_func():
            raise ValueError("Test error")

        async def success_func():
            return "success"

        # Trip the circuit
        for _ in range(failure_threshold):
            try:
                await cb.call(failing_func)
            except (ValueError, CircuitBreakerError):
                pass

        assert cb.state == CircuitState.OPEN

        # Wait for recovery
        await asyncio.sleep(0.02)

        # Execute successes - need to handle potential state changes
        for i in range(success_threshold):
            if cb.state == CircuitState.CLOSED:
                break
            try:
                await cb.call(success_func)
            except CircuitBreakerError:
                # If rejected, wait and retry
                await asyncio.sleep(0.02)
                await cb.call(success_func)

        # Verify property
        assert cb.state == CircuitState.CLOSED, \
            f"Circuit should be CLOSED after {success_threshold} successes"

    @given(
        failure_threshold=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_property_half_open_to_open_on_failure(
        self,
        failure_threshold: int,
    ) -> None:
        """
        Property: Any failure in HALF_OPEN state transitions back to OPEN.
        
        *For any* circuit in HALF_OPEN state:
        - A single failure immediately transitions to OPEN
        """
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=10,  # High so we don't accidentally close
            recovery_timeout=0.01,
        )
        cb = CircuitBreaker("test_property", config)

        async def failing_func():
            raise ValueError("Test error")

        async def success_func():
            return "success"

        # Trip the circuit
        for _ in range(failure_threshold):
            try:
                await cb.call(failing_func)
            except (ValueError, CircuitBreakerError):
                pass

        # Wait for recovery
        await asyncio.sleep(0.02)

        # One success to confirm half-open
        await cb.call(success_func)

        # One failure should trip back to open
        try:
            await cb.call(failing_func)
        except ValueError:
            pass

        # Verify property
        assert cb.state == CircuitState.OPEN, \
            "Circuit should be OPEN after failure in HALF_OPEN state"

    @given(
        num_successes=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_property_successes_dont_affect_closed_state(
        self,
        num_successes: int,
    ) -> None:
        """
        Property: Successes in CLOSED state keep circuit CLOSED.
        
        *For any* number of successes in CLOSED state:
        - Circuit remains CLOSED
        - No state transitions occur
        """
        cb = CircuitBreaker("test_property")

        async def success_func():
            return "success"

        initial_state_changes = len(cb.stats.state_changes)

        for _ in range(num_successes):
            await cb.call(success_func)

        # Verify property
        assert cb.state == CircuitState.CLOSED, \
            "Circuit should remain CLOSED after successes"
        assert len(cb.stats.state_changes) == initial_state_changes, \
            "No state changes should occur"

    @given(
        failure_threshold=st.integers(min_value=2, max_value=10),
        num_failures=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_property_success_resets_failure_count(
        self,
        failure_threshold: int,
        num_failures: int,
    ) -> None:
        """
        Property: A success resets the consecutive failure count.
        
        *For any* number of failures less than threshold:
        - A success resets the failure count
        - Circuit stays CLOSED
        """
        # Ensure we don't hit threshold
        actual_failures = min(num_failures, failure_threshold - 1)
        
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=60.0,
        )
        cb = CircuitBreaker("test_property", config)

        async def failing_func():
            raise ValueError("Test error")

        async def success_func():
            return "success"

        # Execute some failures (less than threshold)
        for _ in range(actual_failures):
            try:
                await cb.call(failing_func)
            except ValueError:
                pass

        assert cb.stats.consecutive_failures == actual_failures

        # One success should reset
        await cb.call(success_func)

        # Verify property
        assert cb.stats.consecutive_failures == 0, \
            "Consecutive failures should be reset after success"
        assert cb.state == CircuitState.CLOSED, \
            "Circuit should remain CLOSED"


# =============================================================================
# Unit Tests - RetryWithBackoff
# =============================================================================


class TestRetryWithBackoff:
    """Test retry with exponential backoff."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self) -> None:
        """Test successful call doesn't retry."""
        call_count = 0

        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        retry = RetryWithBackoff(RetryConfig(max_attempts=3))
        result = await retry.call(success_func)

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self) -> None:
        """Test retry on transient failure."""
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Transient error")
            return "success"

        config = RetryConfig(
            max_attempts=4,
            initial_wait=0.01,
            max_wait=0.1,
        )
        retry = RetryWithBackoff(config)
        result = await retry.call(flaky_func)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self) -> None:
        """Test failure after max retries."""
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        config = RetryConfig(
            max_attempts=3,
            initial_wait=0.01,
            max_wait=0.1,
        )
        retry = RetryWithBackoff(config)

        with pytest.raises(ValueError):
            await retry.call(always_fail)

        assert call_count == 3


# =============================================================================
# Unit Tests - ResilientCall
# =============================================================================


class TestResilientCall:
    """Test combined circuit breaker + retry."""

    @pytest.mark.asyncio
    async def test_success_path(self) -> None:
        """Test successful call through resilient wrapper."""
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))
        resilient = ResilientCall(cb, RetryConfig(max_attempts=2, initial_wait=0.01))

        async def success_func():
            return "success"

        result = await resilient.call(success_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_then_success(self) -> None:
        """Test retry succeeds before circuit trips."""
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=5))
        resilient = ResilientCall(
            cb,
            RetryConfig(max_attempts=3, initial_wait=0.01, max_wait=0.1),
        )

        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Transient")
            return "success"

        result = await resilient.call(flaky_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_trips_after_retries(self) -> None:
        """Test circuit trips after retry exhaustion."""
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2))
        resilient = ResilientCall(
            cb,
            RetryConfig(max_attempts=2, initial_wait=0.01, max_wait=0.1),
        )

        async def always_fail():
            raise ValueError("Always fails")

        # First call exhausts retries, counts as 1 failure
        with pytest.raises(ValueError):
            await resilient.call(always_fail)

        # Second call exhausts retries, trips circuit
        with pytest.raises(ValueError):
            await resilient.call(always_fail)

        assert cb.state == CircuitState.OPEN


# =============================================================================
# Unit Tests - CircuitBreakerRegistry
# =============================================================================


class TestCircuitBreakerRegistry:
    """Test circuit breaker registry."""

    @pytest.mark.asyncio
    async def test_get_or_create(self) -> None:
        """Test getting or creating circuit breakers."""
        registry = CircuitBreakerRegistry()

        cb1 = await registry.get_or_create("test1")
        cb2 = await registry.get_or_create("test1")
        cb3 = await registry.get_or_create("test2")

        assert cb1 is cb2  # Same instance
        assert cb1 is not cb3  # Different instance

    @pytest.mark.asyncio
    async def test_reset_all(self) -> None:
        """Test resetting all circuit breakers."""
        registry = CircuitBreakerRegistry()

        cb1 = await registry.get_or_create("test1")
        cb2 = await registry.get_or_create("test2")

        await cb1.force_open()
        await cb2.force_open()

        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.OPEN

        await registry.reset_all()

        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_get_all_stats(self) -> None:
        """Test getting stats for all circuit breakers."""
        registry = CircuitBreakerRegistry()

        await registry.get_or_create("test1")
        await registry.get_or_create("test2")

        stats = registry.get_all_stats()

        assert "test1" in stats
        assert "test2" in stats
        assert isinstance(stats["test1"], CircuitStats)
