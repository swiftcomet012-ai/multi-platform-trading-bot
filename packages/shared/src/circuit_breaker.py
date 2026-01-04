"""
Circuit Breaker pattern implementation for resilient API connections.

Implements the three states: CLOSED, OPEN, HALF_OPEN.
Uses tenacity for retry with exponential backoff.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, TypeVar

from packages.shared.src.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Circuit tripped, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening circuit
    success_threshold: int = 3  # Successes in half-open before closing
    recovery_timeout: float = 30.0  # Seconds before trying half-open
    half_open_max_calls: int = 3  # Max concurrent calls in half-open


@dataclass
class CircuitStats:
    """Statistics for circuit breaker."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: datetime | None = None
    last_success_time: datetime | None = None
    state_changes: list[tuple[datetime, CircuitState]] = field(default_factory=list)

    def record_success(self) -> None:
        """Record a successful call."""
        self.total_calls += 1
        self.successful_calls += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = datetime.now(UTC)

    def record_failure(self) -> None:
        """Record a failed call."""
        self.total_calls += 1
        self.failed_calls += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = datetime.now(UTC)

    def record_rejection(self) -> None:
        """Record a rejected call (circuit open)."""
        self.rejected_calls += 1

    def record_state_change(self, new_state: CircuitState) -> None:
        """Record a state change."""
        self.state_changes.append((datetime.now(UTC), new_state))

    def reset_consecutive(self) -> None:
        """Reset consecutive counters."""
        self.consecutive_failures = 0
        self.consecutive_successes = 0


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str, state: CircuitState, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.state = state
        self.retry_after = retry_after


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Circuit tripped, all requests fail fast
    - HALF_OPEN: Testing recovery, limited requests allowed

    Transitions:
    - CLOSED -> OPEN: When failure_threshold consecutive failures occur
    - OPEN -> HALF_OPEN: After recovery_timeout seconds
    - HALF_OPEN -> CLOSED: When success_threshold successes occur
    - HALF_OPEN -> OPEN: On any failure
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            name: Identifier for this circuit breaker.
            config: Configuration options.
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._opened_at: datetime | None = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        return self._stats

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._opened_at is None:
            return False
        elapsed = (datetime.now(UTC) - self._opened_at).total_seconds()
        return elapsed >= self.config.recovery_timeout

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._stats.record_state_change(new_state)

        if new_state == CircuitState.OPEN:
            self._opened_at = datetime.now(UTC)
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
        elif new_state == CircuitState.CLOSED:
            self._opened_at = None
            self._stats.reset_consecutive()

        logger.info(
            "circuit_breaker_state_change",
            name=self.name,
            old_state=old_state.value,
            new_state=new_state.value,
            consecutive_failures=self._stats.consecutive_failures,
            consecutive_successes=self._stats.consecutive_successes,
        )

    async def _check_state(self) -> None:
        """Check and potentially update circuit state."""
        if self._state == CircuitState.OPEN and self._should_attempt_reset():
            self._transition_to(CircuitState.HALF_OPEN)

    def _can_execute(self) -> bool:
        """Check if a call can be executed."""
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.OPEN:
            return False
        # HALF_OPEN: limit concurrent calls
        return self._half_open_calls < self.config.half_open_max_calls

    def _get_retry_after(self) -> float | None:
        """Get seconds until retry is allowed."""
        if self._state != CircuitState.OPEN or self._opened_at is None:
            return None
        elapsed = (datetime.now(UTC) - self._opened_at).total_seconds()
        remaining = self.config.recovery_timeout - elapsed
        return max(0, remaining)


    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Async function to execute.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            Result of func.

        Raises:
            CircuitBreakerError: If circuit is open.
            Exception: Any exception from func.
        """
        async with self._lock:
            await self._check_state()

            if not self._can_execute():
                self._stats.record_rejection()
                retry_after = self._get_retry_after()
                logger.warning(
                    "circuit_breaker_rejected",
                    name=self.name,
                    state=self._state.value,
                    retry_after=retry_after,
                )
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is {self._state.value}",
                    state=self._state,
                    retry_after=retry_after,
                )

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure(e)
            raise

    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            self._stats.record_success()

            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

            logger.debug(
                "circuit_breaker_success",
                name=self.name,
                state=self._state.value,
                consecutive_successes=self._stats.consecutive_successes,
            )

    async def _on_failure(self, error: Exception) -> None:
        """Handle failed call."""
        async with self._lock:
            self._stats.record_failure()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open trips the circuit
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._stats.consecutive_failures >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

            logger.warning(
                "circuit_breaker_failure",
                name=self.name,
                state=self._state.value,
                consecutive_failures=self._stats.consecutive_failures,
                error=str(error),
            )

    async def reset(self) -> None:
        """Manually reset circuit to closed state."""
        async with self._lock:
            self._transition_to(CircuitState.CLOSED)
            logger.info("circuit_breaker_manual_reset", name=self.name)

    async def force_open(self) -> None:
        """Manually open the circuit."""
        async with self._lock:
            self._transition_to(CircuitState.OPEN)
            logger.info("circuit_breaker_force_open", name=self.name)

    def __call__(
        self,
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        """
        Decorator to wrap a function with circuit breaker.

        Usage:
            @circuit_breaker
            async def my_api_call():
                ...
        """

        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await self.call(func, *args, **kwargs)

        return wrapper


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]

    async def get(self, name: str) -> CircuitBreaker | None:
        """Get circuit breaker by name."""
        return self._breakers.get(name)

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            await breaker.reset()

    def get_all_stats(self) -> dict[str, CircuitStats]:
        """Get stats for all circuit breakers."""
        return {name: breaker.stats for name, breaker in self._breakers.items()}


# Global registry instance
_registry = CircuitBreakerRegistry()


async def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker from the global registry."""
    return await _registry.get_or_create(name, config)


def circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator factory for circuit breaker.

    Usage:
        @circuit_breaker("binance_api")
        async def call_binance():
            ...
    """
    breaker = CircuitBreaker(name, config)

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        return breaker(func)

    return decorator



# =============================================================================
# Retry with Exponential Backoff
# =============================================================================

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


@dataclass
class RetryConfig:
    """Configuration for retry with exponential backoff."""

    max_attempts: int = 4  # Total attempts (1 initial + 3 retries)
    initial_wait: float = 1.0  # Initial wait in seconds
    max_wait: float = 30.0  # Maximum wait in seconds
    multiplier: float = 2.0  # Exponential multiplier
    retry_exceptions: tuple[type[Exception], ...] = (Exception,)


class RetryWithBackoff:
    """
    Retry wrapper with exponential backoff.

    Backoff sequence with default config: 1s, 2s, 4s, 8s (capped at max_wait)
    """

    def __init__(self, config: RetryConfig | None = None) -> None:
        """
        Initialize retry wrapper.

        Args:
            config: Retry configuration.
        """
        self.config = config or RetryConfig()

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute function with retry and exponential backoff.

        Args:
            func: Async function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result of func.

        Raises:
            RetryError: If all retries exhausted.
        """
        attempt = 0
        async for attempt_state in AsyncRetrying(
            stop=stop_after_attempt(self.config.max_attempts),
            wait=wait_exponential(
                multiplier=self.config.multiplier,
                min=self.config.initial_wait,
                max=self.config.max_wait,
            ),
            retry=retry_if_exception_type(self.config.retry_exceptions),
            reraise=True,
        ):
            with attempt_state:
                attempt += 1
                logger.debug(
                    "retry_attempt",
                    attempt=attempt,
                    max_attempts=self.config.max_attempts,
                    func=func.__name__,
                )
                return await func(*args, **kwargs)

        # Should not reach here, but satisfy type checker
        raise RuntimeError("Retry loop exited unexpectedly")

    def __call__(
        self,
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        """Decorator to wrap function with retry."""

        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await self.call(func, *args, **kwargs)

        return wrapper


def retry_with_backoff(
    max_attempts: int = 4,
    initial_wait: float = 1.0,
    max_wait: float = 30.0,
    multiplier: float = 2.0,
    retry_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator factory for retry with exponential backoff.

    Default backoff: 1s -> 2s -> 4s -> 8s (capped at 30s)

    Usage:
        @retry_with_backoff(max_attempts=4, initial_wait=1.0)
        async def call_api():
            ...
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_wait=initial_wait,
        max_wait=max_wait,
        multiplier=multiplier,
        retry_exceptions=retry_exceptions,
    )
    retry = RetryWithBackoff(config)

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        return retry(func)

    return decorator


# =============================================================================
# Combined Circuit Breaker + Retry
# =============================================================================


class ResilientCall:
    """
    Combines circuit breaker with retry for maximum resilience.

    Order of operations:
    1. Check circuit breaker state
    2. If allowed, execute with retry
    3. Update circuit breaker based on final result
    """

    def __init__(
        self,
        circuit_breaker: CircuitBreaker,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """
        Initialize resilient call wrapper.

        Args:
            circuit_breaker: Circuit breaker instance.
            retry_config: Retry configuration.
        """
        self.circuit_breaker = circuit_breaker
        self.retry = RetryWithBackoff(retry_config)

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute function with circuit breaker and retry.

        Args:
            func: Async function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result of func.

        Raises:
            CircuitBreakerError: If circuit is open.
            RetryError: If all retries exhausted.
        """

        async def wrapped() -> T:
            return await self.retry.call(func, *args, **kwargs)

        return await self.circuit_breaker.call(wrapped)

    def __call__(
        self,
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        """Decorator to wrap function with resilient call."""

        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await self.call(func, *args, **kwargs)

        return wrapper


def resilient(
    name: str,
    circuit_config: CircuitBreakerConfig | None = None,
    retry_config: RetryConfig | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator factory for resilient calls (circuit breaker + retry).

    Usage:
        @resilient("binance_api")
        async def call_binance():
            ...
    """
    cb = CircuitBreaker(name, circuit_config)
    resilient_call = ResilientCall(cb, retry_config)

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        return resilient_call(func)

    return decorator
