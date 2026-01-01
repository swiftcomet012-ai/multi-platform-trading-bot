"""
Utility functions for the trading platform.
"""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from typing import TypeVar
from uuid import uuid4

T = TypeVar("T")


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    uid = str(uuid4())[:12]
    return f"{prefix}_{uid}" if prefix else uid


def generate_idempotency_key(symbol: str, side: str, quantity: str, timestamp: int | None = None) -> str:
    """
    Generate idempotency key for order deduplication.

    Args:
        symbol: Trading symbol
        side: Order side (buy/sell)
        quantity: Order quantity
        timestamp: Unix timestamp (defaults to current time)

    Returns:
        Unique idempotency key
    """
    ts = timestamp or int(time.time())
    data = f"{symbol}:{side}:{quantity}:{ts}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(UTC)


def timestamp_ms() -> int:
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)


def round_decimal(value: Decimal, precision: int, rounding: str = ROUND_DOWN) -> Decimal:
    """
    Round decimal to specified precision.

    Args:
        value: Decimal value to round
        precision: Number of decimal places
        rounding: Rounding mode (default: ROUND_DOWN for safety)

    Returns:
        Rounded Decimal
    """
    quantize_str = "0." + "0" * precision if precision > 0 else "1"
    return value.quantize(Decimal(quantize_str), rounding=rounding)


def calculate_pnl(
    entry_price: Decimal,
    exit_price: Decimal,
    quantity: Decimal,
    side: str,
    fees: Decimal = Decimal("0"),
) -> tuple[Decimal, float]:
    """
    Calculate P&L for a trade.

    Args:
        entry_price: Entry price
        exit_price: Exit price
        quantity: Trade quantity
        side: Trade side ('buy' or 'sell')
        fees: Total fees

    Returns:
        Tuple of (absolute P&L, percentage P&L)
    """
    if side.lower() == "buy":
        pnl = (exit_price - entry_price) * quantity - fees
    else:
        pnl = (entry_price - exit_price) * quantity - fees

    pnl_pct = float(pnl / (entry_price * quantity) * 100) if entry_price * quantity else 0.0
    return pnl, pnl_pct


def calculate_position_size(
    balance: Decimal,
    risk_pct: Decimal,
    entry_price: Decimal,
    stop_loss_price: Decimal,
) -> Decimal:
    """
    Calculate position size based on risk percentage.

    Args:
        balance: Account balance
        risk_pct: Risk percentage (e.g., 0.02 for 2%)
        entry_price: Entry price
        stop_loss_price: Stop loss price

    Returns:
        Position size in base currency
    """
    risk_amount = balance * risk_pct
    price_diff = abs(entry_price - stop_loss_price)

    if price_diff == 0:
        return Decimal("0")

    return risk_amount / price_diff


def format_currency(value: Decimal, symbol: str = "$", precision: int = 2) -> str:
    """Format decimal as currency string."""
    rounded = round_decimal(value, precision)
    sign = "-" if rounded < 0 else ""
    return f"{sign}{symbol}{abs(rounded):,.{precision}f}"


def format_percentage(value: float, precision: int = 2) -> str:
    """Format float as percentage string."""
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{precision}f}%"


def chunk_list(lst: list[T], chunk_size: int) -> list[list[T]]:
    """Split list into chunks of specified size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def safe_divide(numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")) -> Decimal:
    """Safely divide two decimals, returning default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: T, min_value: T, max_value: T) -> T:
    """Clamp value between min and max."""
    return max(min_value, min(value, max_value))


class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""

    def __init__(self, max_requests: int, time_window_seconds: float) -> None:
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in time window
            time_window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window_seconds
        self.tokens = max_requests
        self.last_update = time.monotonic()

    def acquire(self) -> bool:
        """
        Try to acquire a token.

        Returns:
            True if token acquired, False if rate limited
        """
        now = time.monotonic()
        elapsed = now - self.last_update

        # Refill tokens
        self.tokens = min(
            self.max_requests,
            self.tokens + elapsed * (self.max_requests / self.time_window),
        )
        self.last_update = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    def wait_time(self) -> float:
        """Get time to wait before next request is allowed."""
        if self.tokens >= 1:
            return 0.0
        return (1 - self.tokens) * (self.time_window / self.max_requests)
