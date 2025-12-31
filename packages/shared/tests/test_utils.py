"""
Tests for shared utility functions.
"""

from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from packages.shared.src.utils import (
    RateLimiter,
    calculate_pnl,
    calculate_position_size,
    chunk_list,
    clamp,
    format_currency,
    format_percentage,
    generate_id,
    generate_idempotency_key,
    round_decimal,
    safe_divide,
)


class TestGenerateId:
    """Tests for generate_id function."""

    def test_generate_id_unique(self) -> None:
        """Generated IDs should be unique."""
        ids = [generate_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_generate_id_with_prefix(self) -> None:
        """Generated ID should include prefix."""
        id_ = generate_id("order")
        assert id_.startswith("order_")


class TestIdempotencyKey:
    """Tests for idempotency key generation."""

    def test_same_inputs_same_key(self) -> None:
        """Same inputs should produce same key."""
        key1 = generate_idempotency_key("BTCUSDT", "buy", "0.1", timestamp=1000)
        key2 = generate_idempotency_key("BTCUSDT", "buy", "0.1", timestamp=1000)
        assert key1 == key2

    def test_different_inputs_different_key(self) -> None:
        """Different inputs should produce different keys."""
        key1 = generate_idempotency_key("BTCUSDT", "buy", "0.1", timestamp=1000)
        key2 = generate_idempotency_key("BTCUSDT", "sell", "0.1", timestamp=1000)
        assert key1 != key2


class TestRoundDecimal:
    """Tests for round_decimal function."""

    def test_round_down(self) -> None:
        """Test rounding down."""
        result = round_decimal(Decimal("1.2345"), 2)
        assert result == Decimal("1.23")

    def test_round_zero_precision(self) -> None:
        """Test rounding to integer."""
        result = round_decimal(Decimal("1.9"), 0)
        assert result == Decimal("1")


class TestCalculatePnl:
    """Tests for P&L calculation."""

    def test_buy_profit(self) -> None:
        """Test profit on buy trade."""
        pnl, pnl_pct = calculate_pnl(
            entry_price=Decimal("100"),
            exit_price=Decimal("110"),
            quantity=Decimal("1"),
            side="buy",
            fees=Decimal("1"),
        )
        assert pnl == Decimal("9")  # (110-100)*1 - 1
        assert abs(pnl_pct - 9.0) < 0.01

    def test_buy_loss(self) -> None:
        """Test loss on buy trade."""
        pnl, pnl_pct = calculate_pnl(
            entry_price=Decimal("100"),
            exit_price=Decimal("90"),
            quantity=Decimal("1"),
            side="buy",
            fees=Decimal("1"),
        )
        assert pnl == Decimal("-11")  # (90-100)*1 - 1

    def test_sell_profit(self) -> None:
        """Test profit on sell trade."""
        pnl, pnl_pct = calculate_pnl(
            entry_price=Decimal("100"),
            exit_price=Decimal("90"),
            quantity=Decimal("1"),
            side="sell",
            fees=Decimal("1"),
        )
        assert pnl == Decimal("9")  # (100-90)*1 - 1


class TestCalculatePositionSize:
    """Tests for position size calculation."""

    def test_position_size(self) -> None:
        """Test position size calculation."""
        size = calculate_position_size(
            balance=Decimal("10000"),
            risk_pct=Decimal("0.02"),  # 2% risk
            entry_price=Decimal("100"),
            stop_loss_price=Decimal("95"),  # 5% stop loss
        )
        # Risk amount = 10000 * 0.02 = 200
        # Price diff = 100 - 95 = 5
        # Size = 200 / 5 = 40
        assert size == Decimal("40")

    def test_position_size_zero_diff(self) -> None:
        """Test position size with zero price difference."""
        size = calculate_position_size(
            balance=Decimal("10000"),
            risk_pct=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_loss_price=Decimal("100"),
        )
        assert size == Decimal("0")


class TestFormatCurrency:
    """Tests for currency formatting."""

    def test_format_positive(self) -> None:
        """Test formatting positive amount."""
        result = format_currency(Decimal("1234.56"))
        assert result == "$1,234.56"

    def test_format_negative(self) -> None:
        """Test formatting negative amount."""
        result = format_currency(Decimal("-1234.56"))
        assert result == "-$1,234.56"


class TestFormatPercentage:
    """Tests for percentage formatting."""

    def test_format_positive(self) -> None:
        """Test formatting positive percentage."""
        result = format_percentage(5.5)
        assert result == "+5.50%"

    def test_format_negative(self) -> None:
        """Test formatting negative percentage."""
        result = format_percentage(-3.2)
        assert result == "-3.20%"


class TestChunkList:
    """Tests for list chunking."""

    def test_chunk_list(self) -> None:
        """Test chunking a list."""
        result = chunk_list([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_chunk_empty_list(self) -> None:
        """Test chunking empty list."""
        result = chunk_list([], 2)
        assert result == []


class TestSafeDivide:
    """Tests for safe division."""

    def test_normal_division(self) -> None:
        """Test normal division."""
        result = safe_divide(Decimal("10"), Decimal("2"))
        assert result == Decimal("5")

    def test_division_by_zero(self) -> None:
        """Test division by zero returns default."""
        result = safe_divide(Decimal("10"), Decimal("0"))
        assert result == Decimal("0")

    def test_division_by_zero_custom_default(self) -> None:
        """Test division by zero with custom default."""
        result = safe_divide(Decimal("10"), Decimal("0"), default=Decimal("-1"))
        assert result == Decimal("-1")


class TestClamp:
    """Tests for clamp function."""

    def test_clamp_within_range(self) -> None:
        """Test value within range."""
        assert clamp(5, 0, 10) == 5

    def test_clamp_below_min(self) -> None:
        """Test value below minimum."""
        assert clamp(-5, 0, 10) == 0

    def test_clamp_above_max(self) -> None:
        """Test value above maximum."""
        assert clamp(15, 0, 10) == 10


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_acquire_within_limit(self) -> None:
        """Test acquiring tokens within limit."""
        limiter = RateLimiter(max_requests=5, time_window_seconds=1.0)
        for _ in range(5):
            assert limiter.acquire() is True

    def test_acquire_exceeds_limit(self) -> None:
        """Test acquiring tokens exceeds limit."""
        limiter = RateLimiter(max_requests=2, time_window_seconds=1.0)
        assert limiter.acquire() is True
        assert limiter.acquire() is True
        assert limiter.acquire() is False

    def test_wait_time(self) -> None:
        """Test wait time calculation."""
        limiter = RateLimiter(max_requests=1, time_window_seconds=1.0)
        limiter.acquire()
        wait = limiter.wait_time()
        assert wait > 0


# =============================================================================
# Property-Based Tests
# =============================================================================


@pytest.mark.property
class TestPropertyBasedUtils:
    """Property-based tests for utility functions."""

    @given(
        balance=st.decimals(min_value=Decimal("100"), max_value=Decimal("1000000")),
        risk_pct=st.decimals(min_value=Decimal("0.001"), max_value=Decimal("0.1")),
    )
    @settings(max_examples=100)
    def test_position_size_proportional_to_balance(
        self, balance: Decimal, risk_pct: Decimal
    ) -> None:
        """Position size should be proportional to balance."""
        entry = Decimal("100")
        stop_loss = Decimal("95")

        size = calculate_position_size(balance, risk_pct, entry, stop_loss)

        # Size should be positive
        assert size >= 0

        # Risk amount should not exceed balance * risk_pct
        risk_amount = size * (entry - stop_loss)
        assert risk_amount <= balance * risk_pct + Decimal("0.01")  # Small tolerance

    @given(
        numerator=st.decimals(min_value=Decimal("-1000"), max_value=Decimal("1000")),
        denominator=st.decimals(min_value=Decimal("-1000"), max_value=Decimal("1000")),
    )
    @settings(max_examples=100)
    def test_safe_divide_never_raises(
        self, numerator: Decimal, denominator: Decimal
    ) -> None:
        """safe_divide should never raise an exception."""
        # This should never raise
        result = safe_divide(numerator, denominator)
        assert isinstance(result, Decimal)
