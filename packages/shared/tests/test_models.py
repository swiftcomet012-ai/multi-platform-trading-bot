"""
Tests for shared models including property-based tests.

Property 1: Serialization round-trip - validates R52.1
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from packages.shared.src.models import (
    OHLCV,
    ExchangeInfo,
    Order,
    OrderStatus,
    OrderType,
    Platform,
    Position,
    Side,
    Signal,
    SignalAction,
    Trade,
    deserialize,
    serialize,
)


# =============================================================================
# Hypothesis Strategies
# =============================================================================

decimal_strategy = st.decimals(
    min_value=Decimal("0.00000001"),
    max_value=Decimal("1000000"),
    allow_nan=False,
    allow_infinity=False,
    places=8,
)

positive_decimal_strategy = st.decimals(
    min_value=Decimal("0.01"),
    max_value=Decimal("100000"),
    allow_nan=False,
    allow_infinity=False,
    places=8,
)

symbol_strategy = st.sampled_from(["BTCUSDT", "ETHUSDT", "EURUSD", "XAUUSD"])
platform_strategy = st.sampled_from(list(Platform))
side_strategy = st.sampled_from(list(Side))
signal_action_strategy = st.sampled_from(list(SignalAction))
order_type_strategy = st.sampled_from(list(OrderType))
order_status_strategy = st.sampled_from(list(OrderStatus))
confidence_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
datetime_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
    timezones=st.just(timezone.utc),
)


# =============================================================================
# Property-Based Tests - Serialization Round-Trip (Property 1)
# =============================================================================


@pytest.mark.property
class TestSerializationRoundTrip:
    """Property 1: Serialization round-trip tests."""

    @given(
        timestamp=datetime_strategy,
        open_=positive_decimal_strategy,
        high=positive_decimal_strategy,
        low=positive_decimal_strategy,
        close=positive_decimal_strategy,
        volume=positive_decimal_strategy,
        symbol=symbol_strategy,
        platform=platform_strategy,
    )
    @settings(max_examples=100)
    def test_ohlcv_roundtrip(
        self,
        timestamp: datetime,
        open_: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        volume: Decimal,
        symbol: str,
        platform: Platform,
    ) -> None:
        """OHLCV serialization round-trip preserves all data."""
        # Ensure high >= low for valid OHLCV
        if high < low:
            high, low = low, high

        ohlcv = OHLCV(
            timestamp=timestamp,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            symbol=symbol,
            platform=platform,
        )

        # Serialize and deserialize
        data = serialize(ohlcv)
        restored = deserialize(data, OHLCV)

        # Verify all fields match
        assert restored.timestamp == ohlcv.timestamp
        assert restored.open == ohlcv.open
        assert restored.high == ohlcv.high
        assert restored.low == ohlcv.low
        assert restored.close == ohlcv.close
        assert restored.volume == ohlcv.volume
        assert restored.symbol == ohlcv.symbol
        assert restored.platform == ohlcv.platform

    @given(
        symbol=symbol_strategy,
        action=signal_action_strategy,
        confidence=confidence_strategy,
        reasoning=st.text(min_size=1, max_size=100),
        strategy=st.text(min_size=1, max_size=50),
        timestamp=datetime_strategy,
        platform=platform_strategy,
    )
    @settings(max_examples=100)
    def test_signal_roundtrip(
        self,
        symbol: str,
        action: SignalAction,
        confidence: float,
        reasoning: str,
        strategy: str,
        timestamp: datetime,
        platform: Platform,
    ) -> None:
        """Signal serialization round-trip preserves all data."""
        signal = Signal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            strategy=strategy,
            timestamp=timestamp,
            platform=platform,
        )

        data = serialize(signal)
        restored = deserialize(data, Signal)

        assert restored.symbol == signal.symbol
        assert restored.action == signal.action
        assert abs(restored.confidence - signal.confidence) < 1e-6
        assert restored.reasoning == signal.reasoning
        assert restored.strategy == signal.strategy
        assert restored.platform == signal.platform

    @given(
        id_=st.text(min_size=1, max_size=20),
        symbol=symbol_strategy,
        side=side_strategy,
        entry_price=positive_decimal_strategy,
        quantity=positive_decimal_strategy,
        fees=decimal_strategy,
        platform=platform_strategy,
        strategy=st.text(min_size=1, max_size=50),
        timestamp=datetime_strategy,
    )
    @settings(max_examples=100)
    def test_trade_roundtrip(
        self,
        id_: str,
        symbol: str,
        side: Side,
        entry_price: Decimal,
        quantity: Decimal,
        fees: Decimal,
        platform: Platform,
        strategy: str,
        timestamp: datetime,
    ) -> None:
        """Trade serialization round-trip preserves all data."""
        trade = Trade(
            id=id_,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            fees=fees,
            platform=platform,
            strategy=strategy,
            created_at=timestamp,
        )

        data = serialize(trade)
        restored = deserialize(data, Trade)

        assert restored.id == trade.id
        assert restored.symbol == trade.symbol
        assert restored.side == trade.side
        assert restored.entry_price == trade.entry_price
        assert restored.quantity == trade.quantity
        assert restored.fees == trade.fees
        assert restored.platform == trade.platform
        assert restored.strategy == trade.strategy


# =============================================================================
# Unit Tests
# =============================================================================


class TestOHLCV:
    """Unit tests for OHLCV model."""

    def test_create_ohlcv(self) -> None:
        """Test OHLCV creation."""
        ohlcv = OHLCV(
            timestamp=datetime.now(timezone.utc),
            open=Decimal("100.00"),
            high=Decimal("105.00"),
            low=Decimal("99.00"),
            close=Decimal("103.00"),
            volume=Decimal("1000.00"),
            symbol="BTCUSDT",
        )
        assert ohlcv.symbol == "BTCUSDT"
        assert ohlcv.close == Decimal("103.00")

    def test_ohlcv_to_dict(self) -> None:
        """Test OHLCV to_dict method."""
        ohlcv = OHLCV(
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("99"),
            close=Decimal("103"),
            volume=Decimal("1000"),
        )
        d = ohlcv.to_dict()
        assert d["open"] == "100"
        assert d["close"] == "103"
        assert "timestamp" in d


class TestSignal:
    """Unit tests for Signal model."""

    def test_signal_is_actionable(self) -> None:
        """Test Signal.is_actionable method."""
        signal = Signal(
            symbol="BTCUSDT",
            action=SignalAction.BUY,
            confidence=0.8,
            reasoning="Strong uptrend",
            strategy="trend_following",
            timestamp=datetime.now(timezone.utc),
        )
        assert signal.is_actionable(threshold=0.7) is True
        assert signal.is_actionable(threshold=0.9) is False

    def test_hold_signal_not_actionable(self) -> None:
        """Test HOLD signal is never actionable."""
        signal = Signal(
            symbol="BTCUSDT",
            action=SignalAction.HOLD,
            confidence=1.0,
            reasoning="No clear signal",
            strategy="ai",
            timestamp=datetime.now(timezone.utc),
        )
        assert signal.is_actionable() is False


class TestOrder:
    """Unit tests for Order model."""

    def test_order_is_filled(self) -> None:
        """Test Order.is_filled method."""
        order = Order(
            id="order_123",
            symbol="BTCUSDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            status=OrderStatus.FILLED,
        )
        assert order.is_filled() is True

        pending_order = Order(
            id="order_456",
            symbol="BTCUSDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            status=OrderStatus.PENDING,
        )
        assert pending_order.is_filled() is False


class TestTrade:
    """Unit tests for Trade model."""

    def test_trade_is_open(self) -> None:
        """Test Trade.is_open method."""
        open_trade = Trade(
            id="trade_123",
            symbol="BTCUSDT",
            side=Side.BUY,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            fees=Decimal("5"),
            platform=Platform.BINANCE,
            strategy="trend",
            created_at=datetime.now(timezone.utc),
        )
        assert open_trade.is_open() is True

        closed_trade = Trade(
            id="trade_456",
            symbol="BTCUSDT",
            side=Side.BUY,
            entry_price=Decimal("50000"),
            exit_price=Decimal("51000"),
            quantity=Decimal("0.1"),
            fees=Decimal("10"),
            pnl=Decimal("90"),
            platform=Platform.BINANCE,
            strategy="trend",
            created_at=datetime.now(timezone.utc),
            closed_at=datetime.now(timezone.utc),
        )
        assert closed_trade.is_open() is False

    def test_trade_calculate_pnl(self) -> None:
        """Test Trade.calculate_pnl method."""
        trade = Trade(
            id="trade_123",
            symbol="BTCUSDT",
            side=Side.BUY,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            fees=Decimal("5"),
            platform=Platform.BINANCE,
            strategy="trend",
            created_at=datetime.now(timezone.utc),
        )
        # Price went up to 51000
        pnl = trade.calculate_pnl(Decimal("51000"))
        # (51000 - 50000) * 0.1 - 5 = 100 - 5 = 95
        assert pnl == Decimal("95")


class TestExchangeInfo:
    """Unit tests for ExchangeInfo model."""

    def test_validate_quantity_valid(self) -> None:
        """Test quantity validation with valid quantity."""
        info = ExchangeInfo(
            symbol="BTCUSDT",
            platform=Platform.BINANCE,
            min_qty=Decimal("0.001"),
            max_qty=Decimal("100"),
            qty_step=Decimal("0.001"),
            qty_precision=3,
            price_precision=2,
            min_notional=Decimal("10"),
            leverage_options=[1, 5, 10, 20],
            maker_fee=Decimal("0.001"),
            taker_fee=Decimal("0.001"),
            updated_at=datetime.now(timezone.utc),
        )
        valid, msg = info.validate_quantity(Decimal("0.5"))
        assert valid is True
        assert msg == "OK"

    def test_validate_quantity_below_min(self) -> None:
        """Test quantity validation below minimum."""
        info = ExchangeInfo(
            symbol="BTCUSDT",
            platform=Platform.BINANCE,
            min_qty=Decimal("0.001"),
            max_qty=Decimal("100"),
            qty_step=Decimal("0.001"),
            qty_precision=3,
            price_precision=2,
            min_notional=Decimal("10"),
            leverage_options=[1, 5, 10, 20],
            maker_fee=Decimal("0.001"),
            taker_fee=Decimal("0.001"),
            updated_at=datetime.now(timezone.utc),
        )
        valid, msg = info.validate_quantity(Decimal("0.0001"))
        assert valid is False
        assert "below minimum" in msg

    def test_round_quantity(self) -> None:
        """Test quantity rounding to step size."""
        info = ExchangeInfo(
            symbol="BTCUSDT",
            platform=Platform.BINANCE,
            min_qty=Decimal("0.001"),
            max_qty=Decimal("100"),
            qty_step=Decimal("0.001"),
            qty_precision=3,
            price_precision=2,
            min_notional=Decimal("10"),
            leverage_options=[1, 5, 10, 20],
            maker_fee=Decimal("0.001"),
            taker_fee=Decimal("0.001"),
            updated_at=datetime.now(timezone.utc),
        )
        rounded = info.round_quantity(Decimal("0.1234567"))
        assert rounded == Decimal("0.123")
