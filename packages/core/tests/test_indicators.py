"""Tests for technical indicators."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from packages.core.src.indicators import (
    calculate_atr,
    calculate_ema,
    calculate_rsi,
    calculate_sma,
)
from packages.shared.src.models import OHLCV, Platform


def create_candle(
    close: float,
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
) -> OHLCV:
    """Helper to create OHLCV candle."""
    close_d = Decimal(str(close))
    return OHLCV(
        timestamp=datetime.now(UTC),
        open=Decimal(str(open_ or close)),
        high=Decimal(str(high or close)),
        low=Decimal(str(low or close)),
        close=close_d,
        volume=Decimal("1000"),
        symbol="BTCUSDT",
        timeframe="1h",
        platform=Platform.BINANCE,
    )


class TestCalculateRSI:
    """Test RSI calculation."""

    def test_insufficient_data(self) -> None:
        """Test RSI returns None with insufficient data."""
        candles = [create_candle(100) for _ in range(10)]
        result = calculate_rsi(candles, period=14)
        assert result is None

    def test_all_gains(self) -> None:
        """Test RSI = 100 when all gains."""
        # Create 16 candles with increasing prices
        candles = [create_candle(100 + i) for i in range(16)]
        result = calculate_rsi(candles, period=14)
        assert result == Decimal("100")

    def test_all_losses(self) -> None:
        """Test RSI = 0 when all losses."""
        # Create 16 candles with decreasing prices
        candles = [create_candle(100 - i) for i in range(16)]
        result = calculate_rsi(candles, period=14)
        assert result == Decimal("0")

    def test_mixed_movement(self) -> None:
        """Test RSI with mixed price movement."""
        # Alternating up and down
        prices = [100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108, 107, 109]
        candles = [create_candle(p) for p in prices]
        result = calculate_rsi(candles, period=14)
        
        assert result is not None
        assert Decimal("0") <= result <= Decimal("100")

    def test_oversold_condition(self) -> None:
        """Test RSI in oversold territory."""
        # Strong downtrend
        prices = [100, 98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70]
        candles = [create_candle(p) for p in prices]
        result = calculate_rsi(candles, period=14)
        
        assert result is not None
        assert result < Decimal("30")  # Oversold

    def test_overbought_condition(self) -> None:
        """Test RSI in overbought territory."""
        # Strong uptrend
        prices = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130]
        candles = [create_candle(p) for p in prices]
        result = calculate_rsi(candles, period=14)
        
        assert result is not None
        assert result > Decimal("70")  # Overbought


class TestCalculateSMA:
    """Test SMA calculation."""

    def test_insufficient_data(self) -> None:
        """Test SMA returns None with insufficient data."""
        candles = [create_candle(100) for _ in range(5)]
        result = calculate_sma(candles, period=20)
        assert result is None

    def test_simple_average(self) -> None:
        """Test SMA calculation."""
        candles = [create_candle(100 + i) for i in range(20)]
        result = calculate_sma(candles, period=20)
        
        # Average of 100-119 = 109.5
        assert result is not None
        assert result == Decimal("109.5")

    def test_uses_last_n_candles(self) -> None:
        """Test SMA uses only last N candles."""
        candles = [create_candle(50) for _ in range(10)]  # Old candles
        candles += [create_candle(100) for _ in range(5)]  # Recent candles
        
        result = calculate_sma(candles, period=5)
        assert result == Decimal("100")


class TestCalculateEMA:
    """Test EMA calculation."""

    def test_insufficient_data(self) -> None:
        """Test EMA returns None with insufficient data."""
        candles = [create_candle(100) for _ in range(5)]
        result = calculate_ema(candles, period=20)
        assert result is None

    def test_ema_weights_recent_more(self) -> None:
        """Test EMA gives more weight to recent prices."""
        # Create data with sudden jump at the end
        # First 20 candles at 100, then 10 candles at 200
        candles = [create_candle(100) for _ in range(20)]
        candles += [create_candle(200) for _ in range(10)]
        
        sma = calculate_sma(candles, period=15)  # Last 15: 5x100 + 10x200 = 166.67
        ema = calculate_ema(candles, period=15)  # EMA reacts faster to 200s
        
        # EMA should be higher than SMA because it weights recent 200s more
        assert ema is not None
        assert sma is not None
        # With sudden jump, EMA should be higher (closer to recent prices)
        assert ema > sma


class TestCalculateATR:
    """Test ATR calculation."""

    def test_insufficient_data(self) -> None:
        """Test ATR returns None with insufficient data."""
        candles = [create_candle(100, high=101, low=99) for _ in range(10)]
        result = calculate_atr(candles, period=14)
        assert result is None

    def test_atr_calculation(self) -> None:
        """Test ATR with known values."""
        # Create candles with consistent range
        candles = []
        for i in range(16):
            candles.append(create_candle(
                close=100,
                high=102,
                low=98,
                open_=100,
            ))
        
        result = calculate_atr(candles, period=14)
        
        assert result is not None
        # True range should be 4 (high - low) for each candle
        assert result == Decimal("4")

    def test_atr_with_gaps(self) -> None:
        """Test ATR accounts for gaps."""
        candles = [
            create_candle(close=100, high=101, low=99),
            create_candle(close=105, high=106, low=104),  # Gap up
        ]
        # Add more candles to meet period requirement
        for _ in range(14):
            candles.append(create_candle(close=105, high=106, low=104))
        
        result = calculate_atr(candles, period=14)
        assert result is not None
        assert result > Decimal("0")
