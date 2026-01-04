"""Tests for simple RSI strategy."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from packages.core.src.simple_strategy import SimpleRSIStrategy, StrategyConfig
from packages.shared.src.models import OHLCV, Platform, SignalAction


def create_candle(close: float) -> OHLCV:
    """Helper to create OHLCV candle."""
    return OHLCV(
        timestamp=datetime.now(UTC),
        open=Decimal(str(close)),
        high=Decimal(str(close * 1.01)),
        low=Decimal(str(close * 0.99)),
        close=Decimal(str(close)),
        volume=Decimal("1000"),
        symbol="BTCUSDT",
        timeframe="1h",
        platform=Platform.BINANCE,
    )


class TestSimpleRSIStrategy:
    """Test SimpleRSIStrategy."""

    def test_hold_on_insufficient_data(self) -> None:
        """Test HOLD signal when insufficient data."""
        strategy = SimpleRSIStrategy()
        candles = [create_candle(100) for _ in range(5)]
        
        signal = strategy.analyze(candles, "BTCUSDT")
        
        assert signal.action == SignalAction.HOLD
        assert "Insufficient data" in signal.reasoning

    def test_buy_signal_on_oversold(self) -> None:
        """Test BUY signal when RSI is oversold."""
        strategy = SimpleRSIStrategy()
        
        # Create strong downtrend (RSI < 30)
        prices = [100 - i * 2 for i in range(20)]
        candles = [create_candle(p) for p in prices]
        
        signal = strategy.analyze(candles, "BTCUSDT")
        
        assert signal.action == SignalAction.BUY
        assert "oversold" in signal.reasoning.lower()
        assert signal.confidence >= 0.6

    def test_sell_signal_on_overbought(self) -> None:
        """Test SELL signal when RSI is overbought."""
        strategy = SimpleRSIStrategy()
        
        # Create strong uptrend (RSI > 70)
        prices = [100 + i * 2 for i in range(20)]
        candles = [create_candle(p) for p in prices]
        
        signal = strategy.analyze(candles, "BTCUSDT")
        
        assert signal.action == SignalAction.SELL
        assert "overbought" in signal.reasoning.lower()
        assert signal.confidence >= 0.6

    def test_hold_signal_on_neutral(self) -> None:
        """Test HOLD signal when RSI is neutral."""
        strategy = SimpleRSIStrategy()
        
        # Create sideways movement (RSI around 50)
        prices = [100, 101, 100, 101, 100, 101, 100, 101, 100, 101,
                  100, 101, 100, 101, 100, 101, 100, 101, 100, 101]
        candles = [create_candle(p) for p in prices]
        
        signal = strategy.analyze(candles, "BTCUSDT")
        
        assert signal.action == SignalAction.HOLD

    def test_custom_thresholds(self) -> None:
        """Test custom RSI thresholds."""
        config = StrategyConfig(
            rsi_oversold=Decimal("20"),
            rsi_overbought=Decimal("80"),
        )
        strategy = SimpleRSIStrategy(config)
        
        # Moderate downtrend (RSI around 25-35)
        prices = [100 - i for i in range(20)]
        candles = [create_candle(p) for p in prices]
        
        signal = strategy.analyze(candles, "BTCUSDT")
        
        # With default thresholds (30), this would be BUY
        # With custom threshold (20), might be HOLD
        assert signal.action in [SignalAction.BUY, SignalAction.HOLD]

    def test_signal_metadata(self) -> None:
        """Test signal contains metadata."""
        strategy = SimpleRSIStrategy()
        
        prices = [100 + i for i in range(20)]
        candles = [create_candle(p) for p in prices]
        
        signal = strategy.analyze(candles, "BTCUSDT")
        
        assert signal.metadata is not None
        assert "rsi" in signal.metadata
        assert signal.strategy == "simple_rsi"
        assert signal.symbol == "BTCUSDT"

    def test_confidence_capped_at_95(self) -> None:
        """Test confidence is capped at 0.95."""
        strategy = SimpleRSIStrategy()
        
        # Extreme downtrend
        prices = [100 - i * 5 for i in range(20)]
        candles = [create_candle(p) for p in prices]
        
        signal = strategy.analyze(candles, "BTCUSDT")
        
        assert signal.confidence <= 0.95


class TestStopLossCalculation:
    """Test stop-loss calculation."""

    def test_stop_loss_for_buy(self) -> None:
        """Test stop-loss below entry for BUY."""
        strategy = SimpleRSIStrategy()
        
        entry_price = Decimal("100")
        atr = Decimal("2")
        
        stop_loss = strategy.calculate_stop_loss(entry_price, atr, SignalAction.BUY)
        
        assert stop_loss is not None
        assert stop_loss < entry_price
        # Default ATR multiplier is 2.0, so stop = 100 - (2 * 2) = 96
        assert stop_loss == Decimal("96")

    def test_stop_loss_for_sell(self) -> None:
        """Test stop-loss above entry for SELL."""
        strategy = SimpleRSIStrategy()
        
        entry_price = Decimal("100")
        atr = Decimal("2")
        
        stop_loss = strategy.calculate_stop_loss(entry_price, atr, SignalAction.SELL)
        
        assert stop_loss is not None
        assert stop_loss > entry_price
        # stop = 100 + (2 * 2) = 104
        assert stop_loss == Decimal("104")

    def test_stop_loss_none_for_hold(self) -> None:
        """Test no stop-loss for HOLD."""
        strategy = SimpleRSIStrategy()
        
        stop_loss = strategy.calculate_stop_loss(
            Decimal("100"), Decimal("2"), SignalAction.HOLD
        )
        
        assert stop_loss is None

    def test_stop_loss_none_without_atr(self) -> None:
        """Test no stop-loss when ATR is None."""
        strategy = SimpleRSIStrategy()
        
        stop_loss = strategy.calculate_stop_loss(
            Decimal("100"), None, SignalAction.BUY
        )
        
        assert stop_loss is None
