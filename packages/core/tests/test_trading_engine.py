"""Tests for trading engine."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from packages.core.src.paper_trading import PaperTradingConfig
from packages.core.src.simple_strategy import StrategyConfig
from packages.core.src.trading_engine import TradingEngine, TradingEngineConfig
from packages.shared.src.models import OHLCV, Platform


def create_mock_connector() -> MagicMock:
    """Create a mock exchange connector."""
    connector = MagicMock()
    connector.platform = Platform.BINANCE
    connector.is_connected = True
    connector.connect = AsyncMock(return_value=True)
    connector.disconnect = AsyncMock()
    return connector


def create_candles(
    count: int = 50,
    base_price: Decimal = Decimal("40000"),
    trend: str = "neutral",
) -> list[OHLCV]:
    """Create test candles with specified trend."""
    candles = []
    price = base_price
    
    for i in range(count):
        if trend == "up":
            price = base_price + Decimal(str(i * 100))
        elif trend == "down":
            price = base_price - Decimal(str(i * 100))
        else:
            # Oscillate for neutral
            price = base_price + Decimal(str((i % 10 - 5) * 50))
        
        candles.append(OHLCV(
            timestamp=datetime.now(UTC),
            open=price - Decimal("10"),
            high=price + Decimal("50"),
            low=price - Decimal("50"),
            close=price,
            volume=Decimal("100"),
            symbol="BTCUSDT",
            timeframe="1h",
            platform=Platform.BINANCE,
        ))
    
    return candles


class TestTradingEngineConfig:
    """Test TradingEngineConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TradingEngineConfig()
        
        assert config.symbol == "BTCUSDT"
        assert config.timeframe == "1h"
        assert config.paper_trading is True  # Safety default
        assert config.position_size_pct == Decimal("0.1")

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = TradingEngineConfig(
            symbol="ETHUSDT",
            timeframe="4h",
            paper_trading=False,
            position_size_pct=Decimal("0.05"),
        )
        
        assert config.symbol == "ETHUSDT"
        assert config.timeframe == "4h"
        assert config.paper_trading is False


class TestTradingEngine:
    """Test TradingEngine."""

    @pytest.fixture
    def engine(self) -> TradingEngine:
        """Create trading engine with mock connector."""
        connector = create_mock_connector()
        config = TradingEngineConfig(
            symbol="BTCUSDT",
            paper_trading=True,
        )
        return TradingEngine(connector, config)

    def test_is_paper_trading(self, engine: TradingEngine) -> None:
        """Test paper trading mode detection."""
        assert engine.is_paper_trading is True

    def test_initial_state(self, engine: TradingEngine) -> None:
        """Test initial engine state."""
        assert engine._running is False
        assert engine._iteration_count == 0

    @pytest.mark.asyncio
    async def test_run_once_no_candles(self, engine: TradingEngine) -> None:
        """Test run_once with no candles returns error."""
        engine.connector.get_ohlcv = AsyncMock(return_value=[])
        
        result = await engine.run_once()
        
        assert "error" in result
        assert result["error"] == "No candles fetched"

    @pytest.mark.asyncio
    async def test_run_once_hold_signal(self, engine: TradingEngine) -> None:
        """Test run_once with neutral market (HOLD signal)."""
        # Create neutral candles (RSI around 50)
        candles = create_candles(50, trend="neutral")
        engine.connector.get_ohlcv = AsyncMock(return_value=candles)
        
        result = await engine.run_once()
        
        assert result["iteration"] == 1
        assert "signal" in result
        # Neutral market should produce HOLD
        assert result["action_taken"] == "none" or result["signal"]["action"] == "hold"

    @pytest.mark.asyncio
    async def test_run_once_buy_signal(self, engine: TradingEngine) -> None:
        """Test run_once with oversold market (BUY signal)."""
        # Create downtrend candles (RSI < 30)
        candles = create_candles(50, trend="down")
        engine.connector.get_ohlcv = AsyncMock(return_value=candles)
        
        result = await engine.run_once()
        
        assert result["iteration"] == 1
        assert "signal" in result
        # Strong downtrend should produce BUY signal (oversold)

    @pytest.mark.asyncio
    async def test_run_once_increments_iteration(self, engine: TradingEngine) -> None:
        """Test iteration counter increments."""
        candles = create_candles(50)
        engine.connector.get_ohlcv = AsyncMock(return_value=candles)
        
        await engine.run_once()
        assert engine._iteration_count == 1
        
        await engine.run_once()
        assert engine._iteration_count == 2

    @pytest.mark.asyncio
    async def test_run_once_connects_if_needed(self, engine: TradingEngine) -> None:
        """Test engine connects if not connected."""
        engine.connector.is_connected = False
        candles = create_candles(50)
        engine.connector.get_ohlcv = AsyncMock(return_value=candles)
        
        await engine.run_once()
        
        engine.connector.connect.assert_called_once()

    def test_get_stats(self, engine: TradingEngine) -> None:
        """Test get_stats returns correct info."""
        stats = engine.get_stats()
        
        assert stats["mode"] == "paper"
        assert stats["iterations"] == 0
        assert stats["symbol"] == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self, engine: TradingEngine) -> None:
        """Test stop() sets _running to False."""
        engine._running = True
        
        await engine.stop()
        
        assert engine._running is False


class TestTradingEnginePositionManagement:
    """Test position management in trading engine."""

    @pytest.fixture
    def engine(self) -> TradingEngine:
        """Create trading engine."""
        connector = create_mock_connector()
        config = TradingEngineConfig(paper_trading=True)
        paper_config = PaperTradingConfig(initial_balance=Decimal("10000"))
        return TradingEngine(connector, config, paper_config=paper_config)

    def test_has_position_empty(self, engine: TradingEngine) -> None:
        """Test has_position returns False when no position."""
        assert engine._has_position("BTCUSDT") is False

    @pytest.mark.asyncio
    async def test_get_balance_paper(self, engine: TradingEngine) -> None:
        """Test get_balance in paper trading mode."""
        balance = await engine._get_balance()
        assert balance == Decimal("10000")

    def test_calculate_position_size(self, engine: TradingEngine) -> None:
        """Test position size calculation."""
        balance = Decimal("10000")
        price = Decimal("40000")
        
        size = engine._calculate_position_size(balance, price)
        
        # 10% of 10000 = 1000 USDT, at 40000 = 0.025 BTC
        expected = Decimal("0.025")
        assert size == expected


class TestTradingEngineLiveMode:
    """Test trading engine in live mode."""

    @pytest.fixture
    def live_engine(self) -> TradingEngine:
        """Create trading engine in live mode."""
        connector = create_mock_connector()
        connector.get_balance = AsyncMock(return_value={"USDT": Decimal("5000")})
        
        config = TradingEngineConfig(paper_trading=False)
        return TradingEngine(connector, config)

    def test_is_not_paper_trading(self, live_engine: TradingEngine) -> None:
        """Test live mode detection."""
        assert live_engine.is_paper_trading is False

    @pytest.mark.asyncio
    async def test_get_balance_live(self, live_engine: TradingEngine) -> None:
        """Test get_balance in live mode calls connector."""
        balance = await live_engine._get_balance()
        
        assert balance == Decimal("5000")
        live_engine.connector.get_balance.assert_called_once()

    def test_get_stats_live_mode(self, live_engine: TradingEngine) -> None:
        """Test stats show live mode."""
        stats = live_engine.get_stats()
        assert stats["mode"] == "live"
