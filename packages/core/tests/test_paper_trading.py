"""Tests for paper trading engine."""

from __future__ import annotations

from decimal import Decimal

import pytest

from packages.core.src.paper_trading import (
    PaperTradingConfig,
    PaperTradingEngine,
)
from packages.shared.src.models import Order, OrderStatus, OrderType, Platform, Side


class TestPaperTradingEngine:
    """Test PaperTradingEngine."""

    @pytest.fixture
    def engine(self) -> PaperTradingEngine:
        """Create paper trading engine."""
        config = PaperTradingConfig(
            initial_balance=Decimal("10000"),
            maker_fee=Decimal("0.001"),
            taker_fee=Decimal("0.001"),
            slippage=Decimal("0.0005"),
        )
        return PaperTradingEngine(config)

    def test_initial_balance(self, engine: PaperTradingEngine) -> None:
        """Test initial balance is set correctly."""
        assert engine.balance == Decimal("10000")
        assert engine.get_balance() == {"USDT": Decimal("10000")}

    def test_no_initial_positions(self, engine: PaperTradingEngine) -> None:
        """Test no positions initially."""
        assert engine.positions == []
        assert not engine.has_position("BTCUSDT")

    @pytest.mark.asyncio
    async def test_buy_order(self, engine: PaperTradingEngine) -> None:
        """Test executing a buy order."""
        order = Order(
            id="",
            symbol="BTCUSDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )
        
        result = await engine.place_order(order, Decimal("40000"))
        
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == Decimal("0.1")
        assert result.filled_price is not None
        assert engine.has_position("BTCUSDT")
        assert engine.balance < Decimal("10000")  # Balance reduced

    @pytest.mark.asyncio
    async def test_sell_order_closes_position(self, engine: PaperTradingEngine) -> None:
        """Test sell order closes position."""
        # First buy
        buy_order = Order(
            id="",
            symbol="BTCUSDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )
        await engine.place_order(buy_order, Decimal("40000"))
        
        # Then sell
        sell_order = Order(
            id="",
            symbol="BTCUSDT",
            side=Side.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )
        result = await engine.place_order(sell_order, Decimal("41000"))
        
        assert result.status == OrderStatus.FILLED
        assert not engine.has_position("BTCUSDT")
        assert len(engine.trades) == 1

    @pytest.mark.asyncio
    async def test_profitable_trade(self, engine: PaperTradingEngine) -> None:
        """Test P&L calculation for profitable trade."""
        initial_balance = engine.balance
        
        # Buy at 40000
        buy_order = Order(
            id="",
            symbol="BTCUSDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )
        await engine.place_order(buy_order, Decimal("40000"))
        
        # Sell at 42000 (5% profit)
        sell_order = Order(
            id="",
            symbol="BTCUSDT",
            side=Side.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )
        await engine.place_order(sell_order, Decimal("42000"))
        
        # Check profit (minus fees and slippage)
        trade = engine.trades[0]
        assert trade.pnl is not None
        assert trade.pnl > 0
        assert engine.balance > initial_balance

    @pytest.mark.asyncio
    async def test_losing_trade(self, engine: PaperTradingEngine) -> None:
        """Test P&L calculation for losing trade."""
        initial_balance = engine.balance
        
        # Buy at 40000
        buy_order = Order(
            id="",
            symbol="BTCUSDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )
        await engine.place_order(buy_order, Decimal("40000"))
        
        # Sell at 38000 (5% loss)
        sell_order = Order(
            id="",
            symbol="BTCUSDT",
            side=Side.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )
        await engine.place_order(sell_order, Decimal("38000"))
        
        trade = engine.trades[0]
        assert trade.pnl is not None
        assert trade.pnl < 0
        assert engine.balance < initial_balance

    @pytest.mark.asyncio
    async def test_insufficient_balance(self, engine: PaperTradingEngine) -> None:
        """Test order rejected with insufficient balance."""
        order = Order(
            id="",
            symbol="BTCUSDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10"),  # 10 BTC at 40000 = 400000 USDT
        )
        
        result = await engine.place_order(order, Decimal("40000"))
        
        assert result.status == OrderStatus.REJECTED
        assert engine.balance == Decimal("10000")  # Unchanged

    @pytest.mark.asyncio
    async def test_slippage_applied(self, engine: PaperTradingEngine) -> None:
        """Test slippage is applied to fill price."""
        order = Order(
            id="",
            symbol="BTCUSDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )
        
        result = await engine.place_order(order, Decimal("40000"))
        
        # Buy should have higher fill price due to slippage
        assert result.filled_price is not None
        assert result.filled_price > Decimal("40000")

    @pytest.mark.asyncio
    async def test_fees_deducted(self, engine: PaperTradingEngine) -> None:
        """Test fees are deducted from balance."""
        order = Order(
            id="",
            symbol="BTCUSDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )
        
        result = await engine.place_order(order, Decimal("40000"))
        
        assert result.fees > 0
        # Balance should be less than just the notional
        expected_notional = result.filled_price * Decimal("0.1")
        assert engine.balance < Decimal("10000") - expected_notional

    def test_get_stats_empty(self, engine: PaperTradingEngine) -> None:
        """Test stats with no trades."""
        stats = engine.get_stats()
        
        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["current_balance"] == Decimal("10000")

    @pytest.mark.asyncio
    async def test_get_stats_with_trades(self, engine: PaperTradingEngine) -> None:
        """Test stats after trades."""
        # Execute a profitable trade
        buy = Order(id="", symbol="BTCUSDT", side=Side.BUY, 
                   order_type=OrderType.MARKET, quantity=Decimal("0.1"))
        await engine.place_order(buy, Decimal("40000"))
        
        sell = Order(id="", symbol="BTCUSDT", side=Side.SELL,
                    order_type=OrderType.MARKET, quantity=Decimal("0.1"))
        await engine.place_order(sell, Decimal("42000"))
        
        stats = engine.get_stats()
        
        assert stats["total_trades"] == 1
        assert stats["winning_trades"] == 1
        assert stats["win_rate"] == 100.0

    def test_check_stop_loss(self, engine: PaperTradingEngine) -> None:
        """Test stop-loss check."""
        # Manually add a position with stop-loss
        from packages.core.src.paper_trading import PaperPosition
        
        engine._positions["BTCUSDT"] = PaperPosition(
            symbol="BTCUSDT",
            side=Side.BUY,
            quantity=Decimal("0.1"),
            entry_price=Decimal("40000"),
            stop_loss=Decimal("38000"),
        )
        
        # Price above stop-loss
        assert not engine.check_stop_loss_take_profit("BTCUSDT", Decimal("39000"))
        
        # Price at stop-loss
        assert engine.check_stop_loss_take_profit("BTCUSDT", Decimal("38000"))

    def test_check_take_profit(self, engine: PaperTradingEngine) -> None:
        """Test take-profit check."""
        from packages.core.src.paper_trading import PaperPosition
        
        engine._positions["BTCUSDT"] = PaperPosition(
            symbol="BTCUSDT",
            side=Side.BUY,
            quantity=Decimal("0.1"),
            entry_price=Decimal("40000"),
            take_profit=Decimal("44000"),
        )
        
        # Price below take-profit
        assert not engine.check_stop_loss_take_profit("BTCUSDT", Decimal("42000"))
        
        # Price at take-profit
        assert engine.check_stop_loss_take_profit("BTCUSDT", Decimal("44000"))
