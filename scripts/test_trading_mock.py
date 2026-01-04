#!/usr/bin/env python
"""
Test trading loop with mock data (no API keys needed).

This simulates the full trading flow without connecting to Binance.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import UTC, datetime
from decimal import Decimal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages.core.src.indicators import calculate_rsi
from packages.core.src.paper_trading import PaperTradingConfig, PaperTradingEngine
from packages.core.src.simple_strategy import SimpleRSIStrategy, StrategyConfig
from packages.shared.src.models import OHLCV, Platform


def generate_mock_candles(trend: str = "neutral", count: int = 50) -> list[OHLCV]:
    """Generate mock OHLCV data."""
    candles = []
    base_price = Decimal("40000")
    
    for i in range(count):
        if trend == "down":
            # Downtrend - RSI will be low (oversold)
            price = base_price - Decimal(str(i * 200))
        elif trend == "up":
            # Uptrend - RSI will be high (overbought)
            price = base_price + Decimal(str(i * 200))
        else:
            # Neutral - oscillate
            price = base_price + Decimal(str((i % 10 - 5) * 100))
        
        candles.append(OHLCV(
            timestamp=datetime.now(UTC),
            open=price - Decimal("50"),
            high=price + Decimal("100"),
            low=price - Decimal("100"),
            close=price,
            volume=Decimal("1000"),
            symbol="BTCUSDT",
            timeframe="1h",
            platform=Platform.BINANCE,
        ))
    
    return candles


async def test_oversold_scenario():
    """Test BUY signal when market is oversold."""
    print("\n" + "=" * 60)
    print("SCENARIO 1: Oversold Market (Downtrend)")
    print("=" * 60)
    
    # Generate downtrend data
    candles = generate_mock_candles(trend="down")
    
    # Calculate RSI
    rsi = calculate_rsi(candles)
    print(f"RSI: {rsi}")
    
    # Get strategy signal
    strategy = SimpleRSIStrategy()
    signal = strategy.analyze(candles, "BTCUSDT", Platform.BINANCE)
    
    print(f"Signal: {signal.action.value.upper()}")
    print(f"Confidence: {signal.confidence:.1%}")
    print(f"Reasoning: {signal.reasoning}")
    
    # Execute paper trade if BUY
    if signal.action.value == "buy":
        paper_engine = PaperTradingEngine(
            PaperTradingConfig(initial_balance=Decimal("10000"))
        )
        
        from packages.shared.src.models import Order, OrderType, Side
        
        current_price = candles[-1].close
        quantity = Decimal("0.01")  # 0.01 BTC
        
        order = Order(
            id="",
            symbol="BTCUSDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
        )
        
        result = await paper_engine.place_order(order, current_price)
        
        print(f"\n📈 Paper Trade Executed:")
        print(f"   Order ID: {result.id}")
        print(f"   Side: {result.side.value.upper()}")
        print(f"   Quantity: {result.quantity} BTC")
        print(f"   Fill Price: ${result.filled_price:,.2f}")
        print(f"   Fee: ${result.fees:.4f}")
        print(f"   Balance: ${paper_engine.balance:,.2f}")


async def test_overbought_scenario():
    """Test SELL signal when market is overbought."""
    print("\n" + "=" * 60)
    print("SCENARIO 2: Overbought Market (Uptrend)")
    print("=" * 60)
    
    # Generate uptrend data
    candles = generate_mock_candles(trend="up")
    
    # Calculate RSI
    rsi = calculate_rsi(candles)
    print(f"RSI: {rsi}")
    
    # Get strategy signal
    strategy = SimpleRSIStrategy()
    signal = strategy.analyze(candles, "BTCUSDT", Platform.BINANCE)
    
    print(f"Signal: {signal.action.value.upper()}")
    print(f"Confidence: {signal.confidence:.1%}")
    print(f"Reasoning: {signal.reasoning}")


async def test_neutral_scenario():
    """Test HOLD signal when market is neutral."""
    print("\n" + "=" * 60)
    print("SCENARIO 3: Neutral Market (Sideways)")
    print("=" * 60)
    
    # Generate neutral data
    candles = generate_mock_candles(trend="neutral")
    
    # Calculate RSI
    rsi = calculate_rsi(candles)
    print(f"RSI: {rsi}")
    
    # Get strategy signal
    strategy = SimpleRSIStrategy()
    signal = strategy.analyze(candles, "BTCUSDT", Platform.BINANCE)
    
    print(f"Signal: {signal.action.value.upper()}")
    print(f"Confidence: {signal.confidence:.1%}")
    print(f"Reasoning: {signal.reasoning}")


async def test_full_trade_cycle():
    """Test complete buy and sell cycle."""
    print("\n" + "=" * 60)
    print("SCENARIO 4: Full Trade Cycle (Buy → Sell)")
    print("=" * 60)
    
    from packages.shared.src.models import Order, OrderType, Side
    
    paper_engine = PaperTradingEngine(
        PaperTradingConfig(initial_balance=Decimal("10000"))
    )
    
    print(f"Initial Balance: ${paper_engine.balance:,.2f}")
    
    # 1. Buy at 40000
    buy_price = Decimal("40000")
    buy_order = Order(
        id="",
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.1"),
    )
    
    buy_result = await paper_engine.place_order(buy_order, buy_price)
    print(f"\n1. BUY 0.1 BTC @ ${buy_price:,.2f}")
    print(f"   Fill Price: ${buy_result.filled_price:,.2f}")
    print(f"   Balance: ${paper_engine.balance:,.2f}")
    print(f"   Position: {paper_engine.has_position('BTCUSDT')}")
    
    # 2. Sell at 42000 (5% profit)
    sell_price = Decimal("42000")
    sell_order = Order(
        id="",
        symbol="BTCUSDT",
        side=Side.SELL,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.1"),
    )
    
    sell_result = await paper_engine.place_order(sell_order, sell_price)
    print(f"\n2. SELL 0.1 BTC @ ${sell_price:,.2f}")
    print(f"   Fill Price: ${sell_result.filled_price:,.2f}")
    print(f"   Balance: ${paper_engine.balance:,.2f}")
    print(f"   Position: {paper_engine.has_position('BTCUSDT')}")
    
    # 3. Show stats
    stats = paper_engine.get_stats()
    print(f"\n📊 Trading Stats:")
    print(f"   Total Trades: {stats['total_trades']}")
    print(f"   Winning: {stats['winning_trades']}")
    print(f"   Win Rate: {stats['win_rate']:.1f}%")
    print(f"   Total P&L: ${stats['total_pnl']:,.2f}")
    print(f"   P&L %: {stats['total_pnl_pct']:.2f}%")


async def main():
    """Run all test scenarios."""
    print("\n" + "=" * 60)
    print("🤖 TRADING BOT MOCK TEST")
    print("=" * 60)
    print("Testing trading logic without real API connection")
    
    await test_oversold_scenario()
    await test_overbought_scenario()
    await test_neutral_scenario()
    await test_full_trade_cycle()
    
    print("\n" + "=" * 60)
    print("✅ All scenarios completed!")
    print("=" * 60)
    print("\nTo test with real Binance testnet:")
    print("1. Get keys from: https://testnet.binance.vision/")
    print("2. Run: set BINANCE_API_KEY=your_key")
    print("3. Run: set BINANCE_API_SECRET=your_secret")
    print("4. Run: python scripts/run_trading.py --once --testnet")


if __name__ == "__main__":
    asyncio.run(main())
