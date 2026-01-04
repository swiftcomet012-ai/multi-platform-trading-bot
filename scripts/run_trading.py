#!/usr/bin/env python
"""
CLI script to run the minimal trading loop.

Usage:
    # Paper trading (default, safe)
    python scripts/run_trading.py
    
    # With custom symbol
    python scripts/run_trading.py --symbol ETHUSDT
    
    # Single iteration (for testing)
    python scripts/run_trading.py --once
    
    # Live trading (DANGEROUS - requires confirmation)
    python scripts/run_trading.py --live
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from decimal import Decimal

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages.connectors.src.binance_connector import BinanceConnector
from packages.core.src.paper_trading import PaperTradingConfig
from packages.core.src.simple_strategy import StrategyConfig
from packages.core.src.trading_engine import TradingEngine, TradingEngineConfig
from packages.shared.src.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the minimal trading loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)",
    )
    parser.add_argument(
        "--timeframe",
        default="1h",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        help="Candle timeframe (default: 1h)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=10000,
        help="Initial paper trading balance (default: 10000)",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=0.1,
        help="Position size as fraction of balance (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run single iteration and exit",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (DANGEROUS)",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        default=True,
        help="Use Binance testnet (default: True)",
    )
    
    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Safety check for live trading
    if args.live:
        print("\n" + "=" * 60)
        print("⚠️  WARNING: LIVE TRADING MODE ⚠️")
        print("=" * 60)
        print("You are about to trade with REAL MONEY.")
        print("This can result in FINANCIAL LOSS.")
        print("=" * 60)
        
        confirm = input("\nType 'I UNDERSTAND THE RISKS' to continue: ")
        if confirm != "I UNDERSTAND THE RISKS":
            print("Aborted.")
            return 1
    
    # Get API credentials from environment
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    
    if not api_key or not api_secret:
        print("Error: BINANCE_API_KEY and BINANCE_API_SECRET environment variables required")
        print("\nFor testnet, get keys from: https://testnet.binance.vision/")
        return 1
    
    # Create connector
    connector = BinanceConnector(
        api_key=api_key,
        api_secret=api_secret,
        testnet=args.testnet and not args.live,
    )
    
    # Create configs
    engine_config = TradingEngineConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        check_interval_seconds=args.interval,
        position_size_pct=Decimal(str(args.position_size)),
        paper_trading=not args.live,
    )
    
    paper_config = PaperTradingConfig(
        initial_balance=Decimal(str(args.balance)),
    )
    
    strategy_config = StrategyConfig()
    
    # Create engine
    engine = TradingEngine(
        connector=connector,
        config=engine_config,
        strategy_config=strategy_config,
        paper_config=paper_config,
    )
    
    mode = "LIVE 🔴" if args.live else "PAPER 🟢"
    print(f"\n{'=' * 60}")
    print(f"Trading Bot Started - {mode}")
    print(f"{'=' * 60}")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Check Interval: {args.interval}s")
    print(f"Position Size: {args.position_size * 100}%")
    if not args.live:
        print(f"Paper Balance: ${args.balance:,.2f}")
    print(f"{'=' * 60}\n")
    
    try:
        if args.once:
            # Single iteration
            result = await engine.run_once()
            print("\nIteration Result:")
            print("-" * 40)
            for key, value in result.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"{key}: {value}")
            
            print("\nStats:")
            print("-" * 40)
            stats = engine.get_stats()
            for key, value in stats.items():
                print(f"{key}: {value}")
        else:
            # Continuous loop
            print("Press Ctrl+C to stop\n")
            await engine.start()
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
        await engine.stop()
    
    finally:
        await connector.disconnect()
    
    # Print final stats
    print("\nFinal Stats:")
    print("-" * 40)
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
