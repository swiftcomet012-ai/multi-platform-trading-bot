"""
Core trading package.

Contains the minimal trading loop components:
- Indicators (RSI, SMA, EMA, ATR)
- Simple RSI Strategy
- Paper Trading Engine
- Trading Engine (orchestrator)
"""

from packages.core.src.indicators import (
    calculate_atr,
    calculate_ema,
    calculate_rsi,
    calculate_sma,
)
from packages.core.src.paper_trading import (
    PaperPosition,
    PaperTradingConfig,
    PaperTradingEngine,
)
from packages.core.src.simple_strategy import (
    SimpleRSIStrategy,
    StrategyConfig,
)
from packages.core.src.trading_engine import (
    TradingEngine,
    TradingEngineConfig,
)

__all__ = [
    # Indicators
    "calculate_atr",
    "calculate_ema",
    "calculate_rsi",
    "calculate_sma",
    # Paper Trading
    "PaperPosition",
    "PaperTradingConfig",
    "PaperTradingEngine",
    # Strategy
    "SimpleRSIStrategy",
    "StrategyConfig",
    # Trading Engine
    "TradingEngine",
    "TradingEngineConfig",
]
