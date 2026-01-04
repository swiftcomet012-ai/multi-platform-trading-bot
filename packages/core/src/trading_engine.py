"""
Minimal Trading Engine - orchestrates the trading loop.

Flow: Fetch Data → Calculate Indicators → Generate Signal → Execute Order
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from packages.connectors.src.base import ExchangeConnector
from packages.core.src.indicators import calculate_atr
from packages.core.src.paper_trading import PaperTradingConfig, PaperTradingEngine
from packages.core.src.simple_strategy import SimpleRSIStrategy, StrategyConfig
from packages.shared.src.logging import get_logger
from packages.shared.src.models import (
    Order,
    OrderType,
    Platform,
    Side,
    SignalAction,
)

logger = get_logger(__name__)


@dataclass
class TradingEngineConfig:
    """Configuration for trading engine."""
    
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    candles_to_fetch: int = 50
    position_size_pct: Decimal = Decimal("0.1")  # 10% of balance per trade
    min_confidence: float = 0.6
    check_interval_seconds: int = 60  # How often to check for signals
    paper_trading: bool = True  # Safety: default to paper trading


class TradingEngine:
    """
    Minimal trading engine for end-to-end testing.
    
    Supports both paper trading and live trading modes.
    """
    
    def __init__(
        self,
        connector: ExchangeConnector,
        config: TradingEngineConfig | None = None,
        strategy_config: StrategyConfig | None = None,
        paper_config: PaperTradingConfig | None = None,
    ) -> None:
        """
        Initialize trading engine.
        
        Args:
            connector: Exchange connector for market data and orders.
            config: Engine configuration.
            strategy_config: Strategy configuration.
            paper_config: Paper trading configuration.
        """
        self.connector = connector
        self.config = config or TradingEngineConfig()
        self.strategy = SimpleRSIStrategy(strategy_config)
        
        # Paper trading engine (used when paper_trading=True)
        self.paper_engine = PaperTradingEngine(paper_config, connector.platform)
        
        # State
        self._running = False
        self._last_signal_time: datetime | None = None
        self._iteration_count = 0
    
    @property
    def is_paper_trading(self) -> bool:
        """Check if running in paper trading mode."""
        return self.config.paper_trading
    
    async def start(self) -> None:
        """Start the trading loop."""
        if self._running:
            logger.warning("trading_engine_already_running")
            return
        
        self._running = True
        mode = "PAPER" if self.is_paper_trading else "LIVE"
        
        logger.info(
            "trading_engine_started",
            mode=mode,
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            interval=self.config.check_interval_seconds,
        )
        
        # Connect to exchange
        if not self.connector.is_connected:
            await self.connector.connect()
        
        # Main loop
        while self._running:
            try:
                await self._trading_iteration()
            except Exception as e:
                logger.error("trading_iteration_error", error=str(e))
            
            await asyncio.sleep(self.config.check_interval_seconds)
    
    async def stop(self) -> None:
        """Stop the trading loop."""
        self._running = False
        logger.info("trading_engine_stopped", iterations=self._iteration_count)
    
    async def run_once(self) -> dict[str, Any]:
        """
        Run a single trading iteration.
        
        Useful for testing without the loop.
        
        Returns:
            Dict with iteration results.
        """
        if not self.connector.is_connected:
            await self.connector.connect()
        
        return await self._trading_iteration()


    async def _trading_iteration(self) -> dict[str, Any]:
        """Execute one trading iteration."""
        self._iteration_count += 1
        result: dict[str, Any] = {
            "iteration": self._iteration_count,
            "timestamp": datetime.now(UTC).isoformat(),
            "symbol": self.config.symbol,
        }
        
        try:
            # 1. Fetch market data
            candles = await self.connector.get_ohlcv(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                limit=self.config.candles_to_fetch,
            )
            
            if not candles:
                result["error"] = "No candles fetched"
                return result
            
            current_price = candles[-1].close
            result["current_price"] = str(current_price)
            
            # 2. Generate signal
            signal = self.strategy.analyze(
                candles=candles,
                symbol=self.config.symbol,
                platform=self.connector.platform,
            )
            
            result["signal"] = {
                "action": signal.action.value,
                "confidence": signal.confidence,
                "reasoning": signal.reasoning,
            }
            
            # 3. Check if we should act
            if signal.action == SignalAction.HOLD:
                result["action_taken"] = "none"
                return result
            
            if signal.confidence < self.config.min_confidence:
                result["action_taken"] = "skipped_low_confidence"
                return result
            
            # 4. Check existing position
            has_position = self._has_position(self.config.symbol)
            
            # Don't buy if already have position
            if signal.action == SignalAction.BUY and has_position:
                result["action_taken"] = "skipped_already_in_position"
                return result
            
            # Don't sell if no position
            if signal.action == SignalAction.SELL and not has_position:
                result["action_taken"] = "skipped_no_position"
                return result
            
            # 5. Calculate position size
            balance = await self._get_balance()
            position_size = self._calculate_position_size(balance, current_price)
            
            if position_size <= 0:
                result["action_taken"] = "skipped_insufficient_balance"
                return result
            
            # 6. Calculate stop-loss
            atr = calculate_atr(candles, self.strategy.config.atr_period)
            stop_loss = self.strategy.calculate_stop_loss(current_price, atr, signal.action)
            
            # 7. Create and execute order
            order = Order(
                id="",
                symbol=self.config.symbol,
                side=Side.BUY if signal.action == SignalAction.BUY else Side.SELL,
                order_type=OrderType.MARKET,
                quantity=position_size,
                stop_loss=stop_loss,
                platform=self.connector.platform,
            )
            
            executed_order = await self._execute_order(order, current_price)
            
            result["order"] = {
                "id": executed_order.id,
                "side": executed_order.side.value,
                "quantity": str(executed_order.quantity),
                "filled_price": str(executed_order.filled_price) if executed_order.filled_price else None,
                "status": executed_order.status.value,
                "stop_loss": str(stop_loss) if stop_loss else None,
            }
            result["action_taken"] = "order_executed"
            
            self._last_signal_time = datetime.now(UTC)
            
        except Exception as e:
            result["error"] = str(e)
            logger.error("trading_iteration_failed", error=str(e))
        
        return result
    
    def _has_position(self, symbol: str) -> bool:
        """Check if there's an open position."""
        if self.is_paper_trading:
            return self.paper_engine.has_position(symbol)
        # For live trading, would check connector
        return False
    
    async def _get_balance(self) -> Decimal:
        """Get available balance."""
        if self.is_paper_trading:
            return self.paper_engine.balance
        
        balances = await self.connector.get_balance()
        # Assume USDT for now
        return balances.get("USDT", Decimal("0"))
    
    def _calculate_position_size(
        self,
        balance: Decimal,
        price: Decimal,
    ) -> Decimal:
        """Calculate position size based on balance and config."""
        # Amount to risk
        risk_amount = balance * self.config.position_size_pct
        
        # Convert to quantity
        quantity = risk_amount / price
        
        # Round to reasonable precision (will be adjusted by exchange info)
        return quantity.quantize(Decimal("0.00001"))
    
    async def _execute_order(
        self,
        order: Order,
        current_price: Decimal,
    ) -> Order:
        """Execute order through paper or live trading."""
        if self.is_paper_trading:
            return await self.paper_engine.place_order(order, current_price)
        
        # Live trading
        return await self.connector.place_order(order)
    
    def get_stats(self) -> dict[str, Any]:
        """Get trading statistics."""
        stats = {
            "mode": "paper" if self.is_paper_trading else "live",
            "iterations": self._iteration_count,
            "symbol": self.config.symbol,
            "timeframe": self.config.timeframe,
        }
        
        if self.is_paper_trading:
            stats.update(self.paper_engine.get_stats())
        
        return stats
