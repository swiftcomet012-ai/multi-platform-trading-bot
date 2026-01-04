"""
Simple RSI-based strategy for minimal trading loop.

Rules:
- BUY when RSI < 30 (oversold)
- SELL when RSI > 70 (overbought)
- HOLD otherwise
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Sequence

from packages.core.src.indicators import calculate_atr, calculate_rsi
from packages.shared.src.logging import get_logger
from packages.shared.src.models import OHLCV, Platform, Signal, SignalAction

logger = get_logger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for simple RSI strategy."""
    
    rsi_period: int = 14
    rsi_oversold: Decimal = Decimal("30")
    rsi_overbought: Decimal = Decimal("70")
    atr_period: int = 14
    atr_multiplier: Decimal = Decimal("2.0")  # For stop-loss calculation
    min_confidence: Decimal = Decimal("0.6")


class SimpleRSIStrategy:
    """
    Simple RSI-based trading strategy.
    
    Generates BUY/SELL/HOLD signals based on RSI levels.
    """
    
    def __init__(self, config: StrategyConfig | None = None) -> None:
        """Initialize strategy with config."""
        self.config = config or StrategyConfig()
        self.name = "simple_rsi"
    
    def analyze(
        self,
        candles: Sequence[OHLCV],
        symbol: str,
        platform: Platform = Platform.BINANCE,
    ) -> Signal:
        """
        Analyze candles and generate trading signal.
        
        Args:
            candles: Historical OHLCV data (oldest first).
            symbol: Trading symbol.
            platform: Trading platform.
        
        Returns:
            Trading signal with action and confidence.
        """
        # Calculate indicators
        rsi = calculate_rsi(candles, self.config.rsi_period)
        atr = calculate_atr(candles, self.config.atr_period)
        
        if rsi is None:
            logger.warning(
                "strategy_insufficient_data",
                symbol=symbol,
                required=self.config.rsi_period + 1,
                available=len(candles),
            )
            return self._create_signal(
                symbol=symbol,
                platform=platform,
                action=SignalAction.HOLD,
                confidence=0.0,
                reasoning="Insufficient data for RSI calculation",
                metadata={"rsi": None, "atr": str(atr) if atr else None},
            )
        
        # Determine action based on RSI
        action = SignalAction.HOLD
        confidence = 0.5
        reasoning = f"RSI={rsi}"
        
        if rsi < self.config.rsi_oversold:
            action = SignalAction.BUY
            # Higher confidence when RSI is more extreme
            confidence = float(
                self.config.min_confidence + 
                (self.config.rsi_oversold - rsi) / self.config.rsi_oversold * Decimal("0.3")
            )
            reasoning = f"RSI={rsi} < {self.config.rsi_oversold} (oversold)"
            
        elif rsi > self.config.rsi_overbought:
            action = SignalAction.SELL
            confidence = float(
                self.config.min_confidence +
                (rsi - self.config.rsi_overbought) / (100 - self.config.rsi_overbought) * Decimal("0.3")
            )
            reasoning = f"RSI={rsi} > {self.config.rsi_overbought} (overbought)"
        
        # Cap confidence at 0.95
        confidence = min(confidence, 0.95)
        
        logger.info(
            "strategy_signal_generated",
            symbol=symbol,
            action=action.value,
            confidence=confidence,
            rsi=str(rsi),
            atr=str(atr) if atr else None,
        )
        
        return self._create_signal(
            symbol=symbol,
            platform=platform,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "rsi": str(rsi),
                "atr": str(atr) if atr else None,
                "rsi_oversold": str(self.config.rsi_oversold),
                "rsi_overbought": str(self.config.rsi_overbought),
            },
        )
    
    def calculate_stop_loss(
        self,
        entry_price: Decimal,
        atr: Decimal | None,
        side: SignalAction,
    ) -> Decimal | None:
        """
        Calculate stop-loss price based on ATR.
        
        Args:
            entry_price: Entry price.
            atr: Average True Range.
            side: BUY or SELL.
        
        Returns:
            Stop-loss price or None if ATR not available.
        """
        if atr is None:
            return None
        
        stop_distance = atr * self.config.atr_multiplier
        
        if side == SignalAction.BUY:
            return entry_price - stop_distance
        elif side == SignalAction.SELL:
            return entry_price + stop_distance
        
        return None
    
    def _create_signal(
        self,
        symbol: str,
        platform: Platform,
        action: SignalAction,
        confidence: float,
        reasoning: str,
        metadata: dict | None = None,
    ) -> Signal:
        """Create a Signal object."""
        return Signal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            strategy=self.name,
            timestamp=datetime.now(UTC),
            platform=platform,
            metadata=metadata,
        )
