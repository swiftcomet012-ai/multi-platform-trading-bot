"""
Rule-based fallback provider.

Uses technical indicators to generate signals without AI.
This is the fallback when all AI providers fail.
"""

from __future__ import annotations

import time
from decimal import Decimal

from packages.ai_analyzer.src.base import (
    AIAnalysisResult,
    AIProviderType,
    MarketContext,
)
from packages.shared.src.logging import get_logger
from packages.shared.src.models import SignalAction

logger = get_logger(__name__)


class RuleBasedProvider:
    """
    Rule-based trading signal provider.
    
    Uses simple technical analysis rules:
    - RSI < 30: BUY (oversold)
    - RSI > 70: SELL (overbought)
    - Price > SMA: Bullish bias
    - Price < SMA: Bearish bias
    """
    
    def __init__(
        self,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
    ) -> None:
        """
        Initialize rule-based provider.
        
        Args:
            rsi_oversold: RSI threshold for oversold (default: 30).
            rsi_overbought: RSI threshold for overbought (default: 70).
        """
        self._rsi_oversold = rsi_oversold
        self._rsi_overbought = rsi_overbought
    
    @property
    def provider_type(self) -> AIProviderType:
        """Get provider type."""
        return AIProviderType.RULE_BASED
    
    @property
    def is_available(self) -> bool:
        """Always available as fallback."""
        return True
    
    async def analyze(
        self,
        context: MarketContext,
        timeout: float = 30.0,
    ) -> AIAnalysisResult:
        """Analyze market context using rules."""
        start_time = time.time()
        
        action = SignalAction.HOLD
        confidence = 0.5
        reasons = []
        
        # RSI analysis
        if context.rsi is not None:
            rsi = float(context.rsi)
            
            if rsi < self._rsi_oversold:
                action = SignalAction.BUY
                # Higher confidence for more extreme RSI
                confidence = min(0.9, 0.6 + (self._rsi_oversold - rsi) / 100)
                reasons.append(f"RSI={rsi:.1f} < {self._rsi_oversold} (oversold)")
                
            elif rsi > self._rsi_overbought:
                action = SignalAction.SELL
                confidence = min(0.9, 0.6 + (rsi - self._rsi_overbought) / 100)
                reasons.append(f"RSI={rsi:.1f} > {self._rsi_overbought} (overbought)")
                
            else:
                reasons.append(f"RSI={rsi:.1f} (neutral)")
        
        # SMA trend analysis
        if context.sma_20 is not None and context.current_price:
            price = float(context.current_price)
            sma = float(context.sma_20)
            
            if price > sma * 1.02:  # 2% above SMA
                if action == SignalAction.BUY:
                    confidence = min(0.95, confidence + 0.1)
                reasons.append(f"Price above SMA20 (bullish)")
                
            elif price < sma * 0.98:  # 2% below SMA
                if action == SignalAction.SELL:
                    confidence = min(0.95, confidence + 0.1)
                reasons.append(f"Price below SMA20 (bearish)")
        
        # Position check
        if context.has_position:
            if action == SignalAction.BUY:
                action = SignalAction.HOLD
                confidence = 0.5
                reasons.append("Already in position, holding")
        
        latency_ms = (time.time() - start_time) * 1000
        reasoning = "; ".join(reasons) if reasons else "No clear signal"
        
        logger.info(
            "rule_based_analysis_complete",
            action=action.value,
            confidence=confidence,
            reasoning=reasoning,
        )
        
        return AIAnalysisResult(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            provider=AIProviderType.RULE_BASED,
            model="rule_based_v1",
            latency_ms=latency_ms,
        )
