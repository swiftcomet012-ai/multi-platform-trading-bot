"""
Base AI provider interface and types.

Defines the Protocol for all AI providers to implement.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Protocol, runtime_checkable

from packages.shared.src.models import OHLCV, Platform, SignalAction


class AIProviderType(str, Enum):
    """Supported AI providers."""
    
    GEMINI = "gemini"
    OPENAI = "openai"
    GROQ = "groq"
    QWEN = "qwen"
    RULE_BASED = "rule_based"  # Fallback


@dataclass
class MarketContext:
    """Market context for AI analysis."""
    
    symbol: str
    platform: Platform
    candles: list[OHLCV]
    current_price: Decimal
    
    # Technical indicators (pre-calculated)
    rsi: Decimal | None = None
    sma_20: Decimal | None = None
    sma_50: Decimal | None = None
    ema_12: Decimal | None = None
    ema_26: Decimal | None = None
    atr: Decimal | None = None
    
    # Additional context
    timeframe: str = "1h"
    has_position: bool = False
    position_side: str | None = None
    
    def to_prompt_context(self) -> str:
        """Convert to text for AI prompt."""
        lines = [
            f"Symbol: {self.symbol}",
            f"Platform: {self.platform.value}",
            f"Timeframe: {self.timeframe}",
            f"Current Price: {self.current_price}",
            f"Has Position: {self.has_position}",
        ]
        
        if self.rsi is not None:
            lines.append(f"RSI(14): {self.rsi}")
        if self.sma_20 is not None:
            lines.append(f"SMA(20): {self.sma_20}")
        if self.sma_50 is not None:
            lines.append(f"SMA(50): {self.sma_50}")
        if self.atr is not None:
            lines.append(f"ATR(14): {self.atr}")
        
        # Recent price action
        if self.candles:
            recent = self.candles[-5:]
            lines.append("\nRecent Candles (last 5):")
            for c in recent:
                lines.append(f"  O:{c.open} H:{c.high} L:{c.low} C:{c.close}")
        
        return "\n".join(lines)


@dataclass
class AIAnalysisResult:
    """Result from AI analysis."""
    
    action: SignalAction
    confidence: float  # 0.0 - 1.0
    reasoning: str
    provider: AIProviderType
    
    # Optional details
    entry_price: Decimal | None = None
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    
    # Metadata
    model: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    cached: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


@runtime_checkable
class AIProvider(Protocol):
    """Protocol for AI providers."""
    
    @property
    def provider_type(self) -> AIProviderType:
        """Get provider type."""
        ...
    
    @property
    def is_available(self) -> bool:
        """Check if provider is available (has API key, etc.)."""
        ...
    
    async def analyze(
        self,
        context: MarketContext,
        timeout: float = 30.0,
    ) -> AIAnalysisResult:
        """
        Analyze market context and return trading signal.
        
        Args:
            context: Market context with price data and indicators.
            timeout: Request timeout in seconds.
        
        Returns:
            AIAnalysisResult with action, confidence, and reasoning.
        
        Raises:
            AIProviderError: If analysis fails.
        """
        ...


class AIProviderError(Exception):
    """Base exception for AI provider errors."""
    
    def __init__(
        self,
        message: str,
        provider: AIProviderType,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class AIRateLimitError(AIProviderError):
    """Rate limit exceeded."""
    
    def __init__(self, provider: AIProviderType, retry_after: float = 60.0) -> None:
        super().__init__(
            f"Rate limit exceeded for {provider.value}",
            provider,
            retryable=True,
        )
        self.retry_after = retry_after


class AITimeoutError(AIProviderError):
    """Request timed out."""
    
    def __init__(self, provider: AIProviderType, timeout: float) -> None:
        super().__init__(
            f"Request timed out after {timeout}s for {provider.value}",
            provider,
            retryable=True,
        )
        self.timeout = timeout


class AIAuthenticationError(AIProviderError):
    """Authentication failed."""
    
    def __init__(self, provider: AIProviderType) -> None:
        super().__init__(
            f"Authentication failed for {provider.value}",
            provider,
            retryable=False,
        )


# System prompt for trading analysis
TRADING_ANALYSIS_PROMPT = """You are an expert cryptocurrency and forex trading analyst.
Analyze the provided market data and give a trading recommendation.

Your response MUST be in this exact JSON format:
{
    "action": "buy" | "sell" | "hold",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation",
    "entry_price": number or null,
    "stop_loss": number or null,
    "take_profit": number or null
}

Guidelines:
- BUY when: RSI < 30 (oversold), price above SMA, bullish momentum
- SELL when: RSI > 70 (overbought), price below SMA, bearish momentum
- HOLD when: No clear signal, mixed indicators, or uncertain conditions
- Confidence should reflect how strong the signal is
- Always consider risk management (stop-loss, take-profit)

Be conservative. When in doubt, recommend HOLD.
"""
