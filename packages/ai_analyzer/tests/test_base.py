"""Tests for AI analyzer base types."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from packages.ai_analyzer.src.base import (
    AIAnalysisResult,
    AIProviderError,
    AIProviderType,
    AIRateLimitError,
    AITimeoutError,
    AIAuthenticationError,
    MarketContext,
)
from packages.shared.src.models import OHLCV, Platform, SignalAction


class TestMarketContext:
    """Test MarketContext."""

    def test_create_context(self) -> None:
        """Test creating market context."""
        context = MarketContext(
            symbol="BTCUSDT",
            platform=Platform.BINANCE,
            candles=[],
            current_price=Decimal("40000"),
        )
        
        assert context.symbol == "BTCUSDT"
        assert context.platform == Platform.BINANCE
        assert context.current_price == Decimal("40000")

    def test_context_with_indicators(self) -> None:
        """Test context with technical indicators."""
        context = MarketContext(
            symbol="BTCUSDT",
            platform=Platform.BINANCE,
            candles=[],
            current_price=Decimal("40000"),
            rsi=Decimal("45.5"),
            sma_20=Decimal("39500"),
            atr=Decimal("500"),
        )
        
        assert context.rsi == Decimal("45.5")
        assert context.sma_20 == Decimal("39500")
        assert context.atr == Decimal("500")

    def test_to_prompt_context(self) -> None:
        """Test converting context to prompt text."""
        context = MarketContext(
            symbol="BTCUSDT",
            platform=Platform.BINANCE,
            candles=[],
            current_price=Decimal("40000"),
            rsi=Decimal("45.5"),
        )
        
        text = context.to_prompt_context()
        
        assert "BTCUSDT" in text
        assert "40000" in text
        assert "45.5" in text

    def test_to_prompt_context_with_candles(self) -> None:
        """Test prompt context includes recent candles."""
        candles = [
            OHLCV(
                timestamp=datetime.now(UTC),
                open=Decimal("39900"),
                high=Decimal("40100"),
                low=Decimal("39800"),
                close=Decimal("40000"),
                volume=Decimal("100"),
                symbol="BTCUSDT",
            )
        ]
        
        context = MarketContext(
            symbol="BTCUSDT",
            platform=Platform.BINANCE,
            candles=candles,
            current_price=Decimal("40000"),
        )
        
        text = context.to_prompt_context()
        
        assert "Recent Candles" in text
        assert "39900" in text


class TestAIAnalysisResult:
    """Test AIAnalysisResult."""

    def test_create_result(self) -> None:
        """Test creating analysis result."""
        result = AIAnalysisResult(
            action=SignalAction.BUY,
            confidence=0.85,
            reasoning="RSI oversold",
            provider=AIProviderType.GEMINI,
        )
        
        assert result.action == SignalAction.BUY
        assert result.confidence == 0.85
        assert result.provider == AIProviderType.GEMINI

    def test_result_with_price_levels(self) -> None:
        """Test result with entry/stop/take-profit."""
        result = AIAnalysisResult(
            action=SignalAction.BUY,
            confidence=0.85,
            reasoning="Strong buy signal",
            provider=AIProviderType.OPENAI,
            entry_price=Decimal("40000"),
            stop_loss=Decimal("39000"),
            take_profit=Decimal("42000"),
        )
        
        assert result.entry_price == Decimal("40000")
        assert result.stop_loss == Decimal("39000")
        assert result.take_profit == Decimal("42000")


class TestAIProviderErrors:
    """Test AI provider error types."""

    def test_provider_error(self) -> None:
        """Test base provider error."""
        error = AIProviderError(
            "Test error",
            AIProviderType.GEMINI,
            retryable=True,
        )
        
        assert str(error) == "Test error"
        assert error.provider == AIProviderType.GEMINI
        assert error.retryable is True

    def test_rate_limit_error(self) -> None:
        """Test rate limit error."""
        error = AIRateLimitError(AIProviderType.OPENAI, retry_after=60.0)
        
        assert error.provider == AIProviderType.OPENAI
        assert error.retry_after == 60.0
        assert error.retryable is True

    def test_timeout_error(self) -> None:
        """Test timeout error."""
        error = AITimeoutError(AIProviderType.GROQ, timeout=30.0)
        
        assert error.provider == AIProviderType.GROQ
        assert error.timeout == 30.0
        assert error.retryable is True

    def test_authentication_error(self) -> None:
        """Test authentication error."""
        error = AIAuthenticationError(AIProviderType.GEMINI)
        
        assert error.provider == AIProviderType.GEMINI
        assert error.retryable is False  # Auth errors are not retryable
