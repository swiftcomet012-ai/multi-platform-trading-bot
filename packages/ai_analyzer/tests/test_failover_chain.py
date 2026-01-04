"""Tests for AI failover chain."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from packages.ai_analyzer.src.base import (
    AIAnalysisResult,
    AIProviderError,
    AIProviderType,
    AIRateLimitError,
    MarketContext,
)
from packages.ai_analyzer.src.failover_chain import (
    AIFailoverChain,
    FailoverChainConfig,
)
from packages.shared.src.models import Platform, SignalAction


def create_mock_context() -> MarketContext:
    """Create a mock market context."""
    return MarketContext(
        symbol="BTCUSDT",
        platform=Platform.BINANCE,
        candles=[],
        current_price=Decimal("40000"),
        rsi=Decimal("45"),
    )


def create_mock_result(
    action: SignalAction = SignalAction.HOLD,
    provider: AIProviderType = AIProviderType.GEMINI,
) -> AIAnalysisResult:
    """Create a mock analysis result."""
    return AIAnalysisResult(
        action=action,
        confidence=0.8,
        reasoning="Test reasoning",
        provider=provider,
    )


class TestFailoverChainConfig:
    """Test FailoverChainConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = FailoverChainConfig()
        
        assert AIProviderType.GEMINI in config.provider_order
        assert config.cache_enabled is True
        assert config.cache_ttl_seconds == 300
        assert config.use_rule_based_fallback is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = FailoverChainConfig(
            provider_order=[AIProviderType.OPENAI],
            cache_enabled=False,
            timeout_seconds=60.0,
        )
        
        assert config.provider_order == [AIProviderType.OPENAI]
        assert config.cache_enabled is False
        assert config.timeout_seconds == 60.0


class TestAIFailoverChain:
    """Test AIFailoverChain."""

    @pytest.fixture
    def chain(self) -> AIFailoverChain:
        """Create failover chain."""
        return AIFailoverChain()

    def test_initialization(self, chain: AIFailoverChain) -> None:
        """Test chain initialization."""
        assert chain.config is not None
        assert len(chain._providers) > 0

    def test_get_provider_stats(self, chain: AIFailoverChain) -> None:
        """Test getting provider stats."""
        stats = chain.get_provider_stats()
        
        assert "gemini" in stats
        assert "openai" in stats
        assert "groq" in stats
        assert "cache_size" in stats

    def test_clear_cache(self, chain: AIFailoverChain) -> None:
        """Test clearing cache."""
        # Add something to cache
        chain._cache["test"] = MagicMock()
        
        chain.clear_cache()
        
        assert len(chain._cache) == 0

    def test_reset_provider_stats(self, chain: AIFailoverChain) -> None:
        """Test resetting provider stats."""
        chain._provider_failures[AIProviderType.GEMINI] = 5
        
        chain.reset_provider_stats()
        
        assert len(chain._provider_failures) == 0


class TestFailoverChainAnalysis:
    """Test failover chain analysis."""

    @pytest.mark.asyncio
    async def test_uses_rule_based_when_no_api_keys(self) -> None:
        """Test falls back to rule-based when no API keys."""
        chain = AIFailoverChain()
        context = create_mock_context()
        
        # Without API keys, should use rule-based
        result = await chain.analyze(context)
        
        assert result.provider == AIProviderType.RULE_BASED

    @pytest.mark.asyncio
    async def test_caches_result(self) -> None:
        """Test results are cached."""
        chain = AIFailoverChain()
        context = create_mock_context()
        
        # First call
        result1 = await chain.analyze(context)
        
        # Second call should be cached
        result2 = await chain.analyze(context)
        
        # Both should be from rule-based (no API keys)
        assert result1.provider == result2.provider

    @pytest.mark.asyncio
    async def test_cache_disabled(self) -> None:
        """Test cache can be disabled."""
        config = FailoverChainConfig(cache_enabled=False)
        chain = AIFailoverChain(config)
        context = create_mock_context()
        
        await chain.analyze(context)
        
        assert len(chain._cache) == 0

    @pytest.mark.asyncio
    async def test_failover_on_error(self) -> None:
        """Test failover to next provider on error."""
        chain = AIFailoverChain()
        context = create_mock_context()
        
        # Mock Gemini to fail
        mock_gemini = AsyncMock()
        mock_gemini.is_available = True
        mock_gemini.analyze.side_effect = AIProviderError(
            "Test error",
            AIProviderType.GEMINI,
        )
        
        # Mock OpenAI to succeed
        mock_openai = AsyncMock()
        mock_openai.is_available = True
        mock_openai.analyze.return_value = create_mock_result(
            provider=AIProviderType.OPENAI
        )
        
        chain._providers[AIProviderType.GEMINI] = mock_gemini
        chain._providers[AIProviderType.OPENAI] = mock_openai
        
        result = await chain.analyze(context)
        
        assert result.provider == AIProviderType.OPENAI

    @pytest.mark.asyncio
    async def test_tracks_failures(self) -> None:
        """Test provider failures are tracked."""
        chain = AIFailoverChain()
        context = create_mock_context()
        
        # Mock provider to fail
        mock_provider = AsyncMock()
        mock_provider.is_available = True
        mock_provider.analyze.side_effect = AIProviderError(
            "Test error",
            AIProviderType.GEMINI,
            retryable=False,
        )
        
        chain._providers[AIProviderType.GEMINI] = mock_provider
        
        await chain.analyze(context)
        
        assert chain._provider_failures.get(AIProviderType.GEMINI, 0) > 0


class TestFailoverChainPropertyTests:
    """Property-based tests for failover chain."""

    @pytest.mark.asyncio
    async def test_always_returns_result(self) -> None:
        """Test chain always returns a result (never raises)."""
        chain = AIFailoverChain()
        
        # Various contexts
        contexts = [
            MarketContext(
                symbol="BTCUSDT",
                platform=Platform.BINANCE,
                candles=[],
                current_price=Decimal("40000"),
                rsi=Decimal("25"),  # Oversold
            ),
            MarketContext(
                symbol="ETHUSDT",
                platform=Platform.BINANCE,
                candles=[],
                current_price=Decimal("2000"),
                rsi=Decimal("75"),  # Overbought
            ),
            MarketContext(
                symbol="BTCUSDT",
                platform=Platform.BINANCE,
                candles=[],
                current_price=Decimal("40000"),
                rsi=None,  # No RSI
            ),
        ]
        
        for context in contexts:
            result = await chain.analyze(context)
            
            # Should always get a valid result
            assert result is not None
            assert result.action in [SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD]
            assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_failover_chain_property(self) -> None:
        """
        Property: If all AI providers fail, rule-based fallback is used.
        
        **Property 4: AI failover chain**
        **Validates: Requirements 59.1, 59.3**
        """
        config = FailoverChainConfig(
            provider_order=[AIProviderType.GEMINI, AIProviderType.OPENAI],
            use_rule_based_fallback=True,
        )
        chain = AIFailoverChain(config)
        
        # Mock all AI providers to fail
        for provider_type in [AIProviderType.GEMINI, AIProviderType.OPENAI]:
            mock = AsyncMock()
            mock.is_available = True
            mock.analyze.side_effect = AIProviderError(
                "Test failure",
                provider_type,
            )
            chain._providers[provider_type] = mock
        
        context = create_mock_context()
        result = await chain.analyze(context)
        
        # Should fall back to rule-based
        assert result.provider == AIProviderType.RULE_BASED
        assert result.action in [SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD]
