"""Tests for rule-based provider."""

from __future__ import annotations

from decimal import Decimal

import pytest

from packages.ai_analyzer.src.base import AIProviderType, MarketContext
from packages.ai_analyzer.src.rule_based_provider import RuleBasedProvider
from packages.shared.src.models import Platform, SignalAction


class TestRuleBasedProvider:
    """Test RuleBasedProvider."""

    @pytest.fixture
    def provider(self) -> RuleBasedProvider:
        """Create rule-based provider."""
        return RuleBasedProvider()

    def test_provider_type(self, provider: RuleBasedProvider) -> None:
        """Test provider type."""
        assert provider.provider_type == AIProviderType.RULE_BASED

    def test_always_available(self, provider: RuleBasedProvider) -> None:
        """Test provider is always available."""
        assert provider.is_available is True

    @pytest.mark.asyncio
    async def test_buy_signal_on_oversold(self, provider: RuleBasedProvider) -> None:
        """Test BUY signal when RSI is oversold."""
        context = MarketContext(
            symbol="BTCUSDT",
            platform=Platform.BINANCE,
            candles=[],
            current_price=Decimal("40000"),
            rsi=Decimal("25"),  # Oversold
        )
        
        result = await provider.analyze(context)
        
        assert result.action == SignalAction.BUY
        assert result.confidence > 0.6
        assert "oversold" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_sell_signal_on_overbought(self, provider: RuleBasedProvider) -> None:
        """Test SELL signal when RSI is overbought."""
        context = MarketContext(
            symbol="BTCUSDT",
            platform=Platform.BINANCE,
            candles=[],
            current_price=Decimal("40000"),
            rsi=Decimal("75"),  # Overbought
        )
        
        result = await provider.analyze(context)
        
        assert result.action == SignalAction.SELL
        assert result.confidence > 0.6
        assert "overbought" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_hold_signal_on_neutral(self, provider: RuleBasedProvider) -> None:
        """Test HOLD signal when RSI is neutral."""
        context = MarketContext(
            symbol="BTCUSDT",
            platform=Platform.BINANCE,
            candles=[],
            current_price=Decimal("40000"),
            rsi=Decimal("50"),  # Neutral
        )
        
        result = await provider.analyze(context)
        
        assert result.action == SignalAction.HOLD
        assert "neutral" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_hold_when_already_in_position(self, provider: RuleBasedProvider) -> None:
        """Test HOLD when already in position."""
        context = MarketContext(
            symbol="BTCUSDT",
            platform=Platform.BINANCE,
            candles=[],
            current_price=Decimal("40000"),
            rsi=Decimal("25"),  # Would be BUY
            has_position=True,  # But already in position
        )
        
        result = await provider.analyze(context)
        
        assert result.action == SignalAction.HOLD
        assert "position" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_confidence_increases_with_extreme_rsi(
        self,
        provider: RuleBasedProvider,
    ) -> None:
        """Test confidence is higher for more extreme RSI."""
        # Moderately oversold
        context1 = MarketContext(
            symbol="BTCUSDT",
            platform=Platform.BINANCE,
            candles=[],
            current_price=Decimal("40000"),
            rsi=Decimal("28"),
        )
        
        # Extremely oversold
        context2 = MarketContext(
            symbol="BTCUSDT",
            platform=Platform.BINANCE,
            candles=[],
            current_price=Decimal("40000"),
            rsi=Decimal("10"),
        )
        
        result1 = await provider.analyze(context1)
        result2 = await provider.analyze(context2)
        
        assert result2.confidence > result1.confidence

    @pytest.mark.asyncio
    async def test_sma_confirmation_increases_confidence(
        self,
        provider: RuleBasedProvider,
    ) -> None:
        """Test SMA confirmation increases confidence."""
        # BUY signal with price above SMA (confirmation)
        context = MarketContext(
            symbol="BTCUSDT",
            platform=Platform.BINANCE,
            candles=[],
            current_price=Decimal("41000"),  # Above SMA
            rsi=Decimal("25"),  # Oversold
            sma_20=Decimal("40000"),
        )
        
        result = await provider.analyze(context)
        
        assert result.action == SignalAction.BUY
        assert result.confidence > 0.7
        assert "bullish" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_custom_thresholds(self) -> None:
        """Test custom RSI thresholds."""
        provider = RuleBasedProvider(
            rsi_oversold=25.0,
            rsi_overbought=75.0,
        )
        
        # RSI 28 would be oversold with default (30) but not with 25
        context = MarketContext(
            symbol="BTCUSDT",
            platform=Platform.BINANCE,
            candles=[],
            current_price=Decimal("40000"),
            rsi=Decimal("28"),
        )
        
        result = await provider.analyze(context)
        
        # Should be HOLD since 28 > 25
        assert result.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_no_rsi_returns_hold(self, provider: RuleBasedProvider) -> None:
        """Test HOLD when no RSI available."""
        context = MarketContext(
            symbol="BTCUSDT",
            platform=Platform.BINANCE,
            candles=[],
            current_price=Decimal("40000"),
            rsi=None,  # No RSI
        )
        
        result = await provider.analyze(context)
        
        assert result.action == SignalAction.HOLD

    @pytest.mark.asyncio
    async def test_result_metadata(self, provider: RuleBasedProvider) -> None:
        """Test result includes correct metadata."""
        context = MarketContext(
            symbol="BTCUSDT",
            platform=Platform.BINANCE,
            candles=[],
            current_price=Decimal("40000"),
            rsi=Decimal("50"),
        )
        
        result = await provider.analyze(context)
        
        assert result.provider == AIProviderType.RULE_BASED
        assert result.model == "rule_based_v1"
        assert result.latency_ms >= 0
