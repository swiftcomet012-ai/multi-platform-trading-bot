"""
AI failover chain with caching.

Tries multiple AI providers in order, falling back to rule-based analysis.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from packages.ai_analyzer.src.base import (
    AIAnalysisResult,
    AIProvider,
    AIProviderError,
    AIProviderType,
    MarketContext,
)
from packages.ai_analyzer.src.gemini_provider import GeminiProvider
from packages.ai_analyzer.src.groq_provider import GroqProvider
from packages.ai_analyzer.src.openai_provider import OpenAIProvider
from packages.ai_analyzer.src.rule_based_provider import RuleBasedProvider
from packages.shared.src.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry for AI analysis results."""
    
    result: AIAnalysisResult
    expires_at: datetime
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() > self.expires_at


@dataclass
class FailoverChainConfig:
    """Configuration for failover chain."""
    
    # Provider order (first available is tried first)
    provider_order: list[AIProviderType] = field(default_factory=lambda: [
        AIProviderType.GEMINI,
        AIProviderType.OPENAI,
        AIProviderType.GROQ,
    ])
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    
    # Retry settings
    max_retries_per_provider: int = 2
    timeout_seconds: float = 30.0
    
    # Fallback
    use_rule_based_fallback: bool = True


class AIFailoverChain:
    """
    AI provider failover chain.
    
    Tries providers in order:
    1. Gemini (fast, cheap)
    2. OpenAI (reliable)
    3. Groq (fast inference)
    4. Rule-based (always available)
    
    Features:
    - Automatic failover on errors
    - Response caching (TTL: 5 min)
    - Provider health tracking
    """
    
    def __init__(
        self,
        config: FailoverChainConfig | None = None,
    ) -> None:
        """Initialize failover chain."""
        self.config = config or FailoverChainConfig()
        
        # Initialize providers
        self._providers: dict[AIProviderType, AIProvider] = {
            AIProviderType.GEMINI: GeminiProvider(),
            AIProviderType.OPENAI: OpenAIProvider(),
            AIProviderType.GROQ: GroqProvider(),
            AIProviderType.RULE_BASED: RuleBasedProvider(),
        }
        
        # Cache
        self._cache: dict[str, CacheEntry] = {}
        
        # Provider health tracking
        self._provider_failures: dict[AIProviderType, int] = {}
        self._provider_last_success: dict[AIProviderType, datetime] = {}
    
    def _get_cache_key(self, context: MarketContext) -> str:
        """Generate cache key from context."""
        # Use symbol, timeframe, and recent price for cache key
        key_data = f"{context.symbol}:{context.timeframe}:{context.current_price}"
        if context.rsi:
            key_data += f":{context.rsi}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached(self, key: str) -> AIAnalysisResult | None:
        """Get cached result if available and not expired."""
        if not self.config.cache_enabled:
            return None
        
        entry = self._cache.get(key)
        if entry and not entry.is_expired():
            logger.debug("ai_cache_hit", key=key[:8])
            return entry.result
        
        # Clean up expired entry
        if entry:
            del self._cache[key]
        
        return None
    
    def _set_cached(self, key: str, result: AIAnalysisResult) -> None:
        """Cache analysis result."""
        if not self.config.cache_enabled:
            return
        
        result.cached = True
        self._cache[key] = CacheEntry(
            result=result,
            expires_at=datetime.now() + timedelta(seconds=self.config.cache_ttl_seconds),
        )
        logger.debug("ai_cache_set", key=key[:8])
    
    def _get_available_providers(self) -> list[AIProviderType]:
        """Get list of available providers in order."""
        available = []
        
        for provider_type in self.config.provider_order:
            provider = self._providers.get(provider_type)
            if provider and provider.is_available:
                available.append(provider_type)
        
        return available
    
    async def analyze(
        self,
        context: MarketContext,
    ) -> AIAnalysisResult:
        """
        Analyze market context using failover chain.
        
        Tries providers in order until one succeeds.
        Falls back to rule-based analysis if all fail.
        
        Args:
            context: Market context with price data and indicators.
        
        Returns:
            AIAnalysisResult from first successful provider.
        """
        # Check cache first
        cache_key = self._get_cache_key(context)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        # Get available providers
        providers = self._get_available_providers()
        
        if not providers:
            logger.warning("no_ai_providers_available")
            providers = []
        
        # Try each provider
        last_error: Exception | None = None
        
        for provider_type in providers:
            provider = self._providers[provider_type]
            
            for attempt in range(self.config.max_retries_per_provider):
                try:
                    logger.debug(
                        "ai_provider_attempt",
                        provider=provider_type.value,
                        attempt=attempt + 1,
                    )
                    
                    result = await provider.analyze(
                        context,
                        timeout=self.config.timeout_seconds,
                    )
                    
                    # Success - cache and return
                    self._provider_last_success[provider_type] = datetime.now()
                    self._provider_failures[provider_type] = 0
                    
                    self._set_cached(cache_key, result)
                    
                    logger.info(
                        "ai_analysis_success",
                        provider=provider_type.value,
                        action=result.action.value,
                        confidence=result.confidence,
                    )
                    
                    return result
                    
                except AIProviderError as e:
                    last_error = e
                    self._provider_failures[provider_type] = (
                        self._provider_failures.get(provider_type, 0) + 1
                    )
                    
                    logger.warning(
                        "ai_provider_error",
                        provider=provider_type.value,
                        error=str(e),
                        retryable=e.retryable,
                        attempt=attempt + 1,
                    )
                    
                    # Don't retry non-retryable errors
                    if not e.retryable:
                        break
                    
                except Exception as e:
                    last_error = e
                    logger.error(
                        "ai_provider_unexpected_error",
                        provider=provider_type.value,
                        error=str(e),
                    )
                    break
        
        # All providers failed - use rule-based fallback
        if self.config.use_rule_based_fallback:
            logger.warning(
                "ai_failover_to_rule_based",
                last_error=str(last_error) if last_error else None,
            )
            
            rule_based = self._providers[AIProviderType.RULE_BASED]
            result = await rule_based.analyze(context)
            
            # Don't cache rule-based results (we want to retry AI next time)
            return result
        
        # No fallback - raise error
        raise AIProviderError(
            f"All AI providers failed. Last error: {last_error}",
            AIProviderType.RULE_BASED,
            retryable=True,
        )
    
    def get_provider_stats(self) -> dict[str, Any]:
        """Get provider health statistics."""
        stats = {}
        
        for provider_type in self.config.provider_order:
            provider = self._providers.get(provider_type)
            stats[provider_type.value] = {
                "available": provider.is_available if provider else False,
                "failures": self._provider_failures.get(provider_type, 0),
                "last_success": (
                    self._provider_last_success[provider_type].isoformat()
                    if provider_type in self._provider_last_success
                    else None
                ),
            }
        
        stats["cache_size"] = len(self._cache)
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()
        logger.info("ai_cache_cleared")
    
    def reset_provider_stats(self) -> None:
        """Reset provider failure counts."""
        self._provider_failures.clear()
        logger.info("ai_provider_stats_reset")
