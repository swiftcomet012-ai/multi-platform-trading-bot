"""AI Analyzer package for trading signal generation."""

from packages.ai_analyzer.src.base import (
    AIAnalysisResult,
    AIProvider,
    AIProviderError,
    AIProviderType,
    AIRateLimitError,
    AITimeoutError,
    AIAuthenticationError,
    MarketContext,
)
from packages.ai_analyzer.src.failover_chain import (
    AIFailoverChain,
    FailoverChainConfig,
)
from packages.ai_analyzer.src.gemini_provider import GeminiProvider
from packages.ai_analyzer.src.openai_provider import OpenAIProvider
from packages.ai_analyzer.src.groq_provider import GroqProvider
from packages.ai_analyzer.src.rule_based_provider import RuleBasedProvider

__all__ = [
    # Base types
    "AIAnalysisResult",
    "AIProvider",
    "AIProviderError",
    "AIProviderType",
    "AIRateLimitError",
    "AITimeoutError",
    "AIAuthenticationError",
    "MarketContext",
    # Failover chain
    "AIFailoverChain",
    "FailoverChainConfig",
    # Providers
    "GeminiProvider",
    "OpenAIProvider",
    "GroqProvider",
    "RuleBasedProvider",
]
