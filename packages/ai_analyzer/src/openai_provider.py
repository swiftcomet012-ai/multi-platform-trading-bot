"""
OpenAI AI provider implementation.

Supports GPT-4, GPT-4o, GPT-3.5-turbo.
"""

from __future__ import annotations

import json
import os
import time
from decimal import Decimal

from packages.ai_analyzer.src.base import (
    AIAnalysisResult,
    AIAuthenticationError,
    AIProviderError,
    AIProviderType,
    AIRateLimitError,
    AITimeoutError,
    MarketContext,
    TRADING_ANALYSIS_PROMPT,
)
from packages.shared.src.logging import get_logger
from packages.shared.src.models import SignalAction

logger = get_logger(__name__)

# Lazy import
_openai = None


def _get_openai():
    """Lazy load openai."""
    global _openai
    if _openai is None:
        try:
            import openai
            _openai = openai
        except ImportError:
            raise ImportError(
                "openai not installed. Run: pip install openai"
            )
    return _openai


class OpenAIProvider:
    """OpenAI GPT provider."""
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (or from OPENAI_API_KEY env var).
            model: Model to use (default: gpt-4o-mini).
        """
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._model_name = model
        self._client = None
    
    @property
    def provider_type(self) -> AIProviderType:
        """Get provider type."""
        return AIProviderType.OPENAI
    
    @property
    def is_available(self) -> bool:
        """Check if provider is available."""
        return bool(self._api_key)
    
    def _ensure_client(self):
        """Initialize client if needed."""
        if self._client is None:
            if not self._api_key:
                raise AIAuthenticationError(AIProviderType.OPENAI)
            
            openai = _get_openai()
            self._client = openai.AsyncOpenAI(api_key=self._api_key)
    
    async def analyze(
        self,
        context: MarketContext,
        timeout: float = 30.0,
    ) -> AIAnalysisResult:
        """Analyze market context using OpenAI."""
        start_time = time.time()
        
        try:
            self._ensure_client()
            
            # Build messages
            messages = [
                {"role": "system", "content": TRADING_ANALYSIS_PROMPT},
                {"role": "user", "content": f"Market Data:\n{context.to_prompt_context()}"},
            ]
            
            # Call OpenAI
            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
                timeout=timeout,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Parse response
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            result = self._parse_response(content, latency_ms, tokens_used)
            
            logger.info(
                "openai_analysis_complete",
                action=result.action.value,
                confidence=result.confidence,
                latency_ms=latency_ms,
                tokens=tokens_used,
            )
            
            return result
            
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "quota" in error_str:
                raise AIRateLimitError(AIProviderType.OPENAI)
            if "timeout" in error_str:
                raise AITimeoutError(AIProviderType.OPENAI, timeout)
            if "api key" in error_str or "authentication" in error_str:
                raise AIAuthenticationError(AIProviderType.OPENAI)
            raise AIProviderError(
                f"OpenAI analysis failed: {e}",
                AIProviderType.OPENAI,
            )
    
    def _parse_response(
        self,
        text: str,
        latency_ms: float,
        tokens_used: int,
    ) -> AIAnalysisResult:
        """Parse OpenAI response into AIAnalysisResult."""
        try:
            text = text.strip()
            
            # Handle markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            data = json.loads(text)
            
            # Parse action
            action_str = data.get("action", "hold").lower()
            if action_str == "buy":
                action = SignalAction.BUY
            elif action_str == "sell":
                action = SignalAction.SELL
            else:
                action = SignalAction.HOLD
            
            # Parse optional price levels
            entry_price = None
            stop_loss = None
            take_profit = None
            
            if data.get("entry_price"):
                entry_price = Decimal(str(data["entry_price"]))
            if data.get("stop_loss"):
                stop_loss = Decimal(str(data["stop_loss"]))
            if data.get("take_profit"):
                take_profit = Decimal(str(data["take_profit"]))
            
            return AIAnalysisResult(
                action=action,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "No reasoning provided"),
                provider=AIProviderType.OPENAI,
                model=self._model_name,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("openai_parse_error", error=str(e), response=text[:200])
            return AIAnalysisResult(
                action=SignalAction.HOLD,
                confidence=0.3,
                reasoning=f"Failed to parse response: {text[:100]}",
                provider=AIProviderType.OPENAI,
                model=self._model_name,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
            )
