"""
Groq AI provider implementation.

Supports Llama, Mixtral models with fast inference.
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
_groq = None


def _get_groq():
    """Lazy load groq."""
    global _groq
    if _groq is None:
        try:
            import groq
            _groq = groq
        except ImportError:
            raise ImportError(
                "groq not installed. Run: pip install groq"
            )
    return _groq


class GroqProvider:
    """Groq AI provider for fast inference."""
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "llama-3.1-70b-versatile",
    ) -> None:
        """
        Initialize Groq provider.
        
        Args:
            api_key: Groq API key (or from GROQ_API_KEY env var).
            model: Model to use (default: llama-3.1-70b-versatile).
        """
        self._api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self._model_name = model
        self._client = None
    
    @property
    def provider_type(self) -> AIProviderType:
        """Get provider type."""
        return AIProviderType.GROQ
    
    @property
    def is_available(self) -> bool:
        """Check if provider is available."""
        return bool(self._api_key)
    
    def _ensure_client(self):
        """Initialize client if needed."""
        if self._client is None:
            if not self._api_key:
                raise AIAuthenticationError(AIProviderType.GROQ)
            
            groq = _get_groq()
            self._client = groq.AsyncGroq(api_key=self._api_key)
    
    async def analyze(
        self,
        context: MarketContext,
        timeout: float = 30.0,
    ) -> AIAnalysisResult:
        """Analyze market context using Groq."""
        start_time = time.time()
        
        try:
            self._ensure_client()
            
            # Build messages
            messages = [
                {"role": "system", "content": TRADING_ANALYSIS_PROMPT},
                {"role": "user", "content": f"Market Data:\n{context.to_prompt_context()}"},
            ]
            
            # Call Groq
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
                "groq_analysis_complete",
                action=result.action.value,
                confidence=result.confidence,
                latency_ms=latency_ms,
                tokens=tokens_used,
            )
            
            return result
            
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "quota" in error_str:
                raise AIRateLimitError(AIProviderType.GROQ)
            if "timeout" in error_str:
                raise AITimeoutError(AIProviderType.GROQ, timeout)
            if "api key" in error_str or "authentication" in error_str:
                raise AIAuthenticationError(AIProviderType.GROQ)
            raise AIProviderError(
                f"Groq analysis failed: {e}",
                AIProviderType.GROQ,
            )
    
    def _parse_response(
        self,
        text: str,
        latency_ms: float,
        tokens_used: int,
    ) -> AIAnalysisResult:
        """Parse Groq response into AIAnalysisResult."""
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
                provider=AIProviderType.GROQ,
                model=self._model_name,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("groq_parse_error", error=str(e), response=text[:200])
            return AIAnalysisResult(
                action=SignalAction.HOLD,
                confidence=0.3,
                reasoning=f"Failed to parse response: {text[:100]}",
                provider=AIProviderType.GROQ,
                model=self._model_name,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
            )
