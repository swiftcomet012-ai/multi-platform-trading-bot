"""
Google Gemini AI provider implementation.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

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

# Lazy import to avoid dependency issues
_genai = None


def _get_genai():
    """Lazy load google.generativeai."""
    global _genai
    if _genai is None:
        try:
            import google.generativeai as genai
            _genai = genai
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. "
                "Run: pip install google-generativeai"
            )
    return _genai


class GeminiProvider:
    """Google Gemini AI provider."""
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-1.5-flash",
    ) -> None:
        """
        Initialize Gemini provider.
        
        Args:
            api_key: Gemini API key (or from GEMINI_API_KEY env var).
            model: Model to use (default: gemini-1.5-flash).
        """
        self._api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self._model_name = model
        self._model = None
        self._initialized = False
    
    @property
    def provider_type(self) -> AIProviderType:
        """Get provider type."""
        return AIProviderType.GEMINI
    
    @property
    def is_available(self) -> bool:
        """Check if provider is available."""
        return bool(self._api_key)
    
    def _ensure_initialized(self) -> None:
        """Initialize the model if not already done."""
        if self._initialized:
            return
        
        if not self._api_key:
            raise AIAuthenticationError(AIProviderType.GEMINI)
        
        genai = _get_genai()
        genai.configure(api_key=self._api_key)
        
        self._model = genai.GenerativeModel(
            model_name=self._model_name,
            generation_config={
                "temperature": 0.3,
                "top_p": 0.95,
                "max_output_tokens": 1024,
            },
        )
        self._initialized = True
    
    async def analyze(
        self,
        context: MarketContext,
        timeout: float = 30.0,
    ) -> AIAnalysisResult:
        """Analyze market context using Gemini."""
        start_time = time.time()
        
        try:
            self._ensure_initialized()
            
            # Build prompt
            prompt = f"{TRADING_ANALYSIS_PROMPT}\n\nMarket Data:\n{context.to_prompt_context()}"
            
            # Call Gemini (run in executor since it's sync)
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._model.generate_content(prompt),
                ),
                timeout=timeout,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Parse response
            result = self._parse_response(response.text, latency_ms)
            
            logger.info(
                "gemini_analysis_complete",
                action=result.action.value,
                confidence=result.confidence,
                latency_ms=latency_ms,
            )
            
            return result
            
        except asyncio.TimeoutError:
            raise AITimeoutError(AIProviderType.GEMINI, timeout)
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "quota" in error_str:
                raise AIRateLimitError(AIProviderType.GEMINI)
            if "api key" in error_str or "authentication" in error_str:
                raise AIAuthenticationError(AIProviderType.GEMINI)
            raise AIProviderError(
                f"Gemini analysis failed: {e}",
                AIProviderType.GEMINI,
            )
    
    def _parse_response(self, text: str, latency_ms: float) -> AIAnalysisResult:
        """Parse Gemini response into AIAnalysisResult."""
        try:
            # Try to extract JSON from response
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
            
            return AIAnalysisResult(
                action=action,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "No reasoning provided"),
                provider=AIProviderType.GEMINI,
                model=self._model_name,
                latency_ms=latency_ms,
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("gemini_parse_error", error=str(e), response=text[:200])
            # Return HOLD on parse error
            return AIAnalysisResult(
                action=SignalAction.HOLD,
                confidence=0.3,
                reasoning=f"Failed to parse response: {text[:100]}",
                provider=AIProviderType.GEMINI,
                model=self._model_name,
                latency_ms=latency_ms,
            )
