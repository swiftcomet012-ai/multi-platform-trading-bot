"""
Simple technical indicators for minimal trading loop.

Uses pure Python/Decimal for accuracy. No pandas dependency.
"""

from __future__ import annotations

from collections import deque
from decimal import Decimal
from typing import Sequence

from packages.shared.src.models import OHLCV


def calculate_rsi(
    candles: Sequence[OHLCV],
    period: int = 14,
) -> Decimal | None:
    """
    Calculate RSI (Relative Strength Index).
    
    Args:
        candles: List of OHLCV candles (oldest first).
        period: RSI period (default 14).
    
    Returns:
        RSI value (0-100) or None if not enough data.
    """
    if len(candles) < period + 1:
        return None
    
    # Calculate price changes
    gains: list[Decimal] = []
    losses: list[Decimal] = []
    
    for i in range(1, len(candles)):
        change = candles[i].close - candles[i - 1].close
        if change > 0:
            gains.append(change)
            losses.append(Decimal("0"))
        else:
            gains.append(Decimal("0"))
            losses.append(abs(change))
    
    # Use only the last 'period' changes
    recent_gains = gains[-period:]
    recent_losses = losses[-period:]
    
    avg_gain = sum(recent_gains) / period
    avg_loss = sum(recent_losses) / period
    
    if avg_loss == 0:
        return Decimal("100")
    
    rs = avg_gain / avg_loss
    rsi = Decimal("100") - (Decimal("100") / (1 + rs))
    
    return rsi.quantize(Decimal("0.01"))


def calculate_sma(
    candles: Sequence[OHLCV],
    period: int = 20,
) -> Decimal | None:
    """
    Calculate Simple Moving Average.
    
    Args:
        candles: List of OHLCV candles.
        period: SMA period.
    
    Returns:
        SMA value or None if not enough data.
    """
    if len(candles) < period:
        return None
    
    closes = [c.close for c in candles[-period:]]
    return (sum(closes) / period).quantize(Decimal("0.00000001"))


def calculate_ema(
    candles: Sequence[OHLCV],
    period: int = 20,
) -> Decimal | None:
    """
    Calculate Exponential Moving Average.
    
    Args:
        candles: List of OHLCV candles.
        period: EMA period.
    
    Returns:
        EMA value or None if not enough data.
    """
    if len(candles) < period:
        return None
    
    multiplier = Decimal("2") / (period + 1)
    
    # Start with SMA for first EMA value
    ema = sum(c.close for c in candles[:period]) / period
    
    # Calculate EMA for remaining candles
    for candle in candles[period:]:
        ema = (candle.close - ema) * multiplier + ema
    
    return ema.quantize(Decimal("0.00000001"))


def calculate_atr(
    candles: Sequence[OHLCV],
    period: int = 14,
) -> Decimal | None:
    """
    Calculate Average True Range (for stop-loss calculation).
    
    Args:
        candles: List of OHLCV candles.
        period: ATR period.
    
    Returns:
        ATR value or None if not enough data.
    """
    if len(candles) < period + 1:
        return None
    
    true_ranges: list[Decimal] = []
    
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i - 1].close
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        true_ranges.append(tr)
    
    # Use only the last 'period' true ranges
    recent_tr = true_ranges[-period:]
    atr = sum(recent_tr) / period
    
    return atr.quantize(Decimal("0.00000001"))
