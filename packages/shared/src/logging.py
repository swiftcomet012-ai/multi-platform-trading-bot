"""
Structured logging using structlog.

Provides JSON output for production, colored console for development.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

import structlog

# Context variables for request tracking
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")
trade_id_var: ContextVar[str] = ContextVar("trade_id", default="")


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid4())[:8]


def add_context(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add context variables to log events."""
    correlation_id = correlation_id_var.get()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id

    trade_id = trade_id_var.get()
    if trade_id:
        event_dict["trade_id"] = trade_id

    return event_dict


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: str | None = None,
) -> None:
    """
    Configure structured logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Use JSON format (for production)
        log_file: Optional file path for logging
    """
    # Shared processors
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        add_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        # Production: JSON output
        processors = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Colored console output
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging to use structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    # Optionally add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


# Convenience loggers for common domains
class TradingLogger:
    """Logger for trading operations."""

    def __init__(self) -> None:
        self._logger = get_logger("trading")

    def order_placed(
        self,
        symbol: str,
        side: str,
        quantity: str,
        price: str | None = None,
        order_id: str | None = None,
    ) -> None:
        """Log order placement."""
        self._logger.info(
            "order_placed",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_id=order_id,
        )

    def order_filled(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: str,
        price: str,
        fees: str,
    ) -> None:
        """Log order fill."""
        self._logger.info(
            "order_filled",
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            fees=fees,
        )

    def signal_generated(
        self,
        symbol: str,
        action: str,
        confidence: float,
        strategy: str,
        reasoning: str | None = None,
    ) -> None:
        """Log AI signal generation."""
        self._logger.info(
            "signal_generated",
            symbol=symbol,
            action=action,
            confidence=confidence,
            strategy=strategy,
            reasoning=reasoning[:100] if reasoning else None,
        )

    def risk_check_failed(
        self,
        reason: str,
        symbol: str | None = None,
        details: dict | None = None,
    ) -> None:
        """Log risk check failure."""
        self._logger.warning(
            "risk_check_failed",
            reason=reason,
            symbol=symbol,
            details=details,
        )

    def daily_summary(
        self,
        total_pnl: str,
        win_rate: float,
        total_trades: int,
        platform: str,
    ) -> None:
        """Log daily trading summary."""
        self._logger.info(
            "daily_summary",
            total_pnl=total_pnl,
            win_rate=win_rate,
            total_trades=total_trades,
            platform=platform,
        )


class AILogger:
    """Logger for AI operations."""

    def __init__(self) -> None:
        self._logger = get_logger("ai")

    def provider_called(
        self,
        provider: str,
        model: str,
        prompt_tokens: int | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Log AI provider call."""
        self._logger.info(
            "provider_called",
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            latency_ms=latency_ms,
        )

    def provider_failed(
        self,
        provider: str,
        error: str,
        will_retry: bool = False,
    ) -> None:
        """Log AI provider failure."""
        self._logger.warning(
            "provider_failed",
            provider=provider,
            error=error,
            will_retry=will_retry,
        )

    def failover_triggered(
        self,
        from_provider: str,
        to_provider: str,
        reason: str,
    ) -> None:
        """Log AI failover."""
        self._logger.warning(
            "failover_triggered",
            from_provider=from_provider,
            to_provider=to_provider,
            reason=reason,
        )


# Global logger instances
trading_logger = TradingLogger()
ai_logger = AILogger()
