"""
Custom exceptions for the trading platform.

Organized by domain with error codes for tracking.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Error codes for categorization and tracking."""

    # Connection errors (E001-E099)
    CONNECTION_FAILED = "E001"
    AUTHENTICATION_FAILED = "E002"
    TIMEOUT = "E003"
    RATE_LIMITED = "E004"
    WEBSOCKET_DISCONNECTED = "E005"

    # Trading errors (E100-E199)
    INSUFFICIENT_BALANCE = "E101"
    INVALID_ORDER = "E102"
    ORDER_REJECTED = "E103"
    POSITION_NOT_FOUND = "E104"
    SYMBOL_NOT_FOUND = "E105"
    INVALID_QUANTITY = "E106"
    INVALID_PRICE = "E107"
    ORDER_NOT_FOUND = "E108"

    # Risk errors (E200-E299)
    RISK_LIMIT_EXCEEDED = "E201"
    DAILY_LOSS_EXCEEDED = "E202"
    MAX_POSITIONS_REACHED = "E203"
    POSITION_SIZE_EXCEEDED = "E204"
    LEVERAGE_EXCEEDED = "E205"

    # Data errors (E300-E399)
    DATA_NOT_FOUND = "E301"
    INVALID_DATA = "E302"
    STALE_DATA = "E303"
    DATA_INTEGRITY_ERROR = "E304"

    # AI errors (E400-E499)
    AI_PROVIDER_ERROR = "E401"
    AI_TIMEOUT = "E402"
    AI_RATE_LIMITED = "E403"
    AI_INVALID_RESPONSE = "E404"
    ALL_PROVIDERS_FAILED = "E405"

    # Configuration errors (E500-E599)
    CONFIGURATION_ERROR = "E501"
    INVALID_CONFIGURATION = "E502"
    MISSING_CREDENTIALS = "E503"

    # System errors (E900-E999)
    INTERNAL_ERROR = "E901"
    NOT_IMPLEMENTED = "E902"
    PAPER_MODE_VIOLATION = "E903"


class TradingPlatformError(Exception):
    """Base exception for all trading platform errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/API responses."""
        return {
            "error": self.code.value,
            "message": self.message,
            "details": self.details,
        }


# Connection Errors
class ConnectionError(TradingPlatformError):
    """Exchange connection failed."""

    def __init__(self, message: str, platform: str, details: dict | None = None) -> None:
        super().__init__(
            message,
            ErrorCode.CONNECTION_FAILED,
            {"platform": platform, **(details or {})},
        )


class AuthenticationError(TradingPlatformError):
    """Authentication with exchange failed."""

    def __init__(self, message: str, platform: str) -> None:
        super().__init__(
            message,
            ErrorCode.AUTHENTICATION_FAILED,
            {"platform": platform},
        )


class TimeoutError(TradingPlatformError):
    """Request timed out."""

    def __init__(self, message: str, operation: str, timeout_seconds: float) -> None:
        super().__init__(
            message,
            ErrorCode.TIMEOUT,
            {"operation": operation, "timeout_seconds": timeout_seconds},
        )


class RateLimitError(TradingPlatformError):
    """Rate limit exceeded."""

    def __init__(self, message: str, platform: str, retry_after_seconds: int | None = None) -> None:
        super().__init__(
            message,
            ErrorCode.RATE_LIMITED,
            {"platform": platform, "retry_after_seconds": retry_after_seconds},
        )


# Trading Errors
class InsufficientBalanceError(TradingPlatformError):
    """Insufficient balance for trade."""

    def __init__(
        self, message: str, required: str, available: str, asset: str
    ) -> None:
        super().__init__(
            message,
            ErrorCode.INSUFFICIENT_BALANCE,
            {"required": required, "available": available, "asset": asset},
        )


class InvalidOrderError(TradingPlatformError):
    """Order validation failed."""

    def __init__(self, message: str, order_details: dict | None = None) -> None:
        super().__init__(
            message,
            ErrorCode.INVALID_ORDER,
            {"order": order_details},
        )


class OrderRejectedError(TradingPlatformError):
    """Order rejected by exchange."""

    def __init__(self, message: str, order_id: str, reason: str) -> None:
        super().__init__(
            message,
            ErrorCode.ORDER_REJECTED,
            {"order_id": order_id, "reason": reason},
        )


class PositionNotFoundError(TradingPlatformError):
    """Position not found."""

    def __init__(self, symbol: str, platform: str) -> None:
        super().__init__(
            f"Position not found for {symbol} on {platform}",
            ErrorCode.POSITION_NOT_FOUND,
            {"symbol": symbol, "platform": platform},
        )


class SymbolNotFoundError(TradingPlatformError):
    """Trading symbol not found."""

    def __init__(self, symbol: str, platform: str) -> None:
        super().__init__(
            f"Symbol {symbol} not found on {platform}",
            ErrorCode.SYMBOL_NOT_FOUND,
            {"symbol": symbol, "platform": platform},
        )


# Risk Errors
class RiskLimitExceededError(TradingPlatformError):
    """Risk limit exceeded."""

    def __init__(self, message: str, limit_type: str, current: str, limit: str) -> None:
        super().__init__(
            message,
            ErrorCode.RISK_LIMIT_EXCEEDED,
            {"limit_type": limit_type, "current": current, "limit": limit},
        )


class DailyLossExceededError(TradingPlatformError):
    """Daily loss limit exceeded."""

    def __init__(self, current_loss: str, limit: str) -> None:
        super().__init__(
            f"Daily loss limit exceeded: {current_loss} > {limit}",
            ErrorCode.DAILY_LOSS_EXCEEDED,
            {"current_loss": current_loss, "limit": limit},
        )


class MaxPositionsReachedError(TradingPlatformError):
    """Maximum open positions reached."""

    def __init__(self, current: int, limit: int) -> None:
        super().__init__(
            f"Maximum positions reached: {current}/{limit}",
            ErrorCode.MAX_POSITIONS_REACHED,
            {"current": current, "limit": limit},
        )


# AI Errors
class AIProviderError(TradingPlatformError):
    """AI provider error."""

    def __init__(self, message: str, provider: str, original_error: str | None = None) -> None:
        super().__init__(
            message,
            ErrorCode.AI_PROVIDER_ERROR,
            {"provider": provider, "original_error": original_error},
        )


class AITimeoutError(TradingPlatformError):
    """AI request timed out."""

    def __init__(self, provider: str, timeout_seconds: float) -> None:
        super().__init__(
            f"AI request to {provider} timed out after {timeout_seconds}s",
            ErrorCode.AI_TIMEOUT,
            {"provider": provider, "timeout_seconds": timeout_seconds},
        )


class AllProvidersFailedError(TradingPlatformError):
    """All AI providers failed."""

    def __init__(self, providers: list[str], errors: list[str]) -> None:
        super().__init__(
            f"All AI providers failed: {', '.join(providers)}",
            ErrorCode.ALL_PROVIDERS_FAILED,
            {"providers": providers, "errors": errors},
        )


# Configuration Errors
class ConfigurationError(TradingPlatformError):
    """Configuration error."""

    def __init__(self, message: str, config_key: str | None = None) -> None:
        super().__init__(
            message,
            ErrorCode.CONFIGURATION_ERROR,
            {"config_key": config_key},
        )


class MissingCredentialsError(TradingPlatformError):
    """Required credentials missing."""

    def __init__(self, platform: str, missing_keys: list[str]) -> None:
        super().__init__(
            f"Missing credentials for {platform}: {', '.join(missing_keys)}",
            ErrorCode.MISSING_CREDENTIALS,
            {"platform": platform, "missing_keys": missing_keys},
        )


# Paper Mode
class PaperModeViolationError(TradingPlatformError):
    """Attempted real trade in paper mode."""

    def __init__(self, operation: str) -> None:
        super().__init__(
            f"Cannot perform '{operation}' in paper mode",
            ErrorCode.PAPER_MODE_VIOLATION,
            {"operation": operation},
        )
