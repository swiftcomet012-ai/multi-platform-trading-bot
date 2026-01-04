"""
Base connector interface using Ports and Adapters pattern.

Defines the contract that all exchange connectors must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime  # noqa: TC003 - Required at runtime for type hints
from decimal import Decimal  # noqa: TC003 - Required at runtime for type hints
from typing import Protocol, runtime_checkable

from packages.shared.src.models import (  # noqa: TC001 - Required at runtime
    OHLCV,
    ExchangeInfo,
    Order,
    Platform,
    Position,
)


@runtime_checkable
class ExchangeConnector(Protocol):
    """Protocol defining the exchange connector interface."""

    @property
    def platform(self) -> Platform:
        """Get the platform identifier."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if connected to exchange."""
        ...

    async def connect(self) -> bool:
        """Connect to the exchange."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from the exchange."""
        ...

    async def get_balance(self) -> dict[str, Decimal]:
        """Get account balances."""
        ...

    async def get_positions(self) -> list[Position]:
        """Get open positions."""
        ...

    async def place_order(self, order: Order) -> Order:
        """Place an order."""
        ...

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        ...

    async def get_exchange_info(self, symbol: str) -> ExchangeInfo:
        """Get exchange trading rules for a symbol."""
        ...


class BaseConnector(ABC):
    """Abstract base class for exchange connectors."""

    def __init__(self) -> None:
        self._connected: bool = False

    @property
    @abstractmethod
    def platform(self) -> Platform:
        """Get the platform identifier."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if connected to exchange."""
        return self._connected

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the exchange.

        Returns:
            True if connection successful, False otherwise.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the exchange."""
        ...

    @abstractmethod
    async def get_balance(self) -> dict[str, Decimal]:
        """
        Get account balances.

        Returns:
            Dictionary mapping asset symbol to available balance.
        """
        ...

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """
        Get all open positions.

        Returns:
            List of open positions.
        """
        ...

    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """
        Place an order on the exchange.

        Args:
            order: Order to place.

        Returns:
            Updated order with exchange-assigned ID and status.

        Raises:
            InvalidOrderError: If order validation fails.
            InsufficientBalanceError: If balance is insufficient.
            OrderRejectedError: If exchange rejects the order.
        """
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: Exchange order ID.
            symbol: Trading symbol.

        Returns:
            True if cancellation successful.
        """
        ...

    @abstractmethod
    async def get_exchange_info(self, symbol: str) -> ExchangeInfo:
        """
        Get exchange trading rules for a symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            Exchange info with trading rules.
        """
        ...

    @abstractmethod
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[OHLCV]:
        """
        Get OHLCV candlestick data.

        Args:
            symbol: Trading symbol.
            timeframe: Candlestick timeframe (1m, 5m, 15m, 1h, 4h, 1d).
            limit: Number of candles to fetch.
            start_time: Start time filter.
            end_time: End time filter.

        Returns:
            List of OHLCV candles.
        """
        ...

    @abstractmethod
    async def get_ticker_price(self, symbol: str) -> Decimal:
        """
        Get current ticker price.

        Args:
            symbol: Trading symbol.

        Returns:
            Current price.
        """
        ...
