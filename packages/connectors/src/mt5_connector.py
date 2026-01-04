"""
Exness/MT5 exchange connector implementation.

Supports Forex trading via MetaTrader5 terminal.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from packages.connectors.src.base import BaseConnector
from packages.shared.src.exceptions import (
    AuthenticationError,
    ConnectionError,
    InsufficientBalanceError,
    InvalidOrderError,
    OrderRejectedError,
    PositionNotFoundError,
    SymbolNotFoundError,
    TimeoutError,
)
from packages.shared.src.logging import get_logger
from packages.shared.src.models import (
    OHLCV,
    ExchangeInfo,
    Order,
    OrderStatus,
    OrderType,
    Platform,
    Position,
    Side,
)

logger = get_logger(__name__)

# Timeframe mapping for MT5
TIMEFRAME_MAP = {
    "1m": 1,    # TIMEFRAME_M1
    "5m": 5,    # TIMEFRAME_M5
    "15m": 15,  # TIMEFRAME_M15
    "30m": 30,  # TIMEFRAME_M30
    "1h": 60,   # TIMEFRAME_H1
    "4h": 240,  # TIMEFRAME_H4
    "1d": 1440, # TIMEFRAME_D1
    "1w": 10080, # TIMEFRAME_W1
    "1M": 43200, # TIMEFRAME_MN1
}


class MT5Connector(BaseConnector):
    """MetaTrader5 connector for Exness and other MT5 brokers."""

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        path: str | None = None,
        timeout: int = 60000,
    ) -> None:
        """
        Initialize MT5 connector.

        Args:
            login: MT5 account login number.
            password: MT5 account password.
            server: MT5 server name (e.g., "Exness-MT5Real").
            path: Path to MT5 terminal (optional, auto-detect if None).
            timeout: Connection timeout in milliseconds.
        """
        super().__init__()
        self._login = login
        self._password = password
        self._server = server
        self._path = path
        self._timeout = timeout
        self._mt5: Any = None  # MetaTrader5 module
        self._exchange_info_cache: dict[str, ExchangeInfo] = {}
        self._cache_timestamp: datetime | None = None
        self._cache_ttl_seconds = 3600  # 1 hour

    @property
    def platform(self) -> Platform:
        """Get the platform identifier."""
        return Platform.EXNESS

    def _import_mt5(self) -> Any:
        """Import MetaTrader5 module lazily."""
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5
                self._mt5 = mt5
            except ImportError as e:
                raise ConnectionError(
                    "MetaTrader5 library not installed. Install with: pip install MetaTrader5",
                    platform="exness",
                ) from e
        return self._mt5

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
    )
    async def connect(self) -> bool:
        """Connect to MT5 terminal."""
        mt5 = self._import_mt5()
        
        # Run MT5 operations in thread pool (MT5 is synchronous)
        loop = asyncio.get_event_loop()
        
        try:
            # Initialize MT5
            init_kwargs: dict[str, Any] = {
                "login": self._login,
                "password": self._password,
                "server": self._server,
                "timeout": self._timeout,
            }
            if self._path:
                init_kwargs["path"] = self._path

            initialized = await loop.run_in_executor(
                None, lambda: mt5.initialize(**init_kwargs)
            )
            
            if not initialized:
                error = await loop.run_in_executor(None, mt5.last_error)
                error_code, error_msg = error if error else (0, "Unknown error")
                
                if error_code == 10004:  # Invalid account
                    raise AuthenticationError(
                        f"Invalid login credentials: {error_msg}",
                        platform="exness",
                    )
                raise ConnectionError(
                    f"Failed to initialize MT5: {error_msg}",
                    platform="exness",
                )

            # Verify connection by getting account info
            account_info = await loop.run_in_executor(None, mt5.account_info)
            if account_info is None:
                error = await loop.run_in_executor(None, mt5.last_error)
                raise ConnectionError(
                    f"Failed to get account info: {error}",
                    platform="exness",
                )

            self._connected = True
            logger.info(
                "mt5_connected",
                login=self._login,
                server=self._server,
                balance=account_info.balance,
                leverage=account_info.leverage,
            )
            return True

        except (AuthenticationError, ConnectionError):
            raise
        except Exception as e:
            raise ConnectionError(
                f"MT5 connection failed: {e!s}",
                platform="exness",
            ) from e

    async def disconnect(self) -> None:
        """Disconnect from MT5 terminal."""
        if self._mt5 and self._connected:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._mt5.shutdown)
        self._connected = False
        logger.info("mt5_disconnected")

    def _ensure_connected(self) -> None:
        """Ensure MT5 is connected."""
        if not self._connected or not self._mt5:
            raise ConnectionError("Not connected to MT5", platform="exness")

    async def get_balance(self) -> dict[str, Decimal]:
        """Get account balances."""
        self._ensure_connected()
        loop = asyncio.get_event_loop()
        
        account_info = await loop.run_in_executor(None, self._mt5.account_info)
        if account_info is None:
            error = await loop.run_in_executor(None, self._mt5.last_error)
            raise ConnectionError(f"Failed to get account info: {error}", platform="exness")

        # MT5 returns balance in account currency
        return {
            account_info.currency: Decimal(str(account_info.balance)),
            f"{account_info.currency}_equity": Decimal(str(account_info.equity)),
            f"{account_info.currency}_margin": Decimal(str(account_info.margin)),
            f"{account_info.currency}_free_margin": Decimal(str(account_info.margin_free)),
        }

    async def get_account_info(self) -> dict[str, Any]:
        """Get detailed account information."""
        self._ensure_connected()
        loop = asyncio.get_event_loop()
        
        account_info = await loop.run_in_executor(None, self._mt5.account_info)
        if account_info is None:
            error = await loop.run_in_executor(None, self._mt5.last_error)
            raise ConnectionError(f"Failed to get account info: {error}", platform="exness")

        return {
            "login": account_info.login,
            "server": account_info.server,
            "currency": account_info.currency,
            "balance": Decimal(str(account_info.balance)),
            "equity": Decimal(str(account_info.equity)),
            "margin": Decimal(str(account_info.margin)),
            "free_margin": Decimal(str(account_info.margin_free)),
            "margin_level": account_info.margin_level,
            "leverage": account_info.leverage,
            "profit": Decimal(str(account_info.profit)),
        }


    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        self._ensure_connected()
        loop = asyncio.get_event_loop()
        
        positions = await loop.run_in_executor(None, self._mt5.positions_get)
        if positions is None:
            # No positions or error
            error = await loop.run_in_executor(None, self._mt5.last_error)
            if error and error[0] != 0:
                raise ConnectionError(f"Failed to get positions: {error}", platform="exness")
            return []

        result = []
        for pos in positions:
            # MT5 position types: 0=BUY, 1=SELL
            side = Side.BUY if pos.type == 0 else Side.SELL
            entry_price = Decimal(str(pos.price_open))
            current_price = Decimal(str(pos.price_current))
            volume = Decimal(str(pos.volume))
            profit = Decimal(str(pos.profit))
            
            # Calculate unrealized PnL percentage
            if entry_price > 0 and volume > 0:
                if side == Side.BUY:
                    pnl_pct = float((current_price - entry_price) / entry_price * 100)
                else:
                    pnl_pct = float((entry_price - current_price) / entry_price * 100)
            else:
                pnl_pct = 0.0

            result.append(
                Position(
                    symbol=pos.symbol,
                    side=side,
                    quantity=volume,
                    entry_price=entry_price,
                    current_price=current_price,
                    unrealized_pnl=profit,
                    unrealized_pnl_pct=pnl_pct,
                    platform=Platform.EXNESS,
                    leverage=1,  # MT5 doesn't expose per-position leverage
                    margin=Decimal(str(pos.margin)) if hasattr(pos, 'margin') else None,
                    updated_at=datetime.now(UTC),
                )
            )
        return result

    async def place_order(self, order: Order) -> Order:
        """Place an order on MT5."""
        self._ensure_connected()
        loop = asyncio.get_event_loop()
        mt5 = self._mt5

        # Get symbol info for validation
        exchange_info = await self.get_exchange_info(order.symbol)
        is_valid, error_msg = exchange_info.validate_quantity(order.quantity)
        if not is_valid:
            raise InvalidOrderError(error_msg)

        # Get current price for market orders
        symbol_info = await loop.run_in_executor(
            None, lambda: mt5.symbol_info(order.symbol)
        )
        if symbol_info is None:
            raise SymbolNotFoundError(order.symbol, platform="exness")

        # Build order request
        request: dict[str, Any] = {
            "symbol": order.symbol,
            "volume": float(exchange_info.round_quantity(order.quantity)),
            "deviation": 20,  # Max price deviation in points
            "magic": 234000,  # Magic number for identification
            "comment": f"trading_bot_{order.idempotency_key or 'auto'}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Set order type and price
        if order.side == Side.BUY:
            request["type"] = mt5.ORDER_TYPE_BUY
            request["price"] = symbol_info.ask
        else:
            request["type"] = mt5.ORDER_TYPE_SELL
            request["price"] = symbol_info.bid

        if order.order_type == OrderType.LIMIT:
            if order.side == Side.BUY:
                request["type"] = mt5.ORDER_TYPE_BUY_LIMIT
            else:
                request["type"] = mt5.ORDER_TYPE_SELL_LIMIT
            request["price"] = float(order.price) if order.price else request["price"]

        # Add stop loss and take profit
        if order.stop_loss:
            request["sl"] = float(order.stop_loss)
        if order.take_profit:
            request["tp"] = float(order.take_profit)

        request["action"] = mt5.TRADE_ACTION_DEAL

        # Execute order
        result = await loop.run_in_executor(
            None, lambda: mt5.order_send(request)
        )

        if result is None:
            error = await loop.run_in_executor(None, mt5.last_error)
            raise OrderRejectedError(
                f"Order failed: {error}",
                order_id="",
                reason=str(error),
            )

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self._handle_order_error(result.retcode, result.comment)

        # Return updated order
        return Order(
            id=str(result.order),
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=Decimal(str(result.price)) if result.price else order.price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            platform=Platform.EXNESS,
            idempotency_key=order.idempotency_key,
            status=OrderStatus.FILLED if result.retcode == mt5.TRADE_RETCODE_DONE else OrderStatus.PENDING,
            created_at=datetime.now(UTC),
            filled_quantity=Decimal(str(result.volume)),
            filled_price=Decimal(str(result.price)) if result.price else None,
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel a pending order."""
        self._ensure_connected()
        loop = asyncio.get_event_loop()
        mt5 = self._mt5

        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": int(order_id),
        }

        result = await loop.run_in_executor(
            None, lambda: mt5.order_send(request)
        )

        if result is None:
            return False

        return result.retcode == mt5.TRADE_RETCODE_DONE

    async def close_position(self, ticket: int, volume: Decimal | None = None) -> bool:
        """
        Close an open position.

        Args:
            ticket: Position ticket number.
            volume: Volume to close (None = close all).

        Returns:
            True if position closed successfully.
        """
        self._ensure_connected()
        loop = asyncio.get_event_loop()
        mt5 = self._mt5

        # Get position info
        position = await loop.run_in_executor(
            None, lambda: mt5.positions_get(ticket=ticket)
        )
        if not position:
            raise PositionNotFoundError(str(ticket), platform="exness")

        pos = position[0]
        close_volume = float(volume) if volume else pos.volume

        # Get current price
        symbol_info = await loop.run_in_executor(
            None, lambda: mt5.symbol_info(pos.symbol)
        )
        if symbol_info is None:
            raise SymbolNotFoundError(pos.symbol, platform="exness")

        # Build close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": close_volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": symbol_info.bid if pos.type == 0 else symbol_info.ask,
            "deviation": 20,
            "magic": 234000,
            "comment": "close_position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = await loop.run_in_executor(
            None, lambda: mt5.order_send(request)
        )

        if result is None:
            return False

        return result.retcode == mt5.TRADE_RETCODE_DONE


    async def get_exchange_info(self, symbol: str) -> ExchangeInfo:
        """Get exchange trading rules for a symbol with caching."""
        self._ensure_connected()
        loop = asyncio.get_event_loop()
        mt5 = self._mt5

        # Check cache
        now = datetime.now(UTC)
        if (
            symbol in self._exchange_info_cache
            and self._cache_timestamp
            and (now - self._cache_timestamp).total_seconds() < self._cache_ttl_seconds
        ):
            return self._exchange_info_cache[symbol]

        # Get symbol info from MT5
        symbol_info = await loop.run_in_executor(
            None, lambda: mt5.symbol_info(symbol)
        )
        if symbol_info is None:
            raise SymbolNotFoundError(symbol, platform="exness")

        # Calculate precision from digits
        price_precision = symbol_info.digits
        volume_step = Decimal(str(symbol_info.volume_step))
        qty_precision = abs(volume_step.as_tuple().exponent) if volume_step < 1 else 0

        exchange_info = ExchangeInfo(
            symbol=symbol,
            platform=Platform.EXNESS,
            min_qty=Decimal(str(symbol_info.volume_min)),
            max_qty=Decimal(str(symbol_info.volume_max)),
            qty_step=volume_step,
            qty_precision=qty_precision,
            price_precision=price_precision,
            min_notional=Decimal("0"),  # MT5 doesn't have min notional
            leverage_options=[1, 10, 50, 100, 200, 500, 1000, 2000],  # Exness typical
            maker_fee=Decimal("0"),  # Forex typically has spread, not fees
            taker_fee=Decimal("0"),
            updated_at=now,
        )

        # Update cache
        self._exchange_info_cache[symbol] = exchange_info
        self._cache_timestamp = now

        return exchange_info

    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Get detailed symbol information including spread."""
        self._ensure_connected()
        loop = asyncio.get_event_loop()
        mt5 = self._mt5

        symbol_info = await loop.run_in_executor(
            None, lambda: mt5.symbol_info(symbol)
        )
        if symbol_info is None:
            raise SymbolNotFoundError(symbol, platform="exness")

        return {
            "symbol": symbol_info.name,
            "bid": Decimal(str(symbol_info.bid)),
            "ask": Decimal(str(symbol_info.ask)),
            "spread": symbol_info.spread,
            "spread_float": Decimal(str(symbol_info.spread)) * Decimal(str(10 ** -symbol_info.digits)),
            "digits": symbol_info.digits,
            "volume_min": Decimal(str(symbol_info.volume_min)),
            "volume_max": Decimal(str(symbol_info.volume_max)),
            "volume_step": Decimal(str(symbol_info.volume_step)),
            "trade_contract_size": symbol_info.trade_contract_size,
            "margin_initial": symbol_info.margin_initial,
            "currency_base": symbol_info.currency_base,
            "currency_profit": symbol_info.currency_profit,
        }

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[OHLCV]:
        """Get OHLCV candlestick data."""
        self._ensure_connected()
        loop = asyncio.get_event_loop()
        mt5 = self._mt5

        # Map timeframe
        tf_value = TIMEFRAME_MAP.get(timeframe)
        if tf_value is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        # Get MT5 timeframe constant
        tf_map = {
            1: mt5.TIMEFRAME_M1,
            5: mt5.TIMEFRAME_M5,
            15: mt5.TIMEFRAME_M15,
            30: mt5.TIMEFRAME_M30,
            60: mt5.TIMEFRAME_H1,
            240: mt5.TIMEFRAME_H4,
            1440: mt5.TIMEFRAME_D1,
            10080: mt5.TIMEFRAME_W1,
            43200: mt5.TIMEFRAME_MN1,
        }
        mt5_timeframe = tf_map.get(tf_value, mt5.TIMEFRAME_H1)

        # Fetch rates
        if start_time and end_time:
            rates = await loop.run_in_executor(
                None,
                lambda: mt5.copy_rates_range(symbol, mt5_timeframe, start_time, end_time)
            )
        else:
            rates = await loop.run_in_executor(
                None,
                lambda: mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, limit)
            )

        if rates is None:
            error = await loop.run_in_executor(None, mt5.last_error)
            raise ConnectionError(f"Failed to get OHLCV: {error}", platform="exness")

        return [
            OHLCV(
                timestamp=datetime.fromtimestamp(rate['time'], tz=UTC),
                open=Decimal(str(rate['open'])),
                high=Decimal(str(rate['high'])),
                low=Decimal(str(rate['low'])),
                close=Decimal(str(rate['close'])),
                volume=Decimal(str(rate['tick_volume'])),
                symbol=symbol,
                timeframe=timeframe,
                platform=Platform.EXNESS,
            )
            for rate in rates
        ]

    async def get_ticker_price(self, symbol: str) -> Decimal:
        """Get current ticker price (mid price)."""
        self._ensure_connected()
        loop = asyncio.get_event_loop()
        mt5 = self._mt5

        tick = await loop.run_in_executor(
            None, lambda: mt5.symbol_info_tick(symbol)
        )
        if tick is None:
            raise SymbolNotFoundError(symbol, platform="exness")

        # Return mid price
        return (Decimal(str(tick.bid)) + Decimal(str(tick.ask))) / 2

    def _handle_order_error(self, retcode: int, comment: str) -> None:
        """Handle MT5 order error codes."""
        mt5 = self._mt5
        
        error_messages = {
            mt5.TRADE_RETCODE_REQUOTE: "Requote",
            mt5.TRADE_RETCODE_REJECT: "Request rejected",
            mt5.TRADE_RETCODE_CANCEL: "Request canceled by trader",
            mt5.TRADE_RETCODE_PLACED: "Order placed",
            mt5.TRADE_RETCODE_DONE_PARTIAL: "Partially filled",
            mt5.TRADE_RETCODE_ERROR: "Request processing error",
            mt5.TRADE_RETCODE_TIMEOUT: "Request timeout",
            mt5.TRADE_RETCODE_INVALID: "Invalid request",
            mt5.TRADE_RETCODE_INVALID_VOLUME: "Invalid volume",
            mt5.TRADE_RETCODE_INVALID_PRICE: "Invalid price",
            mt5.TRADE_RETCODE_INVALID_STOPS: "Invalid stops",
            mt5.TRADE_RETCODE_TRADE_DISABLED: "Trade disabled",
            mt5.TRADE_RETCODE_MARKET_CLOSED: "Market closed",
            mt5.TRADE_RETCODE_NO_MONEY: "Insufficient funds",
            mt5.TRADE_RETCODE_PRICE_CHANGED: "Price changed",
            mt5.TRADE_RETCODE_PRICE_OFF: "No quotes",
            mt5.TRADE_RETCODE_INVALID_EXPIRATION: "Invalid expiration",
            mt5.TRADE_RETCODE_ORDER_CHANGED: "Order state changed",
            mt5.TRADE_RETCODE_TOO_MANY_REQUESTS: "Too many requests",
        }

        error_msg = error_messages.get(retcode, f"Unknown error: {retcode}")

        if retcode == mt5.TRADE_RETCODE_NO_MONEY:
            raise InsufficientBalanceError(
                error_msg,
                required="unknown",
                available="unknown",
                asset="unknown",
            )
        if retcode in (mt5.TRADE_RETCODE_INVALID_VOLUME, mt5.TRADE_RETCODE_INVALID_PRICE):
            raise InvalidOrderError(f"{error_msg}: {comment}")
        if retcode == mt5.TRADE_RETCODE_TIMEOUT:
            raise TimeoutError(error_msg, operation="order", timeout_seconds=30.0)

        raise OrderRejectedError(
            error_msg,
            order_id="",
            reason=comment,
        )
