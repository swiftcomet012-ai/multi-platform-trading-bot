"""
Binance exchange connector implementation.

Supports both Spot and Futures trading with testnet support.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException
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
    RateLimitError,
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

# Timeframe mapping
TIMEFRAME_MAP = {
    "1m": AsyncClient.KLINE_INTERVAL_1MINUTE,
    "3m": AsyncClient.KLINE_INTERVAL_3MINUTE,
    "5m": AsyncClient.KLINE_INTERVAL_5MINUTE,
    "15m": AsyncClient.KLINE_INTERVAL_15MINUTE,
    "30m": AsyncClient.KLINE_INTERVAL_30MINUTE,
    "1h": AsyncClient.KLINE_INTERVAL_1HOUR,
    "2h": AsyncClient.KLINE_INTERVAL_2HOUR,
    "4h": AsyncClient.KLINE_INTERVAL_4HOUR,
    "6h": AsyncClient.KLINE_INTERVAL_6HOUR,
    "8h": AsyncClient.KLINE_INTERVAL_8HOUR,
    "12h": AsyncClient.KLINE_INTERVAL_12HOUR,
    "1d": AsyncClient.KLINE_INTERVAL_1DAY,
    "3d": AsyncClient.KLINE_INTERVAL_3DAY,
    "1w": AsyncClient.KLINE_INTERVAL_1WEEK,
    "1M": AsyncClient.KLINE_INTERVAL_1MONTH,
}


class BinanceConnector(BaseConnector):
    """Binance exchange connector."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        futures: bool = False,
    ) -> None:
        """
        Initialize Binance connector.

        Args:
            api_key: Binance API key.
            api_secret: Binance API secret.
            testnet: Use testnet (default True for safety).
            futures: Use futures API (default False for spot).
        """
        super().__init__()
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._futures = futures
        self._client: AsyncClient | None = None
        self._bsm: BinanceSocketManager | None = None
        self._exchange_info_cache: dict[str, ExchangeInfo] = {}
        self._cache_timestamp: datetime | None = None
        self._cache_ttl_seconds = 3600  # 1 hour

    @property
    def platform(self) -> Platform:
        """Get the platform identifier."""
        return Platform.BINANCE

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
    )
    async def connect(self) -> bool:
        """Connect to Binance."""
        try:
            self._client = await AsyncClient.create(
                api_key=self._api_key,
                api_secret=self._api_secret,
                testnet=self._testnet,
            )
            # Verify connection by getting account info
            await self._client.get_account()
            self._connected = True
            logger.info(
                "binance_connected",
                testnet=self._testnet,
                futures=self._futures,
            )
            return True
        except BinanceAPIException as e:
            if e.code == -2015:
                raise AuthenticationError(
                    "Invalid API key or secret", platform="binance"
                ) from e
            if e.code == -1003:
                raise RateLimitError(
                    "Rate limit exceeded", platform="binance"
                ) from e
            raise ConnectionError(
                f"Failed to connect: {e.message}", platform="binance"
            ) from e
        except Exception as e:
            raise ConnectionError(
                f"Connection failed: {e!s}", platform="binance"
            ) from e

    async def disconnect(self) -> None:
        """Disconnect from Binance."""
        if self._client:
            await self._client.close_connection()
            self._client = None
        self._connected = False
        logger.info("binance_disconnected")


    def _ensure_connected(self) -> None:
        """Ensure client is connected."""
        if not self._client or not self._connected:
            raise ConnectionError("Not connected to Binance", platform="binance")

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
    )
    async def get_balance(self) -> dict[str, Decimal]:
        """Get account balances."""
        self._ensure_connected()
        try:
            if self._futures:
                account = await self._client.futures_account_balance()
                return {
                    item["asset"]: Decimal(item["balance"])
                    for item in account
                    if Decimal(item["balance"]) > 0
                }
            else:
                account = await self._client.get_account()
                return {
                    item["asset"]: Decimal(item["free"])
                    for item in account["balances"]
                    if Decimal(item["free"]) > 0
                }
        except BinanceAPIException as e:
            self._handle_api_exception(e)
            raise

    async def get_positions(self) -> list[Position]:
        """Get open positions (futures only, spot returns empty)."""
        self._ensure_connected()
        if not self._futures:
            return []  # Spot doesn't have positions concept

        try:
            positions = await self._client.futures_position_information()
            result = []
            for pos in positions:
                qty = Decimal(pos["positionAmt"])
                if qty == 0:
                    continue

                entry_price = Decimal(pos["entryPrice"])
                mark_price = Decimal(pos["markPrice"])
                unrealized_pnl = Decimal(pos["unRealizedProfit"])

                result.append(
                    Position(
                        symbol=pos["symbol"],
                        side=Side.BUY if qty > 0 else Side.SELL,
                        quantity=abs(qty),
                        entry_price=entry_price,
                        current_price=mark_price,
                        unrealized_pnl=unrealized_pnl,
                        unrealized_pnl_pct=(
                            float(unrealized_pnl / (entry_price * abs(qty)) * 100)
                            if entry_price > 0
                            else 0.0
                        ),
                        platform=Platform.BINANCE,
                        leverage=int(pos["leverage"]),
                        liquidation_price=Decimal(pos["liquidationPrice"]),
                        updated_at=datetime.now(UTC),
                    )
                )
            return result
        except BinanceAPIException as e:
            self._handle_api_exception(e)
            raise


    async def place_order(self, order: Order) -> Order:
        """Place an order on Binance."""
        self._ensure_connected()

        # Validate order against exchange rules
        exchange_info = await self.get_exchange_info(order.symbol)
        is_valid, error_msg = exchange_info.validate_quantity(order.quantity)
        if not is_valid:
            raise InvalidOrderError(error_msg)

        try:
            # Build order params
            params: dict[str, Any] = {
                "symbol": order.symbol,
                "side": order.side.value.upper(),
                "quantity": str(exchange_info.round_quantity(order.quantity)),
            }

            # Add idempotency key if provided
            if order.idempotency_key:
                params["newClientOrderId"] = order.idempotency_key

            # Set order type
            if order.order_type == OrderType.MARKET:
                params["type"] = "MARKET"
            elif order.order_type == OrderType.LIMIT:
                params["type"] = "LIMIT"
                params["price"] = str(order.price)
                params["timeInForce"] = "GTC"
            elif order.order_type == OrderType.STOP_LOSS:
                params["type"] = "STOP_LOSS_LIMIT"
                params["stopPrice"] = str(order.stop_loss)
                params["price"] = str(order.price)
                params["timeInForce"] = "GTC"

            # Execute order
            if self._futures:
                result = await self._client.futures_create_order(**params)
            else:
                result = await self._client.create_order(**params)

            # Update order with result
            return Order(
                id=str(result["orderId"]),
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                platform=Platform.BINANCE,
                idempotency_key=order.idempotency_key,
                status=self._map_order_status(result["status"]),
                created_at=datetime.now(UTC),
                filled_quantity=Decimal(result.get("executedQty", "0")),
                filled_price=(
                    Decimal(result["avgPrice"])
                    if result.get("avgPrice")
                    else None
                ),
            )

        except BinanceAPIException as e:
            self._handle_order_exception(e)
            raise

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        self._ensure_connected()
        try:
            if self._futures:
                await self._client.futures_cancel_order(
                    symbol=symbol, orderId=int(order_id)
                )
            else:
                await self._client.cancel_order(symbol=symbol, orderId=int(order_id))
            return True
        except BinanceAPIException as e:
            if e.code == -2011:  # Unknown order
                return False
            self._handle_api_exception(e)
            raise

    def _map_order_status(self, status: str) -> OrderStatus:
        """Map Binance order status to internal status."""
        mapping = {
            "NEW": OrderStatus.PENDING,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        return mapping.get(status, OrderStatus.PENDING)


    async def get_exchange_info(self, symbol: str) -> ExchangeInfo:
        """Get exchange trading rules for a symbol with caching."""
        self._ensure_connected()

        # Check cache
        now = datetime.now(UTC)
        if (
            symbol in self._exchange_info_cache
            and self._cache_timestamp
            and (now - self._cache_timestamp).total_seconds() < self._cache_ttl_seconds
        ):
            return self._exchange_info_cache[symbol]

        try:
            if self._futures:
                info = await self._client.futures_exchange_info()
            else:
                info = await self._client.get_exchange_info()

            # Find symbol info
            symbol_info = None
            for s in info["symbols"]:
                if s["symbol"] == symbol:
                    symbol_info = s
                    break

            if not symbol_info:
                raise SymbolNotFoundError(symbol, platform="binance")

            # Parse filters
            min_qty = Decimal("0")
            max_qty = Decimal("0")
            qty_step = Decimal("0")
            min_notional = Decimal("0")

            for f in symbol_info["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    min_qty = Decimal(f["minQty"])
                    max_qty = Decimal(f["maxQty"])
                    qty_step = Decimal(f["stepSize"])
                elif f["filterType"] == "MIN_NOTIONAL" or f["filterType"] == "NOTIONAL":
                    min_notional = Decimal(f.get("minNotional", "0"))

            # Calculate precision from step size
            qty_precision = abs(qty_step.as_tuple().exponent)
            price_precision = int(symbol_info.get("quotePrecision", 8))

            exchange_info = ExchangeInfo(
                symbol=symbol,
                platform=Platform.BINANCE,
                min_qty=min_qty,
                max_qty=max_qty,
                qty_step=qty_step,
                qty_precision=qty_precision,
                price_precision=price_precision,
                min_notional=min_notional,
                leverage_options=[1, 2, 3, 5, 10, 20, 50, 75, 100, 125],
                maker_fee=Decimal("0.001"),  # Default 0.1%
                taker_fee=Decimal("0.001"),
                updated_at=now,
            )

            # Update cache
            self._exchange_info_cache[symbol] = exchange_info
            self._cache_timestamp = now

            return exchange_info

        except BinanceAPIException as e:
            self._handle_api_exception(e)
            raise

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

        interval = TIMEFRAME_MAP.get(timeframe)
        if not interval:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        try:
            params: dict[str, Any] = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
            }
            if start_time:
                params["startTime"] = int(start_time.timestamp() * 1000)
            if end_time:
                params["endTime"] = int(end_time.timestamp() * 1000)

            if self._futures:
                klines = await self._client.futures_klines(**params)
            else:
                klines = await self._client.get_klines(**params)

            return [
                OHLCV(
                    timestamp=datetime.fromtimestamp(k[0] / 1000, tz=UTC),
                    open=Decimal(k[1]),
                    high=Decimal(k[2]),
                    low=Decimal(k[3]),
                    close=Decimal(k[4]),
                    volume=Decimal(k[5]),
                    symbol=symbol,
                    timeframe=timeframe,
                    platform=Platform.BINANCE,
                )
                for k in klines
            ]
        except BinanceAPIException as e:
            self._handle_api_exception(e)
            raise

    async def get_ticker_price(self, symbol: str) -> Decimal:
        """Get current ticker price."""
        self._ensure_connected()
        try:
            if self._futures:
                ticker = await self._client.futures_symbol_ticker(symbol=symbol)
            else:
                ticker = await self._client.get_symbol_ticker(symbol=symbol)
            return Decimal(ticker["price"])
        except BinanceAPIException as e:
            self._handle_api_exception(e)
            raise


    def _handle_api_exception(self, e: BinanceAPIException) -> None:
        """Handle Binance API exceptions."""
        if e.code == -1003:
            raise RateLimitError("Rate limit exceeded", platform="binance") from e
        if e.code == -1021:
            raise TimeoutError(
                "Request timed out", operation="api_call", timeout_seconds=30.0
            ) from e
        if e.code == -2015:
            raise AuthenticationError("Invalid API key", platform="binance") from e
        raise ConnectionError(f"API error: {e.message}", platform="binance") from e

    def _handle_order_exception(self, e: BinanceAPIException) -> None:
        """Handle order-specific exceptions."""
        if e.code == -2010:
            raise InsufficientBalanceError(
                "Insufficient balance",
                required="unknown",
                available="unknown",
                asset="unknown",
            ) from e
        if e.code == -1013:
            raise InvalidOrderError(f"Invalid quantity: {e.message}") from e
        if e.code == -1111:
            raise InvalidOrderError(f"Invalid precision: {e.message}") from e
        if e.code == -2011:
            raise OrderRejectedError(
                "Order rejected", order_id="unknown", reason=e.message
            ) from e
        self._handle_api_exception(e)
