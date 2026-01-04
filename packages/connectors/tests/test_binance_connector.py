"""
Tests for Binance connector.

Unit tests with mocked API responses.
Integration tests require BINANCE_API_KEY and BINANCE_API_SECRET env vars.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from binance.exceptions import BinanceAPIException
from tenacity import RetryError

from packages.connectors.src.binance_connector import BinanceConnector
from packages.shared.src.exceptions import (
    AuthenticationError,
    ConnectionError,
    InsufficientBalanceError,
    InvalidOrderError,
    RateLimitError,
    SymbolNotFoundError,
)
from packages.shared.src.models import (
    Order,
    OrderStatus,
    OrderType,
    Platform,
    Side,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def connector():
    """Create a Binance connector instance."""
    return BinanceConnector(
        api_key="test_key",
        api_secret="test_secret",
        testnet=True,
    )


@pytest.fixture
def mock_client():
    """Create a mock Binance client."""
    client = AsyncMock()
    client.get_account = AsyncMock(return_value={"balances": []})
    client.close_connection = AsyncMock()
    return client


# =============================================================================
# Unit Tests
# =============================================================================


class TestBinanceConnectorInit:
    """Test connector initialization."""

    def test_init_default_values(self, connector: BinanceConnector) -> None:
        """Test default initialization values."""
        assert connector.platform == Platform.BINANCE
        assert connector.is_connected is False
        assert connector._testnet is True
        assert connector._futures is False

    def test_init_futures_mode(self) -> None:
        """Test futures mode initialization."""
        connector = BinanceConnector(
            api_key="key",
            api_secret="secret",
            testnet=True,
            futures=True,
        )
        assert connector._futures is True


class TestBinanceConnectorConnection:
    """Test connection methods."""

    @pytest.mark.asyncio
    async def test_connect_success(
        self, connector: BinanceConnector, mock_client: AsyncMock
    ) -> None:
        """Test successful connection."""
        with patch(
            "packages.connectors.src.binance_connector.AsyncClient.create",
            return_value=mock_client,
        ):
            result = await connector.connect()
            assert result is True
            assert connector.is_connected is True

    @pytest.mark.asyncio
    async def test_connect_auth_failure(self, connector: BinanceConnector) -> None:
        """Test connection with invalid credentials."""
        mock_client = AsyncMock()
        mock_client.get_account = AsyncMock(
            side_effect=BinanceAPIException(
                response=MagicMock(status_code=400),
                status_code=400,
                text='{"code": -2015, "msg": "Invalid API-key"}',
            )
        )

        with (
            patch(
                "packages.connectors.src.binance_connector.AsyncClient.create",
                return_value=mock_client,
            ),
            pytest.raises(AuthenticationError),
        ):
            await connector.connect()

    @pytest.mark.asyncio
    async def test_disconnect(
        self, connector: BinanceConnector, mock_client: AsyncMock
    ) -> None:
        """Test disconnection."""
        with patch(
            "packages.connectors.src.binance_connector.AsyncClient.create",
            return_value=mock_client,
        ):
            await connector.connect()
            await connector.disconnect()
            assert connector.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_network_error(self, connector: BinanceConnector) -> None:
        """Test connection with network error."""
        with (
            patch(
                "packages.connectors.src.binance_connector.AsyncClient.create",
                side_effect=Exception("Network error"),
            ),
            pytest.raises((ConnectionError, RetryError)),
        ):
            await connector.connect()


class TestBinanceConnectorBalance:
    """Test balance methods."""

    @pytest.mark.asyncio
    async def test_get_balance_spot(
        self, connector: BinanceConnector, mock_client: AsyncMock
    ) -> None:
        """Test getting spot balance."""
        mock_client.get_account = AsyncMock(
            return_value={
                "balances": [
                    {"asset": "BTC", "free": "1.5", "locked": "0.5"},
                    {"asset": "USDT", "free": "10000.0", "locked": "0"},
                    {"asset": "ETH", "free": "0", "locked": "0"},  # Zero balance
                ]
            }
        )

        with patch(
            "packages.connectors.src.binance_connector.AsyncClient.create",
            return_value=mock_client,
        ):
            await connector.connect()
            balance = await connector.get_balance()

            assert balance["BTC"] == Decimal("1.5")
            assert balance["USDT"] == Decimal("10000.0")
            assert "ETH" not in balance  # Zero balances excluded

    @pytest.mark.asyncio
    async def test_get_balance_futures(self, mock_client: AsyncMock) -> None:
        """Test getting futures balance."""
        connector = BinanceConnector(
            api_key="key", api_secret="secret", testnet=True, futures=True
        )
        mock_client.futures_account_balance = AsyncMock(
            return_value=[
                {"asset": "USDT", "balance": "5000.0"},
                {"asset": "BNB", "balance": "10.0"},
                {"asset": "BUSD", "balance": "0"},
            ]
        )

        with patch(
            "packages.connectors.src.binance_connector.AsyncClient.create",
            return_value=mock_client,
        ):
            await connector.connect()
            balance = await connector.get_balance()

            assert balance["USDT"] == Decimal("5000.0")
            assert balance["BNB"] == Decimal("10.0")
            assert "BUSD" not in balance

    @pytest.mark.asyncio
    async def test_get_balance_not_connected(
        self, connector: BinanceConnector
    ) -> None:
        """Test getting balance when not connected."""
        with pytest.raises(ConnectionError):
            await connector.get_balance()


class TestBinanceConnectorPositions:
    """Test position methods."""

    @pytest.mark.asyncio
    async def test_get_positions_spot_returns_empty(
        self, connector: BinanceConnector, mock_client: AsyncMock
    ) -> None:
        """Test spot mode returns empty positions."""
        with patch(
            "packages.connectors.src.binance_connector.AsyncClient.create",
            return_value=mock_client,
        ):
            await connector.connect()
            positions = await connector.get_positions()
            assert positions == []

    @pytest.mark.asyncio
    async def test_get_positions_futures(self, mock_client: AsyncMock) -> None:
        """Test getting futures positions."""
        connector = BinanceConnector(
            api_key="key", api_secret="secret", testnet=True, futures=True
        )
        mock_client.futures_position_information = AsyncMock(
            return_value=[
                {
                    "symbol": "BTCUSDT",
                    "positionAmt": "0.5",
                    "entryPrice": "40000.0",
                    "markPrice": "42000.0",
                    "unRealizedProfit": "1000.0",
                    "leverage": "10",
                    "liquidationPrice": "35000.0",
                },
                {
                    "symbol": "ETHUSDT",
                    "positionAmt": "-2.0",  # Short position
                    "entryPrice": "2500.0",
                    "markPrice": "2400.0",
                    "unRealizedProfit": "200.0",
                    "leverage": "5",
                    "liquidationPrice": "3000.0",
                },
                {
                    "symbol": "BNBUSDT",
                    "positionAmt": "0",  # No position
                    "entryPrice": "0",
                    "markPrice": "300.0",
                    "unRealizedProfit": "0",
                    "leverage": "1",
                    "liquidationPrice": "0",
                },
            ]
        )

        with patch(
            "packages.connectors.src.binance_connector.AsyncClient.create",
            return_value=mock_client,
        ):
            await connector.connect()
            positions = await connector.get_positions()

            assert len(positions) == 2

            # Long position
            btc_pos = positions[0]
            assert btc_pos.symbol == "BTCUSDT"
            assert btc_pos.side == Side.BUY
            assert btc_pos.quantity == Decimal("0.5")
            assert btc_pos.entry_price == Decimal("40000.0")
            assert btc_pos.leverage == 10

            # Short position
            eth_pos = positions[1]
            assert eth_pos.symbol == "ETHUSDT"
            assert eth_pos.side == Side.SELL
            assert eth_pos.quantity == Decimal("2.0")  # Absolute value


class TestBinanceConnectorOrders:
    """Test order methods."""

    @pytest.mark.asyncio
    async def test_place_market_order(
        self, connector: BinanceConnector, mock_client: AsyncMock
    ) -> None:
        """Test placing a market order."""
        mock_client.get_exchange_info = AsyncMock(
            return_value={
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "quotePrecision": 8,
                        "filters": [
                            {
                                "filterType": "LOT_SIZE",
                                "minQty": "0.001",
                                "maxQty": "1000",
                                "stepSize": "0.001",
                            },
                            {"filterType": "MIN_NOTIONAL", "minNotional": "10"},
                        ],
                    }
                ]
            }
        )
        mock_client.create_order = AsyncMock(
            return_value={
                "orderId": 12345,
                "status": "FILLED",
                "executedQty": "0.1",
                "avgPrice": "42000.0",
            }
        )

        with patch(
            "packages.connectors.src.binance_connector.AsyncClient.create",
            return_value=mock_client,
        ):
            await connector.connect()

            order = Order(
                id="",
                symbol="BTCUSDT",
                side=Side.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
            )
            result = await connector.place_order(order)

            assert result.id == "12345"
            assert result.status == OrderStatus.FILLED
            assert result.filled_quantity == Decimal("0.1")

    @pytest.mark.asyncio
    async def test_place_limit_order(
        self, connector: BinanceConnector, mock_client: AsyncMock
    ) -> None:
        """Test placing a limit order."""
        mock_client.get_exchange_info = AsyncMock(
            return_value={
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "quotePrecision": 8,
                        "filters": [
                            {
                                "filterType": "LOT_SIZE",
                                "minQty": "0.001",
                                "maxQty": "1000",
                                "stepSize": "0.001",
                            },
                            {"filterType": "MIN_NOTIONAL", "minNotional": "10"},
                        ],
                    }
                ]
            }
        )
        mock_client.create_order = AsyncMock(
            return_value={
                "orderId": 12346,
                "status": "NEW",
                "executedQty": "0",
            }
        )

        with patch(
            "packages.connectors.src.binance_connector.AsyncClient.create",
            return_value=mock_client,
        ):
            await connector.connect()

            order = Order(
                id="",
                symbol="BTCUSDT",
                side=Side.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
                price=Decimal("40000.0"),
            )
            result = await connector.place_order(order)

            assert result.id == "12346"
            assert result.status == OrderStatus.PENDING
            mock_client.create_order.assert_called_once()
            call_kwargs = mock_client.create_order.call_args.kwargs
            assert call_kwargs["type"] == "LIMIT"
            assert call_kwargs["price"] == "40000.0"
            assert call_kwargs["timeInForce"] == "GTC"

    @pytest.mark.asyncio
    async def test_place_order_invalid_quantity(
        self, connector: BinanceConnector, mock_client: AsyncMock
    ) -> None:
        """Test placing order with invalid quantity."""
        mock_client.get_exchange_info = AsyncMock(
            return_value={
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "quotePrecision": 8,
                        "filters": [
                            {
                                "filterType": "LOT_SIZE",
                                "minQty": "0.001",
                                "maxQty": "1000",
                                "stepSize": "0.001",
                            },
                        ],
                    }
                ]
            }
        )

        with patch(
            "packages.connectors.src.binance_connector.AsyncClient.create",
            return_value=mock_client,
        ):
            await connector.connect()

            order = Order(
                id="",
                symbol="BTCUSDT",
                side=Side.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.0001"),  # Below min
            )

            with pytest.raises(InvalidOrderError):
                await connector.place_order(order)

    @pytest.mark.asyncio
    async def test_place_order_insufficient_balance(
        self, connector: BinanceConnector, mock_client: AsyncMock
    ) -> None:
        """Test placing order with insufficient balance."""
        mock_client.get_exchange_info = AsyncMock(
            return_value={
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "quotePrecision": 8,
                        "filters": [
                            {
                                "filterType": "LOT_SIZE",
                                "minQty": "0.001",
                                "maxQty": "1000",
                                "stepSize": "0.001",
                            },
                        ],
                    }
                ]
            }
        )
        mock_client.create_order = AsyncMock(
            side_effect=BinanceAPIException(
                response=MagicMock(status_code=400),
                status_code=400,
                text='{"code": -2010, "msg": "Insufficient balance"}',
            )
        )

        with patch(
            "packages.connectors.src.binance_connector.AsyncClient.create",
            return_value=mock_client,
        ):
            await connector.connect()

            order = Order(
                id="",
                symbol="BTCUSDT",
                side=Side.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
            )

            with pytest.raises(InsufficientBalanceError):
                await connector.place_order(order)

    @pytest.mark.asyncio
    async def test_cancel_order_success(
        self, connector: BinanceConnector, mock_client: AsyncMock
    ) -> None:
        """Test cancelling an order."""
        mock_client.cancel_order = AsyncMock(return_value={"orderId": 12345})

        with patch(
            "packages.connectors.src.binance_connector.AsyncClient.create",
            return_value=mock_client,
        ):
            await connector.connect()
            result = await connector.cancel_order("12345", "BTCUSDT")
            assert result is True

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(
        self, connector: BinanceConnector, mock_client: AsyncMock
    ) -> None:
        """Test cancelling non-existent order."""
        mock_client.cancel_order = AsyncMock(
            side_effect=BinanceAPIException(
                response=MagicMock(status_code=400),
                status_code=400,
                text='{"code": -2011, "msg": "Unknown order"}',
            )
        )

        with patch(
            "packages.connectors.src.binance_connector.AsyncClient.create",
            return_value=mock_client,
        ):
            await connector.connect()
            result = await connector.cancel_order("99999", "BTCUSDT")
            assert result is False


class TestBinanceConnectorExchangeInfo:
    """Test exchange info methods."""

    @pytest.mark.asyncio
    async def test_get_exchange_info(
        self, connector: BinanceConnector, mock_client: AsyncMock
    ) -> None:
        """Test getting exchange info."""
        mock_client.get_exchange_info = AsyncMock(
            return_value={
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "quotePrecision": 8,
                        "filters": [
                            {
                                "filterType": "LOT_SIZE",
                                "minQty": "0.00001",
                                "maxQty": "9000",
                                "stepSize": "0.00001",
                            },
                            {"filterType": "MIN_NOTIONAL", "minNotional": "10"},
                        ],
                    }
                ]
            }
        )

        with patch(
            "packages.connectors.src.binance_connector.AsyncClient.create",
            return_value=mock_client,
        ):
            await connector.connect()
            info = await connector.get_exchange_info("BTCUSDT")

            assert info.symbol == "BTCUSDT"
            assert info.platform == Platform.BINANCE
            assert info.min_qty == Decimal("0.00001")
            assert info.max_qty == Decimal("9000")
            assert info.qty_step == Decimal("0.00001")
            assert info.min_notional == Decimal("10")

    @pytest.mark.asyncio
    async def test_get_exchange_info_caching(
        self, connector: BinanceConnector, mock_client: AsyncMock
    ) -> None:
        """Test exchange info caching."""
        mock_client.get_exchange_info = AsyncMock(
            return_value={
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "quotePrecision": 8,
                        "filters": [
                            {
                                "filterType": "LOT_SIZE",
                                "minQty": "0.001",
                                "maxQty": "1000",
                                "stepSize": "0.001",
                            },
                        ],
                    }
                ]
            }
        )

        with patch(
            "packages.connectors.src.binance_connector.AsyncClient.create",
            return_value=mock_client,
        ):
            await connector.connect()

            # First call
            await connector.get_exchange_info("BTCUSDT")
            # Second call should use cache
            await connector.get_exchange_info("BTCUSDT")

            # Should only call API once
            assert mock_client.get_exchange_info.call_count == 1

    @pytest.mark.asyncio
    async def test_get_exchange_info_symbol_not_found(
        self, connector: BinanceConnector, mock_client: AsyncMock
    ) -> None:
        """Test getting exchange info for non-existent symbol."""
        mock_client.get_exchange_info = AsyncMock(
            return_value={"symbols": [{"symbol": "BTCUSDT", "filters": []}]}
        )

        with patch(
            "packages.connectors.src.binance_connector.AsyncClient.create",
            return_value=mock_client,
        ):
            await connector.connect()

            with pytest.raises(SymbolNotFoundError):
                await connector.get_exchange_info("INVALIDPAIR")


class TestBinanceConnectorOHLCV:
    """Test OHLCV methods."""

    @pytest.mark.asyncio
    async def test_get_ohlcv(
        self, connector: BinanceConnector, mock_client: AsyncMock
    ) -> None:
        """Test getting OHLCV data."""
        mock_client.get_klines = AsyncMock(
            return_value=[
                [
                    1704067200000,  # timestamp
                    "42000.0",  # open
                    "42500.0",  # high
                    "41800.0",  # low
                    "42300.0",  # close
                    "1000.0",  # volume
                    1704070800000,  # close time
                    "42000000.0",  # quote volume
                    100,  # trades
                    "500.0",  # taker buy base
                    "21000000.0",  # taker buy quote
                    "0",  # ignore
                ],
                [
                    1704070800000,
                    "42300.0",
                    "42800.0",
                    "42100.0",
                    "42600.0",
                    "1200.0",
                    1704074400000,
                    "50400000.0",
                    120,
                    "600.0",
                    "25200000.0",
                    "0",
                ],
            ]
        )

        with patch(
            "packages.connectors.src.binance_connector.AsyncClient.create",
            return_value=mock_client,
        ):
            await connector.connect()
            ohlcv = await connector.get_ohlcv("BTCUSDT", "1h", 2)

            assert len(ohlcv) == 2
            assert ohlcv[0].open == Decimal("42000.0")
            assert ohlcv[0].high == Decimal("42500.0")
            assert ohlcv[0].low == Decimal("41800.0")
            assert ohlcv[0].close == Decimal("42300.0")
            assert ohlcv[0].volume == Decimal("1000.0")
            assert ohlcv[0].symbol == "BTCUSDT"
            assert ohlcv[0].timeframe == "1h"

    @pytest.mark.asyncio
    async def test_get_ohlcv_invalid_timeframe(
        self, connector: BinanceConnector, mock_client: AsyncMock
    ) -> None:
        """Test getting OHLCV with invalid timeframe."""
        with patch(
            "packages.connectors.src.binance_connector.AsyncClient.create",
            return_value=mock_client,
        ):
            await connector.connect()

            with pytest.raises(ValueError, match="Invalid timeframe"):
                await connector.get_ohlcv("BTCUSDT", "invalid", 100)


class TestBinanceConnectorTickerPrice:
    """Test ticker price methods."""

    @pytest.mark.asyncio
    async def test_get_ticker_price(
        self, connector: BinanceConnector, mock_client: AsyncMock
    ) -> None:
        """Test getting ticker price."""
        mock_client.get_symbol_ticker = AsyncMock(
            return_value={"symbol": "BTCUSDT", "price": "42500.50"}
        )

        with patch(
            "packages.connectors.src.binance_connector.AsyncClient.create",
            return_value=mock_client,
        ):
            await connector.connect()
            price = await connector.get_ticker_price("BTCUSDT")

            assert price == Decimal("42500.50")

    @pytest.mark.asyncio
    async def test_get_ticker_price_futures(self, mock_client: AsyncMock) -> None:
        """Test getting futures ticker price."""
        connector = BinanceConnector(
            api_key="key", api_secret="secret", testnet=True, futures=True
        )
        mock_client.futures_symbol_ticker = AsyncMock(
            return_value={"symbol": "BTCUSDT", "price": "42600.00"}
        )

        with patch(
            "packages.connectors.src.binance_connector.AsyncClient.create",
            return_value=mock_client,
        ):
            await connector.connect()
            price = await connector.get_ticker_price("BTCUSDT")

            assert price == Decimal("42600.00")


class TestBinanceConnectorErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_rate_limit_error(
        self, connector: BinanceConnector, mock_client: AsyncMock
    ) -> None:
        """Test rate limit error handling."""
        mock_client.get_account = AsyncMock(
            side_effect=BinanceAPIException(
                response=MagicMock(status_code=429),
                status_code=429,
                text='{"code": -1003, "msg": "Too many requests"}',
            )
        )

        with patch(
            "packages.connectors.src.binance_connector.AsyncClient.create",
            return_value=mock_client,
        ), pytest.raises(RateLimitError):
            await connector.connect()

    @pytest.mark.asyncio
    async def test_order_status_mapping(self, connector: BinanceConnector) -> None:
        """Test order status mapping."""
        assert connector._map_order_status("NEW") == OrderStatus.PENDING
        assert connector._map_order_status("FILLED") == OrderStatus.FILLED
        assert (
            connector._map_order_status("PARTIALLY_FILLED")
            == OrderStatus.PARTIALLY_FILLED
        )
        assert connector._map_order_status("CANCELED") == OrderStatus.CANCELLED
        assert connector._map_order_status("REJECTED") == OrderStatus.REJECTED
        assert connector._map_order_status("EXPIRED") == OrderStatus.EXPIRED
        assert connector._map_order_status("UNKNOWN") == OrderStatus.PENDING
