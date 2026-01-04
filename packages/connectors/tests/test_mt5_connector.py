"""
Tests for MT5/Exness connector.

Unit tests with mocked MT5 module.
Integration tests require MT5 terminal and Exness demo account.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

import pytest

from packages.connectors.src.mt5_connector import MT5Connector
from packages.shared.src.exceptions import (
    AuthenticationError,
    ConnectionError,
    InsufficientBalanceError,
    InvalidOrderError,
    OrderRejectedError,
    PositionNotFoundError,
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
    """Create an MT5 connector instance."""
    return MT5Connector(
        login=12345678,
        password="test_password",
        server="Exness-MT5Demo",
    )


@pytest.fixture
def mock_mt5():
    """Create a mock MT5 module."""
    mt5 = MagicMock()
    
    # Constants
    mt5.ORDER_TYPE_BUY = 0
    mt5.ORDER_TYPE_SELL = 1
    mt5.ORDER_TYPE_BUY_LIMIT = 2
    mt5.ORDER_TYPE_SELL_LIMIT = 3
    mt5.ORDER_TIME_GTC = 0
    mt5.ORDER_FILLING_IOC = 1
    mt5.TRADE_ACTION_DEAL = 1
    mt5.TRADE_ACTION_REMOVE = 2
    mt5.TRADE_RETCODE_DONE = 10009
    mt5.TRADE_RETCODE_REQUOTE = 10004
    mt5.TRADE_RETCODE_REJECT = 10006
    mt5.TRADE_RETCODE_CANCEL = 10007
    mt5.TRADE_RETCODE_PLACED = 10008
    mt5.TRADE_RETCODE_DONE_PARTIAL = 10010
    mt5.TRADE_RETCODE_ERROR = 10011
    mt5.TRADE_RETCODE_TIMEOUT = 10012
    mt5.TRADE_RETCODE_INVALID = 10013
    mt5.TRADE_RETCODE_INVALID_VOLUME = 10014
    mt5.TRADE_RETCODE_INVALID_PRICE = 10015
    mt5.TRADE_RETCODE_INVALID_STOPS = 10016
    mt5.TRADE_RETCODE_TRADE_DISABLED = 10017
    mt5.TRADE_RETCODE_MARKET_CLOSED = 10018
    mt5.TRADE_RETCODE_NO_MONEY = 10019
    mt5.TRADE_RETCODE_PRICE_CHANGED = 10020
    mt5.TRADE_RETCODE_PRICE_OFF = 10021
    mt5.TRADE_RETCODE_INVALID_EXPIRATION = 10022
    mt5.TRADE_RETCODE_ORDER_CHANGED = 10023
    mt5.TRADE_RETCODE_TOO_MANY_REQUESTS = 10024
    
    # Timeframes
    mt5.TIMEFRAME_M1 = 1
    mt5.TIMEFRAME_M5 = 5
    mt5.TIMEFRAME_M15 = 15
    mt5.TIMEFRAME_M30 = 30
    mt5.TIMEFRAME_H1 = 60
    mt5.TIMEFRAME_H4 = 240
    mt5.TIMEFRAME_D1 = 1440
    mt5.TIMEFRAME_W1 = 10080
    mt5.TIMEFRAME_MN1 = 43200
    
    return mt5


@pytest.fixture
def mock_account_info():
    """Create mock account info."""
    return SimpleNamespace(
        login=12345678,
        server="Exness-MT5Demo",
        currency="USD",
        balance=10000.0,
        equity=10500.0,
        margin=1000.0,
        margin_free=9500.0,
        margin_level=1050.0,
        leverage=100,
        profit=500.0,
    )


@pytest.fixture
def mock_symbol_info():
    """Create mock symbol info."""
    return SimpleNamespace(
        name="EURUSD",
        bid=1.08500,
        ask=1.08510,
        spread=10,
        digits=5,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        trade_contract_size=100000,
        margin_initial=0,
        currency_base="EUR",
        currency_profit="USD",
    )


# =============================================================================
# Unit Tests
# =============================================================================


class TestMT5ConnectorInit:
    """Test connector initialization."""

    def test_init_default_values(self, connector: MT5Connector) -> None:
        """Test default initialization values."""
        assert connector.platform == Platform.EXNESS
        assert connector.is_connected is False
        assert connector._login == 12345678
        assert connector._server == "Exness-MT5Demo"

    def test_init_with_path(self) -> None:
        """Test initialization with custom path."""
        connector = MT5Connector(
            login=12345678,
            password="test",
            server="Exness-MT5Demo",
            path="C:/Program Files/MetaTrader 5/terminal64.exe",
        )
        assert connector._path == "C:/Program Files/MetaTrader 5/terminal64.exe"


class TestMT5ConnectorConnection:
    """Test connection methods."""

    @pytest.mark.asyncio
    async def test_connect_success(
        self, connector: MT5Connector, mock_mt5: MagicMock, mock_account_info
    ) -> None:
        """Test successful connection."""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.last_error.return_value = (0, "")

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            result = await connector.connect()
            
            assert result is True
            assert connector.is_connected is True

    @pytest.mark.asyncio
    async def test_connect_auth_failure(
        self, connector: MT5Connector, mock_mt5: MagicMock
    ) -> None:
        """Test connection with invalid credentials."""
        mock_mt5.initialize.return_value = False
        mock_mt5.last_error.return_value = (10004, "Invalid account")

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            with pytest.raises(AuthenticationError):
                await connector.connect()

    @pytest.mark.asyncio
    async def test_connect_failure(
        self, connector: MT5Connector, mock_mt5: MagicMock
    ) -> None:
        """Test connection failure."""
        from tenacity import RetryError
        
        mock_mt5.initialize.return_value = False
        mock_mt5.last_error.return_value = (1, "Connection failed")

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            with pytest.raises((ConnectionError, RetryError)):
                await connector.connect()

    @pytest.mark.asyncio
    async def test_disconnect(
        self, connector: MT5Connector, mock_mt5: MagicMock, mock_account_info
    ) -> None:
        """Test disconnection."""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.shutdown.return_value = None

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()
            await connector.disconnect()
            
            assert connector.is_connected is False
            mock_mt5.shutdown.assert_called_once()



class TestMT5ConnectorBalance:
    """Test balance methods."""

    @pytest.mark.asyncio
    async def test_get_balance(
        self, connector: MT5Connector, mock_mt5: MagicMock, mock_account_info
    ) -> None:
        """Test getting account balance."""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()
            balance = await connector.get_balance()

            assert balance["USD"] == Decimal("10000.0")
            assert balance["USD_equity"] == Decimal("10500.0")
            assert balance["USD_margin"] == Decimal("1000.0")
            assert balance["USD_free_margin"] == Decimal("9500.0")

    @pytest.mark.asyncio
    async def test_get_balance_not_connected(self, connector: MT5Connector) -> None:
        """Test getting balance when not connected."""
        with pytest.raises(ConnectionError):
            await connector.get_balance()

    @pytest.mark.asyncio
    async def test_get_account_info(
        self, connector: MT5Connector, mock_mt5: MagicMock, mock_account_info
    ) -> None:
        """Test getting detailed account info."""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()
            info = await connector.get_account_info()

            assert info["login"] == 12345678
            assert info["server"] == "Exness-MT5Demo"
            assert info["currency"] == "USD"
            assert info["balance"] == Decimal("10000.0")
            assert info["leverage"] == 100


class TestMT5ConnectorPositions:
    """Test position methods."""

    @pytest.mark.asyncio
    async def test_get_positions_empty(
        self, connector: MT5Connector, mock_mt5: MagicMock, mock_account_info
    ) -> None:
        """Test getting positions when none exist."""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.positions_get.return_value = None
        mock_mt5.last_error.return_value = (0, "")

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()
            positions = await connector.get_positions()

            assert positions == []

    @pytest.mark.asyncio
    async def test_get_positions_with_data(
        self, connector: MT5Connector, mock_mt5: MagicMock, mock_account_info
    ) -> None:
        """Test getting positions with data."""
        mock_position = SimpleNamespace(
            symbol="EURUSD",
            type=0,  # BUY
            volume=0.1,
            price_open=1.08000,
            price_current=1.08500,
            profit=50.0,
            margin=100.0,
        )
        
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.positions_get.return_value = [mock_position]

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()
            positions = await connector.get_positions()

            assert len(positions) == 1
            pos = positions[0]
            assert pos.symbol == "EURUSD"
            assert pos.side == Side.BUY
            assert pos.quantity == Decimal("0.1")
            assert pos.entry_price == Decimal("1.08")
            assert pos.current_price == Decimal("1.085")
            assert pos.unrealized_pnl == Decimal("50.0")
            assert pos.platform == Platform.EXNESS

    @pytest.mark.asyncio
    async def test_get_positions_short(
        self, connector: MT5Connector, mock_mt5: MagicMock, mock_account_info
    ) -> None:
        """Test getting short positions."""
        mock_position = SimpleNamespace(
            symbol="GBPUSD",
            type=1,  # SELL
            volume=0.5,
            price_open=1.26000,
            price_current=1.25500,
            profit=250.0,
        )
        
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.positions_get.return_value = [mock_position]

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()
            positions = await connector.get_positions()

            assert len(positions) == 1
            assert positions[0].side == Side.SELL


class TestMT5ConnectorOrders:
    """Test order methods."""

    @pytest.mark.asyncio
    async def test_place_market_order(
        self, connector: MT5Connector, mock_mt5: MagicMock, 
        mock_account_info, mock_symbol_info
    ) -> None:
        """Test placing a market order."""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.symbol_info.return_value = mock_symbol_info
        mock_mt5.order_send.return_value = SimpleNamespace(
            retcode=mock_mt5.TRADE_RETCODE_DONE,
            order=123456,
            volume=0.1,
            price=1.08510,
            comment="",
        )

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()

            order = Order(
                id="",
                symbol="EURUSD",
                side=Side.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
            )
            result = await connector.place_order(order)

            assert result.id == "123456"
            assert result.status == OrderStatus.FILLED
            assert result.filled_quantity == Decimal("0.1")
            assert result.platform == Platform.EXNESS

    @pytest.mark.asyncio
    async def test_place_order_with_sl_tp(
        self, connector: MT5Connector, mock_mt5: MagicMock,
        mock_account_info, mock_symbol_info
    ) -> None:
        """Test placing order with stop loss and take profit."""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.symbol_info.return_value = mock_symbol_info
        mock_mt5.order_send.return_value = SimpleNamespace(
            retcode=mock_mt5.TRADE_RETCODE_DONE,
            order=123457,
            volume=0.1,
            price=1.08510,
            comment="",
        )

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()

            order = Order(
                id="",
                symbol="EURUSD",
                side=Side.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
                stop_loss=Decimal("1.08000"),
                take_profit=Decimal("1.09000"),
            )
            result = await connector.place_order(order)

            assert result.id == "123457"
            # Verify SL/TP were passed
            call_args = mock_mt5.order_send.call_args[0][0]
            assert call_args["sl"] == 1.08
            assert call_args["tp"] == 1.09

    @pytest.mark.asyncio
    async def test_place_order_insufficient_balance(
        self, connector: MT5Connector, mock_mt5: MagicMock,
        mock_account_info, mock_symbol_info
    ) -> None:
        """Test placing order with insufficient balance."""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.symbol_info.return_value = mock_symbol_info
        mock_mt5.order_send.return_value = SimpleNamespace(
            retcode=mock_mt5.TRADE_RETCODE_NO_MONEY,
            comment="Insufficient funds",
        )

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()

            order = Order(
                id="",
                symbol="EURUSD",
                side=Side.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
            )

            with pytest.raises(InsufficientBalanceError):
                await connector.place_order(order)

    @pytest.mark.asyncio
    async def test_place_order_invalid_volume(
        self, connector: MT5Connector, mock_mt5: MagicMock,
        mock_account_info, mock_symbol_info
    ) -> None:
        """Test placing order with invalid volume."""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.symbol_info.return_value = mock_symbol_info

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()

            order = Order(
                id="",
                symbol="EURUSD",
                side=Side.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.001"),  # Below min 0.01
            )

            with pytest.raises(InvalidOrderError):
                await connector.place_order(order)

    @pytest.mark.asyncio
    async def test_cancel_order_success(
        self, connector: MT5Connector, mock_mt5: MagicMock, mock_account_info
    ) -> None:
        """Test cancelling an order."""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.order_send.return_value = SimpleNamespace(
            retcode=mock_mt5.TRADE_RETCODE_DONE,
        )

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()
            result = await connector.cancel_order("123456", "EURUSD")

            assert result is True

    @pytest.mark.asyncio
    async def test_cancel_order_failure(
        self, connector: MT5Connector, mock_mt5: MagicMock, mock_account_info
    ) -> None:
        """Test cancelling non-existent order."""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.order_send.return_value = None

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()
            result = await connector.cancel_order("999999", "EURUSD")

            assert result is False



class TestMT5ConnectorClosePosition:
    """Test close position methods."""

    @pytest.mark.asyncio
    async def test_close_position_success(
        self, connector: MT5Connector, mock_mt5: MagicMock,
        mock_account_info, mock_symbol_info
    ) -> None:
        """Test closing a position."""
        mock_position = SimpleNamespace(
            symbol="EURUSD",
            type=0,  # BUY
            volume=0.1,
        )
        
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.positions_get.return_value = [mock_position]
        mock_mt5.symbol_info.return_value = mock_symbol_info
        mock_mt5.order_send.return_value = SimpleNamespace(
            retcode=mock_mt5.TRADE_RETCODE_DONE,
        )

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()
            result = await connector.close_position(123456)

            assert result is True

    @pytest.mark.asyncio
    async def test_close_position_not_found(
        self, connector: MT5Connector, mock_mt5: MagicMock, mock_account_info
    ) -> None:
        """Test closing non-existent position."""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.positions_get.return_value = None

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()

            with pytest.raises(PositionNotFoundError):
                await connector.close_position(999999)


class TestMT5ConnectorExchangeInfo:
    """Test exchange info methods."""

    @pytest.mark.asyncio
    async def test_get_exchange_info(
        self, connector: MT5Connector, mock_mt5: MagicMock,
        mock_account_info, mock_symbol_info
    ) -> None:
        """Test getting exchange info."""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.symbol_info.return_value = mock_symbol_info

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()
            info = await connector.get_exchange_info("EURUSD")

            assert info.symbol == "EURUSD"
            assert info.platform == Platform.EXNESS
            assert info.min_qty == Decimal("0.01")
            assert info.max_qty == Decimal("100.0")
            assert info.qty_step == Decimal("0.01")
            assert info.price_precision == 5

    @pytest.mark.asyncio
    async def test_get_exchange_info_caching(
        self, connector: MT5Connector, mock_mt5: MagicMock,
        mock_account_info, mock_symbol_info
    ) -> None:
        """Test exchange info caching."""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.symbol_info.return_value = mock_symbol_info

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()

            # First call
            await connector.get_exchange_info("EURUSD")
            # Second call should use cache
            await connector.get_exchange_info("EURUSD")

            # Should only call MT5 once
            assert mock_mt5.symbol_info.call_count == 1

    @pytest.mark.asyncio
    async def test_get_exchange_info_symbol_not_found(
        self, connector: MT5Connector, mock_mt5: MagicMock, mock_account_info
    ) -> None:
        """Test getting exchange info for non-existent symbol."""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.symbol_info.return_value = None

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()

            with pytest.raises(SymbolNotFoundError):
                await connector.get_exchange_info("INVALID")

    @pytest.mark.asyncio
    async def test_get_symbol_info(
        self, connector: MT5Connector, mock_mt5: MagicMock,
        mock_account_info, mock_symbol_info
    ) -> None:
        """Test getting detailed symbol info."""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.symbol_info.return_value = mock_symbol_info

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()
            info = await connector.get_symbol_info("EURUSD")

            assert info["symbol"] == "EURUSD"
            assert info["bid"] == Decimal("1.085")
            assert info["ask"] == Decimal("1.0851")
            assert info["spread"] == 10
            assert info["digits"] == 5


class TestMT5ConnectorOHLCV:
    """Test OHLCV methods."""

    @pytest.mark.asyncio
    async def test_get_ohlcv(
        self, connector: MT5Connector, mock_mt5: MagicMock, mock_account_info
    ) -> None:
        """Test getting OHLCV data."""
        # Create mock rates as list of dicts (simulating numpy structured array)
        mock_rates = [
            {'time': 1704067200, 'open': 1.08500, 'high': 1.08600, 'low': 1.08400, 
             'close': 1.08550, 'tick_volume': 1000},
            {'time': 1704070800, 'open': 1.08550, 'high': 1.08700, 'low': 1.08500, 
             'close': 1.08650, 'tick_volume': 1200},
        ]
        
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.copy_rates_from_pos.return_value = mock_rates

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()
            ohlcv = await connector.get_ohlcv("EURUSD", "1h", 2)

            assert len(ohlcv) == 2
            assert ohlcv[0].open == Decimal("1.085")
            assert ohlcv[0].high == Decimal("1.086")
            assert ohlcv[0].low == Decimal("1.084")
            assert ohlcv[0].close == Decimal("1.0855")
            assert ohlcv[0].volume == Decimal("1000")
            assert ohlcv[0].symbol == "EURUSD"
            assert ohlcv[0].timeframe == "1h"
            assert ohlcv[0].platform == Platform.EXNESS

    @pytest.mark.asyncio
    async def test_get_ohlcv_invalid_timeframe(
        self, connector: MT5Connector, mock_mt5: MagicMock, mock_account_info
    ) -> None:
        """Test getting OHLCV with invalid timeframe."""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()

            with pytest.raises(ValueError, match="Invalid timeframe"):
                await connector.get_ohlcv("EURUSD", "invalid", 100)


class TestMT5ConnectorTickerPrice:
    """Test ticker price methods."""

    @pytest.mark.asyncio
    async def test_get_ticker_price(
        self, connector: MT5Connector, mock_mt5: MagicMock, mock_account_info
    ) -> None:
        """Test getting ticker price."""
        mock_tick = SimpleNamespace(
            bid=1.08500,
            ask=1.08510,
        )
        
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.symbol_info_tick.return_value = mock_tick

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()
            price = await connector.get_ticker_price("EURUSD")

            # Mid price
            assert price == Decimal("1.08505")

    @pytest.mark.asyncio
    async def test_get_ticker_price_symbol_not_found(
        self, connector: MT5Connector, mock_mt5: MagicMock, mock_account_info
    ) -> None:
        """Test getting ticker price for non-existent symbol."""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.symbol_info_tick.return_value = None

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()

            with pytest.raises(SymbolNotFoundError):
                await connector.get_ticker_price("INVALID")


class TestMT5ConnectorErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_order_rejected_error(
        self, connector: MT5Connector, mock_mt5: MagicMock,
        mock_account_info, mock_symbol_info
    ) -> None:
        """Test order rejected error handling."""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.symbol_info.return_value = mock_symbol_info
        mock_mt5.order_send.return_value = SimpleNamespace(
            retcode=mock_mt5.TRADE_RETCODE_REJECT,
            comment="Order rejected by server",
        )

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()

            order = Order(
                id="",
                symbol="EURUSD",
                side=Side.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
            )

            with pytest.raises(OrderRejectedError):
                await connector.place_order(order)

    @pytest.mark.asyncio
    async def test_invalid_price_error(
        self, connector: MT5Connector, mock_mt5: MagicMock,
        mock_account_info, mock_symbol_info
    ) -> None:
        """Test invalid price error handling."""
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account_info
        mock_mt5.symbol_info.return_value = mock_symbol_info
        mock_mt5.order_send.return_value = SimpleNamespace(
            retcode=mock_mt5.TRADE_RETCODE_INVALID_PRICE,
            comment="Invalid price",
        )

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            connector._mt5 = mock_mt5
            await connector.connect()

            order = Order(
                id="",
                symbol="EURUSD",
                side=Side.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
                price=Decimal("0.5"),  # Invalid price
            )

            with pytest.raises(InvalidOrderError):
                await connector.place_order(order)
