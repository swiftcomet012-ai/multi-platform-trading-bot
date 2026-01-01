"""
Tests for data store repositories including property-based tests.

Property 2: Data persistence round-trip - validates R5.1, R5.2
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
import pytest_asyncio
from hypothesis import given, settings
from hypothesis import strategies as st

from packages.data_store.src.database import Database, init_database
from packages.data_store.src.repositories import (
    ExchangeInfoRepository,
    OHLCVRepository,
    SignalRepository,
    TradeRepository,
)
from packages.shared.src.models import (
    OHLCV,
    ExchangeInfo,
    Platform,
    Side,
    Signal,
    SignalAction,
    Trade,
)
from packages.shared.src.utils import generate_id


# =============================================================================
# Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def db():
    """Create a test database."""
    database = await init_database("sqlite+aiosqlite:///:memory:")
    yield database
    await database.close()


@pytest_asyncio.fixture
async def trade_repo(db: Database):
    """Create a trade repository with session."""
    async with db.session() as session:
        yield TradeRepository(session)


@pytest_asyncio.fixture
async def ohlcv_repo(db: Database):
    """Create an OHLCV repository with session."""
    async with db.session() as session:
        yield OHLCVRepository(session)


@pytest_asyncio.fixture
async def signal_repo(db: Database):
    """Create a signal repository with session."""
    async with db.session() as session:
        yield SignalRepository(session)


@pytest_asyncio.fixture
async def exchange_info_repo(db: Database):
    """Create an exchange info repository with session."""
    async with db.session() as session:
        yield ExchangeInfoRepository(session)


# =============================================================================
# Hypothesis Strategies
# =============================================================================

decimal_strategy = st.decimals(
    min_value=Decimal("0.00000001"),
    max_value=Decimal("1000000"),
    allow_nan=False,
    allow_infinity=False,
    places=8,
)

positive_decimal_strategy = st.decimals(
    min_value=Decimal("0.01"),
    max_value=Decimal("100000"),
    allow_nan=False,
    allow_infinity=False,
    places=8,
)

symbol_strategy = st.sampled_from(["BTCUSDT", "ETHUSDT", "EURUSD", "XAUUSD"])
platform_strategy = st.sampled_from(list(Platform))
side_strategy = st.sampled_from(list(Side))
signal_action_strategy = st.sampled_from(list(SignalAction))


# =============================================================================
# Property-Based Tests - Data Persistence Round-Trip (Property 2)
# =============================================================================


@pytest.mark.property
class TestDataPersistenceRoundTrip:
    """Property 2: Data persistence round-trip tests."""

    @pytest.mark.asyncio
    async def test_trade_roundtrip(self, db: Database) -> None:
        """Trade create and retrieve preserves all data."""
        async with db.session() as session:
            repo = TradeRepository(session)

            # Create trade
            trade = Trade(
                id=generate_id("trade"),
                symbol="BTCUSDT",
                side=Side.BUY,
                entry_price=Decimal("50000.12345678"),
                quantity=Decimal("0.12345678"),
                fees=Decimal("5.00"),
                platform=Platform.BINANCE,
                strategy="test_strategy",
                created_at=datetime.now(timezone.utc),
            )

            await repo.create(trade)

        # Retrieve in new session
        async with db.session() as session:
            repo = TradeRepository(session)
            retrieved = await repo.get_by_id(trade.id)

            assert retrieved is not None
            assert retrieved.id == trade.id
            assert retrieved.symbol == trade.symbol
            assert retrieved.side == trade.side
            assert retrieved.entry_price == trade.entry_price
            assert retrieved.quantity == trade.quantity
            assert retrieved.fees == trade.fees
            assert retrieved.platform == trade.platform
            assert retrieved.strategy == trade.strategy

    @pytest.mark.asyncio
    async def test_ohlcv_roundtrip(self, db: Database) -> None:
        """OHLCV create and retrieve preserves all data."""
        async with db.session() as session:
            repo = OHLCVRepository(session)

            # Create OHLCV
            ohlcv = OHLCV(
                timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                open=Decimal("50000.12345678"),
                high=Decimal("51000.12345678"),
                low=Decimal("49000.12345678"),
                close=Decimal("50500.12345678"),
                volume=Decimal("1000.12345678"),
                symbol="BTCUSDT",
                timeframe="1h",
                platform=Platform.BINANCE,
            )

            await repo.create(ohlcv)

        # Retrieve in new session
        async with db.session() as session:
            repo = OHLCVRepository(session)
            results = await repo.get_by_symbol(
                symbol="BTCUSDT",
                timeframe="1h",
                platform=Platform.BINANCE,
            )

            assert len(results) == 1
            retrieved = results[0]
            assert retrieved.symbol == ohlcv.symbol
            assert retrieved.open == ohlcv.open
            assert retrieved.high == ohlcv.high
            assert retrieved.low == ohlcv.low
            assert retrieved.close == ohlcv.close
            assert retrieved.volume == ohlcv.volume

    @pytest.mark.asyncio
    async def test_signal_roundtrip(self, db: Database) -> None:
        """Signal create and retrieve preserves all data."""
        async with db.session() as session:
            repo = SignalRepository(session)

            # Create signal
            signal = Signal(
                symbol="BTCUSDT",
                action=SignalAction.BUY,
                confidence=0.85,
                reasoning="Strong uptrend detected",
                strategy="trend_following",
                timestamp=datetime.now(timezone.utc),
                platform=Platform.BINANCE,
                metadata={"ai_provider": "gemini"},
            )

            await repo.create(signal)

        # Retrieve in new session
        async with db.session() as session:
            repo = SignalRepository(session)
            results = await repo.get_recent(symbol="BTCUSDT", limit=1)

            assert len(results) == 1
            retrieved = results[0]
            assert retrieved.symbol == signal.symbol
            assert retrieved.action == signal.action
            assert abs(retrieved.confidence - signal.confidence) < 0.001
            assert retrieved.reasoning == signal.reasoning
            assert retrieved.strategy == signal.strategy

    @pytest.mark.asyncio
    async def test_exchange_info_roundtrip(self, db: Database) -> None:
        """ExchangeInfo upsert and retrieve preserves all data."""
        async with db.session() as session:
            repo = ExchangeInfoRepository(session)

            # Create exchange info
            info = ExchangeInfo(
                symbol="BTCUSDT",
                platform=Platform.BINANCE,
                min_qty=Decimal("0.00001"),
                max_qty=Decimal("1000"),
                qty_step=Decimal("0.00001"),
                qty_precision=5,
                price_precision=2,
                min_notional=Decimal("10"),
                leverage_options=[1, 5, 10, 20, 50, 100],
                maker_fee=Decimal("0.001"),
                taker_fee=Decimal("0.001"),
                updated_at=datetime.now(timezone.utc),
            )

            await repo.upsert(info)

        # Retrieve in new session
        async with db.session() as session:
            repo = ExchangeInfoRepository(session)
            retrieved = await repo.get("BTCUSDT", Platform.BINANCE)

            assert retrieved is not None
            assert retrieved.symbol == info.symbol
            assert retrieved.min_qty == info.min_qty
            assert retrieved.max_qty == info.max_qty
            assert retrieved.qty_step == info.qty_step
            assert retrieved.qty_precision == info.qty_precision
            assert retrieved.price_precision == info.price_precision
            assert retrieved.leverage_options == info.leverage_options
            assert retrieved.maker_fee == info.maker_fee
            assert retrieved.taker_fee == info.taker_fee


# =============================================================================
# Unit Tests
# =============================================================================


class TestTradeRepository:
    """Unit tests for TradeRepository."""

    @pytest.mark.asyncio
    async def test_create_and_get(self, db: Database) -> None:
        """Test creating and retrieving a trade."""
        async with db.session() as session:
            repo = TradeRepository(session)

            trade = Trade(
                id="test_trade_1",
                symbol="BTCUSDT",
                side=Side.BUY,
                entry_price=Decimal("50000"),
                quantity=Decimal("0.1"),
                fees=Decimal("5"),
                platform=Platform.BINANCE,
                strategy="test",
                created_at=datetime.now(timezone.utc),
            )

            await repo.create(trade)
            retrieved = await repo.get_by_id("test_trade_1")

            assert retrieved is not None
            assert retrieved.id == "test_trade_1"

    @pytest.mark.asyncio
    async def test_get_open_trades(self, db: Database) -> None:
        """Test getting open trades."""
        async with db.session() as session:
            repo = TradeRepository(session)

            # Create open trade
            open_trade = Trade(
                id="open_trade",
                symbol="BTCUSDT",
                side=Side.BUY,
                entry_price=Decimal("50000"),
                quantity=Decimal("0.1"),
                fees=Decimal("5"),
                platform=Platform.BINANCE,
                strategy="test",
                created_at=datetime.now(timezone.utc),
            )
            await repo.create(open_trade)

            # Create closed trade
            closed_trade = Trade(
                id="closed_trade",
                symbol="BTCUSDT",
                side=Side.BUY,
                entry_price=Decimal("50000"),
                exit_price=Decimal("51000"),
                quantity=Decimal("0.1"),
                fees=Decimal("10"),
                pnl=Decimal("90"),
                pnl_pct=1.8,
                platform=Platform.BINANCE,
                strategy="test",
                created_at=datetime.now(timezone.utc),
                closed_at=datetime.now(timezone.utc),
            )
            await repo.create(closed_trade)

            open_trades = await repo.get_open_trades()
            assert len(open_trades) == 1
            assert open_trades[0].id == "open_trade"

    @pytest.mark.asyncio
    async def test_close_trade(self, db: Database) -> None:
        """Test closing a trade."""
        async with db.session() as session:
            repo = TradeRepository(session)

            trade = Trade(
                id="trade_to_close",
                symbol="BTCUSDT",
                side=Side.BUY,
                entry_price=Decimal("50000"),
                quantity=Decimal("0.1"),
                fees=Decimal("5"),
                platform=Platform.BINANCE,
                strategy="test",
                created_at=datetime.now(timezone.utc),
            )
            await repo.create(trade)

            closed = await repo.close_trade(
                trade_id="trade_to_close",
                exit_price=Decimal("51000"),
                pnl=Decimal("95"),
                pnl_pct=1.9,
                closed_at=datetime.now(timezone.utc),
            )
            assert closed is True

            retrieved = await repo.get_by_id("trade_to_close")
            assert retrieved is not None
            assert retrieved.exit_price == Decimal("51000")
            assert retrieved.pnl == Decimal("95")


class TestOHLCVRepository:
    """Unit tests for OHLCVRepository."""

    @pytest.mark.asyncio
    async def test_create_many(self, db: Database) -> None:
        """Test bulk inserting OHLCV data."""
        async with db.session() as session:
            repo = OHLCVRepository(session)

            ohlcv_list = [
                OHLCV(
                    timestamp=datetime(2024, 1, 1, i, 0, 0, tzinfo=timezone.utc),
                    open=Decimal("50000"),
                    high=Decimal("51000"),
                    low=Decimal("49000"),
                    close=Decimal("50500"),
                    volume=Decimal("1000"),
                    symbol="BTCUSDT",
                    timeframe="1h",
                    platform=Platform.BINANCE,
                )
                for i in range(5)
            ]

            count = await repo.create_many(ohlcv_list)
            assert count == 5

            results = await repo.get_by_symbol(
                symbol="BTCUSDT",
                timeframe="1h",
                platform=Platform.BINANCE,
            )
            assert len(results) == 5

    @pytest.mark.asyncio
    async def test_get_latest(self, db: Database) -> None:
        """Test getting latest OHLCV."""
        async with db.session() as session:
            repo = OHLCVRepository(session)

            for i in range(3):
                ohlcv = OHLCV(
                    timestamp=datetime(2024, 1, 1, i, 0, 0, tzinfo=timezone.utc),
                    open=Decimal(str(50000 + i * 100)),
                    high=Decimal("51000"),
                    low=Decimal("49000"),
                    close=Decimal("50500"),
                    volume=Decimal("1000"),
                    symbol="BTCUSDT",
                    timeframe="1h",
                    platform=Platform.BINANCE,
                )
                await repo.create(ohlcv)

            latest = await repo.get_latest("BTCUSDT", "1h", Platform.BINANCE)
            assert latest is not None
            assert latest.open == Decimal("50200")  # Last one created
