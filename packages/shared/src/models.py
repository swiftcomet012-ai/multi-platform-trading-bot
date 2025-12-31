"""
Core data models using msgspec for high-performance serialization.

All financial values use Decimal for precision.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Self

import msgspec


class Platform(str, Enum):
    """Supported trading platforms."""

    BINANCE = "binance"
    EXNESS = "exness"


class Side(str, Enum):
    """Trade side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(str, Enum):
    """Order status."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class SignalAction(str, Enum):
    """AI signal actions."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class MarketRegime(str, Enum):
    """Market regime types."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"


# Custom Decimal encoder/decoder for msgspec
def dec_hook(type_: type, obj: object) -> Decimal:
    """Decode string to Decimal."""
    if type_ is Decimal:
        return Decimal(str(obj))
    raise TypeError(f"Cannot decode {type_}")


def enc_hook(obj: object) -> str:
    """Encode Decimal to string."""
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError(f"Cannot encode {type(obj)}")


class OHLCV(msgspec.Struct, frozen=True, kw_only=True):
    """OHLCV candlestick data."""

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    symbol: str = ""
    timeframe: str = "1h"
    platform: Platform = Platform.BINANCE

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": str(self.volume),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "platform": self.platform.value,
        }


class Signal(msgspec.Struct, frozen=True, kw_only=True):
    """Trading signal from AI or strategy."""

    symbol: str
    action: SignalAction
    confidence: float  # 0.0 - 1.0
    reasoning: str
    strategy: str
    timestamp: datetime
    platform: Platform = Platform.BINANCE
    metadata: dict | None = None

    def is_actionable(self, threshold: float = 0.7) -> bool:
        """Check if signal confidence meets threshold."""
        return self.action != SignalAction.HOLD and self.confidence >= threshold


class Order(msgspec.Struct, kw_only=True):
    """Order to be placed on exchange."""

    id: str
    symbol: str
    side: Side
    order_type: OrderType
    quantity: Decimal
    price: Decimal | None = None  # None for market orders
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    platform: Platform = Platform.BINANCE
    idempotency_key: str | None = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime | None = None
    filled_at: datetime | None = None
    filled_quantity: Decimal = Decimal("0")
    filled_price: Decimal | None = None
    fees: Decimal = Decimal("0")

    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.FILLED


class Trade(msgspec.Struct, frozen=True, kw_only=True):
    """Executed trade record."""

    id: str
    symbol: str
    side: Side
    entry_price: Decimal
    exit_price: Decimal | None = None
    quantity: Decimal
    fees: Decimal
    pnl: Decimal | None = None
    pnl_pct: float | None = None
    platform: Platform
    strategy: str
    signal_id: str | None = None
    created_at: datetime
    closed_at: datetime | None = None

    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.exit_price is None

    def calculate_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized P&L."""
        if self.side == Side.BUY:
            return (current_price - self.entry_price) * self.quantity - self.fees
        return (self.entry_price - current_price) * self.quantity - self.fees


class Position(msgspec.Struct, kw_only=True):
    """Current open position."""

    symbol: str
    side: Side
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_pct: float
    platform: Platform
    leverage: int = 1
    margin: Decimal | None = None
    liquidation_price: Decimal | None = None
    updated_at: datetime | None = None


class ExchangeInfo(msgspec.Struct, frozen=True, kw_only=True):
    """Exchange trading rules for a symbol."""

    symbol: str
    platform: Platform
    min_qty: Decimal
    max_qty: Decimal
    qty_step: Decimal
    qty_precision: int
    price_precision: int
    min_notional: Decimal
    leverage_options: list[int]
    maker_fee: Decimal
    taker_fee: Decimal
    updated_at: datetime

    def validate_quantity(self, qty: Decimal) -> tuple[bool, str]:
        """Validate order quantity against exchange rules."""
        if qty < self.min_qty:
            return False, f"Quantity {qty} below minimum {self.min_qty}"
        if qty > self.max_qty:
            return False, f"Quantity {qty} above maximum {self.max_qty}"
        # Check step size
        remainder = qty % self.qty_step
        if remainder != Decimal("0"):
            return False, f"Quantity {qty} not aligned to step {self.qty_step}"
        return True, "OK"

    def round_quantity(self, qty: Decimal) -> Decimal:
        """Round quantity to valid step size."""
        return (qty // self.qty_step) * self.qty_step


class PerformanceMetrics(msgspec.Struct, frozen=True, kw_only=True):
    """Trading performance metrics."""

    total_pnl: Decimal
    total_pnl_pct: float
    win_rate: float
    profit_factor: float
    avg_profit: Decimal
    avg_loss: Decimal
    max_drawdown: float
    max_drawdown_duration_days: int
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_duration_hours: float
    period_start: datetime
    period_end: datetime


# JSON encoder/decoder instances
encoder = msgspec.json.Encoder(enc_hook=enc_hook)
decoder = msgspec.json.Decoder()


def serialize(obj: msgspec.Struct) -> bytes:
    """Serialize msgspec struct to JSON bytes."""
    return encoder.encode(obj)


def deserialize[T: msgspec.Struct](data: bytes, type_: type[T]) -> T:
    """Deserialize JSON bytes to msgspec struct."""
    return msgspec.json.decode(data, type=type_, dec_hook=dec_hook)
