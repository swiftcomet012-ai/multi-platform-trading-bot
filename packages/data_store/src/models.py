"""
SQLAlchemy ORM models for the trading platform.

All financial values stored as TEXT to preserve Decimal precision.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from packages.data_store.src.database import Base


class TradeModel(Base):
    """Trade record in database."""

    __tablename__ = "trades"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    entry_price: Mapped[str] = mapped_column(Text, nullable=False)  # Decimal as string
    exit_price: Mapped[str | None] = mapped_column(Text, nullable=True)
    quantity: Mapped[str] = mapped_column(Text, nullable=False)
    fees: Mapped[str] = mapped_column(Text, nullable=False, default="0")
    pnl: Mapped[str | None] = mapped_column(Text, nullable=True)
    pnl_pct: Mapped[float | None] = mapped_column(nullable=True)
    platform: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    strategy: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    signal_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_trades_platform_symbol", "platform", "symbol"),
        Index("ix_trades_created_at", "created_at"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "fees": self.fees,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "platform": self.platform,
            "strategy": self.strategy,
            "signal_id": self.signal_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
        }


class OHLCVModel(Base):
    """OHLCV candlestick data."""

    __tablename__ = "ohlcv"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    open: Mapped[str] = mapped_column(Text, nullable=False)
    high: Mapped[str] = mapped_column(Text, nullable=False)
    low: Mapped[str] = mapped_column(Text, nullable=False)
    close: Mapped[str] = mapped_column(Text, nullable=False)
    volume: Mapped[str] = mapped_column(Text, nullable=False)
    platform: Mapped[str] = mapped_column(String(20), nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", "platform", name="uq_ohlcv"),
        Index("ix_ohlcv_symbol_timeframe", "symbol", "timeframe"),
        Index("ix_ohlcv_timestamp", "timestamp"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "platform": self.platform,
        }


class SignalModel(Base):
    """Trading signal log."""

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(10), nullable=False)
    confidence: Mapped[float] = mapped_column(nullable=False)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    strategy: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    ai_provider: Mapped[str | None] = mapped_column(String(30), nullable=True)
    platform: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_signals_created_at", "created_at"),)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "action": self.action,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "strategy": self.strategy,
            "ai_provider": self.ai_provider,
            "platform": self.platform,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ExchangeInfoModel(Base):
    """Exchange trading rules cache."""

    __tablename__ = "exchange_info"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    platform: Mapped[str] = mapped_column(String(20), primary_key=True)
    min_qty: Mapped[str] = mapped_column(Text, nullable=False)
    max_qty: Mapped[str] = mapped_column(Text, nullable=False)
    qty_step: Mapped[str] = mapped_column(Text, nullable=False)
    qty_precision: Mapped[int] = mapped_column(Integer, nullable=False)
    price_precision: Mapped[int] = mapped_column(Integer, nullable=False)
    min_notional: Mapped[str] = mapped_column(Text, nullable=False)
    leverage_options: Mapped[str] = mapped_column(Text, nullable=False)  # JSON array
    maker_fee: Mapped[str] = mapped_column(Text, nullable=False)
    taker_fee: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "platform": self.platform,
            "min_qty": self.min_qty,
            "max_qty": self.max_qty,
            "qty_step": self.qty_step,
            "qty_precision": self.qty_precision,
            "price_precision": self.price_precision,
            "min_notional": self.min_notional,
            "leverage_options": self.leverage_options,
            "maker_fee": self.maker_fee,
            "taker_fee": self.taker_fee,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class StrategyPerformanceModel(Base):
    """Strategy performance metrics."""

    __tablename__ = "strategy_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    platform: Mapped[str | None] = mapped_column(String(20), nullable=True)
    symbol: Mapped[str | None] = mapped_column(String(20), nullable=True)
    total_pnl: Mapped[str] = mapped_column(Text, nullable=False)
    total_pnl_pct: Mapped[float] = mapped_column(nullable=False)
    win_rate: Mapped[float] = mapped_column(nullable=False)
    profit_factor: Mapped[float | None] = mapped_column(nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(nullable=True)
    sharpe_ratio: Mapped[float | None] = mapped_column(nullable=True)
    total_trades: Mapped[int] = mapped_column(Integer, nullable=False)
    period_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "strategy": self.strategy,
            "platform": self.platform,
            "symbol": self.symbol,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": self.total_pnl_pct,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "total_trades": self.total_trades,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AuditLogModel(Base):
    """Immutable audit log for trading decisions."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    action: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(30), nullable=False)
    entity_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    old_value: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    new_value: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    extra_data: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON (renamed from metadata - reserved)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_audit_log_created_at", "created_at"),)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "action": self.action,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "extra_data": self.extra_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
