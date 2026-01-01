"""
Repository pattern for data access.

Provides CRUD operations with Decimal precision preservation.
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from typing import Generic, TypeVar

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from packages.data_store.src.models import (
    AuditLogModel,
    ExchangeInfoModel,
    OHLCVModel,
    SignalModel,
    StrategyPerformanceModel,
    TradeModel,
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

T = TypeVar("T")


class BaseRepository(Generic[T]):
    """Base repository with common CRUD operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session


class TradeRepository(BaseRepository[Trade]):
    """Repository for Trade entities."""

    async def create(self, trade: Trade) -> Trade:
        """Create a new trade record."""
        model = TradeModel(
            id=trade.id,
            symbol=trade.symbol,
            side=trade.side.value,
            entry_price=str(trade.entry_price),
            exit_price=str(trade.exit_price) if trade.exit_price else None,
            quantity=str(trade.quantity),
            fees=str(trade.fees),
            pnl=str(trade.pnl) if trade.pnl else None,
            pnl_pct=trade.pnl_pct,
            platform=trade.platform.value,
            strategy=trade.strategy,
            signal_id=trade.signal_id,
            created_at=trade.created_at,
            closed_at=trade.closed_at,
        )
        self._session.add(model)
        await self._session.flush()
        return trade

    async def get_by_id(self, trade_id: str) -> Trade | None:
        """Get trade by ID."""
        result = await self._session.execute(
            select(TradeModel).where(TradeModel.id == trade_id)
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def get_open_trades(self, platform: Platform | None = None) -> list[Trade]:
        """Get all open trades (no exit_price)."""
        query = select(TradeModel).where(TradeModel.exit_price.is_(None))
        if platform:
            query = query.where(TradeModel.platform == platform.value)
        result = await self._session.execute(query)
        return [self._to_entity(m) for m in result.scalars().all()]

    async def get_by_symbol(
        self,
        symbol: str,
        platform: Platform | None = None,
        limit: int = 100,
    ) -> list[Trade]:
        """Get trades by symbol."""
        query = select(TradeModel).where(TradeModel.symbol == symbol)
        if platform:
            query = query.where(TradeModel.platform == platform.value)
        query = query.order_by(TradeModel.created_at.desc()).limit(limit)
        result = await self._session.execute(query)
        return [self._to_entity(m) for m in result.scalars().all()]

    async def get_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        platform: Platform | None = None,
    ) -> list[Trade]:
        """Get trades within date range."""
        query = select(TradeModel).where(
            TradeModel.created_at >= start_date,
            TradeModel.created_at <= end_date,
        )
        if platform:
            query = query.where(TradeModel.platform == platform.value)
        query = query.order_by(TradeModel.created_at.desc())
        result = await self._session.execute(query)
        return [self._to_entity(m) for m in result.scalars().all()]

    async def close_trade(
        self,
        trade_id: str,
        exit_price: Decimal,
        pnl: Decimal,
        pnl_pct: float,
        closed_at: datetime,
    ) -> bool:
        """Close a trade with exit price and P&L."""
        result = await self._session.execute(
            update(TradeModel)
            .where(TradeModel.id == trade_id)
            .values(
                exit_price=str(exit_price),
                pnl=str(pnl),
                pnl_pct=pnl_pct,
                closed_at=closed_at,
            )
        )
        return result.rowcount > 0

    async def delete(self, trade_id: str) -> bool:
        """Delete a trade."""
        result = await self._session.execute(
            delete(TradeModel).where(TradeModel.id == trade_id)
        )
        return result.rowcount > 0

    def _to_entity(self, model: TradeModel) -> Trade:
        """Convert ORM model to domain entity."""
        return Trade(
            id=model.id,
            symbol=model.symbol,
            side=Side(model.side),
            entry_price=Decimal(model.entry_price),
            exit_price=Decimal(model.exit_price) if model.exit_price else None,
            quantity=Decimal(model.quantity),
            fees=Decimal(model.fees),
            pnl=Decimal(model.pnl) if model.pnl else None,
            pnl_pct=model.pnl_pct,
            platform=Platform(model.platform),
            strategy=model.strategy,
            signal_id=model.signal_id,
            created_at=model.created_at,
            closed_at=model.closed_at,
        )


class OHLCVRepository(BaseRepository[OHLCV]):
    """Repository for OHLCV data."""

    async def create(self, ohlcv: OHLCV) -> OHLCV:
        """Create a new OHLCV record."""
        model = OHLCVModel(
            symbol=ohlcv.symbol,
            timeframe=ohlcv.timeframe,
            timestamp=ohlcv.timestamp,
            open=str(ohlcv.open),
            high=str(ohlcv.high),
            low=str(ohlcv.low),
            close=str(ohlcv.close),
            volume=str(ohlcv.volume),
            platform=ohlcv.platform.value,
        )
        self._session.add(model)
        await self._session.flush()
        return ohlcv

    async def create_many(self, ohlcv_list: list[OHLCV]) -> int:
        """Bulk insert OHLCV records (upsert)."""
        count = 0
        for ohlcv in ohlcv_list:
            # Check if exists
            result = await self._session.execute(
                select(OHLCVModel).where(
                    OHLCVModel.symbol == ohlcv.symbol,
                    OHLCVModel.timeframe == ohlcv.timeframe,
                    OHLCVModel.timestamp == ohlcv.timestamp,
                    OHLCVModel.platform == ohlcv.platform.value,
                )
            )
            existing = result.scalar_one_or_none()
            if not existing:
                await self.create(ohlcv)
                count += 1
        return count

    async def get_by_symbol(
        self,
        symbol: str,
        timeframe: str,
        platform: Platform,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> list[OHLCV]:
        """Get OHLCV data for a symbol."""
        query = select(OHLCVModel).where(
            OHLCVModel.symbol == symbol,
            OHLCVModel.timeframe == timeframe,
            OHLCVModel.platform == platform.value,
        )
        if start_time:
            query = query.where(OHLCVModel.timestamp >= start_time)
        if end_time:
            query = query.where(OHLCVModel.timestamp <= end_time)
        query = query.order_by(OHLCVModel.timestamp.asc()).limit(limit)
        result = await self._session.execute(query)
        return [self._to_entity(m) for m in result.scalars().all()]

    async def get_latest(
        self,
        symbol: str,
        timeframe: str,
        platform: Platform,
    ) -> OHLCV | None:
        """Get the latest OHLCV record."""
        result = await self._session.execute(
            select(OHLCVModel)
            .where(
                OHLCVModel.symbol == symbol,
                OHLCVModel.timeframe == timeframe,
                OHLCVModel.platform == platform.value,
            )
            .order_by(OHLCVModel.timestamp.desc())
            .limit(1)
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def delete_old(self, before: datetime) -> int:
        """Delete OHLCV records older than specified date."""
        result = await self._session.execute(
            delete(OHLCVModel).where(OHLCVModel.timestamp < before)
        )
        return result.rowcount

    def _to_entity(self, model: OHLCVModel) -> OHLCV:
        """Convert ORM model to domain entity."""
        return OHLCV(
            timestamp=model.timestamp,
            open=Decimal(model.open),
            high=Decimal(model.high),
            low=Decimal(model.low),
            close=Decimal(model.close),
            volume=Decimal(model.volume),
            symbol=model.symbol,
            timeframe=model.timeframe,
            platform=Platform(model.platform),
        )


class SignalRepository(BaseRepository[Signal]):
    """Repository for Signal entities."""

    async def create(self, signal: Signal) -> Signal:
        """Create a new signal record."""
        model = SignalModel(
            symbol=signal.symbol,
            action=signal.action.value,
            confidence=signal.confidence,
            reasoning=signal.reasoning,
            strategy=signal.strategy,
            ai_provider=signal.metadata.get("ai_provider") if signal.metadata else None,
            platform=signal.platform.value,
            created_at=signal.timestamp,
        )
        self._session.add(model)
        await self._session.flush()
        return signal

    async def get_recent(
        self,
        symbol: str | None = None,
        platform: Platform | None = None,
        limit: int = 100,
    ) -> list[Signal]:
        """Get recent signals."""
        query = select(SignalModel)
        if symbol:
            query = query.where(SignalModel.symbol == symbol)
        if platform:
            query = query.where(SignalModel.platform == platform.value)
        query = query.order_by(SignalModel.created_at.desc()).limit(limit)
        result = await self._session.execute(query)
        return [self._to_entity(m) for m in result.scalars().all()]

    def _to_entity(self, model: SignalModel) -> Signal:
        """Convert ORM model to domain entity."""
        return Signal(
            symbol=model.symbol,
            action=SignalAction(model.action),
            confidence=model.confidence,
            reasoning=model.reasoning or "",
            strategy=model.strategy,
            timestamp=model.created_at,
            platform=Platform(model.platform),
            metadata={"ai_provider": model.ai_provider} if model.ai_provider else None,
        )


class ExchangeInfoRepository(BaseRepository[ExchangeInfo]):
    """Repository for ExchangeInfo entities."""

    async def upsert(self, info: ExchangeInfo) -> ExchangeInfo:
        """Insert or update exchange info."""
        # Check if exists
        result = await self._session.execute(
            select(ExchangeInfoModel).where(
                ExchangeInfoModel.symbol == info.symbol,
                ExchangeInfoModel.platform == info.platform.value,
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            # Update
            await self._session.execute(
                update(ExchangeInfoModel)
                .where(
                    ExchangeInfoModel.symbol == info.symbol,
                    ExchangeInfoModel.platform == info.platform.value,
                )
                .values(
                    min_qty=str(info.min_qty),
                    max_qty=str(info.max_qty),
                    qty_step=str(info.qty_step),
                    qty_precision=info.qty_precision,
                    price_precision=info.price_precision,
                    min_notional=str(info.min_notional),
                    leverage_options=json.dumps(info.leverage_options),
                    maker_fee=str(info.maker_fee),
                    taker_fee=str(info.taker_fee),
                )
            )
        else:
            # Insert
            model = ExchangeInfoModel(
                symbol=info.symbol,
                platform=info.platform.value,
                min_qty=str(info.min_qty),
                max_qty=str(info.max_qty),
                qty_step=str(info.qty_step),
                qty_precision=info.qty_precision,
                price_precision=info.price_precision,
                min_notional=str(info.min_notional),
                leverage_options=json.dumps(info.leverage_options),
                maker_fee=str(info.maker_fee),
                taker_fee=str(info.taker_fee),
            )
            self._session.add(model)

        await self._session.flush()
        return info

    async def get(self, symbol: str, platform: Platform) -> ExchangeInfo | None:
        """Get exchange info for a symbol."""
        result = await self._session.execute(
            select(ExchangeInfoModel).where(
                ExchangeInfoModel.symbol == symbol,
                ExchangeInfoModel.platform == platform.value,
            )
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def get_all(self, platform: Platform) -> list[ExchangeInfo]:
        """Get all exchange info for a platform."""
        result = await self._session.execute(
            select(ExchangeInfoModel).where(
                ExchangeInfoModel.platform == platform.value
            )
        )
        return [self._to_entity(m) for m in result.scalars().all()]

    def _to_entity(self, model: ExchangeInfoModel) -> ExchangeInfo:
        """Convert ORM model to domain entity."""
        return ExchangeInfo(
            symbol=model.symbol,
            platform=Platform(model.platform),
            min_qty=Decimal(model.min_qty),
            max_qty=Decimal(model.max_qty),
            qty_step=Decimal(model.qty_step),
            qty_precision=model.qty_precision,
            price_precision=model.price_precision,
            min_notional=Decimal(model.min_notional),
            leverage_options=json.loads(model.leverage_options),
            maker_fee=Decimal(model.maker_fee),
            taker_fee=Decimal(model.taker_fee),
            updated_at=model.updated_at,
        )


class AuditLogRepository(BaseRepository[None]):
    """Repository for audit logs (append-only)."""

    async def log(
        self,
        action: str,
        entity_type: str,
        entity_id: str | None = None,
        old_value: dict | None = None,
        new_value: dict | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Create an audit log entry."""
        model = AuditLogModel(
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            old_value=json.dumps(old_value) if old_value else None,
            new_value=json.dumps(new_value) if new_value else None,
            metadata=json.dumps(metadata) if metadata else None,
        )
        self._session.add(model)
        await self._session.flush()
        return model.id

    async def get_by_entity(
        self,
        entity_type: str,
        entity_id: str,
        limit: int = 100,
    ) -> list[dict]:
        """Get audit logs for an entity."""
        result = await self._session.execute(
            select(AuditLogModel)
            .where(
                AuditLogModel.entity_type == entity_type,
                AuditLogModel.entity_id == entity_id,
            )
            .order_by(AuditLogModel.created_at.desc())
            .limit(limit)
        )
        return [m.to_dict() for m in result.scalars().all()]

    async def get_recent(self, limit: int = 100) -> list[dict]:
        """Get recent audit logs."""
        result = await self._session.execute(
            select(AuditLogModel)
            .order_by(AuditLogModel.created_at.desc())
            .limit(limit)
        )
        return [m.to_dict() for m in result.scalars().all()]
