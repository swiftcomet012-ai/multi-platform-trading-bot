"""
Paper trading engine for simulating trades without real money.

Tracks virtual balance, positions, and P&L.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from packages.shared.src.logging import get_logger
from packages.shared.src.models import (
    Order,
    OrderStatus,
    OrderType,
    Platform,
    Position,
    Side,
    Trade,
)
from packages.shared.src.utils import generate_id

logger = get_logger(__name__)


@dataclass
class PaperTradingConfig:
    """Configuration for paper trading."""
    
    initial_balance: Decimal = Decimal("10000")  # Starting balance in quote currency
    maker_fee: Decimal = Decimal("0.001")  # 0.1%
    taker_fee: Decimal = Decimal("0.001")  # 0.1%
    slippage: Decimal = Decimal("0.0005")  # 0.05% slippage simulation


@dataclass
class PaperPosition:
    """Paper trading position."""
    
    symbol: str
    side: Side
    quantity: Decimal
    entry_price: Decimal
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    opened_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class PaperTradingEngine:
    """
    Paper trading engine that simulates real trading.
    
    Features:
    - Virtual balance tracking
    - Position management
    - Fee and slippage simulation
    - P&L calculation
    """
    
    def __init__(
        self,
        config: PaperTradingConfig | None = None,
        platform: Platform = Platform.BINANCE,
    ) -> None:
        """Initialize paper trading engine."""
        self.config = config or PaperTradingConfig()
        self.platform = platform
        
        # State
        self._balance: Decimal = self.config.initial_balance
        self._positions: dict[str, PaperPosition] = {}
        self._trades: list[Trade] = []
        self._orders: list[Order] = []
    
    @property
    def balance(self) -> Decimal:
        """Get current balance."""
        return self._balance
    
    @property
    def positions(self) -> list[Position]:
        """Get all open positions."""
        return [
            Position(
                symbol=pos.symbol,
                side=pos.side,
                quantity=pos.quantity,
                entry_price=pos.entry_price,
                current_price=pos.entry_price,  # Will be updated with real price
                unrealized_pnl=Decimal("0"),
                unrealized_pnl_pct=0.0,
                platform=self.platform,
            )
            for pos in self._positions.values()
        ]
    
    @property
    def trades(self) -> list[Trade]:
        """Get all executed trades."""
        return self._trades.copy()
    
    def get_balance(self) -> dict[str, Decimal]:
        """Get balance as dict (compatible with connector interface)."""
        return {"USDT": self._balance}
    
    def get_position(self, symbol: str) -> PaperPosition | None:
        """Get position for a symbol."""
        return self._positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if there's an open position for symbol."""
        return symbol in self._positions


    async def place_order(
        self,
        order: Order,
        current_price: Decimal,
    ) -> Order:
        """
        Execute a paper trade.
        
        Args:
            order: Order to execute.
            current_price: Current market price.
        
        Returns:
            Updated order with fill information.
        """
        # Simulate slippage
        if order.side == Side.BUY:
            fill_price = current_price * (1 + self.config.slippage)
        else:
            fill_price = current_price * (1 - self.config.slippage)
        
        fill_price = fill_price.quantize(Decimal("0.00000001"))
        
        # Calculate fees
        notional = fill_price * order.quantity
        fee = notional * self.config.taker_fee
        
        # Check balance for BUY
        if order.side == Side.BUY:
            required = notional + fee
            if required > self._balance:
                logger.warning(
                    "paper_trade_insufficient_balance",
                    required=str(required),
                    available=str(self._balance),
                )
                return Order(
                    id=generate_id(),
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type,
                    quantity=order.quantity,
                    price=order.price,
                    platform=self.platform,
                    status=OrderStatus.REJECTED,
                    created_at=datetime.now(UTC),
                )
        
        # Execute the trade
        order_id = generate_id()
        now = datetime.now(UTC)
        
        if order.side == Side.BUY:
            self._open_position(order, fill_price, fee)
        else:
            self._close_position(order, fill_price, fee)
        
        # Create filled order
        filled_order = Order(
            id=order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            platform=self.platform,
            idempotency_key=order.idempotency_key,
            status=OrderStatus.FILLED,
            created_at=now,
            filled_at=now,
            filled_quantity=order.quantity,
            filled_price=fill_price,
            fees=fee,
        )
        
        self._orders.append(filled_order)
        
        logger.info(
            "paper_trade_executed",
            order_id=order_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=str(order.quantity),
            fill_price=str(fill_price),
            fee=str(fee),
            balance=str(self._balance),
        )
        
        return filled_order
    
    def _open_position(
        self,
        order: Order,
        fill_price: Decimal,
        fee: Decimal,
    ) -> None:
        """Open or add to a position."""
        notional = fill_price * order.quantity
        
        # Deduct from balance
        self._balance -= (notional + fee)
        
        # Check if position exists
        if order.symbol in self._positions:
            # Average into existing position
            existing = self._positions[order.symbol]
            total_qty = existing.quantity + order.quantity
            avg_price = (
                (existing.entry_price * existing.quantity + fill_price * order.quantity)
                / total_qty
            )
            existing.quantity = total_qty
            existing.entry_price = avg_price
        else:
            # Create new position
            self._positions[order.symbol] = PaperPosition(
                symbol=order.symbol,
                side=Side.BUY,
                quantity=order.quantity,
                entry_price=fill_price,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
            )
    
    def _close_position(
        self,
        order: Order,
        fill_price: Decimal,
        fee: Decimal,
    ) -> None:
        """Close or reduce a position."""
        if order.symbol not in self._positions:
            logger.warning("paper_trade_no_position", symbol=order.symbol)
            return
        
        position = self._positions[order.symbol]
        close_qty = min(order.quantity, position.quantity)
        
        # Calculate P&L
        if position.side == Side.BUY:
            pnl = (fill_price - position.entry_price) * close_qty - fee
        else:
            pnl = (position.entry_price - fill_price) * close_qty - fee
        
        # Add to balance
        notional = fill_price * close_qty
        self._balance += (notional + pnl)
        
        # Record trade
        trade = Trade(
            id=generate_id(),
            symbol=order.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=fill_price,
            quantity=close_qty,
            fees=fee,
            pnl=pnl,
            pnl_pct=float(pnl / (position.entry_price * close_qty) * 100),
            platform=self.platform,
            strategy="paper_trading",
            created_at=position.opened_at,
            closed_at=datetime.now(UTC),
        )
        self._trades.append(trade)
        
        # Update or remove position
        position.quantity -= close_qty
        if position.quantity <= 0:
            del self._positions[order.symbol]
        
        logger.info(
            "paper_trade_closed",
            symbol=order.symbol,
            pnl=str(pnl),
            pnl_pct=f"{trade.pnl_pct:.2f}%",
            balance=str(self._balance),
        )
    
    def check_stop_loss_take_profit(
        self,
        symbol: str,
        current_price: Decimal,
    ) -> bool:
        """
        Check if stop-loss or take-profit is hit.
        
        Returns:
            True if position was closed.
        """
        if symbol not in self._positions:
            return False
        
        position = self._positions[symbol]
        
        # Check stop-loss
        if position.stop_loss:
            if position.side == Side.BUY and current_price <= position.stop_loss:
                logger.info("paper_trade_stop_loss_hit", symbol=symbol, price=str(current_price))
                return True
            if position.side == Side.SELL and current_price >= position.stop_loss:
                logger.info("paper_trade_stop_loss_hit", symbol=symbol, price=str(current_price))
                return True
        
        # Check take-profit
        if position.take_profit:
            if position.side == Side.BUY and current_price >= position.take_profit:
                logger.info("paper_trade_take_profit_hit", symbol=symbol, price=str(current_price))
                return True
            if position.side == Side.SELL and current_price <= position.take_profit:
                logger.info("paper_trade_take_profit_hit", symbol=symbol, price=str(current_price))
                return True
        
        return False
    
    def get_stats(self) -> dict[str, Any]:
        """Get trading statistics."""
        if not self._trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": Decimal("0"),
                "total_pnl_pct": 0.0,
                "current_balance": self._balance,
                "initial_balance": self.config.initial_balance,
            }
        
        winning = [t for t in self._trades if t.pnl and t.pnl > 0]
        losing = [t for t in self._trades if t.pnl and t.pnl < 0]
        total_pnl = sum(t.pnl for t in self._trades if t.pnl) or Decimal("0")
        
        return {
            "total_trades": len(self._trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(self._trades) * 100 if self._trades else 0.0,
            "total_pnl": total_pnl,
            "total_pnl_pct": float(total_pnl / self.config.initial_balance * 100),
            "current_balance": self._balance,
            "initial_balance": self.config.initial_balance,
        }
