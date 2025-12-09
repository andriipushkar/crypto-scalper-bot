"""
Paper Trading Mode

Simulated trading for testing strategies without real money.
"""
import asyncio
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import uuid
import random

from loguru import logger


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


@dataclass
class PaperOrder:
    """Paper trading order."""
    id: str
    symbol: str
    side: str  # buy, sell
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal("0")
    avg_fill_price: Decimal = Decimal("0")
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    client_order_id: Optional[str] = None


@dataclass
class PaperPosition:
    """Paper trading position."""
    symbol: str
    side: str  # long, short
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    leverage: int = 1
    liquidation_price: Optional[Decimal] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PaperTrade:
    """Completed paper trade."""
    id: str
    symbol: str
    side: str
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    pnl: Decimal
    pnl_pct: Decimal
    entry_time: datetime
    exit_time: datetime
    commission: Decimal


class PaperTradingEngine:
    """Paper trading simulation engine."""

    def __init__(
        self,
        initial_balance: Decimal = Decimal("10000"),
        commission_rate: Decimal = Decimal("0.001"),
        slippage_rate: Decimal = Decimal("0.0005"),
        leverage: int = 1,
        simulate_latency: bool = False,
        latency_ms: int = 50,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.default_leverage = leverage
        self.simulate_latency = simulate_latency
        self.latency_ms = latency_ms

        # State
        self.positions: Dict[str, PaperPosition] = {}
        self.orders: Dict[str, PaperOrder] = {}
        self.trades: List[PaperTrade] = []
        self.order_history: List[PaperOrder] = []

        # Market data
        self.current_prices: Dict[str, Decimal] = {}
        self.bid_prices: Dict[str, Decimal] = {}
        self.ask_prices: Dict[str, Decimal] = {}

        # Callbacks
        self.on_order_filled: Optional[Callable] = None
        self.on_position_closed: Optional[Callable] = None
        self.on_trade_completed: Optional[Callable] = None

        # Stats
        self.stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "total_trades": 0,
            "total_commission": Decimal("0"),
        }

    @property
    def equity(self) -> Decimal:
        """Total equity (balance + unrealized P&L)."""
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return self.balance + unrealized

    @property
    def available_balance(self) -> Decimal:
        """Available balance for new orders."""
        margin_used = sum(
            (p.quantity * p.entry_price) / p.leverage
            for p in self.positions.values()
        )
        return self.balance - margin_used

    def update_price(self, symbol: str, price: Decimal, bid: Optional[Decimal] = None, ask: Optional[Decimal] = None):
        """Update market price for symbol."""
        self.current_prices[symbol] = price
        self.bid_prices[symbol] = bid or price * (1 - self.slippage_rate)
        self.ask_prices[symbol] = ask or price * (1 + self.slippage_rate)

        # Update positions
        if symbol in self.positions:
            self._update_position(symbol, price)

        # Check pending orders
        self._check_pending_orders(symbol)

    def _update_position(self, symbol: str, price: Decimal):
        """Update position with new price."""
        pos = self.positions.get(symbol)
        if not pos:
            return

        pos.current_price = price

        # Calculate unrealized P&L
        if pos.side == "long":
            pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
        else:
            pos.unrealized_pnl = (pos.entry_price - price) * pos.quantity

        # Check liquidation
        if pos.liquidation_price:
            if (pos.side == "long" and price <= pos.liquidation_price) or \
               (pos.side == "short" and price >= pos.liquidation_price):
                logger.warning(f"Position {symbol} liquidated!")
                self._liquidate_position(symbol)

    def _check_pending_orders(self, symbol: str):
        """Check and fill pending orders."""
        price = self.current_prices.get(symbol)
        if not price:
            return

        for order_id, order in list(self.orders.items()):
            if order.symbol != symbol or order.status != OrderStatus.PENDING:
                continue

            should_fill = False
            fill_price = price

            if order.order_type == OrderType.MARKET:
                should_fill = True
                fill_price = self.ask_prices.get(symbol, price) if order.side == "buy" else self.bid_prices.get(symbol, price)

            elif order.order_type == OrderType.LIMIT:
                if order.side == "buy" and price <= order.price:
                    should_fill = True
                    fill_price = order.price
                elif order.side == "sell" and price >= order.price:
                    should_fill = True
                    fill_price = order.price

            elif order.order_type == OrderType.STOP_MARKET:
                if order.side == "buy" and price >= order.stop_price:
                    should_fill = True
                    fill_price = self.ask_prices.get(symbol, price)
                elif order.side == "sell" and price <= order.stop_price:
                    should_fill = True
                    fill_price = self.bid_prices.get(symbol, price)

            elif order.order_type == OrderType.TAKE_PROFIT:
                if order.side == "sell" and price >= order.stop_price:
                    should_fill = True
                    fill_price = self.bid_prices.get(symbol, price)
                elif order.side == "buy" and price <= order.stop_price:
                    should_fill = True
                    fill_price = self.ask_prices.get(symbol, price)

            if should_fill:
                self._fill_order(order, fill_price)

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        client_order_id: Optional[str] = None,
    ) -> PaperOrder:
        """Place a paper order."""
        if self.simulate_latency:
            await asyncio.sleep(self.latency_ms / 1000)

        order = PaperOrder(
            id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=Decimal(str(quantity)),
            price=Decimal(str(price)) if price else None,
            stop_price=Decimal(str(stop_price)) if stop_price else None,
            client_order_id=client_order_id,
        )

        self.orders[order.id] = order
        self.stats["total_orders"] += 1

        logger.info(f"Paper order placed: {order.id} {side} {quantity} {symbol}")

        # Check for immediate fill (market orders)
        if order_type == OrderType.MARKET:
            current_price = self.current_prices.get(symbol)
            if current_price:
                fill_price = self.ask_prices.get(symbol, current_price) if side == "buy" else self.bid_prices.get(symbol, current_price)
                self._fill_order(order, fill_price)

        return order

    def _fill_order(self, order: PaperOrder, fill_price: Decimal):
        """Fill an order."""
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.updated_at = datetime.now()

        # Calculate commission
        commission = order.quantity * fill_price * self.commission_rate
        self.balance -= commission
        self.stats["total_commission"] += commission

        # Update position
        self._update_position_from_order(order, fill_price, commission)

        self.stats["filled_orders"] += 1
        self.order_history.append(order)
        del self.orders[order.id]

        logger.info(f"Paper order filled: {order.id} @ {fill_price}")

        if self.on_order_filled:
            self.on_order_filled(order)

    def _update_position_from_order(self, order: PaperOrder, fill_price: Decimal, commission: Decimal):
        """Update position based on filled order."""
        symbol = order.symbol
        existing = self.positions.get(symbol)

        if existing:
            # Check if closing or adding
            if (existing.side == "long" and order.side == "sell") or \
               (existing.side == "short" and order.side == "buy"):
                # Closing position
                self._close_position(symbol, fill_price, order.quantity, commission)
            else:
                # Adding to position
                total_qty = existing.quantity + order.quantity
                total_cost = existing.quantity * existing.entry_price + order.quantity * fill_price
                existing.entry_price = total_cost / total_qty
                existing.quantity = total_qty

                # Deduct margin
                margin = (order.quantity * fill_price) / existing.leverage
                self.balance -= margin
        else:
            # Open new position
            side = "long" if order.side == "buy" else "short"
            position = PaperPosition(
                symbol=symbol,
                side=side,
                quantity=order.quantity,
                entry_price=fill_price,
                current_price=fill_price,
                leverage=self.default_leverage,
            )

            # Calculate liquidation price
            if self.default_leverage > 1:
                margin_ratio = Decimal("0.5")  # 50% maintenance margin
                if side == "long":
                    position.liquidation_price = fill_price * (1 - margin_ratio / self.default_leverage)
                else:
                    position.liquidation_price = fill_price * (1 + margin_ratio / self.default_leverage)

            self.positions[symbol] = position

            # Deduct margin
            margin = (order.quantity * fill_price) / self.default_leverage
            self.balance -= margin

    def _close_position(
        self,
        symbol: str,
        exit_price: Decimal,
        quantity: Optional[Decimal] = None,
        commission: Decimal = Decimal("0"),
    ):
        """Close a position."""
        pos = self.positions.get(symbol)
        if not pos:
            return

        close_qty = quantity or pos.quantity

        # Calculate P&L
        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) * close_qty
        else:
            pnl = (pos.entry_price - exit_price) * close_qty

        pnl_pct = (pnl / (pos.entry_price * close_qty)) * 100

        # Create trade record
        trade = PaperTrade(
            id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=close_qty,
            pnl=pnl - commission,
            pnl_pct=pnl_pct,
            entry_time=pos.created_at,
            exit_time=datetime.now(),
            commission=commission,
        )
        self.trades.append(trade)
        self.stats["total_trades"] += 1

        # Return margin + P&L
        margin = (close_qty * pos.entry_price) / pos.leverage
        self.balance += margin + pnl

        # Update or remove position
        if close_qty >= pos.quantity:
            del self.positions[symbol]
        else:
            pos.quantity -= close_qty

        logger.info(f"Position closed: {symbol} P&L: {pnl:.2f}")

        if self.on_trade_completed:
            self.on_trade_completed(trade)

    def _liquidate_position(self, symbol: str):
        """Liquidate a position due to margin call."""
        pos = self.positions.get(symbol)
        if not pos:
            return

        # Close at liquidation price
        self._close_position(symbol, pos.liquidation_price or pos.current_price)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if self.simulate_latency:
            await asyncio.sleep(self.latency_ms / 1000)

        order = self.orders.get(order_id)
        if not order or order.status != OrderStatus.PENDING:
            return False

        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()

        self.order_history.append(order)
        del self.orders[order_id]
        self.stats["cancelled_orders"] += 1

        logger.info(f"Paper order cancelled: {order_id}")
        return True

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all pending orders."""
        cancelled = 0
        for order_id in list(self.orders.keys()):
            order = self.orders[order_id]
            if symbol and order.symbol != symbol:
                continue
            if await self.cancel_order(order_id):
                cancelled += 1
        return cancelled

    async def close_position(self, symbol: str, quantity: Optional[Decimal] = None) -> bool:
        """Close a position."""
        pos = self.positions.get(symbol)
        if not pos:
            return False

        price = self.current_prices.get(symbol)
        if not price:
            return False

        # Place close order
        close_qty = quantity or pos.quantity
        side = "sell" if pos.side == "long" else "buy"

        await self.place_order(
            symbol=symbol,
            side=side,
            quantity=close_qty,
            order_type=OrderType.MARKET,
        )

        return True

    async def close_all_positions(self) -> int:
        """Close all positions."""
        closed = 0
        for symbol in list(self.positions.keys()):
            if await self.close_position(symbol):
                closed += 1
        return closed

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for symbol."""
        pos = self.positions.get(symbol)
        if not pos:
            return None

        return {
            "symbol": pos.symbol,
            "side": pos.side,
            "quantity": float(pos.quantity),
            "entry_price": float(pos.entry_price),
            "current_price": float(pos.current_price),
            "unrealized_pnl": float(pos.unrealized_pnl),
            "leverage": pos.leverage,
        }

    def get_all_positions(self) -> List[Dict[str, Any]]:
        """Get all positions."""
        return [self.get_position(s) for s in self.positions.keys()]

    def get_balance(self) -> Dict[str, Any]:
        """Get balance information."""
        return {
            "total": float(self.equity),
            "available": float(self.available_balance),
            "balance": float(self.balance),
            "unrealized_pnl": float(sum(p.unrealized_pnl for p in self.positions.values())),
        }

    def get_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades."""
        trades = self.trades[-limit:]
        return [
            {
                "id": t.id,
                "symbol": t.symbol,
                "side": t.side,
                "entry_price": float(t.entry_price),
                "exit_price": float(t.exit_price),
                "quantity": float(t.quantity),
                "pnl": float(t.pnl),
                "pnl_pct": float(t.pnl_pct),
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
            }
            for t in trades
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get trading statistics."""
        if not self.trades:
            return {
                **self.stats,
                "win_rate": 0,
                "total_pnl": 0,
            }

        wins = [t for t in self.trades if t.pnl > 0]
        total_pnl = sum(t.pnl for t in self.trades)

        return {
            **{k: float(v) if isinstance(v, Decimal) else v for k, v in self.stats.items()},
            "win_rate": len(wins) / len(self.trades) * 100,
            "total_pnl": float(total_pnl),
            "total_return_pct": float((self.equity / self.initial_balance - 1) * 100),
        }

    def reset(self):
        """Reset paper trading state."""
        self.balance = self.initial_balance
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.order_history.clear()
        self.stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "total_trades": 0,
            "total_commission": Decimal("0"),
        }
        logger.info("Paper trading engine reset")
