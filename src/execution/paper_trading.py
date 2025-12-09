"""
Paper Trading Mode - Simulate trading without real money.

Provides a simulated exchange that mimics real exchange behavior
for testing strategies with zero financial risk.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

from loguru import logger

from src.data.models import Side, OrderType, OrderStatus, Trade
from src.core.events import Event, EventType, get_event_bus


# =============================================================================
# Paper Trading Models
# =============================================================================

class PaperOrderStatus(Enum):
    """Paper order status."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class PaperOrder:
    """Simulated order."""
    order_id: str
    symbol: str
    side: Side
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: PaperOrderStatus = PaperOrderStatus.NEW
    filled_quantity: Decimal = Decimal("0")
    avg_fill_price: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    client_order_id: Optional[str] = None

    @property
    def remaining_quantity(self) -> Decimal:
        return self.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        return self.status in [PaperOrderStatus.NEW, PaperOrderStatus.PARTIALLY_FILLED]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "orderId": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.order_type.value,
            "origQty": str(self.quantity),
            "price": str(self.price) if self.price else "0",
            "stopPrice": str(self.stop_price) if self.stop_price else "0",
            "status": self.status.value,
            "executedQty": str(self.filled_quantity),
            "avgPrice": str(self.avg_fill_price),
            "commission": str(self.commission),
            "time": int(self.created_at.timestamp() * 1000),
            "updateTime": int(self.updated_at.timestamp() * 1000),
            "clientOrderId": self.client_order_id or self.order_id,
        }


@dataclass
class PaperPosition:
    """Simulated position."""
    symbol: str
    side: Side
    quantity: Decimal
    entry_price: Decimal
    leverage: int = 1
    unrealized_pnl: Decimal = Decimal("0")
    margin: Decimal = Decimal("0")
    liquidation_price: Optional[Decimal] = None
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def notional_value(self) -> Decimal:
        return self.quantity * self.entry_price

    def update_pnl(self, current_price: Decimal) -> None:
        """Update unrealized P&L based on current price."""
        if self.side == Side.BUY:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "positionSide": "LONG" if self.side == Side.BUY else "SHORT",
            "positionAmt": str(self.quantity if self.side == Side.BUY else -self.quantity),
            "entryPrice": str(self.entry_price),
            "unrealizedProfit": str(self.unrealized_pnl),
            "leverage": str(self.leverage),
            "marginType": "cross",
            "isolatedMargin": "0",
            "liquidationPrice": str(self.liquidation_price) if self.liquidation_price else "0",
        }


@dataclass
class PaperBalance:
    """Simulated account balance."""
    asset: str = "USDT"
    wallet_balance: Decimal = Decimal("100")
    available_balance: Decimal = Decimal("100")
    unrealized_pnl: Decimal = Decimal("0")
    margin_balance: Decimal = Decimal("100")
    used_margin: Decimal = Decimal("0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "walletBalance": str(self.wallet_balance),
            "availableBalance": str(self.available_balance),
            "unrealizedProfit": str(self.unrealized_pnl),
            "marginBalance": str(self.margin_balance),
            "usedMargin": str(self.used_margin),
        }


# =============================================================================
# Paper Trading Engine
# =============================================================================

class PaperTradingEngine:
    """
    Simulated exchange for paper trading.

    Features:
        - Realistic order matching
        - Slippage simulation
        - Commission calculation
        - Position tracking
        - P&L calculation
        - Liquidation simulation

    Usage:
        engine = PaperTradingEngine(initial_balance=100.0)

        # Place orders
        order = await engine.place_order(
            symbol="BTCUSDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.001"),
        )

        # Update with market prices
        await engine.update_price("BTCUSDT", Decimal("50000"))

        # Get account info
        balance = engine.get_balance()
        positions = engine.get_positions()
    """

    def __init__(
        self,
        initial_balance: float = 100.0,
        leverage: int = 10,
        commission_rate: float = 0.0004,
        slippage_bps: float = 1.0,
        enable_liquidation: bool = True,
        maintenance_margin_rate: float = 0.004,
    ):
        self.initial_balance = Decimal(str(initial_balance))
        self.leverage = leverage
        self.commission_rate = Decimal(str(commission_rate))
        self.slippage_bps = Decimal(str(slippage_bps))
        self.enable_liquidation = enable_liquidation
        self.maintenance_margin_rate = Decimal(str(maintenance_margin_rate))

        # State
        self.balance = PaperBalance(
            wallet_balance=self.initial_balance,
            available_balance=self.initial_balance,
            margin_balance=self.initial_balance,
        )
        self.positions: Dict[str, PaperPosition] = {}
        self.orders: Dict[str, PaperOrder] = {}
        self.trades: List[Dict[str, Any]] = []

        # Market prices
        self._prices: Dict[str, Decimal] = {}
        self._order_counter = 0

        # Event bus
        self._event_bus = get_event_bus()

        # Statistics
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": Decimal("0"),
            "total_commission": Decimal("0"),
            "max_drawdown": Decimal("0"),
            "peak_balance": self.initial_balance,
        }

        logger.info(
            f"Paper trading initialized: balance=${initial_balance}, "
            f"leverage={leverage}x, commission={commission_rate*100:.2f}%"
        )

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"PAPER_{int(time.time() * 1000)}_{self._order_counter}"

    def _apply_slippage(self, price: Decimal, side: Side) -> Decimal:
        """Apply slippage to price."""
        slippage = price * self.slippage_bps / Decimal("10000")
        if side == Side.BUY:
            return price + slippage
        else:
            return price - slippage

    def _calculate_commission(self, notional: Decimal) -> Decimal:
        """Calculate commission for trade."""
        return notional * self.commission_rate

    def _calculate_liquidation_price(
        self,
        entry_price: Decimal,
        side: Side,
        leverage: int,
    ) -> Decimal:
        """Calculate liquidation price."""
        margin_rate = Decimal("1") / Decimal(str(leverage))
        maintenance = self.maintenance_margin_rate

        if side == Side.BUY:
            # Long: liquidation when price drops
            liq_price = entry_price * (Decimal("1") - margin_rate + maintenance)
        else:
            # Short: liquidation when price rises
            liq_price = entry_price * (Decimal("1") + margin_rate - maintenance)

        return liq_price

    async def update_price(self, symbol: str, price: Decimal) -> None:
        """
        Update market price and check positions.

        This should be called whenever new price data is received.
        """
        self._prices[symbol] = price

        # Update position P&L
        if symbol in self.positions:
            position = self.positions[symbol]
            position.update_pnl(price)

            # Check for liquidation
            if self.enable_liquidation and position.liquidation_price:
                is_liquidated = False
                if position.side == Side.BUY and price <= position.liquidation_price:
                    is_liquidated = True
                elif position.side == Side.SELL and price >= position.liquidation_price:
                    is_liquidated = True

                if is_liquidated:
                    await self._liquidate_position(symbol, price)

        # Update balance
        self._update_balance()

        # Try to fill pending limit orders
        await self._check_pending_orders(symbol, price)

    async def _check_pending_orders(self, symbol: str, price: Decimal) -> None:
        """Check and fill pending limit orders."""
        for order in list(self.orders.values()):
            if order.symbol != symbol or not order.is_active:
                continue

            should_fill = False

            if order.order_type == OrderType.LIMIT:
                if order.side == Side.BUY and price <= order.price:
                    should_fill = True
                elif order.side == Side.SELL and price >= order.price:
                    should_fill = True

            elif order.order_type == OrderType.STOP_MARKET and order.stop_price:
                if order.side == Side.BUY and price >= order.stop_price:
                    should_fill = True
                elif order.side == Side.SELL and price <= order.stop_price:
                    should_fill = True

            if should_fill:
                await self._fill_order(order, price)

    async def _fill_order(self, order: PaperOrder, fill_price: Decimal) -> None:
        """Fill an order."""
        # Apply slippage for market orders
        if order.order_type in [OrderType.MARKET, OrderType.STOP_MARKET]:
            fill_price = self._apply_slippage(fill_price, order.side)

        notional = order.quantity * fill_price
        commission = self._calculate_commission(notional)

        # Update order
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.commission = commission
        order.status = PaperOrderStatus.FILLED
        order.updated_at = datetime.now()

        # Deduct commission
        self.balance.wallet_balance -= commission
        self.stats["total_commission"] += commission

        # Update position
        await self._update_position(order, fill_price)

        # Record trade
        trade_record = {
            "id": str(uuid.uuid4()),
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": str(order.quantity),
            "price": str(fill_price),
            "commission": str(commission),
            "realized_pnl": "0",
            "timestamp": datetime.now().isoformat(),
        }
        self.trades.append(trade_record)

        # Emit event
        await self._event_bus.publish(Event(
            type=EventType.ORDER_FILLED,
            data={
                "order": order.to_dict(),
                "fill_price": str(fill_price),
                "paper_trading": True,
            },
        ))

        logger.info(
            f"Paper order filled: {order.side.value} {order.quantity} {order.symbol} "
            f"@ {fill_price} (commission: {commission})"
        )

    async def _update_position(self, order: PaperOrder, fill_price: Decimal) -> None:
        """Update position after order fill."""
        symbol = order.symbol
        quantity = order.quantity

        if symbol not in self.positions:
            # New position
            margin = (quantity * fill_price) / Decimal(str(self.leverage))
            liq_price = self._calculate_liquidation_price(fill_price, order.side, self.leverage)

            self.positions[symbol] = PaperPosition(
                symbol=symbol,
                side=order.side,
                quantity=quantity,
                entry_price=fill_price,
                leverage=self.leverage,
                margin=margin,
                liquidation_price=liq_price,
            )

            self.balance.available_balance -= margin
            self.balance.used_margin += margin

            logger.info(f"Paper position opened: {order.side.value} {quantity} {symbol} @ {fill_price}")

        else:
            position = self.positions[symbol]

            if position.side == order.side:
                # Adding to position
                total_quantity = position.quantity + quantity
                total_cost = (position.quantity * position.entry_price) + (quantity * fill_price)
                new_entry = total_cost / total_quantity

                additional_margin = (quantity * fill_price) / Decimal(str(self.leverage))
                position.quantity = total_quantity
                position.entry_price = new_entry
                position.margin += additional_margin
                position.liquidation_price = self._calculate_liquidation_price(
                    new_entry, position.side, self.leverage
                )

                self.balance.available_balance -= additional_margin
                self.balance.used_margin += additional_margin

            else:
                # Closing/reducing position
                if quantity >= position.quantity:
                    # Full close
                    realized_pnl = position.unrealized_pnl
                    self.balance.wallet_balance += realized_pnl
                    self.balance.available_balance += position.margin + realized_pnl
                    self.balance.used_margin -= position.margin

                    # Update stats
                    self.stats["total_trades"] += 1
                    self.stats["total_pnl"] += realized_pnl
                    if realized_pnl > 0:
                        self.stats["winning_trades"] += 1
                    else:
                        self.stats["losing_trades"] += 1

                    # Update peak and drawdown
                    if self.balance.wallet_balance > self.stats["peak_balance"]:
                        self.stats["peak_balance"] = self.balance.wallet_balance
                    else:
                        drawdown = (self.stats["peak_balance"] - self.balance.wallet_balance) / self.stats["peak_balance"]
                        if drawdown > self.stats["max_drawdown"]:
                            self.stats["max_drawdown"] = drawdown

                    # Update trade record with P&L
                    if self.trades:
                        self.trades[-1]["realized_pnl"] = str(realized_pnl)

                    logger.info(
                        f"Paper position closed: {symbol} P&L: {realized_pnl:+.4f} USDT"
                    )

                    del self.positions[symbol]

                else:
                    # Partial close
                    close_ratio = quantity / position.quantity
                    realized_pnl = position.unrealized_pnl * close_ratio
                    released_margin = position.margin * close_ratio

                    position.quantity -= quantity
                    position.margin -= released_margin
                    position.update_pnl(fill_price)

                    self.balance.wallet_balance += realized_pnl
                    self.balance.available_balance += released_margin + realized_pnl
                    self.balance.used_margin -= released_margin

                    self.stats["total_pnl"] += realized_pnl

                    if self.trades:
                        self.trades[-1]["realized_pnl"] = str(realized_pnl)

        self._update_balance()

    async def _liquidate_position(self, symbol: str, price: Decimal) -> None:
        """Liquidate a position."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Calculate liquidation loss
        liquidation_fee = position.notional_value * Decimal("0.005")  # 0.5% fee
        total_loss = position.margin + liquidation_fee

        self.balance.wallet_balance -= total_loss
        self.balance.available_balance = self.balance.wallet_balance
        self.balance.used_margin = Decimal("0")

        self.stats["total_trades"] += 1
        self.stats["losing_trades"] += 1
        self.stats["total_pnl"] -= total_loss

        logger.warning(
            f"Paper position LIQUIDATED: {symbol} @ {price}, "
            f"loss: -{total_loss:.4f} USDT"
        )

        await self._event_bus.publish(Event(
            type=EventType.ERROR,
            data={
                "error": "LIQUIDATION",
                "symbol": symbol,
                "price": str(price),
                "loss": str(total_loss),
                "paper_trading": True,
            },
        ))

        del self.positions[symbol]

    def _update_balance(self) -> None:
        """Update balance with unrealized P&L."""
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        self.balance.unrealized_pnl = total_unrealized
        self.balance.margin_balance = self.balance.wallet_balance + total_unrealized

    async def place_order(
        self,
        symbol: str,
        side: Side,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        client_order_id: Optional[str] = None,
    ) -> PaperOrder:
        """
        Place a paper trading order.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            order_type: MARKET, LIMIT, STOP_MARKET
            quantity: Order quantity
            price: Limit price (for LIMIT orders)
            stop_price: Stop price (for STOP orders)
            client_order_id: Optional client order ID

        Returns:
            PaperOrder object
        """
        order_id = self._generate_order_id()

        order = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            client_order_id=client_order_id,
        )

        # Validate order
        if order_type == OrderType.LIMIT and price is None:
            order.status = PaperOrderStatus.REJECTED
            logger.warning("Limit order rejected: price required")
            return order

        # Check available margin
        current_price = self._prices.get(symbol) or price or Decimal("0")
        required_margin = (quantity * current_price) / Decimal(str(self.leverage))

        if required_margin > self.balance.available_balance:
            order.status = PaperOrderStatus.REJECTED
            logger.warning(
                f"Order rejected: insufficient margin "
                f"(required: {required_margin}, available: {self.balance.available_balance})"
            )
            return order

        self.orders[order_id] = order

        # Market orders fill immediately
        if order_type == OrderType.MARKET:
            if symbol in self._prices:
                await self._fill_order(order, self._prices[symbol])
            else:
                logger.warning(f"Market order pending: no price for {symbol}")

        logger.info(f"Paper order placed: {order_id} - {side.value} {quantity} {symbol}")

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a paper order."""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        if not order.is_active:
            return False

        order.status = PaperOrderStatus.CANCELED
        order.updated_at = datetime.now()

        logger.info(f"Paper order canceled: {order_id}")
        return True

    async def close_position(
        self,
        symbol: str,
        quantity: Optional[Decimal] = None,
    ) -> Optional[PaperOrder]:
        """Close a position (fully or partially)."""
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        close_qty = quantity or position.quantity
        close_side = Side.SELL if position.side == Side.BUY else Side.BUY

        return await self.place_order(
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=close_qty,
        )

    def get_balance(self) -> Dict[str, Any]:
        """Get account balance."""
        self._update_balance()
        return self.balance.to_dict()

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all positions."""
        return [p.to_dict() for p in self.positions.values()]

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for a symbol."""
        if symbol in self.positions:
            return self.positions[symbol].to_dict()
        return None

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders."""
        orders = [o for o in self.orders.values() if o.is_active]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return [o.to_dict() for o in orders]

    def get_trades(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get trade history."""
        trades = self.trades
        if symbol:
            trades = [t for t in trades if t["symbol"] == symbol]
        return trades[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get trading statistics."""
        total = self.stats["total_trades"]
        return {
            "total_trades": total,
            "winning_trades": self.stats["winning_trades"],
            "losing_trades": self.stats["losing_trades"],
            "win_rate": (self.stats["winning_trades"] / total * 100) if total > 0 else 0,
            "total_pnl": float(self.stats["total_pnl"]),
            "total_commission": float(self.stats["total_commission"]),
            "net_pnl": float(self.stats["total_pnl"] - self.stats["total_commission"]),
            "max_drawdown_pct": float(self.stats["max_drawdown"] * 100),
            "current_balance": float(self.balance.wallet_balance),
            "return_pct": float(
                (self.balance.wallet_balance - self.initial_balance) / self.initial_balance * 100
            ),
        }

    def reset(self) -> None:
        """Reset paper trading state."""
        self.balance = PaperBalance(
            wallet_balance=self.initial_balance,
            available_balance=self.initial_balance,
            margin_balance=self.initial_balance,
        )
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self._prices.clear()
        self._order_counter = 0
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": Decimal("0"),
            "total_commission": Decimal("0"),
            "max_drawdown": Decimal("0"),
            "peak_balance": self.initial_balance,
        }
        logger.info("Paper trading reset")


# =============================================================================
# Paper Trading API Wrapper
# =============================================================================

class PaperTradingAPI:
    """
    API wrapper that mimics BinanceFuturesAPI for paper trading.

    Drop-in replacement for real API in paper trading mode.
    """

    def __init__(self, engine: PaperTradingEngine):
        self.engine = engine
        self.testnet = True  # Always testnet for paper trading

    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange info (simulated)."""
        return {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "pricePrecision": 2,
                    "quantityPrecision": 3,
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                },
                {
                    "symbol": "ETHUSDT",
                    "pricePrecision": 2,
                    "quantityPrecision": 3,
                    "baseAsset": "ETH",
                    "quoteAsset": "USDT",
                },
            ]
        }

    async def get_ticker_price(self, symbol: str) -> Dict[str, str]:
        """Get current price."""
        price = self.engine._prices.get(symbol, Decimal("0"))
        return {"symbol": symbol, "price": str(price)}

    async def get_balance(self) -> List[Dict[str, Any]]:
        """Get account balance."""
        return [self.engine.get_balance()]

    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get positions."""
        positions = self.engine.get_positions()
        if symbol:
            positions = [p for p in positions if p["symbol"] == symbol]
        return positions

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Place order."""
        order = await self.engine.place_order(
            symbol=symbol,
            side=Side(side),
            order_type=OrderType(order_type),
            quantity=Decimal(str(quantity)),
            price=Decimal(str(price)) if price else None,
            stop_price=Decimal(str(stop_price)) if stop_price else None,
        )
        return order.to_dict()

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
    ) -> Dict[str, Any]:
        """Place market order."""
        return await self.place_order(symbol, side, "MARKET", quantity)

    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel order."""
        success = await self.engine.cancel_order(order_id)
        return {"orderId": order_id, "status": "CANCELED" if success else "ERROR"}

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders."""
        return self.engine.get_open_orders(symbol)

    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Set leverage (updates engine setting)."""
        self.engine.leverage = leverage
        return {"symbol": symbol, "leverage": leverage}

    async def close(self) -> None:
        """Close (no-op for paper trading)."""
        pass
