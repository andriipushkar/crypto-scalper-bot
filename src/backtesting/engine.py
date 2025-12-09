"""
Backtesting Engine

Core backtesting engine for strategy evaluation.
"""
from typing import Any, Callable, Dict, List, Optional, Type
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import copy

from loguru import logger

from .data import OHLCV, DataIterator


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Backtest order."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    filled: bool = False
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None


@dataclass
class Position:
    """Backtest position."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0
    unrealized_pnl: float = 0
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None


@dataclass
class Trade:
    """Completed trade."""
    id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    commission: float = 0


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_balance: float = 10000.0
    commission_rate: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    leverage: int = 1
    margin_call_level: float = 0.5  # 50% margin
    max_position_size: float = 1.0  # 100% of balance
    allow_shorting: bool = True
    use_limit_orders: bool = False


@dataclass
class BacktestResult:
    """Backtest results."""
    config: BacktestConfig
    start_time: datetime
    end_time: datetime
    initial_balance: float
    final_balance: float
    total_return: float
    total_return_pct: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)

    # Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0
    profit_factor: float = 0
    max_drawdown: float = 0
    max_drawdown_pct: float = 0
    sharpe_ratio: float = 0
    sortino_ratio: float = 0
    avg_trade: float = 0
    avg_win: float = 0
    avg_loss: float = 0
    largest_win: float = 0
    largest_loss: float = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0


class BacktestEngine:
    """Core backtesting engine."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

        # State
        self.balance = self.config.initial_balance
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict[str, Any]] = []

        # Current market state
        self.current_time: Optional[datetime] = None
        self.current_prices: Dict[str, float] = {}
        self.historical_data: Dict[str, List[OHLCV]] = {}

        # Trade counter
        self._trade_counter = 0
        self._order_counter = 0

    def reset(self):
        """Reset engine state."""
        self.balance = self.config.initial_balance
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self.current_time = None
        self.current_prices.clear()
        self._trade_counter = 0
        self._order_counter = 0

    def _next_order_id(self) -> str:
        """Generate next order ID."""
        self._order_counter += 1
        return f"order_{self._order_counter}"

    def _next_trade_id(self) -> str:
        """Generate next trade ID."""
        self._trade_counter += 1
        return f"trade_{self._trade_counter}"

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply slippage to price."""
        slippage = price * self.config.slippage
        if side == OrderSide.BUY:
            return price + slippage
        return price - slippage

    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission for trade."""
        return quantity * price * self.config.commission_rate

    def get_equity(self) -> float:
        """Calculate total equity (balance + unrealized P&L)."""
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return self.balance + unrealized

    def get_available_balance(self) -> float:
        """Get available balance for trading."""
        margin_used = sum(
            p.quantity * p.entry_price / self.config.leverage
            for p in self.positions.values()
        )
        return self.balance - margin_used

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Optional[Order]:
        """Place an order."""
        order = Order(
            id=self._next_order_id(),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            timestamp=self.current_time,
        )

        if order_type == OrderType.MARKET:
            # Execute immediately
            self._execute_order(order)
        else:
            # Add to pending orders
            self.orders.append(order)

        return order

    def _execute_order(self, order: Order) -> bool:
        """Execute an order."""
        current_price = self.current_prices.get(order.symbol)
        if current_price is None:
            logger.warning(f"No price for {order.symbol}")
            return False

        # Apply slippage
        fill_price = self._apply_slippage(current_price, order.side)

        # Calculate cost and commission
        cost = order.quantity * fill_price
        commission = self._calculate_commission(order.quantity, fill_price)

        # Check balance
        if order.side == OrderSide.BUY:
            required = (cost / self.config.leverage) + commission
            if required > self.get_available_balance():
                logger.warning(f"Insufficient balance for order {order.id}")
                return False

        # Check if closing existing position
        existing = self.positions.get(order.symbol)

        if existing:
            if (existing.side == "long" and order.side == OrderSide.SELL) or \
               (existing.side == "short" and order.side == OrderSide.BUY):
                # Close position
                self._close_position(order.symbol, fill_price, order.quantity)
            else:
                # Add to position
                self._add_to_position(order.symbol, order.quantity, fill_price)
        else:
            # Open new position
            self._open_position(
                symbol=order.symbol,
                side="long" if order.side == OrderSide.BUY else "short",
                quantity=order.quantity,
                price=fill_price,
            )

        # Deduct commission
        self.balance -= commission

        order.filled = True
        order.fill_price = fill_price
        order.fill_time = self.current_time

        return True

    def _open_position(self, symbol: str, side: str, quantity: float, price: float):
        """Open a new position."""
        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            entry_time=self.current_time,
            current_price=price,
        )

        # Deduct margin
        margin = (quantity * price) / self.config.leverage
        self.balance -= margin

    def _add_to_position(self, symbol: str, quantity: float, price: float):
        """Add to existing position."""
        pos = self.positions[symbol]

        # Calculate new average price
        total_cost = pos.quantity * pos.entry_price + quantity * price
        total_qty = pos.quantity + quantity
        pos.entry_price = total_cost / total_qty
        pos.quantity = total_qty

        # Deduct margin for new quantity
        margin = (quantity * price) / self.config.leverage
        self.balance -= margin

    def _close_position(self, symbol: str, price: float, quantity: Optional[float] = None):
        """Close a position."""
        pos = self.positions.get(symbol)
        if not pos:
            return

        close_qty = quantity or pos.quantity

        # Calculate P&L
        if pos.side == "long":
            pnl = (price - pos.entry_price) * close_qty
        else:
            pnl = (pos.entry_price - price) * close_qty

        pnl_pct = (pnl / (pos.entry_price * close_qty)) * 100

        # Create trade record
        trade = Trade(
            id=self._next_trade_id(),
            symbol=symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=close_qty,
            entry_time=pos.entry_time,
            exit_time=self.current_time,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=self._calculate_commission(close_qty, price),
        )
        self.trades.append(trade)

        # Return margin + P&L
        margin = (close_qty * pos.entry_price) / self.config.leverage
        self.balance += margin + pnl

        # Update or remove position
        if close_qty >= pos.quantity:
            del self.positions[symbol]
        else:
            pos.quantity -= close_qty

    def set_stop_loss(self, symbol: str, price: float):
        """Set stop loss for position."""
        if symbol in self.positions:
            self.positions[symbol].stop_loss = price

    def set_take_profit(self, symbol: str, price: float):
        """Set take profit for position."""
        if symbol in self.positions:
            self.positions[symbol].take_profit = price

    def _update_positions(self, candle: OHLCV):
        """Update positions with new price data."""
        symbol = candle.timestamp  # Will be fixed in run method

        for sym, pos in list(self.positions.items()):
            price = self.current_prices.get(sym, pos.current_price)
            pos.current_price = price

            # Calculate unrealized P&L
            if pos.side == "long":
                pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
            else:
                pos.unrealized_pnl = (pos.entry_price - price) * pos.quantity

            # Check stop loss
            if pos.stop_loss:
                if (pos.side == "long" and price <= pos.stop_loss) or \
                   (pos.side == "short" and price >= pos.stop_loss):
                    self._close_position(sym, pos.stop_loss)
                    continue

            # Check take profit
            if pos.take_profit:
                if (pos.side == "long" and price >= pos.take_profit) or \
                   (pos.side == "short" and price <= pos.take_profit):
                    self._close_position(sym, pos.take_profit)

    def _check_pending_orders(self, candle: OHLCV):
        """Check and fill pending orders."""
        for order in list(self.orders):
            if order.filled:
                continue

            price = self.current_prices.get(order.symbol)
            if not price:
                continue

            should_fill = False

            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and price <= order.price:
                    should_fill = True
                elif order.side == OrderSide.SELL and price >= order.price:
                    should_fill = True

            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY and price >= order.stop_price:
                    should_fill = True
                elif order.side == OrderSide.SELL and price <= order.stop_price:
                    should_fill = True

            if should_fill:
                self._execute_order(order)
                self.orders.remove(order)

    def _record_equity(self):
        """Record equity curve point."""
        self.equity_curve.append({
            "timestamp": self.current_time.isoformat() if self.current_time else None,
            "equity": self.get_equity(),
            "balance": self.balance,
            "positions": len(self.positions),
        })

    def run(
        self,
        strategy: Callable,
        data: Dict[str, List[OHLCV]],
        warmup_period: int = 100,
    ) -> BacktestResult:
        """Run backtest with strategy.

        Args:
            strategy: Strategy function that receives (engine, symbol, candle, history)
            data: Dictionary of symbol -> candles
            warmup_period: Number of candles for warmup (no trading)
        """
        self.reset()
        self.historical_data = data

        # Get all timestamps
        all_candles = []
        for symbol, candles in data.items():
            for c in candles:
                all_candles.append((c.timestamp, symbol, c))

        # Sort by timestamp
        all_candles.sort(key=lambda x: x[0])

        if not all_candles:
            raise ValueError("No data provided")

        start_time = all_candles[warmup_period][0] if len(all_candles) > warmup_period else all_candles[0][0]
        end_time = all_candles[-1][0]

        logger.info(f"Starting backtest from {start_time} to {end_time}")
        logger.info(f"Total candles: {len(all_candles)}, Warmup: {warmup_period}")

        # Run through data
        history: Dict[str, List[OHLCV]] = {s: [] for s in data.keys()}

        for i, (timestamp, symbol, candle) in enumerate(all_candles):
            self.current_time = timestamp
            self.current_prices[symbol] = candle.close

            # Update positions
            self._update_positions(candle)

            # Check pending orders
            self._check_pending_orders(candle)

            # Add to history
            history[symbol].append(candle)

            # Skip warmup period
            if i < warmup_period:
                continue

            # Call strategy
            try:
                strategy(self, symbol, candle, history[symbol])
            except Exception as e:
                logger.error(f"Strategy error: {e}")

            # Record equity periodically
            if i % 100 == 0:
                self._record_equity()

        # Close all positions at end
        for symbol in list(self.positions.keys()):
            price = self.current_prices.get(symbol, 0)
            self._close_position(symbol, price)

        # Final equity record
        self._record_equity()

        # Calculate statistics
        return self._calculate_results(start_time, end_time)

    def _calculate_results(self, start_time: datetime, end_time: datetime) -> BacktestResult:
        """Calculate backtest statistics."""
        result = BacktestResult(
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            initial_balance=self.config.initial_balance,
            final_balance=self.balance,
            total_return=self.balance - self.config.initial_balance,
            total_return_pct=((self.balance / self.config.initial_balance) - 1) * 100,
            trades=self.trades,
            equity_curve=self.equity_curve,
            total_trades=len(self.trades),
        )

        if not self.trades:
            return result

        # Win/loss stats
        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]

        result.winning_trades = len(winning)
        result.losing_trades = len(losing)
        result.win_rate = (len(winning) / len(self.trades)) * 100

        # P&L stats
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))

        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        result.avg_trade = sum(t.pnl for t in self.trades) / len(self.trades)
        result.avg_win = gross_profit / len(winning) if winning else 0
        result.avg_loss = gross_loss / len(losing) if losing else 0
        result.largest_win = max(t.pnl for t in winning) if winning else 0
        result.largest_loss = abs(min(t.pnl for t in losing)) if losing else 0

        # Drawdown
        equity_values = [e["equity"] for e in self.equity_curve]
        if equity_values:
            peak = equity_values[0]
            max_dd = 0
            for value in equity_values:
                if value > peak:
                    peak = value
                dd = (peak - value)
                if dd > max_dd:
                    max_dd = dd
                    result.max_drawdown_pct = (dd / peak) * 100

            result.max_drawdown = max_dd

        # Consecutive wins/losses
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0

        for trade in self.trades:
            if trade.pnl > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_win_streak = max(max_win_streak, current_streak)
            else:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_loss_streak = max(max_loss_streak, abs(current_streak))

        result.max_consecutive_wins = max_win_streak
        result.max_consecutive_losses = max_loss_streak

        # Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                prev = self.equity_curve[i-1]["equity"]
                curr = self.equity_curve[i]["equity"]
                if prev > 0:
                    returns.append((curr - prev) / prev)

            if returns:
                import statistics
                mean_return = statistics.mean(returns)
                std_return = statistics.stdev(returns) if len(returns) > 1 else 0.0001
                result.sharpe_ratio = (mean_return / std_return) * (252 ** 0.5)  # Annualized

        return result
