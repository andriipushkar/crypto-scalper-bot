"""
Portfolio Analytics

Provides comprehensive portfolio analysis and tracking.
"""
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

from loguru import logger


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    side: str  # long, short
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    entry_time: datetime
    unrealized_pnl: Decimal = Decimal("0")
    unrealized_pnl_pct: Decimal = Decimal("0")
    leverage: int = 1

    def update_price(self, price: Decimal) -> None:
        """Update current price and P&L."""
        self.current_price = price
        if self.side == "long":
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity

        entry_value = self.entry_price * self.quantity
        if entry_value > 0:
            self.unrealized_pnl_pct = (self.unrealized_pnl / entry_value) * 100


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio snapshot."""
    timestamp: datetime
    total_value: Decimal
    cash_balance: Decimal
    positions_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    daily_pnl: Decimal
    daily_pnl_pct: Decimal


@dataclass
class AllocationBreakdown:
    """Asset allocation breakdown."""
    by_symbol: Dict[str, Decimal] = field(default_factory=dict)
    by_side: Dict[str, Decimal] = field(default_factory=dict)
    by_strategy: Dict[str, Decimal] = field(default_factory=dict)
    cash_pct: Decimal = Decimal("0")


class PortfolioAnalytics:
    """Comprehensive portfolio analytics and tracking."""

    def __init__(
        self,
        initial_balance: Decimal = Decimal("10000"),
        risk_free_rate: Decimal = Decimal("0.05"),  # Annual
    ):
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.risk_free_rate = risk_free_rate
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Dict[str, Any]] = []
        self.snapshots: List[PortfolioSnapshot] = []
        self.daily_returns: List[Decimal] = []

    @property
    def total_value(self) -> Decimal:
        """Calculate total portfolio value."""
        positions_value = sum(
            p.current_price * p.quantity
            for p in self.positions.values()
        )
        return self.cash_balance + positions_value

    @property
    def unrealized_pnl(self) -> Decimal:
        """Total unrealized P&L."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def realized_pnl(self) -> Decimal:
        """Total realized P&L."""
        return sum(Decimal(str(t.get('pnl', 0))) for t in self.closed_trades)

    @property
    def total_pnl(self) -> Decimal:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def total_return_pct(self) -> Decimal:
        """Total return percentage."""
        if self.initial_balance == 0:
            return Decimal("0")
        return ((self.total_value - self.initial_balance) / self.initial_balance) * 100

    def open_position(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        leverage: int = 1,
        strategy: Optional[str] = None,
    ) -> Position:
        """Open a new position."""
        position = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            current_price=price,
            entry_time=datetime.now(),
            leverage=leverage,
        )

        # Deduct from cash
        margin = (price * quantity) / leverage
        self.cash_balance -= margin

        self.positions[symbol] = position
        logger.info(f"Opened {side} position: {quantity} {symbol} @ {price}")

        return position

    def close_position(
        self,
        symbol: str,
        price: Decimal,
        quantity: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Close a position (fully or partially)."""
        if symbol not in self.positions:
            raise ValueError(f"No position found for {symbol}")

        position = self.positions[symbol]
        close_qty = quantity or position.quantity

        if close_qty > position.quantity:
            raise ValueError("Cannot close more than position size")

        # Calculate P&L
        if position.side == "long":
            pnl = (price - position.entry_price) * close_qty
        else:
            pnl = (position.entry_price - price) * close_qty

        # Return margin + P&L
        margin = (position.entry_price * close_qty) / position.leverage
        self.cash_balance += margin + pnl

        trade = {
            "symbol": symbol,
            "side": position.side,
            "quantity": float(close_qty),
            "entry_price": float(position.entry_price),
            "exit_price": float(price),
            "entry_time": position.entry_time.isoformat(),
            "exit_time": datetime.now().isoformat(),
            "pnl": float(pnl),
            "pnl_pct": float((pnl / margin) * 100) if margin > 0 else 0,
            "leverage": position.leverage,
        }

        self.closed_trades.append(trade)

        # Update or remove position
        if close_qty >= position.quantity:
            del self.positions[symbol]
        else:
            position.quantity -= close_qty

        logger.info(f"Closed {position.side} {close_qty} {symbol} @ {price}, P&L: {pnl}")

        return trade

    def update_prices(self, prices: Dict[str, Decimal]) -> None:
        """Update all position prices."""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])

    def take_snapshot(self) -> PortfolioSnapshot:
        """Take a portfolio snapshot."""
        positions_value = sum(
            p.current_price * p.quantity
            for p in self.positions.values()
        )

        # Calculate daily P&L
        daily_pnl = Decimal("0")
        daily_pnl_pct = Decimal("0")

        if self.snapshots:
            prev_value = self.snapshots[-1].total_value
            daily_pnl = self.total_value - prev_value
            if prev_value > 0:
                daily_pnl_pct = (daily_pnl / prev_value) * 100
                self.daily_returns.append(daily_pnl_pct)

        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            total_value=self.total_value,
            cash_balance=self.cash_balance,
            positions_value=positions_value,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
        )

        self.snapshots.append(snapshot)
        return snapshot

    def get_allocation(self) -> AllocationBreakdown:
        """Get current portfolio allocation."""
        allocation = AllocationBreakdown()
        total = self.total_value

        if total == 0:
            return allocation

        # By symbol
        for symbol, position in self.positions.items():
            value = position.current_price * position.quantity
            allocation.by_symbol[symbol] = (value / total) * 100

        # By side
        long_value = sum(
            p.current_price * p.quantity
            for p in self.positions.values()
            if p.side == "long"
        )
        short_value = sum(
            p.current_price * p.quantity
            for p in self.positions.values()
            if p.side == "short"
        )

        allocation.by_side["long"] = (long_value / total) * 100
        allocation.by_side["short"] = (short_value / total) * 100
        allocation.cash_pct = (self.cash_balance / total) * 100

        return allocation

    def get_exposure(self) -> Dict[str, Any]:
        """Get portfolio exposure metrics."""
        total = self.total_value

        long_exposure = sum(
            p.current_price * p.quantity * p.leverage
            for p in self.positions.values()
            if p.side == "long"
        )

        short_exposure = sum(
            p.current_price * p.quantity * p.leverage
            for p in self.positions.values()
            if p.side == "short"
        )

        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure

        return {
            "long_exposure": float(long_exposure),
            "short_exposure": float(short_exposure),
            "gross_exposure": float(gross_exposure),
            "net_exposure": float(net_exposure),
            "gross_leverage": float(gross_exposure / total) if total > 0 else 0,
            "net_leverage": float(net_exposure / total) if total > 0 else 0,
        }

    def get_drawdown(self) -> Dict[str, Any]:
        """Calculate drawdown metrics."""
        if not self.snapshots:
            return {
                "current_drawdown": 0,
                "max_drawdown": 0,
                "max_drawdown_duration": 0,
            }

        values = [float(s.total_value) for s in self.snapshots]
        peak = values[0]
        max_dd = 0
        current_dd = 0

        dd_start = None
        max_dd_duration = 0
        current_dd_duration = 0

        for i, value in enumerate(values):
            if value > peak:
                peak = value
                if dd_start is not None:
                    max_dd_duration = max(max_dd_duration, current_dd_duration)
                dd_start = None
                current_dd_duration = 0
            else:
                dd = (peak - value) / peak * 100
                if dd > max_dd:
                    max_dd = dd
                if dd_start is None:
                    dd_start = i
                current_dd_duration = i - dd_start
                current_dd = dd

        return {
            "current_drawdown": current_dd,
            "max_drawdown": max_dd,
            "max_drawdown_duration": max(max_dd_duration, current_dd_duration),
            "peak_value": peak,
        }

    def get_top_performers(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get top performing trades."""
        sorted_trades = sorted(
            self.closed_trades,
            key=lambda t: t.get('pnl', 0),
            reverse=True
        )
        return sorted_trades[:n]

    def get_worst_performers(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get worst performing trades."""
        sorted_trades = sorted(
            self.closed_trades,
            key=lambda t: t.get('pnl', 0)
        )
        return sorted_trades[:n]

    def get_symbol_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by symbol."""
        symbol_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "trades": 0,
            "wins": 0,
            "total_pnl": 0,
            "avg_pnl": 0,
            "win_rate": 0,
        })

        for trade in self.closed_trades:
            symbol = trade.get('symbol', 'UNKNOWN')
            pnl = trade.get('pnl', 0)

            symbol_stats[symbol]["trades"] += 1
            symbol_stats[symbol]["total_pnl"] += pnl
            if pnl > 0:
                symbol_stats[symbol]["wins"] += 1

        for symbol, stats in symbol_stats.items():
            if stats["trades"] > 0:
                stats["avg_pnl"] = stats["total_pnl"] / stats["trades"]
                stats["win_rate"] = (stats["wins"] / stats["trades"]) * 100

        return dict(symbol_stats)

    def get_time_analysis(self) -> Dict[str, Any]:
        """Analyze performance by time."""
        hourly_pnl = defaultdict(list)
        daily_pnl = defaultdict(list)

        for trade in self.closed_trades:
            exit_time = trade.get('exit_time', '')
            if exit_time:
                dt = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
                hour = dt.hour
                day = dt.strftime('%A')
                pnl = trade.get('pnl', 0)

                hourly_pnl[hour].append(pnl)
                daily_pnl[day].append(pnl)

        best_hour = max(hourly_pnl.items(), key=lambda x: sum(x[1]), default=(0, []))
        worst_hour = min(hourly_pnl.items(), key=lambda x: sum(x[1]), default=(0, []))
        best_day = max(daily_pnl.items(), key=lambda x: sum(x[1]), default=('', []))
        worst_day = min(daily_pnl.items(), key=lambda x: sum(x[1]), default=('', []))

        return {
            "best_hour": best_hour[0],
            "best_hour_pnl": sum(best_hour[1]),
            "worst_hour": worst_hour[0],
            "worst_hour_pnl": sum(worst_hour[1]),
            "best_day": best_day[0],
            "best_day_pnl": sum(best_day[1]),
            "worst_day": worst_day[0],
            "worst_day_pnl": sum(worst_day[1]),
            "hourly_breakdown": {h: sum(pnls) for h, pnls in hourly_pnl.items()},
            "daily_breakdown": {d: sum(pnls) for d, pnls in daily_pnl.items()},
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export portfolio state to dictionary."""
        return {
            "total_value": float(self.total_value),
            "cash_balance": float(self.cash_balance),
            "unrealized_pnl": float(self.unrealized_pnl),
            "realized_pnl": float(self.realized_pnl),
            "total_pnl": float(self.total_pnl),
            "total_return_pct": float(self.total_return_pct),
            "positions": {
                symbol: {
                    "side": p.side,
                    "quantity": float(p.quantity),
                    "entry_price": float(p.entry_price),
                    "current_price": float(p.current_price),
                    "unrealized_pnl": float(p.unrealized_pnl),
                    "unrealized_pnl_pct": float(p.unrealized_pnl_pct),
                }
                for symbol, p in self.positions.items()
            },
            "allocation": {
                "by_symbol": {k: float(v) for k, v in self.get_allocation().by_symbol.items()},
                "by_side": {k: float(v) for k, v in self.get_allocation().by_side.items()},
                "cash_pct": float(self.get_allocation().cash_pct),
            },
            "exposure": self.get_exposure(),
            "drawdown": self.get_drawdown(),
            "total_trades": len(self.closed_trades),
        }
