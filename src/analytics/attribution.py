"""
Performance Attribution Analysis.

Breaks down portfolio performance by:
- Strategy
- Symbol/Asset
- Time period
- Market conditions
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import statistics

from loguru import logger


# =============================================================================
# Models
# =============================================================================

@dataclass
class TradeRecord:
    """Record of a completed trade for attribution."""
    trade_id: str
    symbol: str
    strategy: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    commission: float
    leverage: int = 1

    @property
    def net_pnl(self) -> float:
        return self.pnl - self.commission

    @property
    def return_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.side == "LONG":
            return ((self.exit_price - self.entry_price) / self.entry_price) * 100 * self.leverage
        else:
            return ((self.entry_price - self.exit_price) / self.entry_price) * 100 * self.leverage

    @property
    def duration_minutes(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 60


@dataclass
class AttributionResult:
    """Result of attribution analysis."""
    name: str  # Strategy name, symbol, or period
    total_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_trade_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    contribution_pct: float = 0.0  # % of total P&L
    avg_duration_minutes: float = 0.0
    sharpe_contribution: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "total_pnl": self.total_pnl,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "avg_trade_pnl": self.avg_trade_pnl,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "contribution_pct": self.contribution_pct,
            "avg_duration_minutes": self.avg_duration_minutes,
            "sharpe_contribution": self.sharpe_contribution,
        }


@dataclass
class FullAttribution:
    """Complete attribution analysis."""
    # Overall
    total_pnl: float = 0.0
    total_trades: int = 0
    total_commission: float = 0.0

    # By dimension
    by_strategy: Dict[str, AttributionResult] = field(default_factory=dict)
    by_symbol: Dict[str, AttributionResult] = field(default_factory=dict)
    by_day_of_week: Dict[str, AttributionResult] = field(default_factory=dict)
    by_hour: Dict[int, AttributionResult] = field(default_factory=dict)
    by_month: Dict[str, AttributionResult] = field(default_factory=dict)
    by_side: Dict[str, AttributionResult] = field(default_factory=dict)

    # Time series
    daily_pnl: Dict[str, float] = field(default_factory=dict)
    cumulative_pnl: List[Tuple[datetime, float]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "total_pnl": self.total_pnl,
            "total_trades": self.total_trades,
            "total_commission": self.total_commission,
            "by_strategy": {k: v.to_dict() for k, v in self.by_strategy.items()},
            "by_symbol": {k: v.to_dict() for k, v in self.by_symbol.items()},
            "by_day_of_week": {k: v.to_dict() for k, v in self.by_day_of_week.items()},
            "by_hour": {k: v.to_dict() for k, v in self.by_hour.items()},
            "by_month": {k: v.to_dict() for k, v in self.by_month.items()},
            "by_side": {k: v.to_dict() for k, v in self.by_side.items()},
            "daily_pnl": self.daily_pnl,
        }


# =============================================================================
# Attribution Engine
# =============================================================================

class PerformanceAttributor:
    """
    Analyze and attribute trading performance.

    Usage:
        attributor = PerformanceAttributor()

        # Add trades
        attributor.add_trade(TradeRecord(...))

        # Or bulk add
        attributor.add_trades(trades_list)

        # Get attribution
        result = attributor.analyze()

        # Get specific analysis
        by_strategy = attributor.get_strategy_attribution()
        by_symbol = attributor.get_symbol_attribution()
    """

    def __init__(self):
        self._trades: List[TradeRecord] = []

    def reset(self) -> None:
        """Clear all trades."""
        self._trades.clear()

    def add_trade(self, trade: TradeRecord) -> None:
        """Add a trade record."""
        self._trades.append(trade)

    def add_trades(self, trades: List[TradeRecord]) -> None:
        """Add multiple trade records."""
        self._trades.extend(trades)

    def _calculate_attribution(
        self,
        trades: List[TradeRecord],
        name: str,
        total_portfolio_pnl: float,
    ) -> AttributionResult:
        """Calculate attribution for a group of trades."""
        if not trades:
            return AttributionResult(name=name)

        pnls = [t.net_pnl for t in trades]
        winning = [t for t in trades if t.net_pnl > 0]
        losing = [t for t in trades if t.net_pnl < 0]

        gross_profit = sum(t.net_pnl for t in winning)
        gross_loss = abs(sum(t.net_pnl for t in losing))

        total_pnl = sum(pnls)
        avg_duration = statistics.mean([t.duration_minutes for t in trades])

        # Contribution to portfolio
        contribution = (total_pnl / total_portfolio_pnl * 100) if total_portfolio_pnl != 0 else 0

        return AttributionResult(
            name=name,
            total_pnl=total_pnl,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            avg_trade_pnl=total_pnl / len(trades),
            win_rate=len(winning) / len(trades),
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            contribution_pct=contribution,
            avg_duration_minutes=avg_duration,
        )

    def get_strategy_attribution(self) -> Dict[str, AttributionResult]:
        """Get attribution by strategy."""
        total_pnl = sum(t.net_pnl for t in self._trades)

        # Group by strategy
        by_strategy: Dict[str, List[TradeRecord]] = {}
        for trade in self._trades:
            key = trade.strategy or "unknown"
            if key not in by_strategy:
                by_strategy[key] = []
            by_strategy[key].append(trade)

        return {
            strategy: self._calculate_attribution(trades, strategy, total_pnl)
            for strategy, trades in by_strategy.items()
        }

    def get_symbol_attribution(self) -> Dict[str, AttributionResult]:
        """Get attribution by symbol."""
        total_pnl = sum(t.net_pnl for t in self._trades)

        by_symbol: Dict[str, List[TradeRecord]] = {}
        for trade in self._trades:
            if trade.symbol not in by_symbol:
                by_symbol[trade.symbol] = []
            by_symbol[trade.symbol].append(trade)

        return {
            symbol: self._calculate_attribution(trades, symbol, total_pnl)
            for symbol, trades in by_symbol.items()
        }

    def get_temporal_attribution(self) -> Dict[str, Dict[str, AttributionResult]]:
        """Get attribution by time dimensions."""
        total_pnl = sum(t.net_pnl for t in self._trades)

        # Day of week
        by_dow: Dict[str, List[TradeRecord]] = {}
        dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for trade in self._trades:
            dow = dow_names[trade.entry_time.weekday()]
            if dow not in by_dow:
                by_dow[dow] = []
            by_dow[dow].append(trade)

        # Hour of day
        by_hour: Dict[int, List[TradeRecord]] = {}
        for trade in self._trades:
            hour = trade.entry_time.hour
            if hour not in by_hour:
                by_hour[hour] = []
            by_hour[hour].append(trade)

        # Month
        by_month: Dict[str, List[TradeRecord]] = {}
        for trade in self._trades:
            month = trade.entry_time.strftime("%Y-%m")
            if month not in by_month:
                by_month[month] = []
            by_month[month].append(trade)

        return {
            "by_day_of_week": {
                dow: self._calculate_attribution(trades, dow, total_pnl)
                for dow, trades in by_dow.items()
            },
            "by_hour": {
                hour: self._calculate_attribution(trades, f"Hour {hour}", total_pnl)
                for hour, trades in by_hour.items()
            },
            "by_month": {
                month: self._calculate_attribution(trades, month, total_pnl)
                for month, trades in by_month.items()
            },
        }

    def get_side_attribution(self) -> Dict[str, AttributionResult]:
        """Get attribution by trade side (LONG/SHORT)."""
        total_pnl = sum(t.net_pnl for t in self._trades)

        by_side: Dict[str, List[TradeRecord]] = {}
        for trade in self._trades:
            side = trade.side.upper()
            if side not in by_side:
                by_side[side] = []
            by_side[side].append(trade)

        return {
            side: self._calculate_attribution(trades, side, total_pnl)
            for side, trades in by_side.items()
        }

    def get_daily_pnl(self) -> Dict[str, float]:
        """Get daily P&L series."""
        daily: Dict[str, float] = {}

        for trade in self._trades:
            day = trade.exit_time.strftime("%Y-%m-%d")
            daily[day] = daily.get(day, 0) + trade.net_pnl

        return dict(sorted(daily.items()))

    def get_cumulative_pnl(self) -> List[Tuple[datetime, float]]:
        """Get cumulative P&L series."""
        sorted_trades = sorted(self._trades, key=lambda t: t.exit_time)

        cumulative = []
        running_total = 0.0

        for trade in sorted_trades:
            running_total += trade.net_pnl
            cumulative.append((trade.exit_time, running_total))

        return cumulative

    def analyze(self) -> FullAttribution:
        """
        Perform full attribution analysis.

        Returns comprehensive breakdown by all dimensions.
        """
        if not self._trades:
            return FullAttribution()

        total_pnl = sum(t.net_pnl for t in self._trades)
        total_commission = sum(t.commission for t in self._trades)

        temporal = self.get_temporal_attribution()

        return FullAttribution(
            total_pnl=total_pnl,
            total_trades=len(self._trades),
            total_commission=total_commission,
            by_strategy=self.get_strategy_attribution(),
            by_symbol=self.get_symbol_attribution(),
            by_day_of_week=temporal["by_day_of_week"],
            by_hour=temporal["by_hour"],
            by_month=temporal["by_month"],
            by_side=self.get_side_attribution(),
            daily_pnl=self.get_daily_pnl(),
            cumulative_pnl=self.get_cumulative_pnl(),
        )

    def get_best_performers(
        self,
        dimension: str = "strategy",
        top_n: int = 5,
    ) -> List[AttributionResult]:
        """Get top performing groups."""
        if dimension == "strategy":
            data = self.get_strategy_attribution()
        elif dimension == "symbol":
            data = self.get_symbol_attribution()
        elif dimension == "side":
            data = self.get_side_attribution()
        else:
            return []

        sorted_data = sorted(
            data.values(),
            key=lambda x: x.total_pnl,
            reverse=True,
        )

        return sorted_data[:top_n]

    def get_worst_performers(
        self,
        dimension: str = "strategy",
        top_n: int = 5,
    ) -> List[AttributionResult]:
        """Get worst performing groups."""
        if dimension == "strategy":
            data = self.get_strategy_attribution()
        elif dimension == "symbol":
            data = self.get_symbol_attribution()
        elif dimension == "side":
            data = self.get_side_attribution()
        else:
            return []

        sorted_data = sorted(
            data.values(),
            key=lambda x: x.total_pnl,
        )

        return sorted_data[:top_n]

    def get_summary_report(self) -> str:
        """Generate a text summary report."""
        result = self.analyze()

        lines = [
            "=" * 60,
            "PERFORMANCE ATTRIBUTION REPORT",
            "=" * 60,
            "",
            f"Total P&L: ${result.total_pnl:,.2f}",
            f"Total Trades: {result.total_trades}",
            f"Total Commission: ${result.total_commission:,.2f}",
            "",
            "-" * 40,
            "BY STRATEGY",
            "-" * 40,
        ]

        for name, attr in sorted(
            result.by_strategy.items(),
            key=lambda x: x[1].total_pnl,
            reverse=True,
        ):
            lines.append(
                f"  {name}: ${attr.total_pnl:+,.2f} "
                f"({attr.contribution_pct:+.1f}%) "
                f"[{attr.total_trades} trades, {attr.win_rate:.0%} WR]"
            )

        lines.extend([
            "",
            "-" * 40,
            "BY SYMBOL",
            "-" * 40,
        ])

        for name, attr in sorted(
            result.by_symbol.items(),
            key=lambda x: x[1].total_pnl,
            reverse=True,
        ):
            lines.append(
                f"  {name}: ${attr.total_pnl:+,.2f} "
                f"({attr.contribution_pct:+.1f}%) "
                f"[{attr.total_trades} trades]"
            )

        lines.extend([
            "",
            "-" * 40,
            "BY SIDE",
            "-" * 40,
        ])

        for name, attr in result.by_side.items():
            lines.append(
                f"  {name}: ${attr.total_pnl:+,.2f} "
                f"[{attr.total_trades} trades, {attr.win_rate:.0%} WR]"
            )

        lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def attribute_trades(trades: List[TradeRecord]) -> FullAttribution:
    """Quick attribution analysis."""
    attributor = PerformanceAttributor()
    attributor.add_trades(trades)
    return attributor.analyze()


def get_strategy_contribution(
    trades: List[TradeRecord],
) -> Dict[str, float]:
    """Get strategy contribution percentages."""
    attributor = PerformanceAttributor()
    attributor.add_trades(trades)
    result = attributor.get_strategy_attribution()

    return {
        name: attr.contribution_pct
        for name, attr in result.items()
    }
