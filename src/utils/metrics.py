"""
Performance metrics calculation.

Provides functions for calculating trading performance metrics
from trade history.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Any
import math


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_time: datetime
    exit_time: datetime
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    pnl: Decimal
    commission: Decimal
    strategy: str = ""

    @property
    def duration(self) -> timedelta:
        """Trade duration."""
        return self.exit_time - self.entry_time

    @property
    def duration_seconds(self) -> float:
        """Trade duration in seconds."""
        return self.duration.total_seconds()

    @property
    def return_pct(self) -> Decimal:
        """Return percentage."""
        if self.entry_price == 0:
            return Decimal("0")

        if self.side == "LONG":
            return ((self.exit_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.exit_price) / self.entry_price) * 100

    @property
    def is_winner(self) -> bool:
        """Is this a winning trade?"""
        return self.pnl > 0

    @property
    def net_pnl(self) -> Decimal:
        """P&L after commission."""
        return self.pnl - self.commission


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    # Basic counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # P&L
    total_pnl: Decimal = Decimal("0")
    total_commission: Decimal = Decimal("0")
    gross_profit: Decimal = Decimal("0")
    gross_loss: Decimal = Decimal("0")

    # Averages
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    avg_trade: Decimal = Decimal("0")

    # Ratios
    win_rate: float = 0.0
    profit_factor: float = 0.0
    payoff_ratio: float = 0.0  # avg_win / avg_loss

    # Risk metrics
    max_drawdown: Decimal = Decimal("0")
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Time metrics
    avg_trade_duration: float = 0.0  # seconds
    avg_win_duration: float = 0.0
    avg_loss_duration: float = 0.0
    trades_per_hour: float = 0.0

    # Streaks
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_streak: int = 0

    # By symbol
    by_symbol: Dict[str, Dict] = field(default_factory=dict)

    # By strategy
    by_strategy: Dict[str, Dict] = field(default_factory=dict)


def calculate_metrics(
    trades: List[TradeRecord],
    initial_capital: Decimal = Decimal("100"),
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics from trade history.

    Args:
        trades: List of completed trades
        initial_capital: Starting capital for ratio calculations
        risk_free_rate: Annual risk-free rate for Sharpe calculation

    Returns:
        PerformanceMetrics with all calculated values
    """
    if not trades:
        return PerformanceMetrics()

    metrics = PerformanceMetrics()

    # Sort trades by exit time
    sorted_trades = sorted(trades, key=lambda t: t.exit_time)

    # Basic counts
    metrics.total_trades = len(sorted_trades)
    winners = [t for t in sorted_trades if t.is_winner]
    losers = [t for t in sorted_trades if not t.is_winner]

    metrics.winning_trades = len(winners)
    metrics.losing_trades = len(losers)

    # P&L calculations
    metrics.total_pnl = sum(t.pnl for t in sorted_trades)
    metrics.total_commission = sum(t.commission for t in sorted_trades)
    metrics.gross_profit = sum(t.pnl for t in winners) if winners else Decimal("0")
    metrics.gross_loss = abs(sum(t.pnl for t in losers)) if losers else Decimal("0")

    # Averages
    if winners:
        metrics.avg_win = metrics.gross_profit / len(winners)
    if losers:
        metrics.avg_loss = metrics.gross_loss / len(losers)
    metrics.avg_trade = metrics.total_pnl / metrics.total_trades

    # Win rate
    metrics.win_rate = (metrics.winning_trades / metrics.total_trades) * 100

    # Profit factor
    if metrics.gross_loss > 0:
        metrics.profit_factor = float(metrics.gross_profit / metrics.gross_loss)
    else:
        metrics.profit_factor = float("inf") if metrics.gross_profit > 0 else 0.0

    # Payoff ratio
    if metrics.avg_loss > 0:
        metrics.payoff_ratio = float(metrics.avg_win / metrics.avg_loss)

    # Drawdown calculation
    equity_curve = _calculate_equity_curve(sorted_trades, initial_capital)
    metrics.max_drawdown, metrics.max_drawdown_pct = _calculate_max_drawdown(equity_curve)

    # Sharpe & Sortino ratios
    returns = [float(t.return_pct) for t in sorted_trades]
    metrics.sharpe_ratio = _calculate_sharpe_ratio(returns, risk_free_rate)
    metrics.sortino_ratio = _calculate_sortino_ratio(returns, risk_free_rate)

    # Time metrics
    durations = [t.duration_seconds for t in sorted_trades]
    metrics.avg_trade_duration = sum(durations) / len(durations)

    if winners:
        win_durations = [t.duration_seconds for t in winners]
        metrics.avg_win_duration = sum(win_durations) / len(win_durations)

    if losers:
        loss_durations = [t.duration_seconds for t in losers]
        metrics.avg_loss_duration = sum(loss_durations) / len(loss_durations)

    # Trades per hour
    if len(sorted_trades) >= 2:
        total_time = (sorted_trades[-1].exit_time - sorted_trades[0].entry_time).total_seconds()
        if total_time > 0:
            metrics.trades_per_hour = (metrics.total_trades / total_time) * 3600

    # Streaks
    metrics.max_consecutive_wins, metrics.max_consecutive_losses, metrics.current_streak = \
        _calculate_streaks(sorted_trades)

    # By symbol
    metrics.by_symbol = _calculate_by_group(sorted_trades, lambda t: t.symbol)

    # By strategy
    metrics.by_strategy = _calculate_by_group(sorted_trades, lambda t: t.strategy)

    return metrics


def _calculate_equity_curve(
    trades: List[TradeRecord],
    initial_capital: Decimal,
) -> List[Decimal]:
    """Calculate equity curve from trades."""
    equity = [initial_capital]
    current = initial_capital

    for trade in trades:
        current += trade.net_pnl
        equity.append(current)

    return equity


def _calculate_max_drawdown(equity_curve: List[Decimal]) -> tuple:
    """
    Calculate maximum drawdown.

    Returns:
        Tuple of (max_drawdown_value, max_drawdown_percentage)
    """
    if not equity_curve:
        return Decimal("0"), 0.0

    peak = equity_curve[0]
    max_dd = Decimal("0")
    max_dd_pct = 0.0

    for equity in equity_curve:
        if equity > peak:
            peak = equity

        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
            if peak > 0:
                max_dd_pct = float(dd / peak) * 100

    return max_dd, max_dd_pct


def _calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate Sharpe ratio.

    Uses daily returns assumption for annualization.
    """
    if len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)
    excess_return = mean_return - (risk_free_rate / 365)  # Daily risk-free rate

    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance) if variance > 0 else 0

    if std_dev == 0:
        return 0.0

    # Annualize (assuming ~250 trading days)
    sharpe = (excess_return / std_dev) * math.sqrt(250)

    return round(sharpe, 3)


def _calculate_sortino_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate Sortino ratio.

    Only considers downside deviation.
    """
    if len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)
    excess_return = mean_return - (risk_free_rate / 365)

    # Downside deviation (only negative returns)
    negative_returns = [r for r in returns if r < 0]
    if not negative_returns:
        return float("inf") if excess_return > 0 else 0.0

    downside_variance = sum(r ** 2 for r in negative_returns) / len(negative_returns)
    downside_dev = math.sqrt(downside_variance)

    if downside_dev == 0:
        return 0.0

    sortino = (excess_return / downside_dev) * math.sqrt(250)

    return round(sortino, 3)


def _calculate_streaks(trades: List[TradeRecord]) -> tuple:
    """
    Calculate win/loss streaks.

    Returns:
        Tuple of (max_wins, max_losses, current_streak)
    """
    if not trades:
        return 0, 0, 0

    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0

    for trade in trades:
        if trade.is_winner:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)

    # Current streak (positive = wins, negative = losses)
    current_streak = current_wins if current_wins > 0 else -current_losses

    return max_wins, max_losses, current_streak


def _calculate_by_group(
    trades: List[TradeRecord],
    key_func,
) -> Dict[str, Dict]:
    """Calculate metrics grouped by a key function."""
    groups = {}

    for trade in trades:
        key = key_func(trade)
        if not key:
            key = "unknown"

        if key not in groups:
            groups[key] = []
        groups[key].append(trade)

    result = {}
    for key, group_trades in groups.items():
        wins = sum(1 for t in group_trades if t.is_winner)
        total = len(group_trades)
        pnl = sum(t.pnl for t in group_trades)

        result[key] = {
            "trades": total,
            "wins": wins,
            "win_rate": (wins / total * 100) if total > 0 else 0,
            "pnl": float(pnl),
        }

    return result


def format_metrics_report(metrics: PerformanceMetrics) -> str:
    """
    Format metrics as a readable report.

    Args:
        metrics: Performance metrics

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 50,
        "PERFORMANCE REPORT",
        "=" * 50,
        "",
        "SUMMARY",
        "-" * 30,
        f"Total Trades:     {metrics.total_trades}",
        f"Winning Trades:   {metrics.winning_trades}",
        f"Losing Trades:    {metrics.losing_trades}",
        f"Win Rate:         {metrics.win_rate:.1f}%",
        "",
        "PROFIT & LOSS",
        "-" * 30,
        f"Total P&L:        ${float(metrics.total_pnl):.2f}",
        f"Gross Profit:     ${float(metrics.gross_profit):.2f}",
        f"Gross Loss:       ${float(metrics.gross_loss):.2f}",
        f"Net P&L:          ${float(metrics.total_pnl - metrics.total_commission):.2f}",
        f"Commission:       ${float(metrics.total_commission):.2f}",
        "",
        "AVERAGES",
        "-" * 30,
        f"Avg Win:          ${float(metrics.avg_win):.2f}",
        f"Avg Loss:         ${float(metrics.avg_loss):.2f}",
        f"Avg Trade:        ${float(metrics.avg_trade):.2f}",
        f"Payoff Ratio:     {metrics.payoff_ratio:.2f}",
        "",
        "RISK METRICS",
        "-" * 30,
        f"Profit Factor:    {metrics.profit_factor:.2f}",
        f"Max Drawdown:     ${float(metrics.max_drawdown):.2f} ({metrics.max_drawdown_pct:.1f}%)",
        f"Sharpe Ratio:     {metrics.sharpe_ratio:.2f}",
        f"Sortino Ratio:    {metrics.sortino_ratio:.2f}",
        "",
        "TIME METRICS",
        "-" * 30,
        f"Avg Trade Time:   {metrics.avg_trade_duration:.1f}s",
        f"Trades/Hour:      {metrics.trades_per_hour:.1f}",
        "",
        "STREAKS",
        "-" * 30,
        f"Max Win Streak:   {metrics.max_consecutive_wins}",
        f"Max Loss Streak:  {metrics.max_consecutive_losses}",
        f"Current Streak:   {metrics.current_streak}",
        "",
    ]

    # By symbol
    if metrics.by_symbol:
        lines.append("BY SYMBOL")
        lines.append("-" * 30)
        for symbol, data in metrics.by_symbol.items():
            lines.append(
                f"  {symbol}: {data['trades']} trades, "
                f"{data['win_rate']:.1f}% WR, ${data['pnl']:.2f}"
            )
        lines.append("")

    # By strategy
    if metrics.by_strategy:
        lines.append("BY STRATEGY")
        lines.append("-" * 30)
        for strategy, data in metrics.by_strategy.items():
            lines.append(
                f"  {strategy}: {data['trades']} trades, "
                f"{data['win_rate']:.1f}% WR, ${data['pnl']:.2f}"
            )
        lines.append("")

    lines.append("=" * 50)

    return "\n".join(lines)
