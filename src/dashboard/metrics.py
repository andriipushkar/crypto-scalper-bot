"""
Performance Metrics

Calculates trading performance metrics and statistics.
"""
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
import statistics
import math

from loguru import logger


@dataclass
class PerformanceStats:
    """Comprehensive performance statistics."""
    # Returns
    total_return: float
    total_return_pct: float
    annualized_return: float

    # Risk
    volatility: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown
    max_drawdown: float
    max_drawdown_duration: int
    current_drawdown: float

    # Trade Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # P&L
    gross_profit: float
    gross_loss: float
    net_profit: float
    profit_factor: float

    # Trade Metrics
    avg_win: float
    avg_loss: float
    avg_trade: float
    largest_win: float
    largest_loss: float

    # Ratios
    reward_risk_ratio: float
    expectancy: float

    # Streaks
    max_win_streak: int
    max_loss_streak: int
    current_streak: int
    current_streak_type: str


class PerformanceMetrics:
    """Calculate performance metrics from trade history."""

    def __init__(
        self,
        risk_free_rate: float = 0.05,  # Annual
        trading_days: int = 252,
    ):
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def calculate_all(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: Optional[List[float]] = None,
        initial_balance: float = 10000,
    ) -> PerformanceStats:
        """Calculate all performance metrics."""
        if not trades:
            return self._empty_stats()

        # Extract P&L
        pnls = [t.get('pnl', 0) for t in trades]

        # Calculate returns
        returns = self._calculate_returns(trades, equity_curve, initial_balance)
        daily_returns = returns.get('daily', [])

        # Calculate all metrics
        total_return = sum(pnls)
        total_return_pct = (total_return / initial_balance) * 100 if initial_balance > 0 else 0

        # Annualized return
        if trades:
            first_trade = trades[0].get('entry_time', '')
            last_trade = trades[-1].get('exit_time', '')
            days = self._days_between(first_trade, last_trade)
            years = days / 365 if days > 0 else 1
            annualized_return = ((1 + total_return_pct / 100) ** (1 / years) - 1) * 100
        else:
            annualized_return = 0

        # Risk metrics
        volatility = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0
        annualized_vol = volatility * math.sqrt(self.trading_days)

        sharpe = self._sharpe_ratio(daily_returns)
        sortino = self._sortino_ratio(daily_returns)

        # Drawdown
        dd_metrics = self._drawdown_metrics(equity_curve or self._build_equity_curve(trades, initial_balance))
        calmar = annualized_return / dd_metrics['max_drawdown'] if dd_metrics['max_drawdown'] > 0 else 0

        # Trade statistics
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p < 0]

        win_rate = (len(winning) / len(pnls)) * 100 if pnls else 0

        gross_profit = sum(winning)
        gross_loss = abs(sum(losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = statistics.mean(winning) if winning else 0
        avg_loss = abs(statistics.mean(losing)) if losing else 0
        avg_trade = statistics.mean(pnls) if pnls else 0

        reward_risk = avg_win / avg_loss if avg_loss > 0 else float('inf')

        # Expectancy
        win_prob = len(winning) / len(pnls) if pnls else 0
        loss_prob = len(losing) / len(pnls) if pnls else 0
        expectancy = (win_prob * avg_win) - (loss_prob * avg_loss)

        # Streaks
        streaks = self._calculate_streaks(pnls)

        return PerformanceStats(
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            volatility=volatility,
            annualized_volatility=annualized_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=dd_metrics['max_drawdown'],
            max_drawdown_duration=dd_metrics['max_duration'],
            current_drawdown=dd_metrics['current'],
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_profit=total_return,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            largest_win=max(winning) if winning else 0,
            largest_loss=abs(min(losing)) if losing else 0,
            reward_risk_ratio=reward_risk,
            expectancy=expectancy,
            max_win_streak=streaks['max_win'],
            max_loss_streak=streaks['max_loss'],
            current_streak=streaks['current'],
            current_streak_type=streaks['current_type'],
        )

    def _empty_stats(self) -> PerformanceStats:
        """Return empty stats when no trades."""
        return PerformanceStats(
            total_return=0, total_return_pct=0, annualized_return=0,
            volatility=0, annualized_volatility=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            max_drawdown=0, max_drawdown_duration=0, current_drawdown=0,
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
            gross_profit=0, gross_loss=0, net_profit=0, profit_factor=0,
            avg_win=0, avg_loss=0, avg_trade=0,
            largest_win=0, largest_loss=0,
            reward_risk_ratio=0, expectancy=0,
            max_win_streak=0, max_loss_streak=0,
            current_streak=0, current_streak_type='none',
        )

    def _calculate_returns(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: Optional[List[float]],
        initial_balance: float,
    ) -> Dict[str, List[float]]:
        """Calculate daily returns."""
        if equity_curve and len(equity_curve) > 1:
            daily = [
                (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                for i in range(1, len(equity_curve))
                if equity_curve[i-1] > 0
            ]
            return {'daily': daily}

        # Build from trades
        equity = self._build_equity_curve(trades, initial_balance)
        if len(equity) > 1:
            daily = [
                (equity[i] - equity[i-1]) / equity[i-1]
                for i in range(1, len(equity))
                if equity[i-1] > 0
            ]
            return {'daily': daily}

        return {'daily': []}

    def _build_equity_curve(
        self,
        trades: List[Dict[str, Any]],
        initial_balance: float,
    ) -> List[float]:
        """Build equity curve from trades."""
        equity = [initial_balance]
        balance = initial_balance

        for trade in trades:
            balance += trade.get('pnl', 0)
            equity.append(balance)

        return equity

    def _sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0

        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)

        if std_return == 0:
            return 0

        daily_rf = self.risk_free_rate / self.trading_days
        excess_return = mean_return - daily_rf

        return (excess_return / std_return) * math.sqrt(self.trading_days)

    def _sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio (uses downside deviation)."""
        if len(returns) < 2:
            return 0

        mean_return = statistics.mean(returns)
        daily_rf = self.risk_free_rate / self.trading_days

        # Downside returns
        downside = [r for r in returns if r < daily_rf]
        if not downside:
            return float('inf') if mean_return > daily_rf else 0

        downside_std = statistics.stdev(downside) if len(downside) > 1 else abs(downside[0])

        if downside_std == 0:
            return 0

        excess_return = mean_return - daily_rf
        return (excess_return / downside_std) * math.sqrt(self.trading_days)

    def _drawdown_metrics(self, equity_curve: List[float]) -> Dict[str, Any]:
        """Calculate drawdown metrics."""
        if not equity_curve:
            return {'max_drawdown': 0, 'max_duration': 0, 'current': 0}

        peak = equity_curve[0]
        max_dd = 0
        max_duration = 0
        current_duration = 0
        dd_start = None

        for i, value in enumerate(equity_curve):
            if value > peak:
                peak = value
                if dd_start is not None:
                    max_duration = max(max_duration, current_duration)
                dd_start = None
                current_duration = 0
            else:
                dd = (peak - value) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)
                if dd_start is None:
                    dd_start = i
                current_duration = i - dd_start

        current_dd = (peak - equity_curve[-1]) / peak * 100 if peak > 0 else 0

        return {
            'max_drawdown': max_dd,
            'max_duration': max(max_duration, current_duration),
            'current': current_dd,
        }

    def _calculate_streaks(self, pnls: List[float]) -> Dict[str, Any]:
        """Calculate win/loss streaks."""
        if not pnls:
            return {'max_win': 0, 'max_loss': 0, 'current': 0, 'current_type': 'none'}

        max_win = 0
        max_loss = 0
        current = 0
        current_type = 'win' if pnls[0] > 0 else 'loss'

        win_streak = 0
        loss_streak = 0

        for pnl in pnls:
            if pnl > 0:
                win_streak += 1
                loss_streak = 0
                max_win = max(max_win, win_streak)
            elif pnl < 0:
                loss_streak += 1
                win_streak = 0
                max_loss = max(max_loss, loss_streak)
            else:
                win_streak = 0
                loss_streak = 0

        current = win_streak if win_streak > 0 else loss_streak
        current_type = 'win' if win_streak > 0 else ('loss' if loss_streak > 0 else 'none')

        return {
            'max_win': max_win,
            'max_loss': max_loss,
            'current': current,
            'current_type': current_type,
        }

    def _days_between(self, start: str, end: str) -> int:
        """Calculate days between two ISO timestamps."""
        try:
            if not start or not end:
                return 1

            start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))

            return max(1, (end_dt - start_dt).days)
        except (ValueError, TypeError):
            return 1

    def calculate_rolling_metrics(
        self,
        trades: List[Dict[str, Any]],
        window: int = 20,
    ) -> List[Dict[str, Any]]:
        """Calculate rolling performance metrics."""
        if len(trades) < window:
            return []

        results = []

        for i in range(window, len(trades) + 1):
            window_trades = trades[i-window:i]
            pnls = [t.get('pnl', 0) for t in window_trades]

            winning = [p for p in pnls if p > 0]
            win_rate = (len(winning) / len(pnls)) * 100 if pnls else 0

            results.append({
                'index': i,
                'timestamp': window_trades[-1].get('exit_time', ''),
                'total_pnl': sum(pnls),
                'avg_pnl': statistics.mean(pnls) if pnls else 0,
                'win_rate': win_rate,
                'volatility': statistics.stdev(pnls) if len(pnls) > 1 else 0,
            })

        return results

    def compare_periods(
        self,
        trades: List[Dict[str, Any]],
        period1_trades: List[Dict[str, Any]],
        period2_trades: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare performance between two periods."""
        stats1 = self.calculate_all(period1_trades)
        stats2 = self.calculate_all(period2_trades)

        return {
            'period1': {
                'trades': stats1.total_trades,
                'return': stats1.total_return,
                'win_rate': stats1.win_rate,
                'sharpe': stats1.sharpe_ratio,
            },
            'period2': {
                'trades': stats2.total_trades,
                'return': stats2.total_return,
                'win_rate': stats2.win_rate,
                'sharpe': stats2.sharpe_ratio,
            },
            'comparison': {
                'return_diff': stats2.total_return - stats1.total_return,
                'win_rate_diff': stats2.win_rate - stats1.win_rate,
                'sharpe_diff': stats2.sharpe_ratio - stats1.sharpe_ratio,
            },
        }
