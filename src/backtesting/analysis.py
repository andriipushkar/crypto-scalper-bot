"""
Backtest Analysis

Analyze and visualize backtest results.
"""
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
import json

from loguru import logger

from .engine import BacktestResult, Trade


class BacktestAnalyzer:
    """Analyze backtest results."""

    def __init__(self, result: BacktestResult):
        self.result = result

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "period": {
                "start": self.result.start_time.isoformat(),
                "end": self.result.end_time.isoformat(),
                "duration_days": (self.result.end_time - self.result.start_time).days,
            },
            "returns": {
                "initial_balance": self.result.initial_balance,
                "final_balance": self.result.final_balance,
                "total_return": self.result.total_return,
                "total_return_pct": self.result.total_return_pct,
            },
            "trades": {
                "total": self.result.total_trades,
                "winning": self.result.winning_trades,
                "losing": self.result.losing_trades,
                "win_rate": self.result.win_rate,
            },
            "risk": {
                "max_drawdown": self.result.max_drawdown,
                "max_drawdown_pct": self.result.max_drawdown_pct,
                "sharpe_ratio": self.result.sharpe_ratio,
                "sortino_ratio": self.result.sortino_ratio,
                "profit_factor": self.result.profit_factor,
            },
            "averages": {
                "avg_trade": self.result.avg_trade,
                "avg_win": self.result.avg_win,
                "avg_loss": self.result.avg_loss,
                "largest_win": self.result.largest_win,
                "largest_loss": self.result.largest_loss,
            },
            "streaks": {
                "max_consecutive_wins": self.result.max_consecutive_wins,
                "max_consecutive_losses": self.result.max_consecutive_losses,
            },
        }

    def trade_analysis(self) -> Dict[str, Any]:
        """Analyze individual trades."""
        if not self.result.trades:
            return {}

        # By symbol
        by_symbol: Dict[str, List[Trade]] = {}
        for trade in self.result.trades:
            if trade.symbol not in by_symbol:
                by_symbol[trade.symbol] = []
            by_symbol[trade.symbol].append(trade)

        symbol_stats = {}
        for symbol, trades in by_symbol.items():
            wins = [t for t in trades if t.pnl > 0]
            symbol_stats[symbol] = {
                "trades": len(trades),
                "wins": len(wins),
                "win_rate": len(wins) / len(trades) * 100,
                "total_pnl": sum(t.pnl for t in trades),
                "avg_pnl": sum(t.pnl for t in trades) / len(trades),
            }

        # By side
        longs = [t for t in self.result.trades if t.side == "long"]
        shorts = [t for t in self.result.trades if t.side == "short"]

        long_wins = [t for t in longs if t.pnl > 0]
        short_wins = [t for t in shorts if t.pnl > 0]

        side_stats = {
            "long": {
                "trades": len(longs),
                "win_rate": len(long_wins) / len(longs) * 100 if longs else 0,
                "total_pnl": sum(t.pnl for t in longs),
            },
            "short": {
                "trades": len(shorts),
                "win_rate": len(short_wins) / len(shorts) * 100 if shorts else 0,
                "total_pnl": sum(t.pnl for t in shorts),
            },
        }

        # Duration analysis
        durations = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in self.result.trades]
        avg_duration = sum(durations) / len(durations) if durations else 0

        # P&L distribution
        pnls = [t.pnl for t in self.result.trades]

        return {
            "by_symbol": symbol_stats,
            "by_side": side_stats,
            "duration": {
                "avg_minutes": avg_duration,
                "min_minutes": min(durations) if durations else 0,
                "max_minutes": max(durations) if durations else 0,
            },
            "pnl_distribution": {
                "min": min(pnls) if pnls else 0,
                "max": max(pnls) if pnls else 0,
                "median": sorted(pnls)[len(pnls)//2] if pnls else 0,
            },
        }

    def monthly_returns(self) -> Dict[str, float]:
        """Calculate monthly returns."""
        if not self.result.trades:
            return {}

        monthly: Dict[str, float] = {}

        for trade in self.result.trades:
            key = trade.exit_time.strftime("%Y-%m")
            monthly[key] = monthly.get(key, 0) + trade.pnl

        return monthly

    def drawdown_periods(self) -> List[Dict[str, Any]]:
        """Identify drawdown periods."""
        if not self.result.equity_curve:
            return []

        periods = []
        in_drawdown = False
        peak = self.result.equity_curve[0]["equity"]
        dd_start = None
        max_dd = 0

        for point in self.result.equity_curve:
            equity = point["equity"]

            if equity > peak:
                if in_drawdown:
                    # Drawdown ended
                    periods.append({
                        "start": dd_start,
                        "end": point["timestamp"],
                        "max_drawdown": max_dd,
                        "max_drawdown_pct": (max_dd / peak) * 100,
                    })
                    in_drawdown = False
                    max_dd = 0
                peak = equity
            else:
                dd = peak - equity
                if dd > 0 and not in_drawdown:
                    in_drawdown = True
                    dd_start = point["timestamp"]
                if dd > max_dd:
                    max_dd = dd

        return periods

    def risk_metrics(self) -> Dict[str, float]:
        """Calculate risk metrics."""
        if len(self.result.equity_curve) < 2:
            return {}

        # Calculate returns
        returns = []
        for i in range(1, len(self.result.equity_curve)):
            prev = self.result.equity_curve[i-1]["equity"]
            curr = self.result.equity_curve[i]["equity"]
            if prev > 0:
                returns.append((curr - prev) / prev)

        if not returns:
            return {}

        import statistics

        # Basic stats
        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns) if len(returns) > 1 else 0

        # Downside returns
        downside_returns = [r for r in returns if r < 0]
        downside_std = statistics.stdev(downside_returns) if len(downside_returns) > 1 else 0

        # VaR (95%)
        sorted_returns = sorted(returns)
        var_95_idx = int(len(sorted_returns) * 0.05)
        var_95 = sorted_returns[var_95_idx] if var_95_idx < len(sorted_returns) else 0

        # Calmar ratio
        calmar = 0
        if self.result.max_drawdown_pct > 0:
            annualized_return = mean_return * 252
            calmar = annualized_return / self.result.max_drawdown_pct

        return {
            "mean_return": mean_return,
            "std_return": std_return,
            "downside_std": downside_std,
            "var_95": var_95,
            "calmar_ratio": calmar,
            "risk_reward": abs(mean_return / std_return) if std_return > 0 else 0,
        }

    def generate_report(self) -> str:
        """Generate text report."""
        summary = self.summary()
        trades = self.trade_analysis()
        monthly = self.monthly_returns()

        report = f"""
================================================================================
                           BACKTEST REPORT
================================================================================

PERIOD
------
Start:    {summary['period']['start']}
End:      {summary['period']['end']}
Duration: {summary['period']['duration_days']} days

RETURNS
-------
Initial Balance:  ${summary['returns']['initial_balance']:,.2f}
Final Balance:    ${summary['returns']['final_balance']:,.2f}
Total Return:     ${summary['returns']['total_return']:,.2f} ({summary['returns']['total_return_pct']:.2f}%)

TRADES
------
Total Trades:  {summary['trades']['total']}
Winning:       {summary['trades']['winning']}
Losing:        {summary['trades']['losing']}
Win Rate:      {summary['trades']['win_rate']:.1f}%

RISK METRICS
------------
Max Drawdown:    ${summary['risk']['max_drawdown']:,.2f} ({summary['risk']['max_drawdown_pct']:.2f}%)
Sharpe Ratio:    {summary['risk']['sharpe_ratio']:.2f}
Profit Factor:   {summary['risk']['profit_factor']:.2f}

TRADE STATISTICS
----------------
Average Trade:   ${summary['averages']['avg_trade']:,.2f}
Average Win:     ${summary['averages']['avg_win']:,.2f}
Average Loss:    ${summary['averages']['avg_loss']:,.2f}
Largest Win:     ${summary['averages']['largest_win']:,.2f}
Largest Loss:    ${summary['averages']['largest_loss']:,.2f}

Max Consecutive Wins:   {summary['streaks']['max_consecutive_wins']}
Max Consecutive Losses: {summary['streaks']['max_consecutive_losses']}

MONTHLY RETURNS
---------------
"""
        for month, pnl in sorted(monthly.items()):
            report += f"{month}: ${pnl:+,.2f}\n"

        report += """
================================================================================
"""
        return report

    def to_json(self) -> str:
        """Export analysis to JSON."""
        return json.dumps({
            "summary": self.summary(),
            "trade_analysis": self.trade_analysis(),
            "monthly_returns": self.monthly_returns(),
            "risk_metrics": self.risk_metrics(),
        }, indent=2, default=str)

    def save_report(self, filepath: str):
        """Save report to file."""
        report = self.generate_report()
        with open(filepath, "w") as f:
            f.write(report)
        logger.info(f"Saved report to {filepath}")

    def save_trades_csv(self, filepath: str):
        """Save trades to CSV."""
        import csv

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trade_id", "symbol", "side", "entry_time", "exit_time",
                "entry_price", "exit_price", "quantity", "pnl", "pnl_pct"
            ])

            for trade in self.result.trades:
                writer.writerow([
                    trade.id,
                    trade.symbol,
                    trade.side,
                    trade.entry_time.isoformat(),
                    trade.exit_time.isoformat(),
                    trade.entry_price,
                    trade.exit_price,
                    trade.quantity,
                    trade.pnl,
                    trade.pnl_pct,
                ])

        logger.info(f"Saved trades to {filepath}")
