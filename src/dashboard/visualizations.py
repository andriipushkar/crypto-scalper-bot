"""
Dashboard Visualizations

Chart generation for portfolio analytics.
"""
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from decimal import Decimal
import io
import base64

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from loguru import logger


class ChartGenerator:
    """Generate charts for portfolio analytics."""

    def __init__(
        self,
        style: str = "dark_background",
        figsize: Tuple[int, int] = (12, 6),
        dpi: int = 100,
    ):
        self.style = style
        self.figsize = figsize
        self.dpi = dpi

        if HAS_MATPLOTLIB:
            plt.style.use(style)

    def _check_matplotlib(self) -> None:
        """Check if matplotlib is available."""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib not installed. Install with: pip install matplotlib")

    def _fig_to_base64(self, fig: "Figure") -> str:
        """Convert figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64

    def _fig_to_bytes(self, fig: "Figure") -> bytes:
        """Convert figure to bytes."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        data = buf.read()
        plt.close(fig)
        return data

    def equity_curve(
        self,
        timestamps: List[datetime],
        values: List[float],
        title: str = "Equity Curve",
        as_base64: bool = True,
    ) -> str | bytes:
        """Generate equity curve chart."""
        self._check_matplotlib()

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(timestamps, values, linewidth=2, color='#00ff88')
        ax.fill_between(timestamps, values, alpha=0.3, color='#00ff88')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()

        ax.grid(True, alpha=0.3)

        if as_base64:
            return self._fig_to_base64(fig)
        return self._fig_to_bytes(fig)

    def drawdown_chart(
        self,
        timestamps: List[datetime],
        drawdowns: List[float],
        title: str = "Drawdown",
        as_base64: bool = True,
    ) -> str | bytes:
        """Generate drawdown chart."""
        self._check_matplotlib()

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.fill_between(timestamps, drawdowns, 0, alpha=0.7, color='#ff4444')
        ax.plot(timestamps, drawdowns, linewidth=1, color='#ff6666')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        ax.grid(True, alpha=0.3)
        ax.set_ylim(min(drawdowns) * 1.1 if drawdowns else -10, 0)

        if as_base64:
            return self._fig_to_base64(fig)
        return self._fig_to_bytes(fig)

    def daily_pnl_chart(
        self,
        dates: List[datetime],
        pnls: List[float],
        title: str = "Daily P&L",
        as_base64: bool = True,
    ) -> str | bytes:
        """Generate daily P&L bar chart."""
        self._check_matplotlib()

        fig, ax = plt.subplots(figsize=self.figsize)

        colors = ['#00ff88' if p >= 0 else '#ff4444' for p in pnls]
        ax.bar(dates, pnls, color=colors, alpha=0.8)

        ax.axhline(y=0, color='white', linewidth=0.5)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('P&L ($)')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        ax.grid(True, alpha=0.3, axis='y')

        if as_base64:
            return self._fig_to_base64(fig)
        return self._fig_to_bytes(fig)

    def returns_distribution(
        self,
        returns: List[float],
        title: str = "Returns Distribution",
        bins: int = 50,
        as_base64: bool = True,
    ) -> str | bytes:
        """Generate returns distribution histogram."""
        self._check_matplotlib()

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.hist(returns, bins=bins, color='#4488ff', alpha=0.7, edgecolor='white')

        # Add mean line
        if returns:
            mean = sum(returns) / len(returns)
            ax.axvline(x=mean, color='#00ff88', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}%')
            ax.axvline(x=0, color='white', linestyle='-', linewidth=1)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Frequency')

        ax.legend()
        ax.grid(True, alpha=0.3)

        if as_base64:
            return self._fig_to_base64(fig)
        return self._fig_to_bytes(fig)

    def allocation_pie(
        self,
        labels: List[str],
        values: List[float],
        title: str = "Portfolio Allocation",
        as_base64: bool = True,
    ) -> str | bytes:
        """Generate allocation pie chart."""
        self._check_matplotlib()

        fig, ax = plt.subplots(figsize=(8, 8))

        colors = plt.cm.Set3(range(len(labels)))

        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
        )

        ax.set_title(title, fontsize=14, fontweight='bold')

        if as_base64:
            return self._fig_to_base64(fig)
        return self._fig_to_bytes(fig)

    def performance_heatmap(
        self,
        data: Dict[str, Dict[str, float]],
        title: str = "Performance Heatmap",
        as_base64: bool = True,
    ) -> str | bytes:
        """Generate performance heatmap (day of week vs hour)."""
        self._check_matplotlib()
        import numpy as np

        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        hours = list(range(24))

        # Build matrix
        matrix = np.zeros((7, 24))
        for day_idx, day in enumerate(days):
            for hour in hours:
                key = f"{day}_{hour}"
                matrix[day_idx, hour] = data.get(day, {}).get(str(hour), 0)

        fig, ax = plt.subplots(figsize=(14, 6))

        cmap = plt.cm.RdYlGn
        im = ax.imshow(matrix, cmap=cmap, aspect='auto')

        ax.set_xticks(range(24))
        ax.set_xticklabels(hours)
        ax.set_yticks(range(7))
        ax.set_yticklabels(days)

        ax.set_xlabel('Hour (UTC)')
        ax.set_ylabel('Day of Week')
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.colorbar(im, ax=ax, label='P&L ($)')

        if as_base64:
            return self._fig_to_base64(fig)
        return self._fig_to_bytes(fig)

    def win_rate_by_symbol(
        self,
        symbols: List[str],
        win_rates: List[float],
        trades_count: List[int],
        title: str = "Win Rate by Symbol",
        as_base64: bool = True,
    ) -> str | bytes:
        """Generate win rate by symbol chart."""
        self._check_matplotlib()

        fig, ax = plt.subplots(figsize=self.figsize)

        colors = ['#00ff88' if wr >= 50 else '#ff4444' for wr in win_rates]
        bars = ax.bar(symbols, win_rates, color=colors, alpha=0.8)

        # Add trade count labels
        for bar, count in zip(bars, trades_count):
            height = bar.get_height()
            ax.annotate(
                f'n={count}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9,
            )

        ax.axhline(y=50, color='white', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Symbol')
        ax.set_ylabel('Win Rate (%)')
        ax.set_ylim(0, 100)

        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)

        if as_base64:
            return self._fig_to_base64(fig)
        return self._fig_to_bytes(fig)

    def cumulative_returns(
        self,
        timestamps: List[datetime],
        returns: List[float],
        benchmark_returns: Optional[List[float]] = None,
        title: str = "Cumulative Returns",
        as_base64: bool = True,
    ) -> str | bytes:
        """Generate cumulative returns chart."""
        self._check_matplotlib()
        import numpy as np

        fig, ax = plt.subplots(figsize=self.figsize)

        # Calculate cumulative returns
        cum_returns = np.cumprod([1 + r/100 for r in returns]) - 1
        cum_returns = [r * 100 for r in cum_returns]

        ax.plot(timestamps, cum_returns, linewidth=2, color='#00ff88', label='Strategy')

        if benchmark_returns:
            cum_bench = np.cumprod([1 + r/100 for r in benchmark_returns]) - 1
            cum_bench = [r * 100 for r in cum_bench]
            ax.plot(timestamps, cum_bench, linewidth=2, color='#4488ff', label='Benchmark', alpha=0.7)

        ax.axhline(y=0, color='white', linewidth=0.5)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (%)')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        ax.legend()
        ax.grid(True, alpha=0.3)

        if as_base64:
            return self._fig_to_base64(fig)
        return self._fig_to_bytes(fig)

    def rolling_sharpe(
        self,
        timestamps: List[datetime],
        sharpe_values: List[float],
        window: int = 20,
        title: str = "Rolling Sharpe Ratio",
        as_base64: bool = True,
    ) -> str | bytes:
        """Generate rolling Sharpe ratio chart."""
        self._check_matplotlib()

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(timestamps, sharpe_values, linewidth=2, color='#4488ff')
        ax.fill_between(
            timestamps,
            sharpe_values,
            0,
            where=[s >= 0 for s in sharpe_values],
            alpha=0.3,
            color='#00ff88',
        )
        ax.fill_between(
            timestamps,
            sharpe_values,
            0,
            where=[s < 0 for s in sharpe_values],
            alpha=0.3,
            color='#ff4444',
        )

        ax.axhline(y=0, color='white', linewidth=0.5)
        ax.axhline(y=1, color='#00ff88', linewidth=1, linestyle='--', alpha=0.5, label='Good (>1)')
        ax.axhline(y=2, color='#00ff88', linewidth=1, linestyle='--', alpha=0.7, label='Excellent (>2)')

        ax.set_title(f'{title} ({window}-period)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sharpe Ratio')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        ax.legend()
        ax.grid(True, alpha=0.3)

        if as_base64:
            return self._fig_to_base64(fig)
        return self._fig_to_bytes(fig)

    def trade_scatter(
        self,
        durations: List[float],
        pnls: List[float],
        sizes: Optional[List[float]] = None,
        title: str = "Trade Analysis",
        as_base64: bool = True,
    ) -> str | bytes:
        """Generate trade duration vs P&L scatter plot."""
        self._check_matplotlib()

        fig, ax = plt.subplots(figsize=self.figsize)

        colors = ['#00ff88' if p >= 0 else '#ff4444' for p in pnls]

        if sizes:
            scatter = ax.scatter(durations, pnls, c=colors, s=sizes, alpha=0.6)
        else:
            scatter = ax.scatter(durations, pnls, c=colors, s=50, alpha=0.6)

        ax.axhline(y=0, color='white', linewidth=0.5)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Duration (minutes)')
        ax.set_ylabel('P&L ($)')

        ax.grid(True, alpha=0.3)

        if as_base64:
            return self._fig_to_base64(fig)
        return self._fig_to_bytes(fig)

    def monthly_returns_table(
        self,
        monthly_data: Dict[str, Dict[str, float]],
        title: str = "Monthly Returns",
        as_base64: bool = True,
    ) -> str | bytes:
        """Generate monthly returns table visualization."""
        self._check_matplotlib()
        import numpy as np

        years = sorted(monthly_data.keys())
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Build matrix
        matrix = []
        for year in years:
            row = []
            for i, month in enumerate(months, 1):
                value = monthly_data.get(year, {}).get(str(i), 0)
                row.append(value)
            matrix.append(row)

        matrix = np.array(matrix)

        fig, ax = plt.subplots(figsize=(14, len(years) + 2))

        cmap = plt.cm.RdYlGn
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=-20, vmax=20)

        ax.set_xticks(range(12))
        ax.set_xticklabels(months)
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years)

        # Add text annotations
        for i in range(len(years)):
            for j in range(12):
                value = matrix[i, j]
                color = 'white' if abs(value) > 10 else 'black'
                ax.text(j, i, f'{value:.1f}%', ha='center', va='center', color=color, fontsize=9)

        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Return (%)')

        if as_base64:
            return self._fig_to_base64(fig)
        return self._fig_to_bytes(fig)
