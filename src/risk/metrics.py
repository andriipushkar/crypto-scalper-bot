"""
Advanced risk metrics for portfolio analysis.

Implements:
- Value at Risk (VaR)
- Conditional VaR (CVaR) / Expected Shortfall
- Maximum Drawdown
- Calmar Ratio
- Tail Risk Metrics
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import statistics

from loguru import logger


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    # VaR metrics (as percentage losses)
    var_95: float = 0.0  # 95% VaR
    var_99: float = 0.0  # 99% VaR
    var_99_9: float = 0.0  # 99.9% VaR

    # CVaR / Expected Shortfall (average loss beyond VaR)
    cvar_95: float = 0.0  # Expected shortfall at 95%
    cvar_99: float = 0.0  # Expected shortfall at 99%

    # Drawdown metrics
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    current_drawdown: float = 0.0

    # Tail risk
    skewness: float = 0.0  # Negative = fat left tail
    kurtosis: float = 0.0  # High = fat tails
    tail_ratio: float = 0.0  # Right tail / left tail

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0

    # Other metrics
    volatility_annual: float = 0.0
    downside_deviation: float = 0.0
    ulcer_index: float = 0.0  # Measures drawdown pain
    pain_index: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "var_95": self.var_95,
            "var_99": self.var_99,
            "var_99_9": self.var_99_9,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "max_drawdown": self.max_drawdown,
            "avg_drawdown": self.avg_drawdown,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "current_drawdown": self.current_drawdown,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "tail_ratio": self.tail_ratio,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "omega_ratio": self.omega_ratio,
            "volatility_annual": self.volatility_annual,
            "downside_deviation": self.downside_deviation,
            "ulcer_index": self.ulcer_index,
            "pain_index": self.pain_index,
        }


# =============================================================================
# Risk Calculator
# =============================================================================

class RiskCalculator:
    """
    Calculate advanced risk metrics from returns.

    Usage:
        calc = RiskCalculator(risk_free_rate=0.05)

        # Add returns
        calc.add_return(0.02)
        calc.add_return(-0.01)
        calc.add_return(0.015)

        # Or batch
        calc.set_returns([0.02, -0.01, 0.015, 0.005, -0.02])

        # Calculate metrics
        metrics = calc.calculate()
        print(f"VaR 95%: {metrics.var_95:.2%}")
        print(f"CVaR 95%: {metrics.cvar_95:.2%}")
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,  # Annual risk-free rate
        periods_per_year: int = 252,  # Trading days per year
    ):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

        self._returns: List[float] = []
        self._equity_curve: List[float] = []
        self._timestamps: List[datetime] = []

    def reset(self) -> None:
        """Reset all data."""
        self._returns.clear()
        self._equity_curve.clear()
        self._timestamps.clear()

    def add_return(
        self,
        ret: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Add a return observation."""
        self._returns.append(ret)
        if timestamp:
            self._timestamps.append(timestamp)

        # Update equity curve
        if not self._equity_curve:
            self._equity_curve.append(1.0 + ret)
        else:
            self._equity_curve.append(self._equity_curve[-1] * (1.0 + ret))

    def set_returns(self, returns: List[float]) -> None:
        """Set returns from list."""
        self.reset()
        for ret in returns:
            self.add_return(ret)

    def set_equity_curve(self, equity: List[float]) -> None:
        """Set equity curve and derive returns."""
        self.reset()
        self._equity_curve = list(equity)

        for i in range(1, len(equity)):
            if equity[i - 1] != 0:
                ret = (equity[i] - equity[i - 1]) / equity[i - 1]
                self._returns.append(ret)

    # =========================================================================
    # VaR Calculations
    # =========================================================================

    def _historical_var(self, confidence: float) -> float:
        """
        Calculate Historical VaR.

        VaR is the maximum loss at a given confidence level.
        Negative values indicate losses.
        """
        if len(self._returns) < 10:
            return 0.0

        sorted_returns = sorted(self._returns)
        index = int((1 - confidence) * len(sorted_returns))
        return abs(sorted_returns[index])  # Return as positive loss

    def _parametric_var(self, confidence: float) -> float:
        """
        Calculate Parametric (Gaussian) VaR.

        Assumes returns are normally distributed.
        """
        if len(self._returns) < 10:
            return 0.0

        mean = statistics.mean(self._returns)
        std = statistics.stdev(self._returns)

        # Z-scores for confidence levels
        z_scores = {
            0.90: 1.282,
            0.95: 1.645,
            0.99: 2.326,
            0.999: 3.090,
        }

        z = z_scores.get(confidence, 1.645)
        var = mean - z * std

        return abs(min(var, 0))  # Return as positive loss

    def _cvar(self, confidence: float) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).

        CVaR is the average loss in the worst (1-confidence)% of cases.
        """
        if len(self._returns) < 10:
            return 0.0

        sorted_returns = sorted(self._returns)
        cutoff = int((1 - confidence) * len(sorted_returns))

        if cutoff < 1:
            cutoff = 1

        tail_returns = sorted_returns[:cutoff]
        return abs(statistics.mean(tail_returns))

    # =========================================================================
    # Drawdown Calculations
    # =========================================================================

    def _calculate_drawdowns(self) -> Tuple[List[float], float, float, int]:
        """
        Calculate drawdown series and metrics.

        Returns:
            (drawdown_series, max_dd, avg_dd, max_duration_periods)
        """
        if not self._equity_curve:
            return [], 0.0, 0.0, 0

        drawdowns = []
        peak = self._equity_curve[0]
        max_dd = 0.0
        current_dd_start = 0
        max_duration = 0
        current_duration = 0

        for i, equity in enumerate(self._equity_curve):
            if equity > peak:
                peak = equity
                current_dd_start = i
                current_duration = 0
            else:
                dd = (peak - equity) / peak
                drawdowns.append(dd)
                max_dd = max(max_dd, dd)
                current_duration = i - current_dd_start
                max_duration = max(max_duration, current_duration)

        avg_dd = statistics.mean(drawdowns) if drawdowns else 0.0

        return drawdowns, max_dd, avg_dd, max_duration

    def _current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if not self._equity_curve:
            return 0.0

        peak = max(self._equity_curve)
        current = self._equity_curve[-1]

        if peak == 0:
            return 0.0

        return (peak - current) / peak

    # =========================================================================
    # Distribution Metrics
    # =========================================================================

    def _skewness(self) -> float:
        """
        Calculate skewness of returns.

        Negative skewness = fat left tail (more extreme losses)
        Positive skewness = fat right tail (more extreme gains)
        """
        if len(self._returns) < 3:
            return 0.0

        n = len(self._returns)
        mean = statistics.mean(self._returns)
        std = statistics.stdev(self._returns)

        if std == 0:
            return 0.0

        skew = sum((r - mean) ** 3 for r in self._returns) / n
        return skew / (std ** 3)

    def _kurtosis(self) -> float:
        """
        Calculate excess kurtosis.

        High kurtosis = fat tails (more extreme events)
        Normal distribution has kurtosis = 3 (excess = 0)
        """
        if len(self._returns) < 4:
            return 0.0

        n = len(self._returns)
        mean = statistics.mean(self._returns)
        std = statistics.stdev(self._returns)

        if std == 0:
            return 0.0

        kurt = sum((r - mean) ** 4 for r in self._returns) / n
        return (kurt / (std ** 4)) - 3  # Excess kurtosis

    def _tail_ratio(self) -> float:
        """
        Calculate tail ratio (right tail / left tail).

        Ratio > 1 means more extreme gains than losses
        Ratio < 1 means more extreme losses than gains
        """
        if len(self._returns) < 20:
            return 1.0

        sorted_returns = sorted(self._returns)
        n = len(sorted_returns)
        cutoff = max(1, int(n * 0.05))

        left_tail = sorted_returns[:cutoff]
        right_tail = sorted_returns[-cutoff:]

        left_avg = abs(statistics.mean(left_tail)) if left_tail else 1
        right_avg = statistics.mean(right_tail) if right_tail else 1

        if left_avg == 0:
            return float("inf")

        return right_avg / left_avg

    # =========================================================================
    # Risk-Adjusted Returns
    # =========================================================================

    def _sharpe_ratio(self) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(self._returns) < 2:
            return 0.0

        mean_return = statistics.mean(self._returns)
        std_return = statistics.stdev(self._returns)

        if std_return == 0:
            return 0.0

        rf_per_period = self.risk_free_rate / self.periods_per_year
        excess_return = mean_return - rf_per_period

        return (excess_return / std_return) * math.sqrt(self.periods_per_year)

    def _sortino_ratio(self) -> float:
        """
        Calculate annualized Sortino ratio.

        Uses downside deviation instead of total volatility.
        """
        if len(self._returns) < 2:
            return 0.0

        mean_return = statistics.mean(self._returns)
        rf_per_period = self.risk_free_rate / self.periods_per_year

        # Downside returns only
        downside_returns = [min(r - rf_per_period, 0) ** 2 for r in self._returns]
        downside_dev = math.sqrt(statistics.mean(downside_returns))

        if downside_dev == 0:
            return float("inf") if mean_return > rf_per_period else 0.0

        excess_return = mean_return - rf_per_period
        return (excess_return / downside_dev) * math.sqrt(self.periods_per_year)

    def _calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio.

        Annual return / Maximum drawdown
        """
        _, max_dd, _, _ = self._calculate_drawdowns()

        if max_dd == 0 or len(self._returns) < 2:
            return 0.0

        # Annualized return
        total_return = self._equity_curve[-1] / self._equity_curve[0] - 1 if self._equity_curve else 0
        periods = len(self._returns)
        annual_return = (1 + total_return) ** (self.periods_per_year / periods) - 1

        return annual_return / max_dd

    def _omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio.

        Sum of gains above threshold / Sum of losses below threshold
        """
        if len(self._returns) < 2:
            return 1.0

        gains = sum(max(r - threshold, 0) for r in self._returns)
        losses = sum(abs(min(r - threshold, 0)) for r in self._returns)

        if losses == 0:
            return float("inf")

        return gains / losses

    # =========================================================================
    # Additional Metrics
    # =========================================================================

    def _volatility(self) -> float:
        """Calculate annualized volatility."""
        if len(self._returns) < 2:
            return 0.0

        std = statistics.stdev(self._returns)
        return std * math.sqrt(self.periods_per_year)

    def _downside_deviation(self) -> float:
        """Calculate annualized downside deviation."""
        if len(self._returns) < 2:
            return 0.0

        rf_per_period = self.risk_free_rate / self.periods_per_year
        downside_returns = [min(r - rf_per_period, 0) ** 2 for r in self._returns]

        if not downside_returns:
            return 0.0

        dd = math.sqrt(statistics.mean(downside_returns))
        return dd * math.sqrt(self.periods_per_year)

    def _ulcer_index(self) -> float:
        """
        Calculate Ulcer Index.

        Measures the depth and duration of drawdowns.
        Lower is better.
        """
        drawdowns, _, _, _ = self._calculate_drawdowns()

        if not drawdowns:
            return 0.0

        squared_dd = [dd ** 2 for dd in drawdowns]
        return math.sqrt(statistics.mean(squared_dd)) * 100

    def _pain_index(self) -> float:
        """
        Calculate Pain Index.

        Average drawdown, representing the "pain" of investing.
        """
        drawdowns, _, avg_dd, _ = self._calculate_drawdowns()
        return avg_dd * 100

    # =========================================================================
    # Main Calculation
    # =========================================================================

    def calculate(self) -> RiskMetrics:
        """
        Calculate all risk metrics.

        Returns:
            RiskMetrics object with all calculated values
        """
        if len(self._returns) < 10:
            logger.warning("Insufficient data for reliable risk metrics")

        drawdowns, max_dd, avg_dd, max_duration = self._calculate_drawdowns()

        return RiskMetrics(
            # VaR
            var_95=self._historical_var(0.95),
            var_99=self._historical_var(0.99),
            var_99_9=self._historical_var(0.999),

            # CVaR
            cvar_95=self._cvar(0.95),
            cvar_99=self._cvar(0.99),

            # Drawdown
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            max_drawdown_duration_days=max_duration,
            current_drawdown=self._current_drawdown(),

            # Distribution
            skewness=self._skewness(),
            kurtosis=self._kurtosis(),
            tail_ratio=self._tail_ratio(),

            # Risk-adjusted
            sharpe_ratio=self._sharpe_ratio(),
            sortino_ratio=self._sortino_ratio(),
            calmar_ratio=self._calmar_ratio(),
            omega_ratio=self._omega_ratio(),

            # Other
            volatility_annual=self._volatility(),
            downside_deviation=self._downside_deviation(),
            ulcer_index=self._ulcer_index(),
            pain_index=self._pain_index(),
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def calculate_var(
    returns: List[float],
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Calculate Value at Risk.

    Args:
        returns: List of returns
        confidence: Confidence level (0.95 = 95%)
        method: "historical" or "parametric"

    Returns:
        VaR as positive percentage
    """
    calc = RiskCalculator()
    calc.set_returns(returns)

    if method == "parametric":
        return calc._parametric_var(confidence)
    return calc._historical_var(confidence)


def calculate_cvar(
    returns: List[float],
    confidence: float = 0.95,
) -> float:
    """
    Calculate Conditional VaR (Expected Shortfall).

    Args:
        returns: List of returns
        confidence: Confidence level

    Returns:
        CVaR as positive percentage
    """
    calc = RiskCalculator()
    calc.set_returns(returns)
    return calc._cvar(confidence)


def calculate_max_drawdown(equity_curve: List[float]) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown.

    Args:
        equity_curve: List of equity values

    Returns:
        (max_drawdown, peak_index, trough_index)
    """
    if not equity_curve:
        return 0.0, 0, 0

    peak = equity_curve[0]
    peak_idx = 0
    max_dd = 0.0
    max_dd_peak_idx = 0
    max_dd_trough_idx = 0

    for i, equity in enumerate(equity_curve):
        if equity > peak:
            peak = equity
            peak_idx = i
        else:
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
                max_dd_peak_idx = peak_idx
                max_dd_trough_idx = i

    return max_dd, max_dd_peak_idx, max_dd_trough_idx


def calculate_sharpe(
    returns: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: List of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Annualized Sharpe ratio
    """
    calc = RiskCalculator(risk_free_rate, periods_per_year)
    calc.set_returns(returns)
    return calc._sharpe_ratio()
