"""
Walk-Forward Optimization for trading strategies.

Implements rolling window optimization to:
- Avoid overfitting to historical data
- Automatically retrain strategies on new data
- Validate out-of-sample performance
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable

from loguru import logger

from src.backtest import BacktestEngine, BacktestConfig, BacktestResult
from src.strategy.base import BaseStrategy


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class WalkForwardConfig:
    """Walk-forward optimization configuration."""
    # Window sizes
    train_period_days: int = 30  # Training window
    test_period_days: int = 7   # Out-of-sample test window
    step_days: int = 7          # How far to step forward each iteration

    # Optimization settings
    optimization_trials: int = 50
    optimization_metric: str = "sharpe_ratio"
    min_trades: int = 10  # Minimum trades in test period

    # Data settings
    symbol: str = "BTCUSDT"
    data_path: str = "data/raw/market_data.db"

    # Backtest settings
    initial_capital: float = 100.0
    leverage: int = 10
    slippage_bps: float = 1.0
    commission_rate: float = 0.0004

    # Output
    output_dir: str = "data/walk_forward"
    save_results: bool = True


@dataclass
class WalkForwardWindow:
    """A single walk-forward window."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # Results
    best_params: Optional[Dict[str, Any]] = None
    train_metrics: Optional[Dict[str, Any]] = None
    test_metrics: Optional[Dict[str, Any]] = None
    test_trades: int = 0


@dataclass
class WalkForwardResult:
    """Walk-forward optimization result."""
    strategy_name: str
    config: WalkForwardConfig
    windows: List[WalkForwardWindow]
    start_date: datetime
    end_date: datetime

    # Aggregate metrics
    total_return: float = 0.0
    avg_sharpe: float = 0.0
    avg_win_rate: float = 0.0
    total_trades: int = 0
    windows_profitable: int = 0

    # Best parameters history
    param_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy_name,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_return": self.total_return,
            "avg_sharpe": self.avg_sharpe,
            "avg_win_rate": self.avg_win_rate,
            "total_trades": self.total_trades,
            "windows_profitable": self.windows_profitable,
            "windows_total": len(self.windows),
            "windows": [
                {
                    "id": w.window_id,
                    "train_start": w.train_start.isoformat(),
                    "train_end": w.train_end.isoformat(),
                    "test_start": w.test_start.isoformat(),
                    "test_end": w.test_end.isoformat(),
                    "best_params": w.best_params,
                    "train_metrics": w.train_metrics,
                    "test_metrics": w.test_metrics,
                    "test_trades": w.test_trades,
                }
                for w in self.windows
            ],
            "param_history": self.param_history,
        }


# =============================================================================
# Walk-Forward Optimizer
# =============================================================================

class WalkForwardOptimizer:
    """
    Walk-forward optimization engine.

    Performs rolling window optimization where:
    1. Optimize parameters on training window
    2. Test on out-of-sample window
    3. Roll forward and repeat

    This helps avoid overfitting and provides realistic performance estimates.

    Usage:
        optimizer = WalkForwardOptimizer(
            strategy_name="orderbook_imbalance",
            config=WalkForwardConfig(
                train_period_days=30,
                test_period_days=7,
            ),
        )

        result = optimizer.run(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
        )

        print(f"Total return: {result.total_return:.2f}%")
        print(f"Avg Sharpe: {result.avg_sharpe:.2f}")
    """

    def __init__(
        self,
        strategy_name: str,
        strategy_class: type,
        param_space_fn: Callable,
        config: WalkForwardConfig = None,
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            strategy_name: Name of the strategy
            strategy_class: Strategy class to instantiate
            param_space_fn: Function that takes Optuna trial and returns params
            config: Walk-forward configuration
        """
        self.strategy_name = strategy_name
        self.strategy_class = strategy_class
        self.param_space_fn = param_space_fn
        self.config = config or WalkForwardConfig()

        self._windows: List[WalkForwardWindow] = []
        self._result: Optional[WalkForwardResult] = None

    def _create_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[WalkForwardWindow]:
        """Create walk-forward windows."""
        windows = []
        window_id = 0

        current_date = start_date

        while True:
            train_start = current_date
            train_end = train_start + timedelta(days=self.config.train_period_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.config.test_period_days)

            # Check if we have enough data
            if test_end > end_date:
                break

            windows.append(WalkForwardWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            ))

            window_id += 1
            current_date += timedelta(days=self.config.step_days)

        logger.info(f"Created {len(windows)} walk-forward windows")
        return windows

    def _optimize_window(
        self,
        window: WalkForwardWindow,
    ) -> Dict[str, Any]:
        """
        Optimize parameters for a single window.

        Uses Optuna for hyperparameter optimization.
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            logger.error("Optuna required for walk-forward optimization")
            return {}

        def objective(trial) -> float:
            # Get parameters from param space function
            params = self.param_space_fn(trial)

            # Create strategy
            strategy = self.strategy_class(params)

            # Run backtest
            backtest_config = BacktestConfig(
                initial_capital=Decimal(str(self.config.initial_capital)),
                leverage=self.config.leverage,
                slippage_bps=self.config.slippage_bps,
                commission_rate=self.config.commission_rate,
            )

            engine = BacktestEngine(backtest_config)

            try:
                result = engine.run(
                    strategy=strategy,
                    symbol=self.config.symbol,
                    start=window.train_start,
                    end=window.train_end,
                    data_path=self.config.data_path,
                )
            except Exception as e:
                logger.warning(f"Backtest failed: {e}")
                return float("-inf")

            # Get metric
            metric_map = {
                "sharpe_ratio": result.metrics.sharpe_ratio,
                "sortino_ratio": result.metrics.sortino_ratio,
                "total_return": float(result.total_return),
                "profit_factor": result.metrics.profit_factor,
                "win_rate": result.metrics.win_rate,
            }

            return metric_map.get(self.config.optimization_metric, 0)

        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42 + window.window_id),
        )

        # Optimize
        study.optimize(
            objective,
            n_trials=self.config.optimization_trials,
            show_progress_bar=False,
        )

        return study.best_params

    def _test_window(
        self,
        window: WalkForwardWindow,
        params: Dict[str, Any],
    ) -> BacktestResult:
        """Test strategy with optimized parameters on out-of-sample data."""
        strategy = self.strategy_class(params)

        backtest_config = BacktestConfig(
            initial_capital=Decimal(str(self.config.initial_capital)),
            leverage=self.config.leverage,
            slippage_bps=self.config.slippage_bps,
            commission_rate=self.config.commission_rate,
        )

        engine = BacktestEngine(backtest_config)

        return engine.run(
            strategy=strategy,
            symbol=self.config.symbol,
            start=window.test_start,
            end=window.test_end,
            data_path=self.config.data_path,
        )

    def run(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization.

        Args:
            start_date: Start of entire period
            end_date: End of entire period

        Returns:
            WalkForwardResult with aggregate metrics
        """
        logger.info(
            f"Starting walk-forward optimization: {self.strategy_name}"
        )
        logger.info(f"Period: {start_date} to {end_date}")

        # Create windows
        self._windows = self._create_windows(start_date, end_date)

        if not self._windows:
            logger.error("No valid windows created")
            return WalkForwardResult(
                strategy_name=self.strategy_name,
                config=self.config,
                windows=[],
                start_date=start_date,
                end_date=end_date,
            )

        # Process each window
        all_returns = []
        all_sharpe = []
        all_win_rates = []
        param_history = []
        profitable_windows = 0

        for window in self._windows:
            logger.info(f"Processing window {window.window_id + 1}/{len(self._windows)}")

            # Optimize on training period
            try:
                best_params = self._optimize_window(window)
                window.best_params = best_params
                param_history.append({
                    "window_id": window.window_id,
                    "params": best_params,
                })

                logger.info(f"Window {window.window_id}: Best params = {best_params}")

            except Exception as e:
                logger.error(f"Optimization failed for window {window.window_id}: {e}")
                continue

            # Test on out-of-sample period
            try:
                result = self._test_window(window, best_params)

                window.test_metrics = {
                    "total_return": float(result.total_return),
                    "total_return_pct": result.total_return_pct,
                    "sharpe_ratio": result.metrics.sharpe_ratio,
                    "win_rate": result.metrics.win_rate,
                    "profit_factor": result.metrics.profit_factor,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "total_trades": result.metrics.total_trades,
                }
                window.test_trades = result.metrics.total_trades

                # Aggregate metrics
                all_returns.append(result.total_return_pct)
                if result.metrics.sharpe_ratio and result.metrics.sharpe_ratio != float("-inf"):
                    all_sharpe.append(result.metrics.sharpe_ratio)
                all_win_rates.append(result.metrics.win_rate)

                if result.total_return > 0:
                    profitable_windows += 1

                logger.info(
                    f"Window {window.window_id} OOS: "
                    f"return={result.total_return_pct:.2f}%, "
                    f"sharpe={result.metrics.sharpe_ratio:.2f}, "
                    f"trades={result.metrics.total_trades}"
                )

            except Exception as e:
                logger.error(f"Testing failed for window {window.window_id}: {e}")
                continue

        # Calculate aggregate results
        total_return = sum(all_returns) if all_returns else 0
        avg_sharpe = sum(all_sharpe) / len(all_sharpe) if all_sharpe else 0
        avg_win_rate = sum(all_win_rates) / len(all_win_rates) if all_win_rates else 0
        total_trades = sum(w.test_trades for w in self._windows)

        self._result = WalkForwardResult(
            strategy_name=self.strategy_name,
            config=self.config,
            windows=self._windows,
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            avg_sharpe=avg_sharpe,
            avg_win_rate=avg_win_rate,
            total_trades=total_trades,
            windows_profitable=profitable_windows,
            param_history=param_history,
        )

        # Save results
        if self.config.save_results:
            self._save_results()

        logger.info(
            f"Walk-forward complete: "
            f"return={total_return:.2f}%, "
            f"sharpe={avg_sharpe:.2f}, "
            f"profitable={profitable_windows}/{len(self._windows)}"
        )

        return self._result

    def _save_results(self) -> None:
        """Save results to file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.strategy_name}_wf_{timestamp}.json"

        with open(output_dir / filename, "w") as f:
            json.dump(self._result.to_dict(), f, indent=2, default=str)

        logger.info(f"Results saved to {output_dir / filename}")

    def get_latest_params(self) -> Optional[Dict[str, Any]]:
        """Get the most recent optimized parameters."""
        if not self._windows:
            return None

        # Find most recent window with params
        for window in reversed(self._windows):
            if window.best_params:
                return window.best_params

        return None

    def get_param_stability(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze parameter stability across windows.

        Returns statistics on how much each parameter varies.
        """
        if not self._result or not self._result.param_history:
            return {}

        # Collect all parameter values
        param_values: Dict[str, List[float]] = {}

        for entry in self._result.param_history:
            params = entry.get("params", {})
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    if key not in param_values:
                        param_values[key] = []
                    param_values[key].append(float(value))

        # Calculate statistics
        stability = {}
        for param, values in param_values.items():
            if len(values) > 1:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                std = variance ** 0.5
                cv = std / mean if mean != 0 else 0  # Coefficient of variation

                stability[param] = {
                    "mean": mean,
                    "std": std,
                    "min": min(values),
                    "max": max(values),
                    "cv": cv,  # Lower = more stable
                }

        return stability


# =============================================================================
# Auto-Retraining Scheduler
# =============================================================================

class AutoRetrainer:
    """
    Automatic retraining scheduler.

    Monitors performance and triggers retraining when needed.
    """

    def __init__(
        self,
        optimizer: WalkForwardOptimizer,
        retrain_interval_days: int = 7,
        performance_threshold: float = -0.05,  # Retrain if return < -5%
    ):
        self.optimizer = optimizer
        self.retrain_interval_days = retrain_interval_days
        self.performance_threshold = performance_threshold

        self._last_retrain: Optional[datetime] = None
        self._current_params: Optional[Dict[str, Any]] = None
        self._performance_since_retrain: float = 0.0

    def should_retrain(self, current_performance: float) -> bool:
        """
        Check if retraining is needed.

        Returns True if:
        - Never trained before
        - Interval elapsed since last training
        - Performance dropped below threshold
        """
        now = datetime.now()

        # Never trained
        if self._last_retrain is None:
            return True

        # Check interval
        days_since = (now - self._last_retrain).days
        if days_since >= self.retrain_interval_days:
            return True

        # Check performance
        if current_performance < self.performance_threshold:
            logger.info(
                f"Performance {current_performance:.2%} below threshold, "
                f"triggering retrain"
            )
            return True

        return False

    def retrain(
        self,
        lookback_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Run retraining.

        Args:
            lookback_days: How many days of data to use for training

        Returns:
            New optimized parameters
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 7)

        # Run walk-forward with single window
        self.optimizer.config.train_period_days = lookback_days
        self.optimizer.config.test_period_days = 7

        result = self.optimizer.run(start_date, end_date)

        self._last_retrain = datetime.now()
        self._current_params = self.optimizer.get_latest_params()
        self._performance_since_retrain = 0.0

        return self._current_params or {}

    def update_performance(self, period_return: float) -> None:
        """Update cumulative performance since last retrain."""
        self._performance_since_retrain += period_return

    @property
    def current_params(self) -> Optional[Dict[str, Any]]:
        """Get current optimized parameters."""
        return self._current_params

    @property
    def days_since_retrain(self) -> int:
        """Get days since last retrain."""
        if self._last_retrain is None:
            return -1
        return (datetime.now() - self._last_retrain).days
