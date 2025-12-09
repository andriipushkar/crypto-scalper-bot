"""
Optuna-based hyperparameter optimization for trading strategies.

Automatically finds optimal parameters by running backtests.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Type

from loguru import logger

# Try to import Optuna
try:
    import optuna
    from optuna.trial import Trial
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_slice,
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Parameter optimization disabled.")

from src.backtest import BacktestEngine, BacktestConfig, HistoricalDataLoader
from src.strategy.base import BaseStrategy
from src.strategy.orderbook_imbalance import OrderBookImbalanceStrategy
from src.strategy.volume_spike import VolumeSpikeStrategy
from src.strategy.grid_trading import GridTradingStrategy
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.dca_strategy import DCAStrategy


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for optimization."""
    # Optimization settings
    n_trials: int = 100
    timeout: int = 3600  # seconds
    n_jobs: int = 1  # parallel jobs

    # Objective
    objective_metric: str = "sharpe_ratio"  # sharpe_ratio, total_return, profit_factor
    minimize: bool = False

    # Pruning
    enable_pruning: bool = True
    min_trials_for_pruning: int = 10

    # Data
    symbol: str = "BTCUSDT"
    data_path: str = "data/raw/market_data.db"

    # Date range for optimization (use recent data)
    train_days: int = 14
    validation_days: int = 7

    # Backtest config
    initial_capital: float = 100.0
    leverage: int = 10
    slippage_bps: float = 1.0
    commission_rate: float = 0.0004


# =============================================================================
# Parameter Space Definitions
# =============================================================================

def get_orderbook_imbalance_params(trial: 'Trial') -> Dict[str, Any]:
    """Define parameter space for OrderBookImbalanceStrategy."""
    return {
        "enabled": True,
        "imbalance_threshold": trial.suggest_float("imbalance_threshold", 1.1, 3.0),
        "max_spread": trial.suggest_float("max_spread", 0.0001, 0.001),
        "signal_cooldown": trial.suggest_float("signal_cooldown", 5.0, 30.0),
        "levels": trial.suggest_int("levels", 3, 10),
        "min_volume_usd": trial.suggest_float("min_volume_usd", 1000, 20000),
        "min_strength": trial.suggest_float("min_strength", 0.3, 0.7),
    }


def get_volume_spike_params(trial: 'Trial') -> Dict[str, Any]:
    """Define parameter space for VolumeSpikeStrategy."""
    return {
        "enabled": True,
        "volume_multiplier": trial.suggest_float("volume_multiplier", 2.0, 5.0),
        "lookback_seconds": trial.suggest_int("lookback_seconds", 30, 120),
        "min_volume_usd": trial.suggest_float("min_volume_usd", 5000, 30000),
        "signal_cooldown": trial.suggest_float("signal_cooldown", 10.0, 60.0),
        "min_strength": trial.suggest_float("min_strength", 0.3, 0.7),
    }


def get_combined_strategy_params(trial: 'Trial') -> Dict[str, Any]:
    """Define parameter space for combined strategy."""
    return {
        "orderbook_imbalance": get_orderbook_imbalance_params(trial),
        "volume_spike": get_volume_spike_params(trial),
        "combined_threshold": trial.suggest_float("combined_threshold", 0.4, 0.8),
    }


def get_grid_trading_params(trial: 'Trial') -> Dict[str, Any]:
    """Define parameter space for GridTradingStrategy."""
    return {
        "enabled": True,
        "auto_range": True,
        "range_percent": trial.suggest_float("range_percent", 0.02, 0.10),
        "grid_levels": trial.suggest_int("grid_levels", 5, 20),
        "quantity_per_grid": trial.suggest_float("quantity_per_grid", 0.001, 0.01),
        "direction": trial.suggest_categorical("direction", ["neutral", "long", "short"]),
        "take_profit_grids": trial.suggest_int("take_profit_grids", 1, 3),
        "trailing_up": trial.suggest_categorical("trailing_up", [True, False]),
        "trailing_down": trial.suggest_categorical("trailing_down", [True, False]),
        "signal_cooldown": trial.suggest_float("signal_cooldown", 5.0, 60.0),
        "min_strength": trial.suggest_float("min_strength", 0.3, 0.7),
    }


def get_mean_reversion_params(trial: 'Trial') -> Dict[str, Any]:
    """Define parameter space for MeanReversionStrategy."""
    return {
        "enabled": True,
        "lookback_period": trial.suggest_int("lookback_period", 10, 50),
        "std_dev_multiplier": trial.suggest_float("std_dev_multiplier", 1.5, 3.0),
        "entry_z_score": trial.suggest_float("entry_z_score", 1.5, 3.0),
        "exit_z_score": trial.suggest_float("exit_z_score", 0.2, 1.0),
        "rsi_period": trial.suggest_int("rsi_period", 7, 21),
        "rsi_oversold": trial.suggest_float("rsi_oversold", 20, 40),
        "rsi_overbought": trial.suggest_float("rsi_overbought", 60, 80),
        "use_rsi_confirmation": trial.suggest_categorical("use_rsi_confirmation", [True, False]),
        "max_position_bars": trial.suggest_int("max_position_bars", 20, 100),
        "signal_cooldown": trial.suggest_float("signal_cooldown", 5.0, 30.0),
        "min_strength": trial.suggest_float("min_strength", 0.3, 0.7),
    }


def get_dca_params(trial: 'Trial') -> Dict[str, Any]:
    """Define parameter space for DCAStrategy."""
    return {
        "enabled": True,
        "mode": trial.suggest_categorical("mode", ["time_based", "price_based", "hybrid"]),
        "direction": trial.suggest_categorical("direction", ["accumulate", "distribute"]),
        "interval_minutes": trial.suggest_int("interval_minutes", 15, 240),
        "dip_threshold": trial.suggest_float("dip_threshold", 0.01, 0.05),
        "spike_threshold": trial.suggest_float("spike_threshold", 0.01, 0.05),
        "base_order_size": trial.suggest_float("base_order_size", 0.001, 0.01),
        "safety_order_size": trial.suggest_float("safety_order_size", 0.001, 0.02),
        "max_safety_orders": trial.suggest_int("max_safety_orders", 1, 5),
        "safety_order_step": trial.suggest_float("safety_order_step", 0.01, 0.05),
        "safety_order_multiplier": trial.suggest_float("safety_order_multiplier", 1.0, 2.0),
        "take_profit_percent": trial.suggest_float("take_profit_percent", 0.01, 0.05),
        "trailing_take_profit": trial.suggest_categorical("trailing_take_profit", [True, False]),
        "trailing_deviation": trial.suggest_float("trailing_deviation", 0.002, 0.01),
        "signal_cooldown": trial.suggest_float("signal_cooldown", 60.0, 600.0),
        "min_strength": trial.suggest_float("min_strength", 0.3, 0.7),
    }


STRATEGY_PARAM_FUNCTIONS = {
    "orderbook_imbalance": get_orderbook_imbalance_params,
    "volume_spike": get_volume_spike_params,
    "combined": get_combined_strategy_params,
    "grid_trading": get_grid_trading_params,
    "mean_reversion": get_mean_reversion_params,
    "dca": get_dca_params,
}


# =============================================================================
# Strategy Optimizer
# =============================================================================

class StrategyOptimizer:
    """
    Optimize strategy parameters using Optuna.

    Usage:
        optimizer = StrategyOptimizer(
            strategy_name="orderbook_imbalance",
            config=OptimizationConfig(n_trials=50),
        )

        best_params = optimizer.optimize()
        optimizer.save_results("optimization_results.json")
    """

    def __init__(
        self,
        strategy_name: str,
        config: OptimizationConfig = None,
    ):
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna required for optimization")

        self.strategy_name = strategy_name
        self.config = config or OptimizationConfig()

        # Get parameter function
        if strategy_name not in STRATEGY_PARAM_FUNCTIONS:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        self.param_fn = STRATEGY_PARAM_FUNCTIONS[strategy_name]

        # Study
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None

        # Results storage
        self._results: List[Dict[str, Any]] = []

    def _create_strategy(self, params: Dict[str, Any]) -> BaseStrategy:
        """Create strategy instance with given parameters."""
        if self.strategy_name == "orderbook_imbalance":
            return OrderBookImbalanceStrategy(params)
        elif self.strategy_name == "volume_spike":
            return VolumeSpikeStrategy(params)
        elif self.strategy_name == "grid_trading":
            return GridTradingStrategy(params)
        elif self.strategy_name == "mean_reversion":
            return MeanReversionStrategy(params)
        elif self.strategy_name == "dca":
            return DCAStrategy(params)
        elif self.strategy_name == "combined":
            from src.strategy.base import CompositeStrategy
            strategies = [
                OrderBookImbalanceStrategy(params.get("orderbook_imbalance", {})),
                VolumeSpikeStrategy(params.get("volume_spike", {})),
            ]
            return CompositeStrategy(params, strategies)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")

    def _objective(self, trial: 'Trial') -> float:
        """Objective function for Optuna."""
        # Get parameters for this trial
        params = self.param_fn(trial)

        # Create strategy
        strategy = self._create_strategy(params)

        # Run backtest
        backtest_config = BacktestConfig(
            initial_capital=Decimal(str(self.config.initial_capital)),
            leverage=self.config.leverage,
            slippage_bps=self.config.slippage_bps,
            commission_rate=self.config.commission_rate,
        )

        engine = BacktestEngine(backtest_config)

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.train_days)

        try:
            result = engine.run(
                strategy=strategy,
                symbol=self.config.symbol,
                start=start_date,
                end=end_date,
                data_path=self.config.data_path,
            )
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return float('-inf') if not self.config.minimize else float('inf')

        # Get objective metric
        metrics = result.metrics
        metric_map = {
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "total_return": float(result.total_return),
            "total_return_pct": result.total_return_pct,
            "profit_factor": metrics.profit_factor,
            "win_rate": metrics.win_rate,
            "total_trades": metrics.total_trades,
        }

        value = metric_map.get(self.config.objective_metric, 0)

        # Store result
        self._results.append({
            "trial": trial.number,
            "params": params,
            "value": value,
            "metrics": {
                "total_return": float(result.total_return),
                "total_trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown_pct": result.max_drawdown_pct,
            },
        })

        # Log progress
        logger.info(
            f"Trial {trial.number}: {self.config.objective_metric}={value:.4f}, "
            f"trades={metrics.total_trades}, win_rate={metrics.win_rate:.1f}%"
        )

        # Pruning check
        if self.config.enable_pruning and trial.number >= self.config.min_trials_for_pruning:
            # Prune if no trades
            if metrics.total_trades == 0:
                raise optuna.TrialPruned()

        return value

    def optimize(self) -> Dict[str, Any]:
        """
        Run optimization.

        Returns:
            Best parameters found
        """
        logger.info(f"Starting optimization for {self.strategy_name}")
        logger.info(f"Trials: {self.config.n_trials}, Metric: {self.config.objective_metric}")

        # Create study
        direction = "minimize" if self.config.minimize else "maximize"
        self.study = optuna.create_study(
            direction=direction,
            study_name=f"{self.strategy_name}_optimization",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner() if self.config.enable_pruning else None,
        )

        # Run optimization
        self.study.optimize(
            self._objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True,
        )

        # Get best parameters
        self.best_params = self.study.best_params
        best_value = self.study.best_value

        logger.info(f"Optimization complete!")
        logger.info(f"Best {self.config.objective_metric}: {best_value:.4f}")
        logger.info(f"Best params: {self.best_params}")

        return self.best_params

    def save_results(self, path: str) -> None:
        """Save optimization results to JSON."""
        results = {
            "strategy": self.strategy_name,
            "config": {
                "n_trials": self.config.n_trials,
                "objective_metric": self.config.objective_metric,
                "symbol": self.config.symbol,
                "train_days": self.config.train_days,
            },
            "best_params": self.best_params,
            "best_value": self.study.best_value if self.study else None,
            "trials": self._results,
        }

        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {path}")

    def load_results(self, path: str) -> Dict[str, Any]:
        """Load optimization results from JSON."""
        with open(path) as f:
            results = json.load(f)

        self.best_params = results.get("best_params")
        self._results = results.get("trials", [])

        return results

    def get_importance(self) -> Dict[str, float]:
        """Get parameter importance scores."""
        if not self.study:
            return {}

        try:
            importance = optuna.importance.get_param_importances(self.study)
            return dict(importance)
        except Exception as e:
            logger.warning(f"Could not calculate importance: {e}")
            return {}

    def plot_history(self, save_path: str = None):
        """Plot optimization history."""
        if not self.study:
            return

        fig = plot_optimization_history(self.study)

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_importance(self, save_path: str = None):
        """Plot parameter importance."""
        if not self.study:
            return

        try:
            fig = plot_param_importances(self.study)

            if save_path:
                fig.write_html(save_path)
            else:
                fig.show()
        except Exception as e:
            logger.warning(f"Could not plot importance: {e}")


# =============================================================================
# Convenience Function
# =============================================================================

def optimize_strategy(
    strategy_name: str,
    n_trials: int = 50,
    symbol: str = "BTCUSDT",
    data_path: str = "data/raw/market_data.db",
    objective: str = "sharpe_ratio",
    output_path: str = None,
) -> Dict[str, Any]:
    """
    Convenience function to optimize a strategy.

    Args:
        strategy_name: Name of strategy (orderbook_imbalance, volume_spike, combined)
        n_trials: Number of trials
        symbol: Trading symbol
        data_path: Path to data file
        objective: Objective metric
        output_path: Optional path to save results

    Returns:
        Best parameters
    """
    config = OptimizationConfig(
        n_trials=n_trials,
        symbol=symbol,
        data_path=data_path,
        objective_metric=objective,
    )

    optimizer = StrategyOptimizer(strategy_name, config)
    best_params = optimizer.optimize()

    if output_path:
        optimizer.save_results(output_path)

    return best_params
