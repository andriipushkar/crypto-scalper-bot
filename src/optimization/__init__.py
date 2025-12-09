"""Parameter optimization module."""

from src.optimization.optuna_optimizer import (
    StrategyOptimizer,
    OptimizationConfig,
    optimize_strategy,
    STRATEGY_PARAM_FUNCTIONS,
    get_orderbook_imbalance_params,
    get_volume_spike_params,
    get_grid_trading_params,
    get_mean_reversion_params,
)
from src.optimization.walk_forward import (
    WalkForwardOptimizer,
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardWindow,
    AutoRetrainer,
)

__all__ = [
    # Optuna
    "StrategyOptimizer",
    "OptimizationConfig",
    "optimize_strategy",
    "STRATEGY_PARAM_FUNCTIONS",
    "get_orderbook_imbalance_params",
    "get_volume_spike_params",
    "get_grid_trading_params",
    "get_mean_reversion_params",
    # Walk-Forward
    "WalkForwardOptimizer",
    "WalkForwardConfig",
    "WalkForwardResult",
    "WalkForwardWindow",
    "AutoRetrainer",
]
