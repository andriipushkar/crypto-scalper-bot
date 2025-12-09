"""Trading strategies."""

from src.strategy.base import BaseStrategy, CompositeStrategy, StrategyConfig
from src.strategy.orderbook_imbalance import OrderBookImbalanceStrategy
from src.strategy.volume_spike import VolumeSpikeStrategy

# ML strategy (optional - requires PyTorch)
try:
    from src.strategy.ml_strategy import MLStrategy, train_model
    ML_AVAILABLE = True
except ImportError:
    MLStrategy = None
    train_model = None
    ML_AVAILABLE = False

__all__ = [
    "BaseStrategy",
    "CompositeStrategy",
    "StrategyConfig",
    "OrderBookImbalanceStrategy",
    "VolumeSpikeStrategy",
    "MLStrategy",
    "train_model",
    "ML_AVAILABLE",
]
