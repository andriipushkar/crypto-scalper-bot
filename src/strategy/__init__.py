"""Trading strategies."""

from src.strategy.base import BaseStrategy, CompositeStrategy, StrategyConfig
from src.strategy.orderbook_imbalance import OrderBookImbalanceStrategy
from src.strategy.volume_spike import VolumeSpikeStrategy
from src.strategy.grid_trading import GridTradingStrategy, GridDirection
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.dca_strategy import DCAStrategy, DCAMode, DCADirection

# Advanced scalping strategies
from src.strategy.impulse_scalping import (
    ImpulseScalpingStrategy,
    LeaderAsset,
    ImpulseEvent,
)
from src.strategy.advanced_orderbook import (
    AdvancedOrderBookStrategy,
    OrderBookWall,
    WallType,
    WallReaction,
    BidAskFlipType,
    IcebergOrder,
    FrontRunOpportunity,
    WallReactionEvent,
    BidAskFlipEvent,
)
from src.strategy.hybrid_scalping import (
    HybridScalpingStrategy,
    HybridSignal,
    SignalSource,
)

# Book-based strategies (from "Скальпинг: практическое руководство трейдера")
from src.strategy.range_trading import (
    RangeTradingStrategy,
    Range,
    RangeState,
    BreakoutType,
    BreakoutEvent as RangeBreakoutEvent,
)
from src.strategy.session_trading import (
    SessionTradingStrategy,
    TradingSession,
    SessionLevels,
    SessionGap,
)
from src.strategy.trendline_breakout import (
    TrendlineBreakoutStrategy,
    TrendDirection,
    PivotType,
    PivotPoint,
    Trendline,
)

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
    "GridTradingStrategy",
    "GridDirection",
    "MeanReversionStrategy",
    "DCAStrategy",
    "DCAMode",
    "DCADirection",
    "MLStrategy",
    "train_model",
    "ML_AVAILABLE",
    # Advanced scalping
    "ImpulseScalpingStrategy",
    "LeaderAsset",
    "ImpulseEvent",
    "AdvancedOrderBookStrategy",
    "OrderBookWall",
    "WallType",
    "WallReaction",
    "BidAskFlipType",
    "IcebergOrder",
    "FrontRunOpportunity",
    "WallReactionEvent",
    "BidAskFlipEvent",
    "HybridScalpingStrategy",
    "HybridSignal",
    "SignalSource",
    # Book-based strategies
    "RangeTradingStrategy",
    "Range",
    "RangeState",
    "BreakoutType",
    "RangeBreakoutEvent",
    "SessionTradingStrategy",
    "TradingSession",
    "SessionLevels",
    "SessionGap",
    "TrendlineBreakoutStrategy",
    "TrendDirection",
    "PivotType",
    "PivotPoint",
    "Trendline",
]
