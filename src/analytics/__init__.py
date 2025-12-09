"""Analytics module for trading performance analysis."""

from src.analytics.trade_journal import (
    TradeJournal,
    TradeEntry,
    TradeOutcome,
    TradeQuality,
    EmotionalState,
)
from src.analytics.attribution import (
    PerformanceAttributor,
    TradeRecord,
    AttributionResult,
    FullAttribution,
    attribute_trades,
    get_strategy_contribution,
)
from src.analytics.sentiment import (
    SentimentAnalyzer,
    SentimentScore,
    SentimentLevel,
    AggregateSentiment,
    FearGreedIndexProvider,
    get_market_sentiment,
    get_fear_greed_index,
    sentiment_to_signal_modifier,
)

__all__ = [
    # Trade Journal
    "TradeJournal",
    "TradeEntry",
    "TradeOutcome",
    "TradeQuality",
    "EmotionalState",
    # Attribution
    "PerformanceAttributor",
    "TradeRecord",
    "AttributionResult",
    "FullAttribution",
    "attribute_trades",
    "get_strategy_contribution",
    # Sentiment
    "SentimentAnalyzer",
    "SentimentScore",
    "SentimentLevel",
    "AggregateSentiment",
    "FearGreedIndexProvider",
    "get_market_sentiment",
    "get_fear_greed_index",
    "sentiment_to_signal_modifier",
]
