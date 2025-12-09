"""
Trading module.

Contains trading strategies and event-based trading systems.
"""

from .news_event_trading import (
    NewsEventTrader,
    EconomicCalendarProvider,
    TokenUnlockProvider,
    NewsSentimentAnalyzer,
    EconomicEvent,
    TokenUnlock,
    NewsItem,
    EventTradeSignal,
    EventTradingConfig,
    EventType,
    EventImpact,
    TradingAction,
    create_event_trader,
)

__all__ = [
    "NewsEventTrader",
    "EconomicCalendarProvider",
    "TokenUnlockProvider",
    "NewsSentimentAnalyzer",
    "EconomicEvent",
    "TokenUnlock",
    "NewsItem",
    "EventTradeSignal",
    "EventTradingConfig",
    "EventType",
    "EventImpact",
    "TradingAction",
    "create_event_trader",
]
