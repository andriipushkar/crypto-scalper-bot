"""
News and Event Trading Module.

Provides automated trading strategies based on:
- Economic calendar events (CPI, FOMC, NFP, etc.)
- Token unlocks and vesting schedules
- News sentiment analysis
- Pre-event positioning and post-event volatility trading
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
import json
import aiohttp
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of tradeable events."""

    # Economic calendar events
    CPI = "cpi"  # Consumer Price Index
    FOMC = "fomc"  # Federal Open Market Committee
    NFP = "nfp"  # Non-Farm Payrolls
    GDP = "gdp"  # Gross Domestic Product
    PPI = "ppi"  # Producer Price Index
    RETAIL_SALES = "retail_sales"
    UNEMPLOYMENT = "unemployment"
    PMI = "pmi"  # Purchasing Managers Index

    # Crypto-specific events
    TOKEN_UNLOCK = "token_unlock"
    HALVING = "halving"
    HARD_FORK = "hard_fork"
    MAINNET_LAUNCH = "mainnet_launch"
    EXCHANGE_LISTING = "exchange_listing"
    AIRDROP = "airdrop"

    # Market events
    ETF_DECISION = "etf_decision"
    REGULATORY = "regulatory"
    EARNINGS = "earnings"  # For crypto companies

    # Other
    CUSTOM = "custom"


class EventImpact(Enum):
    """Impact level of events."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TradingAction(Enum):
    """Trading actions for event strategies."""

    WAIT = "wait"  # No action, wait for better setup
    LONG_PRE_EVENT = "long_pre_event"
    SHORT_PRE_EVENT = "short_pre_event"
    STRADDLE = "straddle"  # Long volatility
    FADE_MOVE = "fade_move"  # Counter-trend after initial move
    BREAKOUT = "breakout"  # Follow the breakout
    REDUCE_EXPOSURE = "reduce_exposure"


@dataclass
class EconomicEvent:
    """Economic calendar event."""

    event_id: str
    event_type: EventType
    name: str
    datetime_utc: datetime
    currency: str  # USD, EUR, etc.
    impact: EventImpact

    # Forecast vs actual
    previous: Optional[str] = None
    forecast: Optional[str] = None
    actual: Optional[str] = None

    # Trading parameters
    expected_volatility_pct: Decimal = Decimal("2.0")
    affected_symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])

    # Meta
    source: str = "unknown"
    description: str = ""

    @property
    def time_until(self) -> timedelta:
        """Time until the event."""
        return self.datetime_utc - datetime.utcnow()

    @property
    def is_past(self) -> bool:
        """Check if event has passed."""
        return datetime.utcnow() > self.datetime_utc

    @property
    def surprise_factor(self) -> Optional[float]:
        """Calculate surprise factor if actual is known."""
        if not self.actual or not self.forecast:
            return None
        try:
            actual = float(self.actual.replace("%", "").replace(",", ""))
            forecast = float(self.forecast.replace("%", "").replace(",", ""))
            if forecast == 0:
                return None
            return (actual - forecast) / abs(forecast)
        except (ValueError, ZeroDivisionError):
            return None


@dataclass
class TokenUnlock:
    """Token unlock/vesting event."""

    unlock_id: str
    symbol: str
    token_name: str
    datetime_utc: datetime

    # Unlock details
    unlock_amount: Decimal
    unlock_value_usd: Decimal
    unlock_pct_of_supply: Decimal
    unlock_pct_of_mcap: Decimal

    # Recipient info
    recipient_type: str  # team, investor, foundation, etc.
    recipient_name: Optional[str] = None

    # Historical behavior
    historical_sell_pressure_pct: Decimal = Decimal("50")
    expected_price_impact_pct: Decimal = Decimal("-2")

    # Vesting schedule
    is_cliff: bool = False
    vesting_period_days: Optional[int] = None
    total_vesting_amount: Optional[Decimal] = None

    @property
    def time_until(self) -> timedelta:
        """Time until unlock."""
        return self.datetime_utc - datetime.utcnow()

    @property
    def is_significant(self) -> bool:
        """Check if unlock is significant (>1% of market cap)."""
        return self.unlock_pct_of_mcap >= Decimal("1.0")


@dataclass
class NewsItem:
    """News item with sentiment."""

    news_id: str
    title: str
    content: str
    source: str
    published_at: datetime
    url: str

    # Sentiment analysis
    sentiment_score: float  # -1 to 1
    sentiment_magnitude: float  # 0 to 1

    # Relevance
    symbols: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    category: str = "general"

    # Impact assessment
    expected_impact: EventImpact = EventImpact.LOW

    @property
    def is_bullish(self) -> bool:
        """Check if sentiment is bullish."""
        return self.sentiment_score > 0.3

    @property
    def is_bearish(self) -> bool:
        """Check if sentiment is bearish."""
        return self.sentiment_score < -0.3


@dataclass
class EventTradingConfig:
    """Configuration for event trading."""

    # Pre-event trading
    pre_event_hours: int = 24  # Hours before event to start considering positions
    max_pre_event_position_pct: Decimal = Decimal("25")  # Max position as % of capital

    # Post-event trading
    post_event_cooldown_minutes: int = 15  # Wait after event before trading
    fade_threshold_pct: Decimal = Decimal("3")  # Min move % to consider fading
    breakout_confirmation_minutes: int = 5  # Time to confirm breakout

    # Risk management
    event_sl_multiplier: Decimal = Decimal("1.5")  # Wider stops around events
    reduce_exposure_hours_before: int = 1  # Reduce exposure before high-impact events
    max_event_trades_per_day: int = 5

    # Token unlocks
    min_unlock_pct_for_trade: Decimal = Decimal("0.5")  # Min unlock % of mcap
    unlock_short_hours_before: int = 48
    unlock_cover_hours_after: int = 72

    # News trading
    min_sentiment_magnitude: float = 0.5
    news_trade_window_minutes: int = 30  # Window to trade on news

    # Straddle strategy
    straddle_days_before: int = 3
    straddle_iv_threshold: Decimal = Decimal("80")  # Min IV percentile

    # General
    enabled_event_types: List[EventType] = field(default_factory=lambda: list(EventType))
    blacklist_symbols: List[str] = field(default_factory=list)


@dataclass
class EventTradeSignal:
    """Trading signal generated from event."""

    signal_id: str
    event_type: EventType
    event_id: str
    symbol: str
    action: TradingAction

    # Trade details
    direction: str  # long, short, both (for straddle)
    entry_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    position_size_pct: Decimal = Decimal("10")

    # Timing
    valid_from: datetime = field(default_factory=datetime.utcnow)
    valid_until: Optional[datetime] = None

    # Confidence
    confidence: float = 0.5
    reasoning: str = ""

    # Execution
    executed: bool = False
    execution_price: Optional[Decimal] = None
    execution_time: Optional[datetime] = None


class EconomicCalendarProvider:
    """Provider for economic calendar events."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://api.tradingeconomics.com"
    ):
        self.api_key = api_key
        self.api_url = api_url
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl = timedelta(minutes=30)

    async def get_upcoming_events(
        self,
        days_ahead: int = 7,
        min_impact: EventImpact = EventImpact.MEDIUM
    ) -> List[EconomicEvent]:
        """
        Get upcoming economic events.

        Falls back to mock data if API not configured.
        """
        if not self.api_key:
            return self._get_mock_events(days_ahead, min_impact)

        cache_key = f"events_{days_ahead}_{min_impact.value}"
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if datetime.utcnow() - cached_time < self._cache_ttl:
                return cached_data

        try:
            async with aiohttp.ClientSession() as session:
                end_date = datetime.utcnow() + timedelta(days=days_ahead)
                url = f"{self.api_url}/calendar/country/united states"
                params = {
                    "c": self.api_key,
                    "d1": datetime.utcnow().strftime("%Y-%m-%d"),
                    "d2": end_date.strftime("%Y-%m-%d")
                }

                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"Calendar API returned {response.status}")
                        return self._get_mock_events(days_ahead, min_impact)

                    data = await response.json()
                    events = self._parse_events(data, min_impact)
                    self._cache[cache_key] = (datetime.utcnow(), events)
                    return events

        except Exception as e:
            logger.error(f"Error fetching economic calendar: {e}")
            return self._get_mock_events(days_ahead, min_impact)

    def _parse_events(
        self,
        data: List[Dict],
        min_impact: EventImpact
    ) -> List[EconomicEvent]:
        """Parse API response into EconomicEvent objects."""
        events = []

        impact_map = {
            1: EventImpact.LOW,
            2: EventImpact.MEDIUM,
            3: EventImpact.HIGH
        }

        event_type_map = {
            "cpi": EventType.CPI,
            "consumer price index": EventType.CPI,
            "fomc": EventType.FOMC,
            "interest rate": EventType.FOMC,
            "non-farm": EventType.NFP,
            "nonfarm": EventType.NFP,
            "gdp": EventType.GDP,
            "ppi": EventType.PPI,
            "retail": EventType.RETAIL_SALES,
            "unemployment": EventType.UNEMPLOYMENT,
            "pmi": EventType.PMI,
        }

        for item in data:
            try:
                # Determine event type
                event_name = item.get("Event", "").lower()
                event_type = EventType.CUSTOM
                for keyword, etype in event_type_map.items():
                    if keyword in event_name:
                        event_type = etype
                        break

                # Determine impact
                importance = item.get("Importance", 1)
                impact = impact_map.get(importance, EventImpact.LOW)

                # Filter by min impact
                impact_order = [EventImpact.LOW, EventImpact.MEDIUM, EventImpact.HIGH, EventImpact.CRITICAL]
                if impact_order.index(impact) < impact_order.index(min_impact):
                    continue

                event = EconomicEvent(
                    event_id=f"{item.get('CalendarId', '')}",
                    event_type=event_type,
                    name=item.get("Event", "Unknown"),
                    datetime_utc=datetime.fromisoformat(item.get("Date", "").replace("Z", "+00:00")),
                    currency=item.get("Currency", "USD"),
                    impact=impact,
                    previous=str(item.get("Previous", "")),
                    forecast=str(item.get("Forecast", "")),
                    actual=str(item.get("Actual", "")) if item.get("Actual") else None,
                    source="tradingeconomics"
                )
                events.append(event)

            except Exception as e:
                logger.warning(f"Error parsing event: {e}")
                continue

        return sorted(events, key=lambda e: e.datetime_utc)

    def _get_mock_events(
        self,
        days_ahead: int,
        min_impact: EventImpact
    ) -> List[EconomicEvent]:
        """Generate mock events for testing."""
        now = datetime.utcnow()
        events = []

        # Mock high-impact events
        mock_data = [
            (EventType.CPI, "US CPI MoM", EventImpact.HIGH, 1, "0.2%", "0.3%"),
            (EventType.FOMC, "FOMC Rate Decision", EventImpact.CRITICAL, 3, "5.25%", "5.25%"),
            (EventType.NFP, "Non-Farm Payrolls", EventImpact.HIGH, 5, "200K", "180K"),
            (EventType.GDP, "US GDP QoQ", EventImpact.MEDIUM, 7, "2.1%", "2.3%"),
        ]

        impact_order = [EventImpact.LOW, EventImpact.MEDIUM, EventImpact.HIGH, EventImpact.CRITICAL]

        for event_type, name, impact, days_offset, prev, forecast in mock_data:
            if days_offset <= days_ahead and impact_order.index(impact) >= impact_order.index(min_impact):
                events.append(EconomicEvent(
                    event_id=f"mock_{event_type.value}_{days_offset}",
                    event_type=event_type,
                    name=name,
                    datetime_utc=now + timedelta(days=days_offset, hours=14, minutes=30),
                    currency="USD",
                    impact=impact,
                    previous=prev,
                    forecast=forecast,
                    expected_volatility_pct=Decimal("3.0") if impact == EventImpact.HIGH else Decimal("5.0"),
                    source="mock"
                ))

        return events


class TokenUnlockProvider:
    """Provider for token unlock events."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://api.tokenunlocks.app"
    ):
        self.api_key = api_key
        self.api_url = api_url
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl = timedelta(hours=1)

    async def get_upcoming_unlocks(
        self,
        days_ahead: int = 30,
        min_value_usd: Decimal = Decimal("1000000")
    ) -> List[TokenUnlock]:
        """Get upcoming token unlocks."""
        if not self.api_key:
            return self._get_mock_unlocks(days_ahead, min_value_usd)

        cache_key = f"unlocks_{days_ahead}_{min_value_usd}"
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if datetime.utcnow() - cached_time < self._cache_ttl:
                return cached_data

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                url = f"{self.api_url}/v1/unlocks/upcoming"
                params = {"days": days_ahead}

                async with session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        return self._get_mock_unlocks(days_ahead, min_value_usd)

                    data = await response.json()
                    unlocks = self._parse_unlocks(data, min_value_usd)
                    self._cache[cache_key] = (datetime.utcnow(), unlocks)
                    return unlocks

        except Exception as e:
            logger.error(f"Error fetching token unlocks: {e}")
            return self._get_mock_unlocks(days_ahead, min_value_usd)

    def _parse_unlocks(
        self,
        data: List[Dict],
        min_value_usd: Decimal
    ) -> List[TokenUnlock]:
        """Parse API response into TokenUnlock objects."""
        unlocks = []

        for item in data:
            try:
                value_usd = Decimal(str(item.get("value_usd", 0)))
                if value_usd < min_value_usd:
                    continue

                unlock = TokenUnlock(
                    unlock_id=item.get("id", ""),
                    symbol=item.get("symbol", "").upper() + "USDT",
                    token_name=item.get("name", ""),
                    datetime_utc=datetime.fromisoformat(item.get("unlock_date", "")),
                    unlock_amount=Decimal(str(item.get("amount", 0))),
                    unlock_value_usd=value_usd,
                    unlock_pct_of_supply=Decimal(str(item.get("pct_of_supply", 0))),
                    unlock_pct_of_mcap=Decimal(str(item.get("pct_of_mcap", 0))),
                    recipient_type=item.get("recipient_type", "unknown"),
                    recipient_name=item.get("recipient_name"),
                    is_cliff=item.get("is_cliff", False)
                )
                unlocks.append(unlock)

            except Exception as e:
                logger.warning(f"Error parsing unlock: {e}")
                continue

        return sorted(unlocks, key=lambda u: u.datetime_utc)

    def _get_mock_unlocks(
        self,
        days_ahead: int,
        min_value_usd: Decimal
    ) -> List[TokenUnlock]:
        """Generate mock unlocks for testing."""
        now = datetime.utcnow()
        unlocks = []

        mock_data = [
            ("ARB", "Arbitrum", 2, Decimal("50000000"), Decimal("2.5"), "investor"),
            ("OP", "Optimism", 5, Decimal("30000000"), Decimal("1.8"), "team"),
            ("APT", "Aptos", 10, Decimal("80000000"), Decimal("3.2"), "foundation"),
            ("SUI", "Sui", 15, Decimal("25000000"), Decimal("1.2"), "investor"),
        ]

        for symbol, name, days_offset, value, mcap_pct, recipient in mock_data:
            if days_offset <= days_ahead and value >= min_value_usd:
                unlocks.append(TokenUnlock(
                    unlock_id=f"mock_{symbol}_{days_offset}",
                    symbol=f"{symbol}USDT",
                    token_name=name,
                    datetime_utc=now + timedelta(days=days_offset),
                    unlock_amount=value / Decimal("1.5"),  # Estimate
                    unlock_value_usd=value,
                    unlock_pct_of_supply=mcap_pct * 2,
                    unlock_pct_of_mcap=mcap_pct,
                    recipient_type=recipient,
                    historical_sell_pressure_pct=Decimal("40") if recipient == "team" else Decimal("60"),
                    expected_price_impact_pct=Decimal("-1.5") * mcap_pct
                ))

        return unlocks


class NewsSentimentAnalyzer:
    """Analyzer for news sentiment."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        news_sources: Optional[List[str]] = None
    ):
        self.api_key = api_key
        self.news_sources = news_sources or [
            "coindesk",
            "cointelegraph",
            "theblock",
            "decrypt"
        ]
        self._sentiment_keywords = {
            "positive": [
                "bullish", "surge", "rally", "breakthrough", "adoption",
                "partnership", "launch", "approval", "success", "milestone",
                "record", "growth", "upgrade", "integration", "investment"
            ],
            "negative": [
                "bearish", "crash", "dump", "hack", "exploit", "ban",
                "lawsuit", "fraud", "scam", "bankruptcy", "layoff",
                "delay", "reject", "fail", "investigation", "warning"
            ]
        }

    def analyze_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Simple rule-based sentiment analysis.

        Returns:
            Tuple of (sentiment_score, magnitude)
            - sentiment_score: -1 (bearish) to 1 (bullish)
            - magnitude: 0 (neutral) to 1 (strong)
        """
        text_lower = text.lower()

        positive_count = sum(
            1 for word in self._sentiment_keywords["positive"]
            if word in text_lower
        )
        negative_count = sum(
            1 for word in self._sentiment_keywords["negative"]
            if word in text_lower
        )

        total = positive_count + negative_count
        if total == 0:
            return 0.0, 0.0

        sentiment_score = (positive_count - negative_count) / total
        magnitude = min(total / 5, 1.0)  # Cap at 1.0

        return sentiment_score, magnitude

    async def get_recent_news(
        self,
        symbols: Optional[List[str]] = None,
        hours_back: int = 24
    ) -> List[NewsItem]:
        """
        Get recent news items with sentiment analysis.

        This is a simplified implementation - in production,
        you would integrate with actual news APIs.
        """
        # Mock news for demonstration
        return self._get_mock_news(symbols, hours_back)

    def _get_mock_news(
        self,
        symbols: Optional[List[str]],
        hours_back: int
    ) -> List[NewsItem]:
        """Generate mock news items."""
        now = datetime.utcnow()
        news_items = []

        mock_news = [
            (
                "Bitcoin ETF sees record inflows as institutional adoption accelerates",
                0.7, 0.8, ["BTCUSDT"], EventImpact.HIGH
            ),
            (
                "Ethereum upgrade successfully deployed on mainnet",
                0.5, 0.6, ["ETHUSDT"], EventImpact.MEDIUM
            ),
            (
                "Major exchange faces regulatory investigation",
                -0.6, 0.7, ["BTCUSDT", "ETHUSDT"], EventImpact.HIGH
            ),
            (
                "New Layer 2 solution announces partnership with top DeFi protocol",
                0.4, 0.5, ["ARBUSDT", "OPUSDT"], EventImpact.MEDIUM
            ),
        ]

        for i, (title, sentiment, magnitude, syms, impact) in enumerate(mock_news):
            # Filter by symbols if specified
            if symbols and not any(s in syms for s in symbols):
                continue

            news_items.append(NewsItem(
                news_id=f"mock_news_{i}",
                title=title,
                content=title,  # Simplified
                source="mock",
                published_at=now - timedelta(hours=i * 2),
                url=f"https://example.com/news/{i}",
                sentiment_score=sentiment,
                sentiment_magnitude=magnitude,
                symbols=syms,
                expected_impact=impact
            ))

        return news_items


class NewsEventTrader:
    """
    Main trading engine for news and events.

    Combines economic calendar, token unlocks, and news sentiment
    to generate trading signals.
    """

    def __init__(
        self,
        config: EventTradingConfig,
        calendar_provider: Optional[EconomicCalendarProvider] = None,
        unlock_provider: Optional[TokenUnlockProvider] = None,
        sentiment_analyzer: Optional[NewsSentimentAnalyzer] = None,
        signal_callback: Optional[Callable[[EventTradeSignal], None]] = None
    ):
        self.config = config
        self.calendar = calendar_provider or EconomicCalendarProvider()
        self.unlocks = unlock_provider or TokenUnlockProvider()
        self.sentiment = sentiment_analyzer or NewsSentimentAnalyzer()
        self.signal_callback = signal_callback

        self._running = False
        self._signals: List[EventTradeSignal] = []
        self._trades_today: int = 0
        self._last_trade_date: Optional[datetime] = None

        # Event handlers
        self._event_handlers: Dict[EventType, Callable] = {
            EventType.FOMC: self._handle_fomc,
            EventType.CPI: self._handle_cpi,
            EventType.NFP: self._handle_nfp,
            EventType.TOKEN_UNLOCK: self._handle_token_unlock,
        }

    async def start(self) -> None:
        """Start the event trading monitor."""
        self._running = True
        logger.info("News/Event trader started")

        asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop the event trading monitor."""
        self._running = False
        logger.info("News/Event trader stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Reset daily trade counter
                today = datetime.utcnow().date()
                if self._last_trade_date != today:
                    self._trades_today = 0
                    self._last_trade_date = today

                # Check for events
                await self._check_economic_events()
                await self._check_token_unlocks()
                await self._check_news_sentiment()

                # Wait before next check
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in event monitor loop: {e}")
                await asyncio.sleep(60)

    async def _check_economic_events(self) -> None:
        """Check and process economic calendar events."""
        events = await self.calendar.get_upcoming_events(
            days_ahead=7,
            min_impact=EventImpact.MEDIUM
        )

        for event in events:
            if event.event_type not in self.config.enabled_event_types:
                continue

            # Check if we need to act
            hours_until = event.time_until.total_seconds() / 3600

            if hours_until <= 0:
                # Event has passed - check for post-event opportunities
                if abs(hours_until) <= 1:  # Within 1 hour after event
                    await self._process_post_event(event)
            elif hours_until <= self.config.pre_event_hours:
                # Pre-event window
                await self._process_pre_event(event)

    async def _check_token_unlocks(self) -> None:
        """Check and process token unlock events."""
        if EventType.TOKEN_UNLOCK not in self.config.enabled_event_types:
            return

        unlocks = await self.unlocks.get_upcoming_unlocks(
            days_ahead=30,
            min_value_usd=Decimal("1000000")
        )

        for unlock in unlocks:
            if unlock.symbol in self.config.blacklist_symbols:
                continue

            if unlock.unlock_pct_of_mcap < self.config.min_unlock_pct_for_trade:
                continue

            hours_until = unlock.time_until.total_seconds() / 3600

            if 0 < hours_until <= self.config.unlock_short_hours_before:
                # Consider short position before unlock
                await self._handle_token_unlock(unlock)

    async def _check_news_sentiment(self) -> None:
        """Check recent news for trading opportunities."""
        news_items = await self.sentiment.get_recent_news(hours_back=24)

        for news in news_items:
            if news.sentiment_magnitude < self.config.min_sentiment_magnitude:
                continue

            # Check if news is recent enough to trade
            news_age_minutes = (datetime.utcnow() - news.published_at).total_seconds() / 60
            if news_age_minutes > self.config.news_trade_window_minutes:
                continue

            await self._process_news_signal(news)

    async def _process_pre_event(self, event: EconomicEvent) -> None:
        """Process pre-event positioning."""
        if self._trades_today >= self.config.max_event_trades_per_day:
            return

        hours_until = event.time_until.total_seconds() / 3600

        # High impact events - reduce exposure or set up straddle
        if event.impact in [EventImpact.HIGH, EventImpact.CRITICAL]:
            if hours_until <= self.config.reduce_exposure_hours_before:
                signal = EventTradeSignal(
                    signal_id=f"pre_{event.event_id}_{datetime.utcnow().timestamp()}",
                    event_type=event.event_type,
                    event_id=event.event_id,
                    symbol="BTCUSDT",  # Primary symbol
                    action=TradingAction.REDUCE_EXPOSURE,
                    direction="neutral",
                    position_size_pct=Decimal("0"),
                    confidence=0.8,
                    reasoning=f"Reducing exposure {hours_until:.1f}h before {event.name}"
                )
                await self._emit_signal(signal)

        # Use event-specific handler if available
        handler = self._event_handlers.get(event.event_type)
        if handler:
            await handler(event)

    async def _process_post_event(self, event: EconomicEvent) -> None:
        """Process post-event trading opportunities."""
        if self._trades_today >= self.config.max_event_trades_per_day:
            return

        # Wait for cooldown
        minutes_since = abs(event.time_until.total_seconds()) / 60
        if minutes_since < self.config.post_event_cooldown_minutes:
            return

        # Check for surprise factor if data available
        surprise = event.surprise_factor
        if surprise is not None:
            if abs(surprise) > 0.5:  # Significant surprise
                direction = "long" if surprise > 0 else "short"
                signal = EventTradeSignal(
                    signal_id=f"post_{event.event_id}_{datetime.utcnow().timestamp()}",
                    event_type=event.event_type,
                    event_id=event.event_id,
                    symbol="BTCUSDT",
                    action=TradingAction.BREAKOUT,
                    direction=direction,
                    position_size_pct=Decimal("15"),
                    confidence=min(abs(surprise), 0.9),
                    reasoning=f"Surprise factor {surprise:.2f} on {event.name}"
                )
                await self._emit_signal(signal)

    async def _handle_fomc(self, event: EconomicEvent) -> None:
        """Handle FOMC rate decision events."""
        hours_until = event.time_until.total_seconds() / 3600

        # FOMC typically causes high volatility
        # Strategy: Straddle or reduce exposure
        if hours_until <= 24 and hours_until > self.config.reduce_exposure_hours_before:
            signal = EventTradeSignal(
                signal_id=f"fomc_{event.event_id}_{datetime.utcnow().timestamp()}",
                event_type=EventType.FOMC,
                event_id=event.event_id,
                symbol="BTCUSDT",
                action=TradingAction.STRADDLE,
                direction="both",
                position_size_pct=Decimal("10"),
                confidence=0.7,
                reasoning=f"FOMC volatility play - {hours_until:.1f}h until decision"
            )
            await self._emit_signal(signal)

    async def _handle_cpi(self, event: EconomicEvent) -> None:
        """Handle CPI release events."""
        hours_until = event.time_until.total_seconds() / 3600

        # CPI typically causes quick moves based on above/below expectations
        if hours_until <= 4:
            signal = EventTradeSignal(
                signal_id=f"cpi_{event.event_id}_{datetime.utcnow().timestamp()}",
                event_type=EventType.CPI,
                event_id=event.event_id,
                symbol="BTCUSDT",
                action=TradingAction.STRADDLE,
                direction="both",
                position_size_pct=Decimal("10"),
                confidence=0.65,
                reasoning=f"CPI release in {hours_until:.1f}h - expect volatility"
            )
            await self._emit_signal(signal)

    async def _handle_nfp(self, event: EconomicEvent) -> None:
        """Handle Non-Farm Payrolls events."""
        hours_until = event.time_until.total_seconds() / 3600

        if hours_until <= 2:
            signal = EventTradeSignal(
                signal_id=f"nfp_{event.event_id}_{datetime.utcnow().timestamp()}",
                event_type=EventType.NFP,
                event_id=event.event_id,
                symbol="BTCUSDT",
                action=TradingAction.REDUCE_EXPOSURE,
                direction="neutral",
                position_size_pct=Decimal("0"),
                confidence=0.75,
                reasoning=f"NFP in {hours_until:.1f}h - reducing exposure"
            )
            await self._emit_signal(signal)

    async def _handle_token_unlock(self, unlock: TokenUnlock) -> None:
        """Handle token unlock events."""
        hours_until = unlock.time_until.total_seconds() / 3600

        # Short position before significant unlocks
        if unlock.is_significant and hours_until <= self.config.unlock_short_hours_before:
            # Calculate expected price impact
            expected_move = unlock.expected_price_impact_pct

            signal = EventTradeSignal(
                signal_id=f"unlock_{unlock.unlock_id}_{datetime.utcnow().timestamp()}",
                event_type=EventType.TOKEN_UNLOCK,
                event_id=unlock.unlock_id,
                symbol=unlock.symbol,
                action=TradingAction.SHORT_PRE_EVENT,
                direction="short",
                position_size_pct=min(
                    Decimal("20"),
                    unlock.unlock_pct_of_mcap * Decimal("5")
                ),
                confidence=float(
                    min(Decimal("0.8"), unlock.unlock_pct_of_mcap / Decimal("5"))
                ),
                reasoning=(
                    f"{unlock.token_name} unlock: {unlock.unlock_pct_of_mcap}% of mcap "
                    f"(${unlock.unlock_value_usd:,.0f}) in {hours_until:.1f}h. "
                    f"Expected impact: {expected_move}%"
                )
            )
            await self._emit_signal(signal)

    async def _process_news_signal(self, news: NewsItem) -> None:
        """Process news item for trading signal."""
        if self._trades_today >= self.config.max_event_trades_per_day:
            return

        for symbol in news.symbols:
            if symbol in self.config.blacklist_symbols:
                continue

            direction = "long" if news.is_bullish else "short" if news.is_bearish else None
            if not direction:
                continue

            signal = EventTradeSignal(
                signal_id=f"news_{news.news_id}_{datetime.utcnow().timestamp()}",
                event_type=EventType.CUSTOM,
                event_id=news.news_id,
                symbol=symbol,
                action=TradingAction.BREAKOUT,
                direction=direction,
                position_size_pct=Decimal("10"),
                confidence=abs(news.sentiment_score) * news.sentiment_magnitude,
                valid_until=news.published_at + timedelta(
                    minutes=self.config.news_trade_window_minutes
                ),
                reasoning=f"News signal: {news.title[:100]}"
            )
            await self._emit_signal(signal)

    async def _emit_signal(self, signal: EventTradeSignal) -> None:
        """Emit trading signal."""
        self._signals.append(signal)
        self._trades_today += 1

        logger.info(
            f"Event signal: {signal.action.value} {signal.direction} "
            f"{signal.symbol} (confidence: {signal.confidence:.2f})"
        )

        if self.signal_callback:
            try:
                self.signal_callback(signal)
            except Exception as e:
                logger.error(f"Error in signal callback: {e}")

    def get_signals(self, active_only: bool = True) -> List[EventTradeSignal]:
        """Get generated signals."""
        if not active_only:
            return self._signals.copy()

        now = datetime.utcnow()
        return [
            s for s in self._signals
            if not s.executed and (s.valid_until is None or s.valid_until > now)
        ]

    def get_upcoming_events(self) -> Dict[str, Any]:
        """Get summary of upcoming events."""
        return {
            "economic_events": [],  # Filled by caller
            "token_unlocks": [],
            "high_impact_count": 0,
            "next_major_event": None
        }

    async def analyze_event_impact(
        self,
        symbol: str,
        lookback_days: int = 90
    ) -> Dict[str, Any]:
        """
        Analyze historical impact of events on a symbol.

        This would require historical price data integration.
        """
        return {
            "symbol": symbol,
            "lookback_days": lookback_days,
            "fomc_avg_move_pct": 2.5,
            "cpi_avg_move_pct": 1.8,
            "nfp_avg_move_pct": 1.5,
            "unlock_avg_move_pct": -3.2,
            "best_strategy": "straddle",
            "note": "Historical analysis requires price data integration"
        }


# Convenience function to create trader
def create_event_trader(
    config: Optional[EventTradingConfig] = None,
    calendar_api_key: Optional[str] = None,
    unlock_api_key: Optional[str] = None,
    signal_callback: Optional[Callable[[EventTradeSignal], None]] = None
) -> NewsEventTrader:
    """Create and configure news/event trader."""
    config = config or EventTradingConfig()

    calendar = EconomicCalendarProvider(api_key=calendar_api_key)
    unlocks = TokenUnlockProvider(api_key=unlock_api_key)
    sentiment = NewsSentimentAnalyzer()

    return NewsEventTrader(
        config=config,
        calendar_provider=calendar,
        unlock_provider=unlocks,
        sentiment_analyzer=sentiment,
        signal_callback=signal_callback
    )
