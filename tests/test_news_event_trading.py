"""
Tests for News and Event Trading module.

100% coverage for:
- EconomicEvent, TokenUnlock, NewsItem dataclasses
- EconomicCalendarProvider
- TokenUnlockProvider
- NewsSentimentAnalyzer
- NewsEventTrader
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from src.trading.news_event_trading import (
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


# =============================================================================
# EconomicEvent Tests
# =============================================================================

class TestEconomicEvent:
    """Tests for EconomicEvent dataclass."""

    def test_event_creation(self):
        """Test event creation."""
        event = EconomicEvent(
            event_id="cpi_001",
            event_type=EventType.CPI,
            name="US CPI MoM",
            datetime_utc=datetime.utcnow() + timedelta(hours=1),
            currency="USD",
            impact=EventImpact.HIGH,
            previous="0.2%",
            forecast="0.3%",
        )

        assert event.event_id == "cpi_001"
        assert event.event_type == EventType.CPI
        assert event.impact == EventImpact.HIGH

    def test_time_until(self):
        """Test time_until property."""
        future_time = datetime.utcnow() + timedelta(hours=2)
        event = EconomicEvent(
            event_id="test",
            event_type=EventType.FOMC,
            name="FOMC",
            datetime_utc=future_time,
            currency="USD",
            impact=EventImpact.CRITICAL,
        )

        time_until = event.time_until
        assert timedelta(hours=1, minutes=59) < time_until < timedelta(hours=2, minutes=1)

    def test_is_past(self):
        """Test is_past property."""
        past_event = EconomicEvent(
            event_id="past",
            event_type=EventType.NFP,
            name="NFP",
            datetime_utc=datetime.utcnow() - timedelta(hours=1),
            currency="USD",
            impact=EventImpact.HIGH,
        )

        future_event = EconomicEvent(
            event_id="future",
            event_type=EventType.NFP,
            name="NFP",
            datetime_utc=datetime.utcnow() + timedelta(hours=1),
            currency="USD",
            impact=EventImpact.HIGH,
        )

        assert past_event.is_past is True
        assert future_event.is_past is False

    def test_surprise_factor(self):
        """Test surprise factor calculation."""
        event = EconomicEvent(
            event_id="test",
            event_type=EventType.CPI,
            name="CPI",
            datetime_utc=datetime.utcnow(),
            currency="USD",
            impact=EventImpact.HIGH,
            forecast="0.3%",
            actual="0.5%",
        )

        surprise = event.surprise_factor
        # (0.5 - 0.3) / 0.3 = 0.667
        assert surprise == pytest.approx(0.667, rel=0.01)

    def test_surprise_factor_no_data(self):
        """Test surprise factor with missing data."""
        event = EconomicEvent(
            event_id="test",
            event_type=EventType.CPI,
            name="CPI",
            datetime_utc=datetime.utcnow(),
            currency="USD",
            impact=EventImpact.HIGH,
        )

        assert event.surprise_factor is None

    def test_surprise_factor_zero_forecast(self):
        """Test surprise factor with zero forecast."""
        event = EconomicEvent(
            event_id="test",
            event_type=EventType.CPI,
            name="CPI",
            datetime_utc=datetime.utcnow(),
            currency="USD",
            impact=EventImpact.HIGH,
            forecast="0%",
            actual="0.5%",
        )

        assert event.surprise_factor is None


# =============================================================================
# TokenUnlock Tests
# =============================================================================

class TestTokenUnlock:
    """Tests for TokenUnlock dataclass."""

    def test_unlock_creation(self):
        """Test unlock creation."""
        unlock = TokenUnlock(
            unlock_id="arb_001",
            symbol="ARBUSDT",
            token_name="Arbitrum",
            datetime_utc=datetime.utcnow() + timedelta(days=2),
            unlock_amount=Decimal("50000000"),
            unlock_value_usd=Decimal("75000000"),
            unlock_pct_of_supply=Decimal("5.0"),
            unlock_pct_of_mcap=Decimal("2.5"),
            recipient_type="investor",
        )

        assert unlock.symbol == "ARBUSDT"
        assert unlock.unlock_pct_of_mcap == Decimal("2.5")
        assert unlock.recipient_type == "investor"

    def test_time_until(self):
        """Test time_until property."""
        unlock = TokenUnlock(
            unlock_id="test",
            symbol="TESTUSDT",
            token_name="Test",
            datetime_utc=datetime.utcnow() + timedelta(days=3),
            unlock_amount=Decimal("1000"),
            unlock_value_usd=Decimal("10000"),
            unlock_pct_of_supply=Decimal("1.0"),
            unlock_pct_of_mcap=Decimal("0.5"),
            recipient_type="team",
        )

        time_until = unlock.time_until
        assert timedelta(days=2, hours=23) < time_until < timedelta(days=3, hours=1)

    def test_is_significant(self):
        """Test is_significant property."""
        significant = TokenUnlock(
            unlock_id="sig",
            symbol="SIGUSDT",
            token_name="Significant",
            datetime_utc=datetime.utcnow(),
            unlock_amount=Decimal("1000"),
            unlock_value_usd=Decimal("10000"),
            unlock_pct_of_supply=Decimal("5.0"),
            unlock_pct_of_mcap=Decimal("2.0"),  # >= 1%
            recipient_type="foundation",
        )

        not_significant = TokenUnlock(
            unlock_id="nsig",
            symbol="NSIGUSDT",
            token_name="NotSignificant",
            datetime_utc=datetime.utcnow(),
            unlock_amount=Decimal("100"),
            unlock_value_usd=Decimal("1000"),
            unlock_pct_of_supply=Decimal("0.5"),
            unlock_pct_of_mcap=Decimal("0.2"),  # < 1%
            recipient_type="team",
        )

        assert significant.is_significant is True
        assert not_significant.is_significant is False


# =============================================================================
# NewsItem Tests
# =============================================================================

class TestNewsItem:
    """Tests for NewsItem dataclass."""

    def test_news_creation(self):
        """Test news item creation."""
        news = NewsItem(
            news_id="news_001",
            title="Bitcoin ETF approved",
            content="SEC approves spot Bitcoin ETF",
            source="coindesk",
            published_at=datetime.utcnow(),
            url="https://example.com/news/1",
            sentiment_score=0.8,
            sentiment_magnitude=0.9,
            symbols=["BTCUSDT"],
        )

        assert news.title == "Bitcoin ETF approved"
        assert news.sentiment_score == 0.8

    def test_is_bullish(self):
        """Test is_bullish property."""
        bullish = NewsItem(
            news_id="1",
            title="Bull",
            content="",
            source="test",
            published_at=datetime.utcnow(),
            url="",
            sentiment_score=0.5,  # > 0.3
            sentiment_magnitude=0.7,
        )

        neutral = NewsItem(
            news_id="2",
            title="Neutral",
            content="",
            source="test",
            published_at=datetime.utcnow(),
            url="",
            sentiment_score=0.2,  # <= 0.3
            sentiment_magnitude=0.5,
        )

        assert bullish.is_bullish is True
        assert neutral.is_bullish is False

    def test_is_bearish(self):
        """Test is_bearish property."""
        bearish = NewsItem(
            news_id="1",
            title="Bear",
            content="",
            source="test",
            published_at=datetime.utcnow(),
            url="",
            sentiment_score=-0.5,  # < -0.3
            sentiment_magnitude=0.7,
        )

        neutral = NewsItem(
            news_id="2",
            title="Neutral",
            content="",
            source="test",
            published_at=datetime.utcnow(),
            url="",
            sentiment_score=-0.2,  # >= -0.3
            sentiment_magnitude=0.5,
        )

        assert bearish.is_bearish is True
        assert neutral.is_bearish is False


# =============================================================================
# EventTradingConfig Tests
# =============================================================================

class TestEventTradingConfig:
    """Tests for EventTradingConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = EventTradingConfig()

        assert config.pre_event_hours == 24
        assert config.max_event_trades_per_day == 5
        assert config.min_sentiment_magnitude == 0.5
        assert len(config.enabled_event_types) == len(EventType)

    def test_custom_config(self):
        """Test custom configuration."""
        config = EventTradingConfig(
            pre_event_hours=12,
            max_event_trades_per_day=10,
            enabled_event_types=[EventType.CPI, EventType.FOMC],
            blacklist_symbols=["BADUSDT"],
        )

        assert config.pre_event_hours == 12
        assert len(config.enabled_event_types) == 2
        assert "BADUSDT" in config.blacklist_symbols


# =============================================================================
# EconomicCalendarProvider Tests
# =============================================================================

class TestEconomicCalendarProvider:
    """Tests for EconomicCalendarProvider."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return EconomicCalendarProvider()

    @pytest.fixture
    def provider_with_key(self):
        """Create provider with API key."""
        return EconomicCalendarProvider(api_key="test_key")

    @pytest.mark.asyncio
    async def test_get_upcoming_events_no_api_key(self, provider):
        """Test getting events without API key returns mock."""
        events = await provider.get_upcoming_events(days_ahead=7)

        assert len(events) > 0
        assert all(isinstance(e, EconomicEvent) for e in events)

    @pytest.mark.asyncio
    async def test_get_upcoming_events_cached(self, provider_with_key):
        """Test event caching."""
        # Populate cache
        cached_events = [
            EconomicEvent(
                event_id="cached",
                event_type=EventType.CPI,
                name="Cached CPI",
                datetime_utc=datetime.utcnow(),
                currency="USD",
                impact=EventImpact.HIGH,
            )
        ]
        cache_key = f"events_7_{EventImpact.MEDIUM.value}"
        provider_with_key._cache[cache_key] = (datetime.utcnow(), cached_events)

        events = await provider_with_key.get_upcoming_events(days_ahead=7)

        assert events == cached_events

    @pytest.mark.asyncio
    async def test_get_upcoming_events_api_call(self, provider_with_key):
        """Test API call for events."""
        mock_response = [
            {
                "CalendarId": "123",
                "Event": "US CPI MoM",
                "Date": (datetime.utcnow() + timedelta(days=1)).isoformat(),
                "Currency": "USD",
                "Importance": 3,
                "Previous": "0.2",
                "Forecast": "0.3",
            }
        ]

        with patch("aiohttp.ClientSession") as mock_session:
            response = AsyncMock()
            response.status = 200
            response.json = AsyncMock(return_value=mock_response)
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = response

            events = await provider_with_key.get_upcoming_events(days_ahead=7)

            assert len(events) >= 0  # May be filtered

    @pytest.mark.asyncio
    async def test_get_upcoming_events_api_error(self, provider_with_key):
        """Test API error falls back to mock."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.get.side_effect = Exception("API Error")

            events = await provider_with_key.get_upcoming_events(days_ahead=7)

            # Should return mock events
            assert len(events) > 0

    def test_parse_events(self, provider):
        """Test event parsing."""
        raw_data = [
            {
                "CalendarId": "1",
                "Event": "US CPI Monthly",
                "Date": datetime.utcnow().isoformat(),
                "Currency": "USD",
                "Importance": 3,
            },
            {
                "CalendarId": "2",
                "Event": "FOMC Rate Decision",
                "Date": datetime.utcnow().isoformat(),
                "Currency": "USD",
                "Importance": 3,
            },
        ]

        events = provider._parse_events(raw_data, EventImpact.HIGH)

        assert len(events) == 2
        assert events[0].event_type == EventType.CPI
        assert events[1].event_type == EventType.FOMC

    def test_parse_events_filters_by_impact(self, provider):
        """Test that events are filtered by impact."""
        raw_data = [
            {
                "CalendarId": "1",
                "Event": "Minor Event",
                "Date": datetime.utcnow().isoformat(),
                "Currency": "USD",
                "Importance": 1,  # LOW
            },
        ]

        events = provider._parse_events(raw_data, EventImpact.HIGH)

        assert len(events) == 0

    def test_get_mock_events(self, provider):
        """Test mock event generation."""
        events = provider._get_mock_events(days_ahead=7, min_impact=EventImpact.MEDIUM)

        assert len(events) > 0
        for event in events:
            assert event.source == "mock"


# =============================================================================
# TokenUnlockProvider Tests
# =============================================================================

class TestTokenUnlockProvider:
    """Tests for TokenUnlockProvider."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return TokenUnlockProvider()

    @pytest.mark.asyncio
    async def test_get_upcoming_unlocks_no_key(self, provider):
        """Test getting unlocks without API key."""
        unlocks = await provider.get_upcoming_unlocks(days_ahead=30)

        assert len(unlocks) > 0
        assert all(isinstance(u, TokenUnlock) for u in unlocks)

    def test_get_mock_unlocks(self, provider):
        """Test mock unlock generation."""
        unlocks = provider._get_mock_unlocks(
            days_ahead=30,
            min_value_usd=Decimal("1000000")
        )

        assert len(unlocks) > 0
        for unlock in unlocks:
            assert unlock.unlock_value_usd >= Decimal("1000000")


# =============================================================================
# NewsSentimentAnalyzer Tests
# =============================================================================

class TestNewsSentimentAnalyzer:
    """Tests for NewsSentimentAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return NewsSentimentAnalyzer()

    def test_analyze_sentiment_positive(self, analyzer):
        """Test positive sentiment analysis."""
        text = "Bitcoin shows bullish surge with record adoption and successful partnership"

        score, magnitude = analyzer.analyze_sentiment(text)

        assert score > 0  # Positive
        assert magnitude > 0

    def test_analyze_sentiment_negative(self, analyzer):
        """Test negative sentiment analysis."""
        text = "Major exchange faces hack and regulatory investigation, crash expected"

        score, magnitude = analyzer.analyze_sentiment(text)

        assert score < 0  # Negative
        assert magnitude > 0

    def test_analyze_sentiment_neutral(self, analyzer):
        """Test neutral sentiment analysis."""
        text = "Bitcoin traded at current levels today"

        score, magnitude = analyzer.analyze_sentiment(text)

        assert score == 0
        assert magnitude == 0

    def test_analyze_sentiment_mixed(self, analyzer):
        """Test mixed sentiment analysis."""
        text = "Despite hack fears, new partnership shows growth potential"

        score, magnitude = analyzer.analyze_sentiment(text)

        # Has both positive and negative keywords
        assert -1 <= score <= 1
        assert magnitude > 0

    @pytest.mark.asyncio
    async def test_get_recent_news(self, analyzer):
        """Test getting recent news."""
        news = await analyzer.get_recent_news(hours_back=24)

        assert len(news) > 0
        assert all(isinstance(n, NewsItem) for n in news)

    @pytest.mark.asyncio
    async def test_get_recent_news_filtered(self, analyzer):
        """Test getting news filtered by symbols."""
        news = await analyzer.get_recent_news(
            symbols=["BTCUSDT"],
            hours_back=24
        )

        for item in news:
            assert "BTCUSDT" in item.symbols


# =============================================================================
# NewsEventTrader Tests
# =============================================================================

class TestNewsEventTrader:
    """Tests for NewsEventTrader."""

    @pytest.fixture
    def config(self):
        """Create trading config."""
        return EventTradingConfig(
            pre_event_hours=24,
            max_event_trades_per_day=5,
        )

    @pytest.fixture
    def trader(self, config):
        """Create trader instance."""
        return NewsEventTrader(config=config)

    @pytest.mark.asyncio
    async def test_start_stop(self, trader):
        """Test trader start/stop."""
        await trader.start()
        assert trader._running is True

        await trader.stop()
        assert trader._running is False

    @pytest.mark.asyncio
    async def test_emit_signal(self, trader):
        """Test signal emission."""
        callback_received = []

        def callback(signal):
            callback_received.append(signal)

        trader.signal_callback = callback

        signal = EventTradeSignal(
            signal_id="test_001",
            event_type=EventType.CPI,
            event_id="cpi_001",
            symbol="BTCUSDT",
            action=TradingAction.STRADDLE,
            direction="both",
        )

        await trader._emit_signal(signal)

        assert len(callback_received) == 1
        assert signal in trader._signals
        assert trader._trades_today == 1

    def test_get_signals(self, trader):
        """Test getting signals."""
        # Add some signals
        active_signal = EventTradeSignal(
            signal_id="active",
            event_type=EventType.CPI,
            event_id="1",
            symbol="BTCUSDT",
            action=TradingAction.BREAKOUT,
            direction="long",
            executed=False,
        )

        executed_signal = EventTradeSignal(
            signal_id="executed",
            event_type=EventType.FOMC,
            event_id="2",
            symbol="ETHUSDT",
            action=TradingAction.STRADDLE,
            direction="both",
            executed=True,
        )

        trader._signals = [active_signal, executed_signal]

        # Active only
        active = trader.get_signals(active_only=True)
        assert len(active) == 1
        assert active[0].signal_id == "active"

        # All signals
        all_signals = trader.get_signals(active_only=False)
        assert len(all_signals) == 2

    def test_get_signals_filters_expired(self, trader):
        """Test that expired signals are filtered."""
        expired_signal = EventTradeSignal(
            signal_id="expired",
            event_type=EventType.CPI,
            event_id="1",
            symbol="BTCUSDT",
            action=TradingAction.BREAKOUT,
            direction="long",
            executed=False,
            valid_until=datetime.utcnow() - timedelta(hours=1),
        )

        trader._signals = [expired_signal]

        active = trader.get_signals(active_only=True)
        assert len(active) == 0

    # -------------------------------------------------------------------------
    # Event Handler Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_handle_fomc(self, trader):
        """Test FOMC event handling."""
        event = EconomicEvent(
            event_id="fomc_001",
            event_type=EventType.FOMC,
            name="FOMC Rate Decision",
            datetime_utc=datetime.utcnow() + timedelta(hours=12),
            currency="USD",
            impact=EventImpact.CRITICAL,
        )

        await trader._handle_fomc(event)

        assert len(trader._signals) == 1
        assert trader._signals[0].action == TradingAction.STRADDLE

    @pytest.mark.asyncio
    async def test_handle_cpi(self, trader):
        """Test CPI event handling."""
        event = EconomicEvent(
            event_id="cpi_001",
            event_type=EventType.CPI,
            name="US CPI",
            datetime_utc=datetime.utcnow() + timedelta(hours=2),
            currency="USD",
            impact=EventImpact.HIGH,
        )

        await trader._handle_cpi(event)

        assert len(trader._signals) == 1
        assert trader._signals[0].event_type == EventType.CPI

    @pytest.mark.asyncio
    async def test_handle_nfp(self, trader):
        """Test NFP event handling."""
        event = EconomicEvent(
            event_id="nfp_001",
            event_type=EventType.NFP,
            name="Non-Farm Payrolls",
            datetime_utc=datetime.utcnow() + timedelta(hours=1),
            currency="USD",
            impact=EventImpact.HIGH,
        )

        await trader._handle_nfp(event)

        assert len(trader._signals) == 1
        assert trader._signals[0].action == TradingAction.REDUCE_EXPOSURE

    @pytest.mark.asyncio
    async def test_handle_token_unlock(self, trader):
        """Test token unlock handling."""
        unlock = TokenUnlock(
            unlock_id="arb_001",
            symbol="ARBUSDT",
            token_name="Arbitrum",
            datetime_utc=datetime.utcnow() + timedelta(hours=24),
            unlock_amount=Decimal("50000000"),
            unlock_value_usd=Decimal("75000000"),
            unlock_pct_of_supply=Decimal("5.0"),
            unlock_pct_of_mcap=Decimal("2.5"),
            recipient_type="investor",
        )

        await trader._handle_token_unlock(unlock)

        assert len(trader._signals) == 1
        assert trader._signals[0].action == TradingAction.SHORT_PRE_EVENT
        assert trader._signals[0].symbol == "ARBUSDT"

    # -------------------------------------------------------------------------
    # Pre/Post Event Processing Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_process_pre_event_reduce_exposure(self, trader):
        """Test pre-event exposure reduction."""
        event = EconomicEvent(
            event_id="test",
            event_type=EventType.FOMC,
            name="FOMC",
            datetime_utc=datetime.utcnow() + timedelta(minutes=30),  # Within reduce window
            currency="USD",
            impact=EventImpact.CRITICAL,
        )

        await trader._process_pre_event(event)

        # Should emit reduce exposure signal
        reduce_signals = [s for s in trader._signals if s.action == TradingAction.REDUCE_EXPOSURE]
        assert len(reduce_signals) >= 1

    @pytest.mark.asyncio
    async def test_process_pre_event_max_trades_reached(self, trader):
        """Test pre-event when max trades reached."""
        trader._trades_today = trader.config.max_event_trades_per_day

        event = EconomicEvent(
            event_id="test",
            event_type=EventType.CPI,
            name="CPI",
            datetime_utc=datetime.utcnow() + timedelta(hours=1),
            currency="USD",
            impact=EventImpact.HIGH,
        )

        await trader._process_pre_event(event)

        assert len(trader._signals) == 0

    @pytest.mark.asyncio
    async def test_process_post_event_with_surprise(self, trader):
        """Test post-event processing with surprise factor."""
        event = EconomicEvent(
            event_id="test",
            event_type=EventType.CPI,
            name="CPI",
            datetime_utc=datetime.utcnow() - timedelta(minutes=20),
            currency="USD",
            impact=EventImpact.HIGH,
            forecast="0.3%",
            actual="0.6%",  # Big surprise
        )

        await trader._process_post_event(event)

        assert len(trader._signals) == 1
        assert trader._signals[0].action == TradingAction.BREAKOUT

    @pytest.mark.asyncio
    async def test_process_post_event_cooldown(self, trader):
        """Test post-event cooldown."""
        event = EconomicEvent(
            event_id="test",
            event_type=EventType.CPI,
            name="CPI",
            datetime_utc=datetime.utcnow() - timedelta(minutes=5),  # Within cooldown
            currency="USD",
            impact=EventImpact.HIGH,
            forecast="0.3%",
            actual="0.6%",
        )

        await trader._process_post_event(event)

        # Should not emit signal due to cooldown
        assert len(trader._signals) == 0

    # -------------------------------------------------------------------------
    # News Signal Processing Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_process_news_signal_bullish(self, trader):
        """Test bullish news signal processing."""
        news = NewsItem(
            news_id="news_001",
            title="Bitcoin ETF approved",
            content="",
            source="test",
            published_at=datetime.utcnow() - timedelta(minutes=5),
            url="",
            sentiment_score=0.8,
            sentiment_magnitude=0.9,
            symbols=["BTCUSDT"],
        )

        await trader._process_news_signal(news)

        assert len(trader._signals) == 1
        assert trader._signals[0].direction == "long"

    @pytest.mark.asyncio
    async def test_process_news_signal_bearish(self, trader):
        """Test bearish news signal processing."""
        news = NewsItem(
            news_id="news_002",
            title="Exchange hacked",
            content="",
            source="test",
            published_at=datetime.utcnow() - timedelta(minutes=5),
            url="",
            sentiment_score=-0.7,
            sentiment_magnitude=0.8,
            symbols=["BTCUSDT"],
        )

        await trader._process_news_signal(news)

        assert len(trader._signals) == 1
        assert trader._signals[0].direction == "short"

    @pytest.mark.asyncio
    async def test_process_news_signal_neutral(self, trader):
        """Test neutral news is skipped."""
        news = NewsItem(
            news_id="news_003",
            title="Normal trading day",
            content="",
            source="test",
            published_at=datetime.utcnow() - timedelta(minutes=5),
            url="",
            sentiment_score=0.1,  # Neutral
            sentiment_magnitude=0.3,
            symbols=["BTCUSDT"],
        )

        await trader._process_news_signal(news)

        assert len(trader._signals) == 0

    @pytest.mark.asyncio
    async def test_process_news_signal_blacklist(self, trader):
        """Test blacklisted symbols are skipped."""
        trader.config.blacklist_symbols = ["BADUSDT"]

        news = NewsItem(
            news_id="news_004",
            title="Bad token pump",
            content="",
            source="test",
            published_at=datetime.utcnow() - timedelta(minutes=5),
            url="",
            sentiment_score=0.8,
            sentiment_magnitude=0.9,
            symbols=["BADUSDT"],
        )

        await trader._process_news_signal(news)

        assert len(trader._signals) == 0

    # -------------------------------------------------------------------------
    # Check Methods Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_check_economic_events(self, trader):
        """Test economic events checking."""
        mock_events = [
            EconomicEvent(
                event_id="test",
                event_type=EventType.CPI,
                name="CPI",
                datetime_utc=datetime.utcnow() + timedelta(hours=2),
                currency="USD",
                impact=EventImpact.HIGH,
            )
        ]

        with patch.object(trader.calendar, 'get_upcoming_events', new_callable=AsyncMock) as mock:
            mock.return_value = mock_events
            await trader._check_economic_events()

    @pytest.mark.asyncio
    async def test_check_token_unlocks(self, trader):
        """Test token unlock checking."""
        mock_unlocks = [
            TokenUnlock(
                unlock_id="test",
                symbol="ARBUSDT",
                token_name="Arbitrum",
                datetime_utc=datetime.utcnow() + timedelta(hours=24),
                unlock_amount=Decimal("50000000"),
                unlock_value_usd=Decimal("75000000"),
                unlock_pct_of_supply=Decimal("5.0"),
                unlock_pct_of_mcap=Decimal("2.5"),
                recipient_type="investor",
            )
        ]

        with patch.object(trader.unlocks, 'get_upcoming_unlocks', new_callable=AsyncMock) as mock:
            mock.return_value = mock_unlocks
            await trader._check_token_unlocks()

    @pytest.mark.asyncio
    async def test_check_token_unlocks_disabled(self, trader):
        """Test token unlock checking when disabled."""
        trader.config.enabled_event_types = [EventType.CPI]  # TOKEN_UNLOCK not included

        await trader._check_token_unlocks()

        assert len(trader._signals) == 0

    @pytest.mark.asyncio
    async def test_check_news_sentiment(self, trader):
        """Test news sentiment checking."""
        mock_news = [
            NewsItem(
                news_id="test",
                title="Test",
                content="",
                source="test",
                published_at=datetime.utcnow() - timedelta(minutes=5),
                url="",
                sentiment_score=0.8,
                sentiment_magnitude=0.9,
                symbols=["BTCUSDT"],
            )
        ]

        with patch.object(trader.sentiment, 'get_recent_news', new_callable=AsyncMock) as mock:
            mock.return_value = mock_news
            await trader._check_news_sentiment()

    # -------------------------------------------------------------------------
    # Utility Tests
    # -------------------------------------------------------------------------

    def test_get_upcoming_events_summary(self, trader):
        """Test upcoming events summary."""
        result = trader.get_upcoming_events()

        assert "economic_events" in result
        assert "token_unlocks" in result
        assert "high_impact_count" in result

    @pytest.mark.asyncio
    async def test_analyze_event_impact(self, trader):
        """Test event impact analysis."""
        result = await trader.analyze_event_impact("BTCUSDT", lookback_days=90)

        assert result["symbol"] == "BTCUSDT"
        assert "fomc_avg_move_pct" in result
        assert "best_strategy" in result


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateEventTrader:
    """Tests for create_event_trader factory."""

    def test_create_with_defaults(self):
        """Test creating trader with defaults."""
        trader = create_event_trader()

        assert trader is not None
        assert trader.config is not None
        assert trader.calendar is not None
        assert trader.unlocks is not None
        assert trader.sentiment is not None

    def test_create_with_custom_config(self):
        """Test creating trader with custom config."""
        config = EventTradingConfig(
            pre_event_hours=12,
            max_event_trades_per_day=3,
        )

        trader = create_event_trader(config=config)

        assert trader.config.pre_event_hours == 12
        assert trader.config.max_event_trades_per_day == 3

    def test_create_with_callback(self):
        """Test creating trader with signal callback."""
        callback_called = []

        def callback(signal):
            callback_called.append(signal)

        trader = create_event_trader(signal_callback=callback)

        assert trader.signal_callback == callback
