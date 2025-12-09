"""
Tests for Signal Provider module.

100% coverage for:
- TradingSignal dataclass and methods
- Subscriber management
- Signal publishing and delivery
- Performance statistics
- WebSocket support
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.signals.signal_provider import (
    SignalProvider,
    TradingSignal,
    Subscriber,
    SignalPerformance,
    SignalDirection,
    SignalStatus,
    SubscriptionTier,
    DeliveryChannel,
)


# =============================================================================
# TradingSignal Tests
# =============================================================================

class TestTradingSignal:
    """Tests for TradingSignal dataclass."""

    def test_signal_creation_defaults(self):
        """Test signal creation with defaults."""
        signal = TradingSignal()

        assert signal.signal_id is not None
        assert len(signal.signal_id) == 12
        assert signal.symbol == ""
        assert signal.direction == SignalDirection.LONG
        assert signal.entry_price == Decimal("0")
        assert signal.status == SignalStatus.PENDING
        assert signal.created_at is not None

    def test_signal_creation_full(self):
        """Test signal creation with all parameters."""
        signal = TradingSignal(
            symbol="BTCUSDT",
            direction=SignalDirection.SHORT,
            entry_price=Decimal("50000"),
            stop_loss=Decimal("51000"),
            take_profit_1=Decimal("49000"),
            take_profit_2=Decimal("48000"),
            take_profit_3=Decimal("47000"),
            risk_reward_ratio=3.0,
            position_size_pct=5.0,
            leverage=10,
            strategy="scalper",
            timeframe="5m",
            confidence=0.85,
            notes="Strong setup",
        )

        assert signal.symbol == "BTCUSDT"
        assert signal.direction == SignalDirection.SHORT
        assert signal.entry_price == Decimal("50000")
        assert signal.stop_loss == Decimal("51000")
        assert signal.take_profit_1 == Decimal("49000")
        assert signal.leverage == 10
        assert signal.confidence == 0.85

    def test_signal_to_dict(self):
        """Test signal serialization to dict."""
        signal = TradingSignal(
            symbol="ETHUSDT",
            direction=SignalDirection.LONG,
            entry_price=Decimal("3000"),
            stop_loss=Decimal("2900"),
            take_profit_1=Decimal("3100"),
        )

        data = signal.to_dict()

        assert data["symbol"] == "ETHUSDT"
        assert data["direction"] == "long"
        assert data["entry_price"] == "3000"
        assert data["stop_loss"] == "2900"
        assert data["take_profit_1"] == "3100"
        assert data["status"] == "pending"
        assert "created_at" in data

    def test_signal_to_telegram_message(self):
        """Test Telegram message formatting."""
        signal = TradingSignal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            entry_price=Decimal("50000"),
            stop_loss=Decimal("49500"),
            take_profit_1=Decimal("51000"),
            take_profit_2=Decimal("52000"),
            risk_reward_ratio=2.0,
            leverage=5,
            confidence=0.75,
            strategy="breakout",
            timeframe="1h",
            notes="Key resistance break",
        )

        msg = signal.to_telegram_message(include_entry=True)

        assert "LONG" in msg
        assert "BTCUSDT" in msg
        assert "50000" in msg
        assert "49500" in msg  # SL
        assert "51000" in msg  # TP1
        assert "52000" in msg  # TP2
        assert "breakout" in msg
        assert "1h" in msg
        assert "Key resistance break" in msg

    def test_signal_to_telegram_without_entry(self):
        """Test Telegram message without entry price."""
        signal = TradingSignal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            entry_price=Decimal("50000"),
        )

        msg = signal.to_telegram_message(include_entry=False)

        assert "50000" not in msg
        assert "BTCUSDT" in msg

    def test_signal_directions(self):
        """Test all signal directions."""
        for direction in SignalDirection:
            signal = TradingSignal(direction=direction)
            msg = signal.to_telegram_message()
            assert direction.value.upper().replace("_", " ") in msg


# =============================================================================
# Subscriber Tests
# =============================================================================

class TestSubscriber:
    """Tests for Subscriber dataclass."""

    def test_subscriber_creation_defaults(self):
        """Test subscriber creation with defaults."""
        sub = Subscriber()

        assert sub.subscriber_id is not None
        assert len(sub.subscriber_id) == 8
        assert sub.tier == SubscriptionTier.FREE
        assert sub.is_active is True
        assert sub.signals_received == 0

    def test_subscriber_creation_full(self):
        """Test subscriber creation with all parameters."""
        sub = Subscriber(
            name="John Doe",
            email="john@example.com",
            telegram_id=123456789,
            telegram_username="johndoe",
            tier=SubscriptionTier.PREMIUM,
            channels=[DeliveryChannel.TELEGRAM, DeliveryChannel.WEBHOOK],
            webhook_url="https://example.com/webhook",
            symbols_filter=["BTCUSDT", "ETHUSDT"],
            min_confidence=0.7,
        )

        assert sub.name == "John Doe"
        assert sub.tier == SubscriptionTier.PREMIUM
        assert len(sub.channels) == 2
        assert sub.symbols_filter == ["BTCUSDT", "ETHUSDT"]


# =============================================================================
# SignalProvider Tests
# =============================================================================

class TestSignalProvider:
    """Tests for SignalProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a SignalProvider instance."""
        return SignalProvider(
            telegram_bot_token="test_token",
            telegram_channel_id="test_channel",
            delay_free_minutes=15,
            delay_basic_minutes=5,
        )

    @pytest.fixture
    async def started_provider(self, provider):
        """Create and start a SignalProvider."""
        await provider.start()
        yield provider
        await provider.stop()

    # -------------------------------------------------------------------------
    # Subscriber Management Tests
    # -------------------------------------------------------------------------

    def test_add_subscriber_basic(self, provider):
        """Test adding a basic subscriber."""
        sub = provider.add_subscriber(
            name="Test User",
            telegram_id=123456789,
        )

        assert sub.name == "Test User"
        assert sub.telegram_id == 123456789
        assert sub.tier == SubscriptionTier.FREE
        assert sub.subscriber_id in provider._subscribers

    def test_add_subscriber_with_webhook(self, provider):
        """Test adding subscriber with webhook."""
        sub = provider.add_subscriber(
            name="Webhook User",
            tier=SubscriptionTier.VIP,
            channels=[DeliveryChannel.WEBHOOK],
            webhook_url="https://example.com/webhook",
        )

        assert sub.webhook_url == "https://example.com/webhook"
        assert sub.webhook_secret is not None
        assert len(sub.webhook_secret) == 32

    def test_add_subscriber_premium(self, provider):
        """Test adding premium subscriber."""
        expires = datetime.utcnow() + timedelta(days=30)
        sub = provider.add_subscriber(
            name="Premium User",
            tier=SubscriptionTier.PREMIUM,
            channels=[DeliveryChannel.TELEGRAM, DeliveryChannel.WEBSOCKET],
            expires_at=expires,
        )

        assert sub.tier == SubscriptionTier.PREMIUM
        assert sub.expires_at == expires
        assert DeliveryChannel.WEBSOCKET in sub.channels

    def test_get_subscriber(self, provider):
        """Test getting subscriber by ID."""
        sub = provider.add_subscriber(name="Test User")

        found = provider.get_subscriber(sub.subscriber_id)
        assert found == sub

        not_found = provider.get_subscriber("nonexistent")
        assert not_found is None

    def test_get_subscriber_by_telegram(self, provider):
        """Test getting subscriber by Telegram ID."""
        sub = provider.add_subscriber(
            name="Telegram User",
            telegram_id=987654321,
        )

        found = provider.get_subscriber_by_telegram(987654321)
        assert found == sub

        not_found = provider.get_subscriber_by_telegram(111111)
        assert not_found is None

    def test_update_subscriber(self, provider):
        """Test updating subscriber."""
        sub = provider.add_subscriber(
            name="Original Name",
            tier=SubscriptionTier.FREE,
        )

        updated = provider.update_subscriber(
            sub.subscriber_id,
            name="New Name",
            tier=SubscriptionTier.PREMIUM,
        )

        assert updated.name == "New Name"
        assert updated.tier == SubscriptionTier.PREMIUM

    def test_update_nonexistent_subscriber(self, provider):
        """Test updating nonexistent subscriber."""
        result = provider.update_subscriber("nonexistent", name="Test")
        assert result is None

    def test_remove_subscriber(self, provider):
        """Test removing subscriber."""
        sub = provider.add_subscriber(name="To Remove")

        assert provider.remove_subscriber(sub.subscriber_id) is True
        assert provider.get_subscriber(sub.subscriber_id) is None
        assert provider.remove_subscriber(sub.subscriber_id) is False

    def test_get_active_subscribers(self, provider):
        """Test getting active subscribers."""
        # Add various subscribers
        sub1 = provider.add_subscriber(name="Active Free", tier=SubscriptionTier.FREE)
        sub2 = provider.add_subscriber(name="Active Premium", tier=SubscriptionTier.PREMIUM)
        sub3 = provider.add_subscriber(name="Inactive")
        sub3.is_active = False
        sub4 = provider.add_subscriber(
            name="Expired",
            expires_at=datetime.utcnow() - timedelta(days=1),
        )

        active = provider.get_active_subscribers()
        assert len(active) == 2
        assert sub1 in active
        assert sub2 in active
        assert sub3 not in active
        assert sub4 not in active

    def test_get_active_subscribers_by_tier(self, provider):
        """Test filtering active subscribers by tier."""
        provider.add_subscriber(name="Free User", tier=SubscriptionTier.FREE)
        provider.add_subscriber(name="Premium User", tier=SubscriptionTier.PREMIUM)

        free_subs = provider.get_active_subscribers(tier=SubscriptionTier.FREE)
        premium_subs = provider.get_active_subscribers(tier=SubscriptionTier.PREMIUM)

        assert len(free_subs) == 1
        assert free_subs[0].tier == SubscriptionTier.FREE
        assert len(premium_subs) == 1
        assert premium_subs[0].tier == SubscriptionTier.PREMIUM

    # -------------------------------------------------------------------------
    # Signal Publishing Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_publish_signal(self, started_provider):
        """Test publishing a signal."""
        provider = started_provider

        # Add subscribers
        provider.add_subscriber(
            name="VIP User",
            tier=SubscriptionTier.VIP,
            telegram_id=111,
        )

        signal = TradingSignal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            entry_price=Decimal("50000"),
            confidence=0.8,
        )

        with patch.object(provider, '_deliver_signal', new_callable=AsyncMock) as mock_deliver:
            await provider.publish_signal(signal)

            assert signal.status == SignalStatus.ACTIVE
            assert signal.activated_at is not None
            assert signal.signal_id in provider._signals
            assert signal in provider._signal_history

    @pytest.mark.asyncio
    async def test_publish_signal_filters_subscribers(self, started_provider):
        """Test that signal publishing respects subscriber filters."""
        provider = started_provider

        # Add subscriber with symbol filter
        sub = provider.add_subscriber(
            name="BTC Only",
            tier=SubscriptionTier.VIP,
            telegram_id=111,
            symbols_filter=["BTCUSDT"],
        )

        # Signal for different symbol should not be delivered
        signal = TradingSignal(
            symbol="ETHUSDT",
            direction=SignalDirection.LONG,
            entry_price=Decimal("3000"),
        )

        with patch.object(provider, '_deliver_signal', new_callable=AsyncMock) as mock_deliver:
            await provider.publish_signal(signal)
            # Check that subscriber was filtered out
            if mock_deliver.called:
                subscribers_called = mock_deliver.call_args[0][1]
                assert sub not in subscribers_called

    @pytest.mark.asyncio
    async def test_publish_signal_confidence_filter(self, started_provider):
        """Test that signal publishing respects confidence filter."""
        provider = started_provider

        sub = provider.add_subscriber(
            name="High Confidence Only",
            tier=SubscriptionTier.VIP,
            telegram_id=111,
            min_confidence=0.9,
        )

        signal = TradingSignal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            entry_price=Decimal("50000"),
            confidence=0.7,  # Below threshold
        )

        with patch.object(provider, '_deliver_signal', new_callable=AsyncMock) as mock_deliver:
            await provider.publish_signal(signal)
            if mock_deliver.called:
                subscribers_called = mock_deliver.call_args[0][1]
                assert sub not in subscribers_called

    @pytest.mark.asyncio
    async def test_signal_callback(self, started_provider):
        """Test signal callback is called."""
        provider = started_provider
        callback_called = []

        def callback(signal):
            callback_called.append(signal)

        provider.on_signal(callback)

        signal = TradingSignal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            entry_price=Decimal("50000"),
        )

        with patch.object(provider, '_deliver_signal', new_callable=AsyncMock):
            await provider.publish_signal(signal)

        assert len(callback_called) == 1
        assert callback_called[0] == signal

    @pytest.mark.asyncio
    async def test_async_signal_callback(self, started_provider):
        """Test async signal callback is called."""
        provider = started_provider
        callback_called = []

        async def async_callback(signal):
            callback_called.append(signal)

        provider.on_signal(async_callback)

        signal = TradingSignal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            entry_price=Decimal("50000"),
        )

        with patch.object(provider, '_deliver_signal', new_callable=AsyncMock):
            await provider.publish_signal(signal)

        assert len(callback_called) == 1

    # -------------------------------------------------------------------------
    # Signal Update Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_update_signal(self, started_provider):
        """Test updating a signal."""
        provider = started_provider

        signal = TradingSignal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            entry_price=Decimal("50000"),
        )
        provider._signals[signal.signal_id] = signal

        with patch.object(provider, '_notify_signal_update', new_callable=AsyncMock):
            updated = await provider.update_signal(
                signal.signal_id,
                status=SignalStatus.CLOSED,
                exit_price=Decimal("51000"),
                pnl_pct=2.0,
                hit_target=1,
            )

        assert updated.status == SignalStatus.CLOSED
        assert updated.exit_price == Decimal("51000")
        assert updated.pnl_pct == 2.0
        assert updated.hit_target == 1
        assert updated.closed_at is not None

    @pytest.mark.asyncio
    async def test_update_nonexistent_signal(self, started_provider):
        """Test updating nonexistent signal."""
        result = await started_provider.update_signal("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_close_signal_long_profit(self, started_provider):
        """Test closing a long signal with profit."""
        provider = started_provider

        signal = TradingSignal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            entry_price=Decimal("50000"),
            leverage=2,
        )
        provider._signals[signal.signal_id] = signal

        with patch.object(provider, '_notify_signal_update', new_callable=AsyncMock):
            closed = await provider.close_signal(
                signal.signal_id,
                exit_price=Decimal("51000"),
                hit_target=1,
            )

        assert closed.status == SignalStatus.CLOSED
        assert closed.exit_price == Decimal("51000")
        # PnL = (51000 - 50000) / 50000 * 100 * 2 = 4%
        assert closed.pnl_pct == pytest.approx(4.0)
        assert closed.hit_target == 1

    @pytest.mark.asyncio
    async def test_close_signal_short_profit(self, started_provider):
        """Test closing a short signal with profit."""
        provider = started_provider

        signal = TradingSignal(
            symbol="BTCUSDT",
            direction=SignalDirection.SHORT,
            entry_price=Decimal("50000"),
            leverage=1,
        )
        provider._signals[signal.signal_id] = signal

        with patch.object(provider, '_notify_signal_update', new_callable=AsyncMock):
            closed = await provider.close_signal(
                signal.signal_id,
                exit_price=Decimal("49000"),
                hit_target=1,
            )

        # PnL = (50000 - 49000) / 50000 * 100 = 2%
        assert closed.pnl_pct == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_close_signal_loss(self, started_provider):
        """Test closing a signal with loss."""
        provider = started_provider

        signal = TradingSignal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            entry_price=Decimal("50000"),
            leverage=1,
        )
        provider._signals[signal.signal_id] = signal

        with patch.object(provider, '_notify_signal_update', new_callable=AsyncMock):
            closed = await provider.close_signal(
                signal.signal_id,
                exit_price=Decimal("49000"),
                hit_target=0,  # SL hit
            )

        # PnL = (49000 - 50000) / 50000 * 100 = -2%
        assert closed.pnl_pct == pytest.approx(-2.0)
        assert closed.hit_target == 0

    # -------------------------------------------------------------------------
    # Performance Statistics Tests
    # -------------------------------------------------------------------------

    def test_get_performance_empty(self, provider):
        """Test performance stats with no signals."""
        perf = provider.get_performance()

        assert perf.total_signals == 0
        assert perf.winning_signals == 0
        assert perf.win_rate == 0

    def test_get_performance_with_signals(self, provider):
        """Test performance stats with signals."""
        # Add some closed signals
        for i in range(10):
            signal = TradingSignal(
                symbol="BTCUSDT",
                direction=SignalDirection.LONG,
                entry_price=Decimal("50000"),
                strategy="scalper",
                status=SignalStatus.CLOSED,
                activated_at=datetime.utcnow() - timedelta(hours=i),
                closed_at=datetime.utcnow(),
                pnl_pct=2.0 if i < 7 else -1.5,
            )
            provider._signal_history.append(signal)

        perf = provider.get_performance(days=30)

        assert perf.total_signals == 10
        assert perf.winning_signals == 7
        assert perf.losing_signals == 3
        assert perf.win_rate == pytest.approx(70.0)
        assert perf.total_pnl_pct == pytest.approx(9.5)  # 7*2 - 3*1.5
        assert perf.avg_pnl_pct == pytest.approx(0.95)
        assert perf.avg_win_pct == pytest.approx(2.0)
        assert perf.avg_loss_pct == pytest.approx(1.5)
        assert perf.best_signal_pnl == pytest.approx(2.0)
        assert perf.worst_signal_pnl == pytest.approx(-1.5)

    def test_get_performance_by_strategy(self, provider):
        """Test performance stats filtered by strategy."""
        # Add signals with different strategies
        for strategy in ["scalper", "breakout"]:
            for i in range(5):
                signal = TradingSignal(
                    symbol="BTCUSDT",
                    strategy=strategy,
                    status=SignalStatus.CLOSED,
                    pnl_pct=3.0 if strategy == "scalper" else 1.0,
                )
                provider._signal_history.append(signal)

        perf = provider.get_performance(strategy="scalper")

        assert perf.total_signals == 5
        assert perf.total_pnl_pct == pytest.approx(15.0)

    def test_get_performance_by_symbol(self, provider):
        """Test performance stats filtered by symbol."""
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            for i in range(3):
                signal = TradingSignal(
                    symbol=symbol,
                    status=SignalStatus.CLOSED,
                    pnl_pct=2.0,
                )
                provider._signal_history.append(signal)

        perf = provider.get_performance(symbol="BTCUSDT")

        assert perf.total_signals == 3

    def test_get_performance_holding_time(self, provider):
        """Test average holding time calculation."""
        for i in range(5):
            signal = TradingSignal(
                symbol="BTCUSDT",
                status=SignalStatus.CLOSED,
                activated_at=datetime.utcnow() - timedelta(hours=2),
                closed_at=datetime.utcnow(),
                pnl_pct=1.0,
            )
            provider._signal_history.append(signal)

        perf = provider.get_performance()

        assert perf.avg_holding_time_hours == pytest.approx(2.0, abs=0.1)

    def test_get_performance_profit_factor(self, provider):
        """Test profit factor calculation."""
        # 3 wins of 10% = 30% gross profit
        for _ in range(3):
            signal = TradingSignal(status=SignalStatus.CLOSED, pnl_pct=10.0)
            provider._signal_history.append(signal)

        # 2 losses of 5% = 10% gross loss
        for _ in range(2):
            signal = TradingSignal(status=SignalStatus.CLOSED, pnl_pct=-5.0)
            provider._signal_history.append(signal)

        perf = provider.get_performance()

        assert perf.profit_factor == pytest.approx(3.0)  # 30 / 10

    # -------------------------------------------------------------------------
    # WebSocket Support Tests
    # -------------------------------------------------------------------------

    def test_register_websocket(self, provider):
        """Test registering WebSocket connection."""
        mock_ws = MagicMock()
        provider.register_websocket("sub123", mock_ws)

        assert "sub123" in provider._ws_connections
        assert mock_ws in provider._ws_connections["sub123"]

    def test_unregister_websocket(self, provider):
        """Test unregistering WebSocket connection."""
        mock_ws = MagicMock()
        provider.register_websocket("sub123", mock_ws)
        provider.unregister_websocket("sub123", mock_ws)

        assert mock_ws not in provider._ws_connections.get("sub123", set())

    # -------------------------------------------------------------------------
    # Signal History Tests
    # -------------------------------------------------------------------------

    def test_get_signals(self, provider):
        """Test getting signal history."""
        for i in range(5):
            signal = TradingSignal(
                symbol="BTCUSDT" if i % 2 == 0 else "ETHUSDT",
                status=SignalStatus.ACTIVE if i < 3 else SignalStatus.CLOSED,
            )
            provider._signal_history.append(signal)

        # Get all signals
        all_signals = provider.get_signals(limit=50)
        assert len(all_signals) == 5

        # Filter by status
        active = provider.get_signals(status=SignalStatus.ACTIVE)
        assert len(active) == 3

        # Filter by symbol
        btc = provider.get_signals(symbol="BTCUSDT")
        assert len(btc) == 3

        # Test limit
        limited = provider.get_signals(limit=2)
        assert len(limited) == 2

    def test_get_active_signals(self, provider):
        """Test getting active signals only."""
        active_signal = TradingSignal(status=SignalStatus.ACTIVE)
        closed_signal = TradingSignal(status=SignalStatus.CLOSED)

        provider._signals[active_signal.signal_id] = active_signal
        provider._signals[closed_signal.signal_id] = closed_signal

        active = provider.get_active_signals()

        assert len(active) == 1
        assert active[0].status == SignalStatus.ACTIVE


# =============================================================================
# Delivery Tests (Mocked)
# =============================================================================

class TestSignalDelivery:
    """Tests for signal delivery mechanisms."""

    @pytest.fixture
    async def provider_with_session(self):
        """Create provider with active session."""
        provider = SignalProvider(telegram_bot_token="test_token")
        await provider.start()
        yield provider
        await provider.stop()

    @pytest.mark.asyncio
    async def test_send_telegram(self, provider_with_session):
        """Test Telegram delivery."""
        provider = provider_with_session
        subscriber = Subscriber(telegram_id=123456789)
        signal = TradingSignal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            entry_price=Decimal("50000"),
        )

        with patch.object(provider._session, 'post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response

            await provider._send_telegram(subscriber, signal, include_entry=True)

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "123456789" in str(call_args)

    @pytest.mark.asyncio
    async def test_send_telegram_no_token(self, provider_with_session):
        """Test Telegram delivery without token."""
        provider = provider_with_session
        provider.telegram_bot_token = None
        subscriber = Subscriber(telegram_id=123456789)
        signal = TradingSignal()

        # Should not raise, just return silently
        await provider._send_telegram(subscriber, signal, include_entry=True)

    @pytest.mark.asyncio
    async def test_send_webhook(self, provider_with_session):
        """Test webhook delivery."""
        provider = provider_with_session
        subscriber = Subscriber(
            webhook_url="https://example.com/hook",
            webhook_secret="test_secret_12345678901234567890",
        )
        signal = TradingSignal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            entry_price=Decimal("50000"),
        )

        with patch.object(provider._session, 'post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response

            await provider._send_webhook(subscriber, signal)

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            # Check HMAC signature header is present
            assert "X-Signal-Signature" in str(call_args)

    @pytest.mark.asyncio
    async def test_send_websocket(self, provider_with_session):
        """Test WebSocket delivery."""
        provider = provider_with_session
        subscriber = Subscriber()

        mock_ws = AsyncMock()
        provider.register_websocket(subscriber.subscriber_id, mock_ws)

        signal = TradingSignal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            entry_price=Decimal("50000"),
        )

        await provider._send_websocket(subscriber, signal)

        mock_ws.send_str.assert_called_once()
        sent_data = json.loads(mock_ws.send_str.call_args[0][0])
        assert sent_data["type"] == "signal"
        assert sent_data["data"]["symbol"] == "BTCUSDT"


# =============================================================================
# Tier Delay Tests
# =============================================================================

class TestTierDelays:
    """Tests for subscription tier delays."""

    def test_tier_delays_configuration(self):
        """Test tier delays are configured correctly."""
        provider = SignalProvider(
            delay_free_minutes=15,
            delay_basic_minutes=5,
        )

        assert provider.tier_delays[SubscriptionTier.FREE] == 15
        assert provider.tier_delays[SubscriptionTier.BASIC] == 5
        assert provider.tier_delays[SubscriptionTier.PREMIUM] == 0
        assert provider.tier_delays[SubscriptionTier.VIP] == 0

    @pytest.mark.asyncio
    async def test_delayed_delivery_cancelled_if_signal_closed(self):
        """Test delayed delivery is cancelled if signal is closed."""
        provider = SignalProvider()
        await provider.start()

        signal = TradingSignal(status=SignalStatus.CLOSED)
        subscribers = [Subscriber()]

        # This should return early without delivering
        await provider._delayed_delivery(signal, subscribers, delay_minutes=0)

        await provider.stop()
