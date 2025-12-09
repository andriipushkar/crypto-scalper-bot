"""
Unit tests for Event Bus.
"""

import pytest
import asyncio
from datetime import datetime

from src.core.events import (
    EventType,
    Event,
    EventBus,
    get_event_bus,
    create_trade_event,
    create_signal_event,
)
from src.data.models import Trade, Signal, SignalType
from decimal import Decimal


class TestEvent:
    """Tests for Event class."""

    def test_create_event(self):
        event = Event(
            event_type=EventType.TRADE,
            data={"test": "data"},
            source="test",
        )

        assert event.event_type == EventType.TRADE
        assert event.data == {"test": "data"}
        assert event.source == "test"
        assert event.timestamp is not None

    def test_event_default_timestamp(self):
        event = Event(event_type=EventType.SIGNAL)
        assert isinstance(event.timestamp, datetime)


class TestEventType:
    """Tests for EventType enum."""

    def test_market_data_events(self):
        assert EventType.TRADE.value is not None
        assert EventType.DEPTH_UPDATE.value is not None
        assert EventType.ORDERBOOK_UPDATE.value is not None
        assert EventType.MARK_PRICE.value is not None

    def test_trading_events(self):
        assert EventType.SIGNAL.value is not None
        assert EventType.ORDER_NEW.value is not None
        assert EventType.ORDER_FILLED.value is not None
        assert EventType.POSITION_OPEN.value is not None

    def test_system_events(self):
        assert EventType.CONNECTED.value is not None
        assert EventType.DISCONNECTED.value is not None
        assert EventType.ERROR.value is not None
        assert EventType.SHUTDOWN.value is not None


@pytest.mark.asyncio
class TestEventBus:
    """Tests for EventBus class."""

    async def test_start_stop(self):
        bus = EventBus()

        await bus.start()
        assert bus._running == True

        await bus.stop()
        assert bus._running == False

    async def test_subscribe_handler(self):
        bus = EventBus()
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe(EventType.TRADE, handler)

        assert len(bus._handlers[EventType.TRADE]) == 1

    async def test_subscribe_multiple_types(self):
        bus = EventBus()

        def handler(event):
            pass

        bus.subscribe([EventType.TRADE, EventType.SIGNAL], handler)

        assert len(bus._handlers[EventType.TRADE]) == 1
        assert len(bus._handlers[EventType.SIGNAL]) == 1

    async def test_unsubscribe_handler(self):
        bus = EventBus()

        def handler(event):
            pass

        bus.subscribe(EventType.TRADE, handler)
        assert len(bus._handlers[EventType.TRADE]) == 1

        bus.unsubscribe(EventType.TRADE, handler)
        assert len(bus._handlers[EventType.TRADE]) == 0

    async def test_unsubscribe_group(self):
        bus = EventBus()

        def handler1(event):
            pass

        def handler2(event):
            pass

        bus.subscribe(EventType.TRADE, handler1, group="test_group")
        bus.subscribe(EventType.SIGNAL, handler2, group="test_group")

        assert len(bus._handlers[EventType.TRADE]) == 1
        assert len(bus._handlers[EventType.SIGNAL]) == 1

        bus.unsubscribe_group("test_group")

        assert len(bus._handlers[EventType.TRADE]) == 0
        assert len(bus._handlers[EventType.SIGNAL]) == 0

    async def test_emit_sync_handler(self):
        bus = EventBus()
        received = []

        def handler(event):
            received.append(event.data)

        bus.subscribe(EventType.TRADE, handler)

        event = Event(event_type=EventType.TRADE, data="test_data")
        await bus.emit(event)

        assert len(received) == 1
        assert received[0] == "test_data"

    async def test_emit_async_handler(self):
        bus = EventBus()
        received = []

        async def async_handler(event):
            await asyncio.sleep(0.01)
            received.append(event.data)

        bus.subscribe(EventType.TRADE, async_handler)

        event = Event(event_type=EventType.TRADE, data="async_data")
        await bus.emit(event)

        assert len(received) == 1
        assert received[0] == "async_data"

    async def test_publish_queues_event(self):
        bus = EventBus()
        await bus.start()

        received = []

        def handler(event):
            received.append(event.data)

        bus.subscribe(EventType.TRADE, handler)

        event = Event(event_type=EventType.TRADE, data="queued_data")
        await bus.publish(event)

        # Wait for processing
        await asyncio.sleep(0.1)

        assert len(received) == 1
        await bus.stop()

    async def test_priority_ordering(self):
        bus = EventBus()
        order = []

        def low_priority(event):
            order.append("low")

        def high_priority(event):
            order.append("high")

        bus.subscribe(EventType.TRADE, low_priority, priority=0)
        bus.subscribe(EventType.TRADE, high_priority, priority=10)

        event = Event(event_type=EventType.TRADE)
        await bus.emit(event)

        assert order == ["high", "low"]

    async def test_handler_error_isolation(self):
        bus = EventBus()
        received = []

        def failing_handler(event):
            raise ValueError("Test error")

        def working_handler(event):
            received.append("success")

        bus.subscribe(EventType.TRADE, failing_handler, priority=10)
        bus.subscribe(EventType.TRADE, working_handler, priority=0)

        event = Event(event_type=EventType.TRADE)
        await bus.emit(event)

        # Working handler should still be called
        assert received == ["success"]

    async def test_stats(self):
        bus = EventBus()
        await bus.start()

        def handler(event):
            pass

        bus.subscribe(EventType.TRADE, handler)

        event = Event(event_type=EventType.TRADE)
        await bus.publish(event)
        await asyncio.sleep(0.1)

        stats = bus.stats
        assert stats["events_published"] >= 1
        assert stats["handler_count"] >= 1
        assert stats["running"] == True

        await bus.stop()

    async def test_decorator_subscription(self):
        bus = EventBus()
        received = []

        @bus.on(EventType.SIGNAL)
        def signal_handler(event):
            received.append(event)

        assert len(bus._handlers[EventType.SIGNAL]) == 1

        event = Event(event_type=EventType.SIGNAL, data="decorator_test")
        await bus.emit(event)

        assert len(received) == 1


class TestEventCreation:
    """Tests for event creation helpers."""

    def test_create_trade_event(self):
        trade = Trade(
            symbol="BTCUSDT",
            trade_id=123,
            price=Decimal("50000"),
            quantity=Decimal("0.1"),
            timestamp=datetime.utcnow(),
            is_buyer_maker=False,
        )

        event = create_trade_event(trade, source="test")

        assert event.event_type == EventType.TRADE
        assert event.source == "test"
        assert event.data.trade == trade

    def test_create_signal_event(self):
        signal = Signal(
            strategy="test",
            signal_type=SignalType.LONG,
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            strength=0.8,
            price=Decimal("50000"),
        )

        event = create_signal_event(signal, source="strategy")

        assert event.event_type == EventType.SIGNAL
        assert event.source == "strategy"
        assert event.data.signal == signal


class TestGetEventBus:
    """Tests for global event bus."""

    def test_get_event_bus_singleton(self):
        bus1 = get_event_bus()
        bus2 = get_event_bus()

        assert bus1 is bus2
