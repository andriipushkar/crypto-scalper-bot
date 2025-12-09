"""
Event System for Trading Bot

Defines event types and provides an in-memory event bus with Kafka integration.
"""
import asyncio
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
from decimal import Decimal
import uuid

from loguru import logger


class EventType(Enum):
    """Trading event types."""
    # Market Data Events
    TICKER_UPDATE = "ticker_update"
    ORDERBOOK_UPDATE = "orderbook_update"
    TRADE_UPDATE = "trade_update"
    KLINE_UPDATE = "kline_update"

    # Trading Events
    SIGNAL_GENERATED = "signal_generated"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"

    # Position Events
    POSITION_OPENED = "position_opened"
    POSITION_UPDATED = "position_updated"
    POSITION_CLOSED = "position_closed"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    TAKE_PROFIT_TRIGGERED = "take_profit_triggered"

    # Risk Events
    RISK_LIMIT_REACHED = "risk_limit_reached"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MAX_DRAWDOWN_REACHED = "max_drawdown_reached"

    # System Events
    BOT_STARTED = "bot_started"
    BOT_STOPPED = "bot_stopped"
    EXCHANGE_CONNECTED = "exchange_connected"
    EXCHANGE_DISCONNECTED = "exchange_disconnected"
    ERROR = "error"

    # Alert Events
    ALERT_INFO = "alert_info"
    ALERT_WARNING = "alert_warning"
    ALERT_CRITICAL = "alert_critical"


@dataclass
class Event:
    """Base event class."""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "trading-bot"
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=EventType(data["event_type"]),
            source=data.get("source", "unknown"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data.get("correlation_id"),
            data=data.get("data", {}),
        )


class EventBus:
    """
    In-memory event bus with optional Kafka integration.

    Provides publish/subscribe pattern for trading events.
    """

    def __init__(self, kafka_client=None):
        self._handlers: Dict[EventType, List[Callable]] = {}
        self._global_handlers: List[Callable] = []
        self._kafka = kafka_client
        self._event_history: List[Event] = []
        self._max_history = 1000

    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], None],
    ) -> None:
        """Subscribe to specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"Handler subscribed to {event_type.value}")

    def subscribe_all(self, handler: Callable[[Event], None]) -> None:
        """Subscribe to all events."""
        self._global_handlers.append(handler)
        logger.debug("Handler subscribed to all events")

    def unsubscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], None],
    ) -> None:
        """Unsubscribe from event type."""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass

    async def publish(self, event: Event) -> None:
        """
        Publish event to all subscribers.

        Args:
            event: Event to publish
        """
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        # Call type-specific handlers
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")

        # Call global handlers
        for handler in self._global_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in global event handler: {e}")

        # Publish to Kafka if connected
        if self._kafka and self._kafka.is_connected:
            await self._publish_to_kafka(event)

    async def _publish_to_kafka(self, event: Event) -> None:
        """Publish event to Kafka."""
        # Map event types to Kafka topics
        topic_map = {
            EventType.TICKER_UPDATE: "market_data",
            EventType.ORDERBOOK_UPDATE: "market_data",
            EventType.TRADE_UPDATE: "market_data",
            EventType.KLINE_UPDATE: "market_data",
            EventType.SIGNAL_GENERATED: "signals",
            EventType.ORDER_PLACED: "orders",
            EventType.ORDER_FILLED: "orders",
            EventType.ORDER_CANCELLED: "orders",
            EventType.ORDER_REJECTED: "orders",
            EventType.POSITION_OPENED: "positions",
            EventType.POSITION_UPDATED: "positions",
            EventType.POSITION_CLOSED: "positions",
            EventType.ALERT_INFO: "alerts",
            EventType.ALERT_WARNING: "alerts",
            EventType.ALERT_CRITICAL: "alerts",
        }

        topic = topic_map.get(event.event_type, "audit")

        try:
            await self._kafka.publish(
                topic=topic,
                message=event.to_dict(),
                key=event.data.get("symbol"),
            )
        except Exception as e:
            logger.error(f"Failed to publish to Kafka: {e}")

    def emit(self, event_type: EventType, data: Dict[str, Any]) -> Event:
        """
        Create and publish event synchronously (fire-and-forget).

        For async publishing, use publish() directly.
        """
        event = Event(event_type=event_type, data=data)
        asyncio.create_task(self.publish(event))
        return event

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100,
    ) -> List[Event]:
        """Get event history."""
        events = self._event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()


# Convenience functions for creating events
def signal_event(
    symbol: str,
    strategy: str,
    signal_type: str,
    strength: float,
    **kwargs,
) -> Event:
    """Create signal event."""
    return Event(
        event_type=EventType.SIGNAL_GENERATED,
        data={
            "symbol": symbol,
            "strategy": strategy,
            "signal_type": signal_type,
            "strength": strength,
            **kwargs,
        },
    )


def order_event(
    event_type: EventType,
    order_id: str,
    symbol: str,
    side: str,
    **kwargs,
) -> Event:
    """Create order event."""
    return Event(
        event_type=event_type,
        data={
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            **kwargs,
        },
    )


def position_event(
    event_type: EventType,
    symbol: str,
    side: str,
    quantity: Decimal,
    **kwargs,
) -> Event:
    """Create position event."""
    return Event(
        event_type=event_type,
        data={
            "symbol": symbol,
            "side": side,
            "quantity": str(quantity),
            **kwargs,
        },
    )


def alert_event(
    severity: str,
    message: str,
    **kwargs,
) -> Event:
    """Create alert event."""
    event_type = {
        "info": EventType.ALERT_INFO,
        "warning": EventType.ALERT_WARNING,
        "critical": EventType.ALERT_CRITICAL,
    }.get(severity, EventType.ALERT_INFO)

    return Event(
        event_type=event_type,
        data={
            "severity": severity,
            "message": message,
            **kwargs,
        },
    )
