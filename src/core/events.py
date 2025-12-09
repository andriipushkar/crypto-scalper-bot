"""
Event Bus for inter-component communication.

Implements a simple pub/sub pattern for decoupled event handling.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Union
from weakref import WeakSet

from loguru import logger


class EventType(Enum):
    """All event types in the system."""

    # Market data events
    TRADE = auto()
    DEPTH_UPDATE = auto()
    ORDERBOOK_UPDATE = auto()
    MARK_PRICE = auto()

    # Trading events
    SIGNAL = auto()
    ORDER_NEW = auto()
    ORDER_FILLED = auto()
    ORDER_CANCELED = auto()
    ORDER_REJECTED = auto()
    POSITION_OPEN = auto()
    POSITION_CLOSE = auto()
    POSITION_UPDATE = auto()

    # System events
    CONNECTED = auto()
    DISCONNECTED = auto()
    ERROR = auto()
    SHUTDOWN = auto()


@dataclass
class Event:
    """
    Base event class.

    All events in the system inherit from this.
    """

    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Any = None
    source: str = ""

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


# Type alias for event handlers
EventHandler = Callable[[Event], Any]


class EventBus:
    """
    Central event bus for pub/sub communication.

    Features:
    - Async and sync handler support
    - Priority-based handler execution
    - Event filtering
    - Handler groups for batch unsubscribe
    """

    def __init__(self):
        # event_type -> list of (priority, handler)
        self._handlers: Dict[EventType, List[tuple]] = defaultdict(list)

        # handler groups for batch management
        self._groups: Dict[str, Set[tuple]] = defaultdict(set)

        # Event queue for async processing
        self._queue: asyncio.Queue = None
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None

        # Stats
        self._events_published = 0
        self._events_processed = 0

    @property
    def stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            "events_published": self._events_published,
            "events_processed": self._events_processed,
            "queue_size": self._queue.qsize() if self._queue else 0,
            "handler_count": sum(len(h) for h in self._handlers.values()),
            "running": self._running,
        }

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start the event bus processor."""
        if self._running:
            return

        logger.info("Starting event bus")
        self._queue = asyncio.Queue()
        self._running = True
        self._processor_task = asyncio.create_task(self._process_events())

    async def stop(self) -> None:
        """Stop the event bus processor."""
        if not self._running:
            return

        logger.info("Stopping event bus")
        self._running = False

        # Signal processor to stop
        await self._queue.put(None)

        # Wait for processor to finish
        if self._processor_task:
            await self._processor_task
            self._processor_task = None

        logger.info(f"Event bus stopped. Processed {self._events_processed} events")

    # =========================================================================
    # Subscription
    # =========================================================================

    def subscribe(
        self,
        event_type: Union[EventType, List[EventType]],
        handler: EventHandler,
        priority: int = 0,
        group: str = None,
    ) -> None:
        """
        Subscribe a handler to event type(s).

        Args:
            event_type: Single event type or list of types
            handler: Callback function (sync or async)
            priority: Higher priority handlers execute first (default 0)
            group: Optional group name for batch unsubscribe
        """
        if isinstance(event_type, list):
            for et in event_type:
                self._subscribe_single(et, handler, priority, group)
        else:
            self._subscribe_single(event_type, handler, priority, group)

    def _subscribe_single(
        self,
        event_type: EventType,
        handler: EventHandler,
        priority: int,
        group: str,
    ) -> None:
        """Subscribe to a single event type."""
        entry = (priority, handler)
        self._handlers[event_type].append(entry)

        # Sort by priority (descending)
        self._handlers[event_type].sort(key=lambda x: -x[0])

        # Track in group
        if group:
            self._groups[group].add((event_type, handler))

        logger.debug(
            f"Subscribed {handler.__name__} to {event_type.name} "
            f"(priority={priority}, group={group})"
        )

    def unsubscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> None:
        """Unsubscribe a handler from an event type."""
        handlers = self._handlers[event_type]
        self._handlers[event_type] = [
            (p, h) for p, h in handlers if h != handler
        ]
        logger.debug(f"Unsubscribed {handler.__name__} from {event_type.name}")

    def unsubscribe_group(self, group: str) -> None:
        """Unsubscribe all handlers in a group."""
        if group not in self._groups:
            return

        for event_type, handler in self._groups[group]:
            self.unsubscribe(event_type, handler)

        del self._groups[group]
        logger.debug(f"Unsubscribed all handlers in group '{group}'")

    def unsubscribe_all(self) -> None:
        """Unsubscribe all handlers."""
        self._handlers.clear()
        self._groups.clear()
        logger.debug("Unsubscribed all handlers")

    # =========================================================================
    # Publishing
    # =========================================================================

    async def publish(self, event: Event) -> None:
        """
        Publish an event asynchronously.

        Event is queued for processing by the event processor.
        """
        if not self._running:
            logger.warning("Event bus not running, event dropped")
            return

        self._events_published += 1
        await self._queue.put(event)

    def publish_sync(self, event: Event) -> None:
        """
        Publish an event synchronously (fire and forget).

        Use this when you can't await (e.g., from sync callbacks).
        """
        if not self._running:
            logger.warning("Event bus not running, event dropped")
            return

        self._events_published += 1
        self._queue.put_nowait(event)

    async def emit(self, event: Event) -> None:
        """
        Emit an event and wait for all handlers to complete.

        Use this when you need to ensure handlers finish before continuing.
        """
        self._events_published += 1
        await self._dispatch(event)

    # =========================================================================
    # Processing
    # =========================================================================

    async def _process_events(self) -> None:
        """Main event processing loop."""
        logger.debug("Event processor started")

        while self._running:
            try:
                event = await self._queue.get()

                # None is shutdown signal
                if event is None:
                    break

                await self._dispatch(event)
                self._events_processed += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}")

        logger.debug("Event processor stopped")

    async def _dispatch(self, event: Event) -> None:
        """Dispatch event to all subscribed handlers."""
        handlers = self._handlers.get(event.event_type, [])

        if not handlers:
            return

        for priority, handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)

            except Exception as e:
                logger.error(
                    f"Error in handler {handler.__name__} for "
                    f"{event.event_type.name}: {e}"
                )

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def on(self, event_type: EventType, priority: int = 0, group: str = None):
        """
        Decorator for subscribing handlers.

        Usage:
            @event_bus.on(EventType.TRADE)
            async def handle_trade(event):
                ...
        """

        def decorator(handler: EventHandler) -> EventHandler:
            self.subscribe(event_type, handler, priority, group)
            return handler

        return decorator


# =============================================================================
# Convenience event creation functions
# =============================================================================

def create_trade_event(trade, source: str = "websocket") -> Event:
    """Create a trade event."""
    from src.data.models import TradeEvent

    return Event(
        event_type=EventType.TRADE,
        data=TradeEvent(trade=trade),
        source=source,
    )


def create_orderbook_event(snapshot, source: str = "orderbook") -> Event:
    """Create an order book event."""
    from src.data.models import OrderBookEvent

    return Event(
        event_type=EventType.ORDERBOOK_UPDATE,
        data=OrderBookEvent(snapshot=snapshot),
        source=source,
    )


def create_signal_event(signal, source: str = "strategy") -> Event:
    """Create a signal event."""
    from src.data.models import SignalEvent

    return Event(
        event_type=EventType.SIGNAL,
        data=SignalEvent(signal=signal),
        source=source,
    )


def create_order_event(
    order,
    event_type: EventType = EventType.ORDER_NEW,
    source: str = "execution",
) -> Event:
    """Create an order event."""
    from src.data.models import OrderEvent

    type_map = {
        EventType.ORDER_NEW: "new",
        EventType.ORDER_FILLED: "filled",
        EventType.ORDER_CANCELED: "canceled",
        EventType.ORDER_REJECTED: "rejected",
    }

    return Event(
        event_type=event_type,
        data=OrderEvent(order=order, event_type=type_map.get(event_type, "unknown")),
        source=source,
    )


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get or create the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
