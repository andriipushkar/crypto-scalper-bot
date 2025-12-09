"""
Main trading engine.

Orchestrates all components: data feeds, strategies, risk management, execution.
"""

import asyncio
import signal
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any

from loguru import logger

from src.core.events import EventBus, EventType, Event, get_event_bus
from src.data.websocket import BinanceWebSocket, create_websocket_from_config
from src.data.orderbook import OrderBookManager
from src.data.models import (
    TradeEvent,
    DepthUpdateEvent,
    MarkPriceEvent,
    OrderBookEvent,
    Signal,
    SignalType,
)


class EngineState(Enum):
    """Engine state."""
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    ERROR = auto()


class TradingEngine:
    """
    Main trading engine.

    Coordinates:
    - Market data feeds (WebSocket)
    - Order book management
    - Strategy execution
    - Risk management
    - Order execution
    """

    def __init__(self, config: dict):
        """
        Initialize trading engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._state = EngineState.STOPPED

        # Core components
        self.event_bus: EventBus = get_event_bus()
        self.websocket: Optional[BinanceWebSocket] = None
        self.orderbook_manager: Optional[OrderBookManager] = None

        # Pluggable components (set via methods)
        self.strategies: List[Any] = []
        self.risk_manager: Optional[Any] = None
        self.executor: Optional[Any] = None

        # Runtime state
        self._start_time: Optional[datetime] = None
        self._tasks: List[asyncio.Task] = []

        # Stats
        self._signals_generated = 0
        self._orders_placed = 0
        self._errors = 0

    @property
    def state(self) -> EngineState:
        """Get current engine state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._state == EngineState.RUNNING

    @property
    def uptime(self) -> Optional[float]:
        """Get engine uptime in seconds."""
        if self._start_time:
            return (datetime.utcnow() - self._start_time).total_seconds()
        return None

    @property
    def stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "state": self._state.name,
            "uptime_seconds": self.uptime,
            "signals_generated": self._signals_generated,
            "orders_placed": self._orders_placed,
            "errors": self._errors,
            "event_bus": self.event_bus.stats,
            "websocket": self.websocket.stats if self.websocket else None,
        }

    # =========================================================================
    # Component Registration
    # =========================================================================

    def add_strategy(self, strategy) -> "TradingEngine":
        """
        Add a strategy to the engine.

        Args:
            strategy: Strategy instance (must have on_orderbook method)

        Returns:
            Self for chaining
        """
        self.strategies.append(strategy)
        logger.info(f"Added strategy: {strategy.__class__.__name__}")
        return self

    def set_risk_manager(self, risk_manager) -> "TradingEngine":
        """
        Set the risk manager.

        Args:
            risk_manager: RiskManager instance

        Returns:
            Self for chaining
        """
        self.risk_manager = risk_manager
        logger.info(f"Set risk manager: {risk_manager.__class__.__name__}")
        return self

    def set_executor(self, executor) -> "TradingEngine":
        """
        Set the order executor.

        Args:
            executor: Executor instance

        Returns:
            Self for chaining
        """
        self.executor = executor
        logger.info(f"Set executor: {executor.__class__.__name__}")
        return self

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start the trading engine."""
        if self._state != EngineState.STOPPED:
            logger.warning(f"Cannot start: engine is {self._state.name}")
            return

        logger.info("=" * 50)
        logger.info("Starting Trading Engine")
        logger.info("=" * 50)

        self._state = EngineState.STARTING

        try:
            # Start event bus
            await self.event_bus.start()

            # Initialize components
            await self._init_components()

            # Register event handlers
            self._register_handlers()

            # Sync order books
            await self._sync_orderbooks()

            # Start WebSocket
            ws_task = asyncio.create_task(self.websocket.start())
            self._tasks.append(ws_task)

            self._state = EngineState.RUNNING
            self._start_time = datetime.utcnow()

            logger.info("Trading Engine started successfully")
            logger.info(f"Strategies: {[s.__class__.__name__ for s in self.strategies]}")
            logger.info(f"Risk Manager: {self.risk_manager.__class__.__name__ if self.risk_manager else 'None'}")

            # Wait for WebSocket task
            await ws_task

        except Exception as e:
            logger.error(f"Failed to start engine: {e}")
            self._state = EngineState.ERROR
            self._errors += 1
            raise

    async def stop(self) -> None:
        """Stop the trading engine gracefully."""
        if self._state not in (EngineState.RUNNING, EngineState.ERROR):
            return

        logger.info("Stopping Trading Engine...")
        self._state = EngineState.STOPPING

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Stop WebSocket
        if self.websocket:
            await self.websocket.stop()

        # Stop event bus
        await self.event_bus.stop()

        self._state = EngineState.STOPPED
        self._print_final_stats()
        logger.info("Trading Engine stopped")

    async def _init_components(self) -> None:
        """Initialize all components."""
        exchange_config = self.config.get("exchange", {})
        data_config = self.config.get("data", {})
        trading_config = self.config.get("trading", {})

        # WebSocket
        self.websocket = create_websocket_from_config(self.config)

        # Order book manager
        self.orderbook_manager = OrderBookManager(
            depth=data_config.get("orderbook_depth", 20),
            testnet=exchange_config.get("testnet", True),
        )

        # Pre-create order books for all symbols
        for symbol in trading_config.get("symbols", ["BTCUSDT"]):
            self.orderbook_manager.get_book(symbol)

    def _register_handlers(self) -> None:
        """Register event handlers."""
        # WebSocket callbacks -> Event Bus
        self.websocket.on_trade(self._on_ws_trade)
        self.websocket.on_depth(self._on_ws_depth)
        self.websocket.on_mark_price(self._on_ws_mark_price)

        # Event bus subscriptions
        self.event_bus.subscribe(
            EventType.ORDERBOOK_UPDATE,
            self._on_orderbook_update,
            priority=10,
            group="engine",
        )

        self.event_bus.subscribe(
            EventType.SIGNAL,
            self._on_signal,
            priority=10,
            group="engine",
        )

    async def _sync_orderbooks(self) -> None:
        """Synchronize order books with exchange."""
        logger.info("Syncing order books...")
        await self.orderbook_manager.sync_all()
        logger.info("Order books synced")

    def _print_final_stats(self) -> None:
        """Print final statistics."""
        logger.info("=" * 50)
        logger.info("Trading Engine Final Statistics")
        logger.info("=" * 50)

        if self.uptime:
            hours = self.uptime / 3600
            logger.info(f"Uptime: {self.uptime:.0f}s ({hours:.2f}h)")

        logger.info(f"Signals generated: {self._signals_generated}")
        logger.info(f"Orders placed: {self._orders_placed}")
        logger.info(f"Errors: {self._errors}")

        if self.event_bus:
            logger.info(f"Events processed: {self.event_bus.stats['events_processed']}")

    # =========================================================================
    # WebSocket Event Handlers
    # =========================================================================

    async def _on_ws_trade(self, event: TradeEvent) -> None:
        """Handle trade from WebSocket."""
        # Publish to event bus
        await self.event_bus.publish(Event(
            event_type=EventType.TRADE,
            data=event,
            source="websocket",
        ))

    async def _on_ws_depth(self, event: DepthUpdateEvent) -> None:
        """Handle depth update from WebSocket."""
        # Update order book
        await self.orderbook_manager.handle_depth_update(event)

        # Get updated snapshot and publish
        symbol = event.update.symbol
        book = self.orderbook_manager.get_book(symbol)

        if book.is_synced:
            snapshot = book.get_snapshot()
            await self.event_bus.publish(Event(
                event_type=EventType.ORDERBOOK_UPDATE,
                data=OrderBookEvent(snapshot=snapshot),
                source="orderbook",
            ))

    async def _on_ws_mark_price(self, event: MarkPriceEvent) -> None:
        """Handle mark price from WebSocket."""
        await self.event_bus.publish(Event(
            event_type=EventType.MARK_PRICE,
            data=event,
            source="websocket",
        ))

    # =========================================================================
    # Event Bus Handlers
    # =========================================================================

    async def _on_orderbook_update(self, event: Event) -> None:
        """Handle order book update - run strategies."""
        if not self.is_running:
            return

        orderbook_event: OrderBookEvent = event.data
        snapshot = orderbook_event.snapshot

        # Run each strategy
        for strategy in self.strategies:
            try:
                signal = await self._run_strategy(strategy, snapshot)

                if signal and signal.signal_type != SignalType.NO_ACTION:
                    self._signals_generated += 1
                    await self.event_bus.publish(Event(
                        event_type=EventType.SIGNAL,
                        data=signal,
                        source=strategy.__class__.__name__,
                    ))

            except Exception as e:
                logger.error(f"Strategy error ({strategy.__class__.__name__}): {e}")
                self._errors += 1

    async def _run_strategy(self, strategy, snapshot) -> Optional[Signal]:
        """Run a single strategy."""
        if asyncio.iscoroutinefunction(strategy.on_orderbook):
            return await strategy.on_orderbook(snapshot)
        else:
            return strategy.on_orderbook(snapshot)

    async def _on_signal(self, event: Event) -> None:
        """Handle trading signal - check risk and execute."""
        if not self.is_running:
            return

        signal: Signal = event.data

        logger.info(
            f"Signal: {signal.signal_type.name} {signal.symbol} "
            f"@ {signal.price} (strength={signal.strength:.2f}, "
            f"strategy={signal.strategy})"
        )

        # Risk check
        if self.risk_manager:
            can_trade, reason = self.risk_manager.can_trade(signal)

            if not can_trade:
                logger.warning(f"Risk check failed: {reason}")
                return

            # Calculate position size
            size = self.risk_manager.calculate_position_size(signal)

            if size <= 0:
                logger.warning("Position size is zero, skipping")
                return

            signal.metadata["position_size"] = size

        # Execute
        if self.executor:
            try:
                order = await self.executor.execute_signal(signal)

                if order:
                    self._orders_placed += 1
                    logger.info(f"Order placed: {order.order_id}")

            except Exception as e:
                logger.error(f"Execution error: {e}")
                self._errors += 1


# =============================================================================
# Factory function
# =============================================================================

def create_engine(config: dict) -> TradingEngine:
    """
    Create a configured trading engine.

    Args:
        config: Configuration dictionary

    Returns:
        Configured TradingEngine instance
    """
    return TradingEngine(config)
