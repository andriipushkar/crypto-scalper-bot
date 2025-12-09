"""
Comprehensive tests for TradingEngine.

Tests cover:
- Engine state management
- Component registration
- Lifecycle (start/stop)
- Event handling
- Signal processing
- Statistics tracking
"""

import pytest
import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import sys
sys.path.insert(0, '/home/sssmmmddd/Code/pro/crypto-scalper-bot')

from src.core.engine import TradingEngine, EngineState, create_engine
from src.core.events import Event, EventType
from src.data.models import Signal, SignalType, OrderBookSnapshot, OrderBookLevel


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def engine_config():
    """Default engine configuration."""
    return {
        "exchange": {
            "testnet": True,
            "api_key": "test_key",
            "api_secret": "test_secret",
        },
        "data": {
            "orderbook_depth": 20,
        },
        "trading": {
            "symbols": ["BTCUSDT", "ETHUSDT"],
        },
    }


@pytest.fixture
def engine(engine_config):
    """Create TradingEngine instance."""
    return TradingEngine(engine_config)


@pytest.fixture
def mock_strategy():
    """Create mock strategy."""
    strategy = MagicMock()
    strategy.__class__.__name__ = "MockStrategy"
    strategy.on_orderbook = AsyncMock(return_value=None)
    return strategy


@pytest.fixture
def mock_risk_manager():
    """Create mock risk manager."""
    rm = MagicMock()
    rm.__class__.__name__ = "MockRiskManager"
    rm.can_trade = MagicMock(return_value=(True, "OK"))
    rm.calculate_position_size = MagicMock(return_value=Decimal("0.01"))
    return rm


@pytest.fixture
def mock_executor():
    """Create mock executor."""
    executor = MagicMock()
    executor.__class__.__name__ = "MockExecutor"
    executor.execute_signal = AsyncMock(return_value=MagicMock(order_id="test_order_123"))
    return executor


@pytest.fixture
def sample_signal():
    """Create sample trading signal."""
    return Signal(
        strategy="test_strategy",
        signal_type=SignalType.LONG,
        symbol="BTCUSDT",
        timestamp=datetime.utcnow(),
        strength=0.85,
        price=Decimal("50000"),
        metadata={},
    )


@pytest.fixture
def sample_orderbook_snapshot():
    """Create sample orderbook snapshot."""
    from src.data.models import OrderBookLevel
    return OrderBookSnapshot(
        symbol="BTCUSDT",
        timestamp=datetime.utcnow(),
        bids=[OrderBookLevel(Decimal("49999"), Decimal("1.5")), OrderBookLevel(Decimal("49998"), Decimal("2.0"))],
        asks=[OrderBookLevel(Decimal("50001"), Decimal("1.0")), OrderBookLevel(Decimal("50002"), Decimal("1.5"))],
        last_update_id=12345,
    )


# =============================================================================
# Engine State Tests
# =============================================================================

class TestEngineState:
    """Tests for engine state management."""

    def test_initial_state_stopped(self, engine):
        """Test engine starts in STOPPED state."""
        assert engine.state == EngineState.STOPPED

    def test_is_running_property(self, engine):
        """Test is_running property."""
        assert engine.is_running is False

        engine._state = EngineState.RUNNING
        assert engine.is_running is True

        engine._state = EngineState.STOPPING
        assert engine.is_running is False

    def test_uptime_none_when_not_started(self, engine):
        """Test uptime is None when not started."""
        assert engine.uptime is None

    def test_uptime_calculated_when_running(self, engine):
        """Test uptime is calculated when running."""
        engine._start_time = datetime.utcnow()

        uptime = engine.uptime
        assert uptime is not None
        assert uptime >= 0

    def test_stats_property(self, engine):
        """Test stats property returns correct data."""
        engine._signals_generated = 10
        engine._orders_placed = 5
        engine._errors = 2

        stats = engine.stats

        assert stats["state"] == "STOPPED"
        assert stats["signals_generated"] == 10
        assert stats["orders_placed"] == 5
        assert stats["errors"] == 2


# =============================================================================
# Component Registration Tests
# =============================================================================

class TestComponentRegistration:
    """Tests for component registration."""

    def test_add_strategy(self, engine, mock_strategy):
        """Test adding a strategy."""
        result = engine.add_strategy(mock_strategy)

        assert result is engine  # Chaining
        assert mock_strategy in engine.strategies

    def test_add_multiple_strategies(self, engine, mock_strategy):
        """Test adding multiple strategies."""
        strategy2 = MagicMock()
        strategy2.__class__.__name__ = "MockStrategy2"

        engine.add_strategy(mock_strategy)
        engine.add_strategy(strategy2)

        assert len(engine.strategies) == 2

    def test_set_risk_manager(self, engine, mock_risk_manager):
        """Test setting risk manager."""
        result = engine.set_risk_manager(mock_risk_manager)

        assert result is engine  # Chaining
        assert engine.risk_manager is mock_risk_manager

    def test_set_executor(self, engine, mock_executor):
        """Test setting executor."""
        result = engine.set_executor(mock_executor)

        assert result is engine  # Chaining
        assert engine.executor is mock_executor

    def test_chaining_components(self, engine, mock_strategy, mock_risk_manager, mock_executor):
        """Test method chaining for component registration."""
        result = (
            engine
            .add_strategy(mock_strategy)
            .set_risk_manager(mock_risk_manager)
            .set_executor(mock_executor)
        )

        assert result is engine
        assert len(engine.strategies) == 1
        assert engine.risk_manager is mock_risk_manager
        assert engine.executor is mock_executor


# =============================================================================
# Lifecycle Tests
# =============================================================================

class TestEngineLifecycle:
    """Tests for engine lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_changes_state(self, engine_config):
        """Test that starting engine changes state."""
        engine = TradingEngine(engine_config)

        with patch.object(engine, '_init_components', new_callable=AsyncMock):
            with patch.object(engine, '_register_handlers'):
                with patch.object(engine, '_sync_orderbooks', new_callable=AsyncMock):
                    # Mock websocket
                    engine.websocket = MagicMock()
                    engine.websocket.start = AsyncMock()

                    # Start engine (will wait for websocket, so we need to handle that)
                    task = asyncio.create_task(engine.start())

                    # Wait a bit for state to change
                    await asyncio.sleep(0.1)

                    assert engine.state == EngineState.RUNNING
                    assert engine._start_time is not None

                    # Cancel the task
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    @pytest.mark.asyncio
    async def test_stop_changes_state(self, engine):
        """Test that stopping engine changes state."""
        engine._state = EngineState.RUNNING
        engine.websocket = MagicMock()
        engine.websocket.stop = AsyncMock()

        await engine.stop()

        assert engine.state == EngineState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_cancels_tasks(self, engine):
        """Test that stop cancels running tasks."""
        engine._state = EngineState.RUNNING

        # Create a dummy task
        async def dummy_task():
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                raise

        task = asyncio.create_task(dummy_task())
        engine._tasks.append(task)

        engine.websocket = MagicMock()
        engine.websocket.stop = AsyncMock()

        await engine.stop()

        # Wait a bit for cancellation to complete
        await asyncio.sleep(0.01)

        # Task should be done (either cancelled or finished)
        assert task.done()
        # And it should have been cancelled (not completed normally)
        assert task.cancelled() or task.exception() is not None or not task.result

    @pytest.mark.asyncio
    async def test_cannot_start_when_not_stopped(self, engine):
        """Test cannot start engine when not in STOPPED state."""
        engine._state = EngineState.RUNNING

        # Should return early without error
        await engine.start()

        # State should remain unchanged
        assert engine.state == EngineState.RUNNING

    @pytest.mark.asyncio
    async def test_stop_does_nothing_when_not_running(self, engine):
        """Test stop does nothing when engine not running."""
        engine._state = EngineState.STOPPED

        await engine.stop()

        assert engine.state == EngineState.STOPPED


# =============================================================================
# Signal Processing Tests
# =============================================================================

class TestSignalProcessing:
    """Tests for signal processing."""

    @pytest.mark.asyncio
    async def test_on_signal_without_risk_manager(self, engine, sample_signal):
        """Test signal processing without risk manager."""
        engine._state = EngineState.RUNNING
        engine.executor = AsyncMock()
        engine.executor.execute_signal = AsyncMock(return_value=MagicMock(order_id="123"))

        event = Event(
            event_type=EventType.SIGNAL,
            data=sample_signal,
            source="test",
        )

        await engine._on_signal(event)

        engine.executor.execute_signal.assert_called_once_with(sample_signal)
        assert engine._orders_placed == 1

    @pytest.mark.asyncio
    async def test_on_signal_with_risk_check_pass(self, engine, sample_signal, mock_risk_manager, mock_executor):
        """Test signal processing with passing risk check."""
        engine._state = EngineState.RUNNING
        engine.risk_manager = mock_risk_manager
        engine.executor = mock_executor

        event = Event(
            event_type=EventType.SIGNAL,
            data=sample_signal,
            source="test",
        )

        await engine._on_signal(event)

        mock_risk_manager.can_trade.assert_called_once_with(sample_signal)
        mock_risk_manager.calculate_position_size.assert_called_once()
        mock_executor.execute_signal.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_signal_with_risk_check_fail(self, engine, sample_signal, mock_risk_manager, mock_executor):
        """Test signal processing with failing risk check."""
        engine._state = EngineState.RUNNING
        engine.risk_manager = mock_risk_manager
        engine.executor = mock_executor

        mock_risk_manager.can_trade.return_value = (False, "Max positions reached")

        event = Event(
            event_type=EventType.SIGNAL,
            data=sample_signal,
            source="test",
        )

        await engine._on_signal(event)

        mock_risk_manager.can_trade.assert_called_once()
        mock_executor.execute_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_signal_zero_position_size(self, engine, sample_signal, mock_risk_manager, mock_executor):
        """Test signal processing with zero position size."""
        engine._state = EngineState.RUNNING
        engine.risk_manager = mock_risk_manager
        engine.executor = mock_executor

        mock_risk_manager.calculate_position_size.return_value = Decimal("0")

        event = Event(
            event_type=EventType.SIGNAL,
            data=sample_signal,
            source="test",
        )

        await engine._on_signal(event)

        mock_executor.execute_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_signal_execution_error(self, engine, sample_signal, mock_executor):
        """Test signal processing with execution error."""
        engine._state = EngineState.RUNNING
        engine.executor = mock_executor
        mock_executor.execute_signal.side_effect = Exception("Execution failed")

        event = Event(
            event_type=EventType.SIGNAL,
            data=sample_signal,
            source="test",
        )

        await engine._on_signal(event)

        assert engine._errors == 1
        assert engine._orders_placed == 0

    @pytest.mark.asyncio
    async def test_on_signal_not_running(self, engine, sample_signal):
        """Test signal is ignored when engine not running."""
        engine._state = EngineState.STOPPED
        engine.executor = AsyncMock()

        event = Event(
            event_type=EventType.SIGNAL,
            data=sample_signal,
            source="test",
        )

        await engine._on_signal(event)

        engine.executor.execute_signal.assert_not_called()


# =============================================================================
# Strategy Execution Tests
# =============================================================================

class TestStrategyExecution:
    """Tests for strategy execution."""

    @pytest.mark.asyncio
    async def test_run_async_strategy(self, engine, sample_orderbook_snapshot):
        """Test running async strategy."""
        async_strategy = MagicMock()
        async_strategy.on_orderbook = AsyncMock(return_value=None)

        result = await engine._run_strategy(async_strategy, sample_orderbook_snapshot)

        async_strategy.on_orderbook.assert_called_once_with(sample_orderbook_snapshot)
        assert result is None

    @pytest.mark.asyncio
    async def test_run_sync_strategy(self, engine, sample_orderbook_snapshot):
        """Test running sync strategy."""
        sync_strategy = MagicMock()
        sync_strategy.on_orderbook = MagicMock(return_value=None)

        result = await engine._run_strategy(sync_strategy, sample_orderbook_snapshot)

        sync_strategy.on_orderbook.assert_called_once_with(sample_orderbook_snapshot)
        assert result is None

    @pytest.mark.asyncio
    async def test_run_strategy_returns_signal(self, engine, sample_orderbook_snapshot, sample_signal):
        """Test strategy returning signal."""
        strategy = MagicMock()
        strategy.on_orderbook = MagicMock(return_value=sample_signal)

        result = await engine._run_strategy(strategy, sample_orderbook_snapshot)

        assert result is sample_signal

    @pytest.mark.asyncio
    async def test_on_orderbook_update_runs_strategies(self, engine, sample_orderbook_snapshot, mock_strategy):
        """Test orderbook update runs all strategies."""
        engine._state = EngineState.RUNNING
        engine.strategies = [mock_strategy]

        from src.data.models import OrderBookEvent
        orderbook_event = OrderBookEvent(snapshot=sample_orderbook_snapshot)

        event = Event(
            event_type=EventType.ORDERBOOK_UPDATE,
            data=orderbook_event,
            source="test",
        )

        # Mock event bus publish
        engine.event_bus = MagicMock()
        engine.event_bus.publish = AsyncMock()

        await engine._on_orderbook_update(event)

        mock_strategy.on_orderbook.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_orderbook_update_publishes_signal(self, engine, sample_orderbook_snapshot, sample_signal):
        """Test signal is published when strategy generates one."""
        engine._state = EngineState.RUNNING

        strategy = MagicMock()
        strategy.__class__.__name__ = "TestStrategy"
        strategy.on_orderbook = MagicMock(return_value=sample_signal)
        engine.strategies = [strategy]

        from src.data.models import OrderBookEvent
        orderbook_event = OrderBookEvent(snapshot=sample_orderbook_snapshot)

        event = Event(
            event_type=EventType.ORDERBOOK_UPDATE,
            data=orderbook_event,
            source="test",
        )

        engine.event_bus = MagicMock()
        engine.event_bus.publish = AsyncMock()

        await engine._on_orderbook_update(event)

        assert engine._signals_generated == 1
        engine.event_bus.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_strategy_error_handling(self, engine, sample_orderbook_snapshot):
        """Test strategy errors are handled gracefully."""
        engine._state = EngineState.RUNNING

        failing_strategy = MagicMock()
        failing_strategy.__class__.__name__ = "FailingStrategy"
        failing_strategy.on_orderbook = MagicMock(side_effect=Exception("Strategy error"))
        engine.strategies = [failing_strategy]

        from src.data.models import OrderBookEvent
        orderbook_event = OrderBookEvent(snapshot=sample_orderbook_snapshot)

        event = Event(
            event_type=EventType.ORDERBOOK_UPDATE,
            data=orderbook_event,
            source="test",
        )

        # Should not raise
        await engine._on_orderbook_update(event)

        assert engine._errors == 1


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateEngine:
    """Tests for create_engine factory function."""

    def test_create_engine_returns_instance(self, engine_config):
        """Test factory function creates engine."""
        engine = create_engine(engine_config)

        assert isinstance(engine, TradingEngine)
        assert engine.config == engine_config

    def test_create_engine_empty_config(self):
        """Test factory with empty config."""
        engine = create_engine({})

        assert isinstance(engine, TradingEngine)
        assert engine.strategies == []


# =============================================================================
# Integration Tests
# =============================================================================

class TestEngineIntegration:
    """Integration tests for TradingEngine."""

    @pytest.mark.asyncio
    async def test_full_signal_flow(self, engine, sample_signal, mock_strategy, mock_risk_manager, mock_executor):
        """Test full signal flow from strategy to execution."""
        engine._state = EngineState.RUNNING
        mock_strategy.on_orderbook = MagicMock(return_value=sample_signal)
        engine.strategies = [mock_strategy]
        engine.risk_manager = mock_risk_manager
        engine.executor = mock_executor

        engine.event_bus = MagicMock()

        # Simulate orderbook event handling
        from src.data.models import OrderBookEvent, OrderBookSnapshot

        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            bids=[OrderBookLevel(price=Decimal("50000"), quantity=Decimal("1"))],
            asks=[OrderBookLevel(price=Decimal("50001"), quantity=Decimal("1"))],
            last_update_id=1,
        )

        orderbook_event = OrderBookEvent(snapshot=snapshot)

        # Capture published signals
        published_events = []

        async def capture_publish(event):
            published_events.append(event)

        engine.event_bus.publish = capture_publish

        # Process orderbook update
        event = Event(
            event_type=EventType.ORDERBOOK_UPDATE,
            data=orderbook_event,
            source="test",
        )

        await engine._on_orderbook_update(event)

        # Verify signal was generated and published
        assert engine._signals_generated == 1
        assert len(published_events) == 1

        # Process the signal
        signal_event = published_events[0]
        await engine._on_signal(signal_event)

        # Verify execution
        mock_executor.execute_signal.assert_called_once()
        assert engine._orders_placed == 1

    def test_stats_tracking(self, engine):
        """Test statistics are tracked correctly."""
        engine._signals_generated = 100
        engine._orders_placed = 50
        engine._errors = 5

        stats = engine.stats

        assert stats["signals_generated"] == 100
        assert stats["orders_placed"] == 50
        assert stats["errors"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
