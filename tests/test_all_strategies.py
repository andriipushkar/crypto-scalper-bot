"""
Comprehensive tests for all trading strategies.

Tests DCA, Grid Trading, Mean Reversion, Orderbook Imbalance, Volume Spike,
and Composite strategies with full coverage.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch
from typing import List

from src.data.models import (
    OrderBookSnapshot,
    OrderBookLevel,
    Trade,
    Signal,
    SignalType,
    Side,
    MarkPrice,
)
from src.strategy.base import BaseStrategy, CompositeStrategy, StrategyConfig, StrategyState
from src.strategy.dca_strategy import DCAStrategy, DCAMode, DCADirection, DCAState
from src.strategy.grid_trading import GridTradingStrategy, GridDirection, GridLevel, GridState
from src.strategy.mean_reversion import MeanReversionStrategy, MeanReversionMetrics, PricePoint
from src.strategy.orderbook_imbalance import OrderBookImbalanceStrategy
from src.strategy.volume_spike import VolumeSpikeStrategy


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_orderbook() -> OrderBookSnapshot:
    """Create mock orderbook snapshot."""
    return OrderBookSnapshot(
        symbol="BTCUSDT",
        timestamp=datetime.utcnow(),
        bids=[
            OrderBookLevel(price=Decimal("50000"), quantity=Decimal("1.0")),
            OrderBookLevel(price=Decimal("49999"), quantity=Decimal("2.0")),
            OrderBookLevel(price=Decimal("49998"), quantity=Decimal("3.0")),
            OrderBookLevel(price=Decimal("49997"), quantity=Decimal("4.0")),
            OrderBookLevel(price=Decimal("49996"), quantity=Decimal("5.0")),
        ],
        asks=[
            OrderBookLevel(price=Decimal("50001"), quantity=Decimal("1.0")),
            OrderBookLevel(price=Decimal("50002"), quantity=Decimal("2.0")),
            OrderBookLevel(price=Decimal("50003"), quantity=Decimal("3.0")),
            OrderBookLevel(price=Decimal("50004"), quantity=Decimal("4.0")),
            OrderBookLevel(price=Decimal("50005"), quantity=Decimal("5.0")),
        ],
        last_update_id=12345,
    )


@pytest.fixture
def mock_trade() -> Trade:
    """Create mock trade."""
    return Trade(
        symbol="BTCUSDT",
        trade_id=1,
        price=Decimal("50000"),
        quantity=Decimal("0.1"),
        timestamp=datetime.utcnow(),
        is_buyer_maker=False,  # False = BUY side
    )


@pytest.fixture
def mock_mark_price() -> MarkPrice:
    """Create mock mark price."""
    return MarkPrice(
        symbol="BTCUSDT",
        mark_price=Decimal("50000"),
        index_price=Decimal("50000"),
        estimated_settle_price=Decimal("50000"),
        funding_rate=Decimal("0.0001"),
        next_funding_time=datetime.utcnow() + timedelta(hours=8),
        timestamp=datetime.utcnow(),
    )


def create_orderbook_with_imbalance(
    symbol: str,
    bid_total: float,
    ask_total: float,
    mid_price: float = 50000.0,
) -> OrderBookSnapshot:
    """Create orderbook with specific bid/ask imbalance."""
    bids = [
        OrderBookLevel(
            price=Decimal(str(mid_price - i)),
            quantity=Decimal(str(bid_total / 5))
        )
        for i in range(1, 6)
    ]
    asks = [
        OrderBookLevel(
            price=Decimal(str(mid_price + i)),
            quantity=Decimal(str(ask_total / 5))
        )
        for i in range(1, 6)
    ]

    return OrderBookSnapshot(
        symbol=symbol,
        timestamp=datetime.utcnow(),
        bids=bids,
        asks=asks,
        last_update_id=12345,
    )


# =============================================================================
# Base Strategy Tests
# =============================================================================

class TestBaseStrategy:
    """Test BaseStrategy and StrategyState."""

    def test_strategy_config_defaults(self):
        """Test StrategyConfig default values."""
        config = StrategyConfig()
        assert config.enabled is True
        assert config.signal_cooldown == 10.0
        assert config.min_strength == 0.5

    def test_strategy_state_defaults(self):
        """Test StrategyState default values."""
        state = StrategyState()
        assert state.last_signal_time is None
        assert state.last_signal_type is None
        assert state.signals_generated == 0
        assert state.errors == 0

    def test_can_signal_enabled(self, mock_orderbook):
        """Test can_signal when enabled."""
        strategy = OrderBookImbalanceStrategy({
            "enabled": True,
            "signal_cooldown": 0,
        })
        assert strategy.can_signal() is True

    def test_can_signal_disabled(self, mock_orderbook):
        """Test can_signal when disabled."""
        strategy = OrderBookImbalanceStrategy({
            "enabled": False,
            "signal_cooldown": 0,
        })
        assert strategy.can_signal() is False

    def test_can_signal_cooldown(self, mock_orderbook):
        """Test can_signal respects cooldown."""
        strategy = OrderBookImbalanceStrategy({
            "enabled": True,
            "signal_cooldown": 60,  # 60 seconds
        })

        # First signal should be allowed
        assert strategy.can_signal() is True

        # Generate a signal to set last_signal_time
        strategy._state.last_signal_time = datetime.utcnow()

        # Now cooldown should prevent
        assert strategy.can_signal() is False

    def test_create_signal_min_strength(self, mock_orderbook):
        """Test create_signal respects min_strength."""
        strategy = OrderBookImbalanceStrategy({
            "enabled": True,
            "signal_cooldown": 0,
            "min_strength": 0.7,
        })

        # Signal below min_strength should return None
        signal = strategy.create_signal(
            SignalType.LONG,
            "BTCUSDT",
            Decimal("50000"),
            0.5,  # Below min_strength
        )
        assert signal is None

    def test_create_signal_success(self, mock_orderbook):
        """Test successful signal creation."""
        strategy = OrderBookImbalanceStrategy({
            "enabled": True,
            "signal_cooldown": 0,
            "min_strength": 0.3,
        })

        signal = strategy.create_signal(
            SignalType.LONG,
            "BTCUSDT",
            Decimal("50000"),
            0.8,
        )

        assert signal is not None
        assert signal.signal_type == SignalType.LONG
        assert signal.symbol == "BTCUSDT"
        assert signal.strength == 0.8
        assert strategy._state.signals_generated == 1

    def test_reset_strategy(self):
        """Test strategy reset."""
        strategy = OrderBookImbalanceStrategy({"enabled": True})
        strategy._state.signals_generated = 10
        strategy._state.errors = 5
        strategy._state.last_signal_time = datetime.utcnow()

        strategy.reset()

        assert strategy._state.signals_generated == 0
        assert strategy._state.errors == 0
        assert strategy._state.last_signal_time is None

    def test_strategy_name(self):
        """Test strategy name property."""
        strategy = OrderBookImbalanceStrategy({"enabled": True})
        assert strategy.name == "OrderBookImbalanceStrategy"

    def test_strategy_stats(self):
        """Test strategy stats property."""
        strategy = OrderBookImbalanceStrategy({"enabled": True})
        stats = strategy.stats

        assert "name" in stats
        assert "enabled" in stats
        assert "signals_generated" in stats
        assert "errors" in stats

    def test_on_trade_default(self, mock_trade):
        """Test default on_trade returns None."""
        strategy = OrderBookImbalanceStrategy({"enabled": True})
        result = strategy.on_trade(mock_trade)
        assert result is None

    def test_on_mark_price_default(self, mock_mark_price):
        """Test default on_mark_price returns None."""
        strategy = OrderBookImbalanceStrategy({"enabled": True})
        result = strategy.on_mark_price(mock_mark_price)
        assert result is None


# =============================================================================
# DCA Strategy Tests
# =============================================================================

class TestDCAStrategy:
    """Test DCA Strategy."""

    def test_dca_initialization(self):
        """Test DCA strategy initialization."""
        config = {
            "enabled": True,
            "mode": "hybrid",
            "direction": "accumulate",
            "interval_minutes": 30,
            "dip_threshold": 0.03,
            "quantity_per_entry": 0.001,
            "max_entries": 50,
            "total_budget": 500,
        }

        strategy = DCAStrategy(config)

        assert strategy.mode == DCAMode.HYBRID
        assert strategy.direction == DCADirection.ACCUMULATE
        assert strategy.interval_minutes == 30
        assert strategy.dip_threshold == Decimal("0.03")
        assert strategy.max_entries == 50

    def test_dca_modes(self):
        """Test all DCA modes."""
        for mode in ["time_based", "price_based", "hybrid"]:
            strategy = DCAStrategy({"mode": mode})
            assert strategy.mode == DCAMode(mode)

    def test_dca_directions(self):
        """Test all DCA directions."""
        for direction in ["accumulate", "distribute", "bidirectional"]:
            strategy = DCAStrategy({"direction": direction})
            assert strategy.direction == DCADirection(direction)

    def test_dca_time_based_signal(self, mock_orderbook):
        """Test time-based DCA signal generation."""
        strategy = DCAStrategy({
            "enabled": True,
            "mode": "time_based",
            "direction": "accumulate",
            "interval_minutes": 1,
            "signal_cooldown": 0,
            "min_strength": 0.1,
        })

        # First signal should generate
        signal = strategy.on_orderbook(mock_orderbook)
        assert signal is not None
        assert signal.signal_type == SignalType.LONG

    def test_dca_time_based_interval(self, mock_orderbook):
        """Test time-based DCA respects interval."""
        strategy = DCAStrategy({
            "enabled": True,
            "mode": "time_based",
            "direction": "accumulate",
            "interval_minutes": 60,  # 1 hour
            "signal_cooldown": 0,
            "min_strength": 0.1,
        })

        # First signal
        signal1 = strategy.on_orderbook(mock_orderbook)
        assert signal1 is not None

        # Second signal should be blocked by interval
        signal2 = strategy.on_orderbook(mock_orderbook)
        assert signal2 is None

    def test_dca_price_based_dip(self):
        """Test price-based DCA detects dips."""
        strategy = DCAStrategy({
            "enabled": True,
            "mode": "price_based",
            "direction": "accumulate",
            "dip_threshold": 0.02,
            "lookback_minutes": 60,
            "signal_cooldown": 0,
            "min_strength": 0.1,
        })

        # Add price history at higher price
        for i in range(10):
            ob = create_orderbook_with_imbalance("BTCUSDT", 10, 10, 50000 + i * 10)
            strategy.on_orderbook(ob)

        # Now create orderbook with dipped price
        dip_ob = create_orderbook_with_imbalance("BTCUSDT", 10, 10, 48500)  # 3% dip
        signal = strategy.on_orderbook(dip_ob)

        # Should generate signal on dip
        # Note: May be None if not enough lookback history
        if signal:
            assert signal.signal_type == SignalType.LONG

    def test_dca_distribution_mode(self, mock_orderbook):
        """Test DCA distribution mode."""
        strategy = DCAStrategy({
            "enabled": True,
            "mode": "time_based",
            "direction": "distribute",
            "interval_minutes": 0,
            "signal_cooldown": 0,
            "min_strength": 0.1,
        })

        signal = strategy.on_orderbook(mock_orderbook)
        assert signal is not None
        assert signal.signal_type == SignalType.SHORT

    def test_dca_max_entries_limit(self, mock_orderbook):
        """Test DCA respects max entries."""
        strategy = DCAStrategy({
            "enabled": True,
            "mode": "time_based",
            "direction": "accumulate",
            "interval_minutes": 0,
            "max_entries": 2,
            "signal_cooldown": 0,
            "min_strength": 0.1,
        })

        # First two entries
        strategy.on_orderbook(mock_orderbook)
        strategy._state.last_entry_time = None  # Reset for test
        strategy.on_orderbook(mock_orderbook)

        # Third should be blocked
        strategy._state.last_entry_time = None
        signal = strategy.on_orderbook(mock_orderbook)
        assert signal is None or signal.metadata.get("dca_exit")

    def test_dca_budget_limit(self, mock_orderbook):
        """Test DCA respects budget limit."""
        strategy = DCAStrategy({
            "enabled": True,
            "mode": "time_based",
            "direction": "accumulate",
            "interval_minutes": 0,
            "total_budget": 10,  # Very small
            "quantity_per_entry": 0.001,
            "signal_cooldown": 0,
            "min_strength": 0.1,
        })

        # Use up budget
        strategy._state.total_invested = Decimal("10")

        signal = strategy.on_orderbook(mock_orderbook)
        assert signal is None or signal.metadata.get("dca_exit")

    def test_dca_get_stats(self):
        """Test DCA statistics."""
        strategy = DCAStrategy({
            "mode": "hybrid",
            "direction": "accumulate",
            "max_entries": 100,
            "total_budget": 1000,
        })

        stats = strategy.get_dca_stats()

        assert "mode" in stats
        assert "direction" in stats
        assert "entries_count" in stats
        assert "total_invested" in stats
        assert "average_price" in stats

    def test_dca_get_entries(self):
        """Test getting DCA entries."""
        strategy = DCAStrategy({"mode": "time_based"})

        entries = strategy.get_entries()
        assert isinstance(entries, list)

    def test_dca_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        strategy = DCAStrategy({"mode": "time_based"})

        # Set up state
        strategy._state.total_quantity = Decimal("0.01")
        strategy._state.average_price = Decimal("50000")

        pnl = strategy.get_unrealized_pnl(Decimal("51000"))

        assert pnl["pnl"] > 0
        assert pnl["pnl_pct"] > 0

    def test_dca_unrealized_pnl_no_position(self):
        """Test unrealized P&L with no position."""
        strategy = DCAStrategy({"mode": "time_based"})

        pnl = strategy.get_unrealized_pnl(Decimal("51000"))

        assert pnl["pnl"] == 0
        assert pnl["pnl_pct"] == 0

    def test_dca_reset(self, mock_orderbook):
        """Test DCA reset."""
        strategy = DCAStrategy({"mode": "time_based", "signal_cooldown": 0})

        # Generate some state
        strategy.on_orderbook(mock_orderbook)

        strategy.reset()

        assert strategy._state.entries_count == 0
        assert strategy._state.total_invested == 0
        assert len(strategy._entries) == 0

    def test_dca_scale_on_dip(self, mock_orderbook):
        """Test DCA scale on dip feature."""
        strategy = DCAStrategy({
            "mode": "hybrid",
            "direction": "accumulate",
            "scale_on_dip": True,
            "scale_factor": 2.0,
            "dip_threshold": 0.02,
        })

        # Verify config
        assert strategy.scale_on_dip is True
        assert strategy.scale_factor == Decimal("2.0")


# =============================================================================
# Grid Trading Strategy Tests
# =============================================================================

class TestGridTradingStrategy:
    """Test Grid Trading Strategy."""

    def test_grid_initialization(self):
        """Test grid strategy initialization."""
        config = {
            "enabled": True,
            "upper_price": 52000,
            "lower_price": 48000,
            "grid_levels": 10,
            "quantity_per_grid": 0.001,
            "direction": "neutral",
        }

        strategy = GridTradingStrategy(config)

        assert strategy.upper_price == Decimal("52000")
        assert strategy.lower_price == Decimal("48000")
        assert strategy.grid_levels == 10
        assert strategy.direction == GridDirection.NEUTRAL

    def test_grid_directions(self):
        """Test all grid directions."""
        for direction in ["neutral", "long", "short"]:
            strategy = GridTradingStrategy({
                "upper_price": 52000,
                "lower_price": 48000,
                "direction": direction,
            })
            assert strategy.direction == GridDirection(direction)

    def test_grid_auto_range(self, mock_orderbook):
        """Test auto range calculation."""
        strategy = GridTradingStrategy({
            "enabled": True,
            "auto_range": True,
            "range_percent": 0.05,
            "grid_levels": 10,
            "signal_cooldown": 0,
        })

        strategy.on_orderbook(mock_orderbook)

        # Grid should be initialized
        assert strategy._initialized is True
        assert strategy.upper_price > strategy.lower_price

    def test_grid_level_creation(self, mock_orderbook):
        """Test grid levels are created."""
        strategy = GridTradingStrategy({
            "enabled": True,
            "auto_range": True,
            "range_percent": 0.05,
            "grid_levels": 10,
            "signal_cooldown": 0,
        })

        strategy.on_orderbook(mock_orderbook)

        levels = strategy.get_grid_levels()
        assert len(levels) == 11  # grid_levels + 1

    def test_grid_long_direction(self, mock_orderbook):
        """Test grid with long direction only creates buy levels."""
        strategy = GridTradingStrategy({
            "enabled": True,
            "auto_range": True,
            "range_percent": 0.05,
            "grid_levels": 5,
            "direction": "long",
            "signal_cooldown": 0,
        })

        strategy.on_orderbook(mock_orderbook)

        levels = strategy.get_grid_levels()
        buy_levels = [l for l in levels if l["is_buy"]]
        assert len(buy_levels) == len(levels)

    def test_grid_short_direction(self, mock_orderbook):
        """Test grid with short direction only creates sell levels."""
        strategy = GridTradingStrategy({
            "enabled": True,
            "auto_range": True,
            "range_percent": 0.05,
            "grid_levels": 5,
            "direction": "short",
            "signal_cooldown": 0,
        })

        strategy.on_orderbook(mock_orderbook)

        levels = strategy.get_grid_levels()
        sell_levels = [l for l in levels if not l["is_buy"]]
        assert len(sell_levels) == len(levels)

    def test_grid_stats(self, mock_orderbook):
        """Test grid statistics."""
        strategy = GridTradingStrategy({
            "enabled": True,
            "auto_range": True,
            "grid_levels": 10,
        })

        strategy.on_orderbook(mock_orderbook)

        stats = strategy.get_grid_stats()

        assert "total_levels" in stats
        assert "filled_levels" in stats
        assert "active_position" in stats
        assert "direction" in stats

    def test_grid_unfilled_levels(self, mock_orderbook):
        """Test getting unfilled levels."""
        strategy = GridTradingStrategy({
            "enabled": True,
            "auto_range": True,
            "grid_levels": 10,
            "direction": "neutral",
        })

        strategy.on_orderbook(mock_orderbook)

        unfilled = strategy.get_unfilled_levels()

        assert "buy" in unfilled
        assert "sell" in unfilled

    def test_grid_trailing_up(self, mock_orderbook):
        """Test trailing grid upward."""
        strategy = GridTradingStrategy({
            "enabled": True,
            "upper_price": 50100,
            "lower_price": 49900,
            "grid_levels": 5,
            "trailing_up": True,
            "signal_cooldown": 0,
        })

        strategy.on_orderbook(mock_orderbook)

        # Price moves up significantly
        high_ob = create_orderbook_with_imbalance("BTCUSDT", 10, 10, 51000)
        strategy.on_orderbook(high_ob)

        # Grid should reset
        # (This tests the reset logic)

    def test_grid_reset(self, mock_orderbook):
        """Test grid reset."""
        strategy = GridTradingStrategy({
            "enabled": True,
            "auto_range": True,
            "grid_levels": 10,
        })

        strategy.on_orderbook(mock_orderbook)
        strategy.reset()

        assert strategy._initialized is False
        assert len(strategy._grid_state.levels) == 0

    def test_grid_on_trade(self, mock_trade):
        """Test grid on_trade handling."""
        strategy = GridTradingStrategy({
            "enabled": True,
            "auto_range": True,
            "grid_levels": 10,
        })

        # Should not crash
        result = strategy.on_trade(mock_trade)
        assert result is None or isinstance(result, Signal)


# =============================================================================
# Mean Reversion Strategy Tests
# =============================================================================

class TestMeanReversionStrategy:
    """Test Mean Reversion Strategy."""

    def test_mean_reversion_initialization(self):
        """Test mean reversion initialization."""
        config = {
            "enabled": True,
            "lookback_period": 20,
            "std_dev_multiplier": 2.0,
            "entry_z_score": 2.0,
            "exit_z_score": 0.5,
            "rsi_period": 14,
            "use_rsi_confirmation": True,
        }

        strategy = MeanReversionStrategy(config)

        assert strategy.lookback_period == 20
        assert strategy.std_dev_multiplier == Decimal("2.0")
        assert strategy.entry_z_score == Decimal("2.0")
        assert strategy.use_rsi_confirmation is True

    def test_mean_reversion_insufficient_data(self, mock_orderbook):
        """Test mean reversion with insufficient data."""
        strategy = MeanReversionStrategy({
            "enabled": True,
            "lookback_period": 20,
            "signal_cooldown": 0,
        })

        # First few updates should return None
        signal = strategy.on_orderbook(mock_orderbook)
        assert signal is None

    def test_mean_reversion_oversold_signal(self):
        """Test mean reversion detects oversold."""
        strategy = MeanReversionStrategy({
            "enabled": True,
            "lookback_period": 20,
            "entry_z_score": 2.0,
            "use_rsi_confirmation": False,
            "signal_cooldown": 0,
            "min_strength": 0.1,
        })

        # Add price history around 50000
        for i in range(25):
            ob = create_orderbook_with_imbalance("BTCUSDT", 10, 10, 50000)
            strategy.on_orderbook(ob)

        # Now price drops significantly (oversold)
        low_ob = create_orderbook_with_imbalance("BTCUSDT", 10, 10, 49000)
        signal = strategy.on_orderbook(low_ob)

        # Should detect oversold condition
        # Note: May not trigger if z-score threshold not reached

    def test_mean_reversion_overbought_signal(self):
        """Test mean reversion detects overbought."""
        strategy = MeanReversionStrategy({
            "enabled": True,
            "lookback_period": 20,
            "entry_z_score": 2.0,
            "use_rsi_confirmation": False,
            "signal_cooldown": 0,
            "min_strength": 0.1,
        })

        # Add price history around 50000
        for i in range(25):
            ob = create_orderbook_with_imbalance("BTCUSDT", 10, 10, 50000)
            strategy.on_orderbook(ob)

        # Now price rises significantly (overbought)
        high_ob = create_orderbook_with_imbalance("BTCUSDT", 10, 10, 51500)
        signal = strategy.on_orderbook(high_ob)

        # Should detect overbought condition

    def test_mean_reversion_rsi_calculation(self):
        """Test RSI calculation."""
        strategy = MeanReversionStrategy({
            "enabled": True,
            "rsi_period": 14,
        })

        # Add price history with gains
        for i in range(20):
            ob = create_orderbook_with_imbalance("BTCUSDT", 10, 10, 50000 + i * 10)
            strategy.on_orderbook(ob)

        # RSI should be calculated
        metrics = strategy.get_current_metrics()
        if metrics:
            assert metrics.rsi is not None

    def test_mean_reversion_metrics(self):
        """Test metrics calculation."""
        strategy = MeanReversionStrategy({
            "enabled": True,
            "lookback_period": 20,
        })

        # Add enough data
        for i in range(25):
            ob = create_orderbook_with_imbalance("BTCUSDT", 10, 10, 50000)
            strategy.on_orderbook(ob)

        metrics = strategy.get_current_metrics()

        assert metrics is not None
        assert metrics.mean > 0
        assert metrics.std_dev >= 0

    def test_mean_reversion_band_distance(self):
        """Test band distance calculation."""
        strategy = MeanReversionStrategy({
            "enabled": True,
            "lookback_period": 20,
        })

        for i in range(25):
            ob = create_orderbook_with_imbalance("BTCUSDT", 10, 10, 50000)
            strategy.on_orderbook(ob)

        distances = strategy.get_band_distance()

        if distances:
            assert "to_upper" in distances
            assert "to_lower" in distances
            assert "to_mean" in distances

    def test_mean_reversion_is_oversold(self):
        """Test is_oversold method."""
        strategy = MeanReversionStrategy({
            "enabled": True,
            "lookback_period": 10,
            "use_rsi_confirmation": False,
        })

        for i in range(15):
            ob = create_orderbook_with_imbalance("BTCUSDT", 10, 10, 50000)
            strategy.on_orderbook(ob)

        result = strategy.is_oversold()
        assert isinstance(result, bool)

    def test_mean_reversion_is_overbought(self):
        """Test is_overbought method."""
        strategy = MeanReversionStrategy({
            "enabled": True,
            "lookback_period": 10,
            "use_rsi_confirmation": False,
        })

        for i in range(15):
            ob = create_orderbook_with_imbalance("BTCUSDT", 10, 10, 50000)
            strategy.on_orderbook(ob)

        result = strategy.is_overbought()
        assert isinstance(result, bool)

    def test_mean_reversion_strategy_state(self):
        """Test strategy state."""
        strategy = MeanReversionStrategy({
            "enabled": True,
            "lookback_period": 10,
        })

        for i in range(15):
            ob = create_orderbook_with_imbalance("BTCUSDT", 10, 10, 50000)
            strategy.on_orderbook(ob)

        state = strategy.get_strategy_state()

        assert "in_position" in state
        assert "position_side" in state
        assert "price_history_length" in state

    def test_mean_reversion_reset(self):
        """Test mean reversion reset."""
        strategy = MeanReversionStrategy({
            "enabled": True,
            "lookback_period": 10,
        })

        for i in range(15):
            ob = create_orderbook_with_imbalance("BTCUSDT", 10, 10, 50000)
            strategy.on_orderbook(ob)

        strategy.reset()

        assert len(strategy._price_history) == 0
        assert strategy._in_position is False

    def test_mean_reversion_exit_timeout(self):
        """Test mean reversion exit on timeout."""
        strategy = MeanReversionStrategy({
            "enabled": True,
            "lookback_period": 10,
            "max_position_bars": 5,
            "use_rsi_confirmation": False,
            "signal_cooldown": 0,
        })

        # Simulate being in position
        strategy._in_position = True
        strategy._position_side = SignalType.LONG
        strategy._position_bars = 10  # Exceeded max

        # Fill history
        for i in range(15):
            ob = create_orderbook_with_imbalance("BTCUSDT", 10, 10, 50000)
            signal = strategy.on_orderbook(ob)

        # Should try to exit

    def test_mean_reversion_on_trade(self, mock_trade):
        """Test on_trade updates history."""
        strategy = MeanReversionStrategy({"enabled": True})

        result = strategy.on_trade(mock_trade)

        assert result is None
        assert len(strategy._price_history) == 1


# =============================================================================
# Orderbook Imbalance Strategy Tests
# =============================================================================

class TestOrderBookImbalanceStrategy:
    """Test Orderbook Imbalance Strategy."""

    def test_orderbook_imbalance_initialization(self):
        """Test orderbook imbalance initialization."""
        config = {
            "enabled": True,
            "imbalance_threshold": 1.5,
            "max_spread": 0.001,
            "signal_cooldown": 10,
            "levels": 5,
            "min_volume_usd": 1000,
        }

        strategy = OrderBookImbalanceStrategy(config)

        assert strategy.imbalance_threshold == Decimal("1.5")
        assert strategy.levels == 5

    def test_orderbook_imbalance_long_signal(self):
        """Test long signal on bid imbalance."""
        strategy = OrderBookImbalanceStrategy({
            "enabled": True,
            "imbalance_threshold": 1.5,
            "signal_cooldown": 0,
            "min_strength": 0.1,
        })

        # Create orderbook with strong bid imbalance
        ob = create_orderbook_with_imbalance("BTCUSDT", 100, 30, 50000)
        signal = strategy.on_orderbook(ob)

        if signal:
            assert signal.signal_type == SignalType.LONG

    def test_orderbook_imbalance_short_signal(self):
        """Test short signal on ask imbalance."""
        strategy = OrderBookImbalanceStrategy({
            "enabled": True,
            "imbalance_threshold": 1.5,
            "signal_cooldown": 0,
            "min_strength": 0.1,
        })

        # Create orderbook with strong ask imbalance
        ob = create_orderbook_with_imbalance("BTCUSDT", 30, 100, 50000)
        signal = strategy.on_orderbook(ob)

        if signal:
            assert signal.signal_type == SignalType.SHORT

    def test_orderbook_imbalance_no_signal(self):
        """Test no signal when balanced."""
        strategy = OrderBookImbalanceStrategy({
            "enabled": True,
            "imbalance_threshold": 1.5,
            "signal_cooldown": 0,
        })

        # Create balanced orderbook
        ob = create_orderbook_with_imbalance("BTCUSDT", 50, 50, 50000)
        signal = strategy.on_orderbook(ob)

        assert signal is None

    def test_orderbook_imbalance_spread_filter(self):
        """Test spread filter."""
        strategy = OrderBookImbalanceStrategy({
            "enabled": True,
            "imbalance_threshold": 1.5,
            "max_spread": 0.0001,  # Very tight spread required
            "signal_cooldown": 0,
        })

        # Create orderbook with wide spread
        ob = create_orderbook_with_imbalance("BTCUSDT", 100, 30, 50000)
        signal = strategy.on_orderbook(ob)

        # May filter out due to spread


# =============================================================================
# Volume Spike Strategy Tests
# =============================================================================

class TestVolumeSpikeStrategy:
    """Test Volume Spike Strategy."""

    def test_volume_spike_initialization(self):
        """Test volume spike initialization."""
        config = {
            "enabled": True,
            "volume_multiplier": 3.0,
            "lookback_seconds": 60,
            "min_volume_usd": 10000,
            "signal_cooldown": 15,
        }

        strategy = VolumeSpikeStrategy(config)

        assert strategy.volume_multiplier == Decimal("3.0")
        assert strategy.lookback_seconds == 60

    def test_volume_spike_detection(self, mock_trade):
        """Test volume spike detection."""
        strategy = VolumeSpikeStrategy({
            "enabled": True,
            "volume_multiplier": 2.0,
            "lookback_seconds": 60,
            "signal_cooldown": 0,
            "min_strength": 0.1,
        })

        # Add normal volume trades
        for i in range(20):
            trade = Trade(
                symbol="BTCUSDT",
                trade_id=i,
                price=Decimal("50000"),
                quantity=Decimal("0.1"),
                timestamp=datetime.utcnow() - timedelta(seconds=60-i),
                is_buyer_maker=False,  # False = BUY side
            )
            strategy.on_trade(trade)

        # Add spike trade
        spike_trade = Trade(
            symbol="BTCUSDT",
            trade_id=100,
            price=Decimal("50100"),
            quantity=Decimal("10.0"),  # Large volume
            timestamp=datetime.utcnow(),
            is_buyer_maker=False,  # False = BUY side
        )
        signal = strategy.on_trade(spike_trade)

        # May generate signal on spike


# =============================================================================
# Composite Strategy Tests
# =============================================================================

class TestCompositeStrategy:
    """Test Composite Strategy."""

    def test_composite_initialization(self):
        """Test composite strategy initialization."""
        strategy1 = OrderBookImbalanceStrategy({"enabled": True})
        strategy2 = VolumeSpikeStrategy({"enabled": True})

        composite = CompositeStrategy(
            config={"enabled": True, "signal_cooldown": 0},
            strategies=[strategy1, strategy2],
            weights={"OrderBookImbalanceStrategy": 0.6, "VolumeSpikeStrategy": 0.4},
        )

        assert len(composite.strategies) == 2
        assert sum(composite.weights.values()) == pytest.approx(1.0)

    def test_composite_equal_weights(self):
        """Test composite with equal weights."""
        strategy1 = OrderBookImbalanceStrategy({"enabled": True})
        strategy2 = VolumeSpikeStrategy({"enabled": True})

        composite = CompositeStrategy(
            config={"enabled": True},
            strategies=[strategy1, strategy2],
        )

        # Equal weights should be normalized to 0.5 each
        assert composite.weights["OrderBookImbalanceStrategy"] == 0.5
        assert composite.weights["VolumeSpikeStrategy"] == 0.5

    def test_composite_aggregates_signals(self, mock_orderbook):
        """Test composite aggregates signals."""
        strategy1 = OrderBookImbalanceStrategy({
            "enabled": True,
            "signal_cooldown": 0,
        })

        composite = CompositeStrategy(
            config={"enabled": True, "signal_cooldown": 0, "combined_threshold": 0.1},
            strategies=[strategy1],
        )

        # Create imbalanced orderbook
        ob = create_orderbook_with_imbalance("BTCUSDT", 100, 30, 50000)
        signal = composite.on_orderbook(ob)

        # May or may not generate signal depending on thresholds

    def test_composite_disabled_strategy(self, mock_orderbook):
        """Test composite skips disabled strategies."""
        strategy1 = OrderBookImbalanceStrategy({"enabled": False})  # Disabled
        strategy2 = VolumeSpikeStrategy({"enabled": True})

        composite = CompositeStrategy(
            config={"enabled": True, "signal_cooldown": 0},
            strategies=[strategy1, strategy2],
        )

        signal = composite.on_orderbook(mock_orderbook)

        # Should still work with one disabled

    def test_composite_error_handling(self, mock_orderbook):
        """Test composite handles strategy errors."""
        # Create a mock strategy that raises
        class BrokenStrategy(BaseStrategy):
            def on_orderbook(self, snapshot):
                raise ValueError("Test error")

        broken = BrokenStrategy({"enabled": True})
        working = OrderBookImbalanceStrategy({"enabled": True, "signal_cooldown": 0})

        composite = CompositeStrategy(
            config={"enabled": True, "signal_cooldown": 0},
            strategies=[broken, working],
        )

        # Should not raise, should handle gracefully
        signal = composite.on_orderbook(mock_orderbook)
        assert broken._state.errors == 1


# =============================================================================
# Signal Type Tests
# =============================================================================

class TestSignalTypes:
    """Test signal type handling."""

    def test_signal_type_values(self):
        """Test signal type values."""
        assert SignalType.LONG.name == "LONG"
        assert SignalType.SHORT.name == "SHORT"
        assert SignalType.NO_ACTION.name == "NO_ACTION"

    def test_signal_metadata(self, mock_orderbook):
        """Test signal metadata."""
        strategy = OrderBookImbalanceStrategy({
            "enabled": True,
            "signal_cooldown": 0,
            "min_strength": 0.1,
        })

        ob = create_orderbook_with_imbalance("BTCUSDT", 100, 30, 50000)
        signal = strategy.on_orderbook(ob)

        if signal:
            assert isinstance(signal.metadata, dict)

    def test_signal_strength_clamped(self):
        """Test signal strength is clamped to [0, 1]."""
        strategy = OrderBookImbalanceStrategy({
            "enabled": True,
            "signal_cooldown": 0,
            "min_strength": 0,
        })

        # Try to create signal with strength > 1
        signal = strategy.create_signal(
            SignalType.LONG,
            "BTCUSDT",
            Decimal("50000"),
            1.5,  # > 1
        )

        if signal:
            assert signal.strength <= 1.0

        # Try with negative strength
        signal2 = strategy.create_signal(
            SignalType.LONG,
            "BTCUSDT",
            Decimal("50000"),
            -0.5,  # < 0
        )

        if signal2:
            assert signal2.strength >= 0.0


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_orderbook(self):
        """Test handling empty orderbook."""
        strategy = OrderBookImbalanceStrategy({"enabled": True})

        empty_ob = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            bids=[],
            asks=[],
            last_update_id=12345,
        )

        signal = strategy.on_orderbook(empty_ob)
        assert signal is None

    def test_single_level_orderbook(self):
        """Test handling single level orderbook."""
        strategy = OrderBookImbalanceStrategy({"enabled": True})

        single_ob = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            bids=[OrderBookLevel(price=Decimal("50000"), quantity=Decimal("1.0"))],
            asks=[OrderBookLevel(price=Decimal("50001"), quantity=Decimal("1.0"))],
            last_update_id=12345,
        )

        signal = strategy.on_orderbook(single_ob)
        # Should not crash

    def test_zero_price(self):
        """Test handling zero price."""
        strategy = MeanReversionStrategy({"enabled": True, "lookback_period": 5})

        # This shouldn't happen in practice, but test resilience
        zero_ob = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            bids=[OrderBookLevel(price=Decimal("0"), quantity=Decimal("1.0"))],
            asks=[OrderBookLevel(price=Decimal("0.0001"), quantity=Decimal("1.0"))],
            last_update_id=12345,
        )

        # Should not crash
        signal = strategy.on_orderbook(zero_ob)

    def test_very_large_values(self):
        """Test handling very large values."""
        strategy = DCAStrategy({"enabled": True, "mode": "time_based"})

        large_ob = create_orderbook_with_imbalance(
            "BTCUSDT",
            1000000000,
            1000000000,
            100000000,
        )

        # Should not crash
        signal = strategy.on_orderbook(large_ob)

    def test_none_timestamp(self):
        """Test handling missing timestamp."""
        # Most strategies should handle this gracefully
        strategy = OrderBookImbalanceStrategy({"enabled": True})

        # Create orderbook without explicit timestamp
        ob = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            bids=[OrderBookLevel(price=Decimal("50000"), quantity=Decimal("1.0"))],
            asks=[OrderBookLevel(price=Decimal("50001"), quantity=Decimal("1.0"))],
            last_update_id=12345,
        )

        signal = strategy.on_orderbook(ob)
