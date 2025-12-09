"""
Unit tests for Risk Manager.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from src.risk.manager import RiskManager, RiskConfig, DailyStats
from src.data.models import Signal, SignalType, Side, Position


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def risk_config():
    return {
        "total_capital": 100,
        "max_position_pct": 0.1,
        "risk_per_trade_pct": 0.01,
        "max_loss_per_trade": 2,
        "max_positions": 1,
        "max_daily_trades": 50,
        "max_daily_loss": 10,
        "default_stop_loss_pct": 0.002,
        "default_take_profit_pct": 0.003,
        "max_leverage": 10,
        "default_leverage": 5,
        "cooldown_after_loss": 60,
        "cooldown_after_max_loss": 3600,
    }


@pytest.fixture
def risk_manager(risk_config):
    return RiskManager(risk_config)


@pytest.fixture
def long_signal():
    return Signal(
        strategy="test",
        signal_type=SignalType.LONG,
        symbol="BTCUSDT",
        timestamp=datetime.utcnow(),
        strength=0.8,
        price=Decimal("50000"),
    )


@pytest.fixture
def short_signal():
    return Signal(
        strategy="test",
        signal_type=SignalType.SHORT,
        symbol="BTCUSDT",
        timestamp=datetime.utcnow(),
        strength=0.8,
        price=Decimal("50000"),
    )


@pytest.fixture
def sample_position():
    return Position(
        symbol="BTCUSDT",
        side=Side.BUY,
        size=Decimal("0.001"),
        entry_price=Decimal("50000"),
        mark_price=Decimal("50100"),
        liquidation_price=Decimal("45000"),
        unrealized_pnl=Decimal("0.1"),
        realized_pnl=Decimal("0"),
        leverage=5,
        margin_type="CROSSED",
        updated_at=datetime.utcnow(),
    )


# =============================================================================
# RiskConfig Tests
# =============================================================================

class TestRiskConfig:
    """Tests for RiskConfig dataclass."""

    def test_default_values(self):
        config = RiskConfig()

        assert config.total_capital == Decimal("100")
        assert config.max_positions == 1
        assert config.max_daily_trades == 50

    def test_custom_values(self):
        config = RiskConfig(
            total_capital=Decimal("1000"),
            max_positions=3,
        )

        assert config.total_capital == Decimal("1000")
        assert config.max_positions == 3


# =============================================================================
# DailyStats Tests
# =============================================================================

class TestDailyStats:
    """Tests for DailyStats dataclass."""

    def test_default_values(self):
        stats = DailyStats()

        assert stats.trades == 0
        assert stats.wins == 0
        assert stats.losses == 0
        assert stats.total_pnl == Decimal("0")

    def test_date_default(self):
        stats = DailyStats()
        assert stats.date == datetime.utcnow().date()


# =============================================================================
# RiskManager Tests
# =============================================================================

class TestRiskManager:
    """Tests for RiskManager class."""

    def test_init(self, risk_manager, risk_config):
        assert risk_manager.config.total_capital == Decimal("100")
        assert risk_manager.config.max_positions == 1

    def test_available_capital_no_positions(self, risk_manager):
        assert risk_manager.available_capital == Decimal("100")

    def test_stats(self, risk_manager):
        stats = risk_manager.stats

        assert "total_capital" in stats
        assert "available_capital" in stats
        assert "position_count" in stats
        assert "daily_trades" in stats
        assert stats["total_capital"] == 100


class TestCanTrade:
    """Tests for can_trade validation."""

    def test_can_trade_default(self, risk_manager, long_signal):
        can_trade, reason = risk_manager.can_trade(long_signal)
        assert can_trade == True
        assert reason == "OK"

    def test_cannot_trade_when_paused(self, risk_manager, long_signal):
        risk_manager._paused = True

        can_trade, reason = risk_manager.can_trade(long_signal)

        assert can_trade == False
        assert "paused" in reason.lower()

    def test_cannot_trade_max_daily_trades(self, risk_manager, long_signal):
        risk_manager._daily_stats.trades = 50

        can_trade, reason = risk_manager.can_trade(long_signal)

        assert can_trade == False
        assert "daily trade limit" in reason.lower()

    def test_cannot_trade_max_daily_loss(self, risk_manager, long_signal):
        risk_manager._daily_stats.total_pnl = Decimal("-10")

        can_trade, reason = risk_manager.can_trade(long_signal)

        assert can_trade == False
        assert "daily loss limit" in reason.lower()

    def test_cannot_trade_max_positions(self, risk_manager, long_signal, sample_position):
        risk_manager._positions["BTCUSDT"] = sample_position

        can_trade, reason = risk_manager.can_trade(long_signal)

        assert can_trade == False
        assert "max positions" in reason.lower()

    def test_cannot_trade_existing_position(self, risk_manager, long_signal, sample_position):
        # Increase max positions to test symbol-specific check
        risk_manager.config = RiskConfig(max_positions=5)
        risk_manager._positions["BTCUSDT"] = sample_position

        can_trade, reason = risk_manager.can_trade(long_signal)

        assert can_trade == False
        assert "already have position" in reason.lower()

    def test_loss_cooldown(self, risk_manager, long_signal):
        risk_manager._last_loss_time = datetime.utcnow()

        can_trade, reason = risk_manager.can_trade(long_signal)

        assert can_trade == False
        assert "cooldown" in reason.lower()

    def test_can_trade_after_cooldown(self, risk_manager, long_signal):
        risk_manager._last_loss_time = datetime.utcnow() - timedelta(seconds=120)

        can_trade, reason = risk_manager.can_trade(long_signal)

        assert can_trade == True


class TestPositionSizing:
    """Tests for position sizing calculations."""

    def test_calculate_position_size(self, risk_manager, long_signal):
        size = risk_manager.calculate_position_size(long_signal)

        # size should be positive and reasonable
        assert size > 0
        assert size < Decimal("1")  # Less than 1 BTC for $100 capital

    def test_position_size_respects_max_position(self, risk_manager, long_signal):
        size = risk_manager.calculate_position_size(long_signal)

        # Max position = 10% of $100 * 5x leverage = $50
        # At $50,000/BTC, max qty = 0.001 BTC
        max_qty = (Decimal("100") * Decimal("0.1") * 5) / Decimal("50000")

        assert size <= max_qty * Decimal("1.01")  # Allow 1% margin for rounding

    def test_calculate_stop_loss_long(self, risk_manager):
        entry = Decimal("50000")
        stop = risk_manager.calculate_stop_loss(entry, Side.BUY)

        # 0.2% stop loss for long = 50000 * 0.998 = 49900
        assert stop == Decimal("49900")

    def test_calculate_stop_loss_short(self, risk_manager):
        entry = Decimal("50000")
        stop = risk_manager.calculate_stop_loss(entry, Side.SELL)

        # 0.2% stop loss for short = 50000 * 1.002 = 50100
        assert stop == Decimal("50100")

    def test_calculate_take_profit_long(self, risk_manager):
        entry = Decimal("50000")
        tp = risk_manager.calculate_take_profit(entry, Side.BUY)

        # 0.3% take profit for long = 50000 * 1.003 = 50150
        assert tp == Decimal("50150")

    def test_calculate_take_profit_short(self, risk_manager):
        entry = Decimal("50000")
        tp = risk_manager.calculate_take_profit(entry, Side.SELL)

        # 0.3% take profit for short = 50000 * 0.997 = 49850
        assert tp == Decimal("49850")


class TestPositionManagement:
    """Tests for position management."""

    def test_register_position(self, risk_manager, sample_position):
        risk_manager.register_position(sample_position)

        assert "BTCUSDT" in risk_manager._positions
        assert risk_manager._daily_stats.trades == 1

    def test_update_position(self, risk_manager, sample_position):
        risk_manager.register_position(sample_position)

        updated = Position(
            symbol="BTCUSDT",
            side=Side.BUY,
            size=Decimal("0.001"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50200"),  # Price moved
            liquidation_price=Decimal("45000"),
            unrealized_pnl=Decimal("0.2"),  # PnL increased
            realized_pnl=Decimal("0"),
            leverage=5,
            margin_type="CROSSED",
            updated_at=datetime.utcnow(),
        )

        risk_manager.update_position(updated)

        assert risk_manager._positions["BTCUSDT"].mark_price == Decimal("50200")

    def test_close_position_profit(self, risk_manager, sample_position):
        risk_manager.register_position(sample_position)

        risk_manager.close_position("BTCUSDT", Decimal("5"))  # $5 profit

        assert "BTCUSDT" not in risk_manager._positions
        assert risk_manager._daily_stats.total_pnl == Decimal("5")
        assert risk_manager._daily_stats.wins == 1
        assert risk_manager._daily_stats.losses == 0

    def test_close_position_loss(self, risk_manager, sample_position):
        risk_manager.register_position(sample_position)

        risk_manager.close_position("BTCUSDT", Decimal("-3"))  # $3 loss

        assert risk_manager._daily_stats.total_pnl == Decimal("-3")
        assert risk_manager._daily_stats.wins == 0
        assert risk_manager._daily_stats.losses == 1
        assert risk_manager._last_loss_time is not None

    def test_get_position(self, risk_manager, sample_position):
        risk_manager.register_position(sample_position)

        pos = risk_manager.get_position("BTCUSDT")

        assert pos is not None
        assert pos.symbol == "BTCUSDT"

    def test_get_nonexistent_position(self, risk_manager):
        pos = risk_manager.get_position("ETHUSDT")
        assert pos is None

    def test_get_all_positions(self, risk_manager, sample_position):
        risk_manager.register_position(sample_position)

        positions = risk_manager.get_all_positions()

        assert len(positions) == 1


class TestEmergencyControls:
    """Tests for emergency controls."""

    def test_emergency_stop(self, risk_manager):
        risk_manager.emergency_stop()

        assert risk_manager._paused == True
        assert risk_manager._pause_until is None

    def test_resume_trading(self, risk_manager):
        risk_manager.emergency_stop()
        risk_manager.resume_trading()

        assert risk_manager._paused == False

    def test_reset_daily_stats(self, risk_manager):
        risk_manager._daily_stats.trades = 10
        risk_manager._daily_stats.total_pnl = Decimal("50")

        risk_manager.reset_daily_stats()

        assert risk_manager._daily_stats.trades == 0
        assert risk_manager._daily_stats.total_pnl == Decimal("0")


class TestDrawdownTracking:
    """Tests for drawdown tracking."""

    def test_peak_pnl_tracking(self, risk_manager, sample_position):
        risk_manager.register_position(sample_position)

        # First winning trade
        risk_manager.close_position("BTCUSDT", Decimal("5"))
        assert risk_manager._daily_stats.peak_pnl == Decimal("5")

        # Second winning trade
        risk_manager.register_position(sample_position)
        risk_manager.close_position("BTCUSDT", Decimal("3"))
        assert risk_manager._daily_stats.peak_pnl == Decimal("8")

    def test_max_drawdown_tracking(self, risk_manager, sample_position):
        risk_manager.register_position(sample_position)

        # Win $10
        risk_manager.close_position("BTCUSDT", Decimal("10"))
        assert risk_manager._daily_stats.peak_pnl == Decimal("10")

        # Lose $5
        risk_manager.register_position(sample_position)
        risk_manager.close_position("BTCUSDT", Decimal("-5"))

        assert risk_manager._daily_stats.max_drawdown == Decimal("5")
        assert risk_manager._daily_stats.total_pnl == Decimal("5")
