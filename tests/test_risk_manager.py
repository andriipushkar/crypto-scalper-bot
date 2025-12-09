"""
Comprehensive tests for RiskManager.

Tests cover:
- RiskConfig parsing
- Position sizing calculations
- Stop loss / Take profit calculations
- Trade validation (can_trade)
- Position management
- Daily stats tracking
- Emergency controls
- Cooldowns
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, '/home/sssmmmddd/Code/pro/crypto-scalper-bot')

from src.risk.manager import RiskManager, RiskConfig, DailyStats
from src.data.models import Signal, SignalType, Side, Position


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Default risk configuration."""
    return {
        "total_capital": 1000,
        "max_position_pct": 0.1,
        "risk_per_trade_pct": 0.01,
        "max_loss_per_trade": 10,
        "max_positions": 3,
        "max_daily_trades": 50,
        "max_daily_loss": 50,
        "default_stop_loss_pct": 0.002,
        "default_take_profit_pct": 0.003,
        "max_leverage": 20,
        "default_leverage": 10,
        "cooldown_after_loss": 30,
        "cooldown_after_max_loss": 1800,
    }


@pytest.fixture
def risk_manager(default_config):
    """Create RiskManager instance."""
    return RiskManager(default_config)


@pytest.fixture
def entry_signal():
    """Sample entry signal."""
    return Signal(
        strategy="test_strategy",
        signal_type=SignalType.LONG,
        symbol="BTCUSDT",
        timestamp=datetime.utcnow(),
        strength=0.8,
        price=Decimal("50000"),
    )


@pytest.fixture
def exit_signal():
    """Sample exit signal."""
    return Signal(
        strategy="test_strategy",
        signal_type=SignalType.CLOSE_LONG,
        symbol="BTCUSDT",
        timestamp=datetime.utcnow(),
        strength=0.7,
        price=Decimal("51000"),
    )


@pytest.fixture
def sample_position():
    """Sample position."""
    return Position(
        symbol="BTCUSDT",
        side=Side.BUY,
        size=Decimal("0.01"),
        entry_price=Decimal("50000"),
        mark_price=Decimal("50500"),
        liquidation_price=Decimal("45000"),
        unrealized_pnl=Decimal("5"),
        realized_pnl=Decimal("0"),
        leverage=10,
        margin_type="CROSSED",
        updated_at=datetime.utcnow(),
    )


# =============================================================================
# RiskConfig Tests
# =============================================================================

class TestRiskConfig:
    """Tests for RiskConfig parsing."""

    def test_default_config_values(self, risk_manager):
        """Test that default config values are parsed correctly."""
        config = risk_manager.config
        assert config.total_capital == Decimal("1000")
        assert config.max_position_pct == Decimal("0.1")
        assert config.risk_per_trade_pct == Decimal("0.01")
        assert config.max_positions == 3
        assert config.max_daily_trades == 50

    def test_parse_empty_config(self):
        """Test parsing empty config uses defaults."""
        rm = RiskManager({})
        assert rm.config.total_capital == Decimal("100")
        assert rm.config.max_positions == 1
        assert rm.config.default_leverage == 5

    def test_parse_string_numbers(self):
        """Test parsing string number values."""
        config = {"total_capital": "5000", "max_position_pct": "0.2"}
        rm = RiskManager(config)
        assert rm.config.total_capital == Decimal("5000")
        assert rm.config.max_position_pct == Decimal("0.2")

    def test_parse_float_values(self):
        """Test parsing float values."""
        config = {"total_capital": 2500.5, "risk_per_trade_pct": 0.015}
        rm = RiskManager(config)
        assert rm.config.total_capital == Decimal("2500.5")
        assert rm.config.risk_per_trade_pct == Decimal("0.015")


# =============================================================================
# Position Sizing Tests
# =============================================================================

class TestPositionSizing:
    """Tests for position sizing calculations."""

    def test_calculate_position_size_basic(self, risk_manager, entry_signal):
        """Test basic position sizing."""
        size = risk_manager.calculate_position_size(entry_signal)
        assert size > 0
        assert isinstance(size, Decimal)

    def test_position_size_with_leverage(self, risk_manager, entry_signal):
        """Test position sizing with custom leverage."""
        size_5x = risk_manager.calculate_position_size(entry_signal, leverage=5)
        size_10x = risk_manager.calculate_position_size(entry_signal, leverage=10)

        # Higher leverage should allow larger position
        assert size_10x >= size_5x

    def test_position_size_respects_max_leverage(self, risk_manager, entry_signal):
        """Test that position size respects max leverage."""
        # Max leverage is 20
        size_20x = risk_manager.calculate_position_size(entry_signal, leverage=20)
        size_50x = risk_manager.calculate_position_size(entry_signal, leverage=50)

        # Both should be same since 50 is capped to 20
        assert size_50x == size_20x

    def test_position_size_respects_max_position_pct(self, risk_manager, entry_signal):
        """Test position size is limited by max_position_pct."""
        size = risk_manager.calculate_position_size(entry_signal, leverage=20)

        # Max position value = 1000 * 0.1 = 100
        max_position_value = risk_manager.config.total_capital * risk_manager.config.max_position_pct
        actual_value = size * entry_signal.price

        assert actual_value <= max_position_value

    def test_position_size_zero_price(self, risk_manager):
        """Test position sizing with zero price."""
        signal = Signal(
            strategy="test",
            signal_type=SignalType.LONG,
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            strength=0.5,
            price=Decimal("0"),
        )
        size = risk_manager.calculate_position_size(signal)
        assert size == Decimal("0")

    def test_position_size_considers_available_capital(self, risk_manager, entry_signal, sample_position):
        """Test position sizing considers available capital."""
        # Register a position to reduce available capital
        risk_manager.register_position(sample_position)

        size_with_position = risk_manager.calculate_position_size(entry_signal)

        # Clear position and check size
        risk_manager._positions.clear()
        size_without_position = risk_manager.calculate_position_size(entry_signal)

        # Size should be smaller when capital is partially used
        assert size_with_position <= size_without_position


# =============================================================================
# Stop Loss / Take Profit Tests
# =============================================================================

class TestStopLossTakeProfit:
    """Tests for SL/TP calculations."""

    def test_stop_loss_buy_position(self, risk_manager):
        """Test stop loss for buy position."""
        entry_price = Decimal("50000")
        stop_loss = risk_manager.calculate_stop_loss(entry_price, Side.BUY)

        # Stop loss should be below entry for buy
        assert stop_loss < entry_price
        expected = entry_price * (1 - risk_manager.config.default_stop_loss_pct)
        assert stop_loss == expected

    def test_stop_loss_sell_position(self, risk_manager):
        """Test stop loss for sell position."""
        entry_price = Decimal("50000")
        stop_loss = risk_manager.calculate_stop_loss(entry_price, Side.SELL)

        # Stop loss should be above entry for sell
        assert stop_loss > entry_price
        expected = entry_price * (1 + risk_manager.config.default_stop_loss_pct)
        assert stop_loss == expected

    def test_take_profit_buy_position(self, risk_manager):
        """Test take profit for buy position."""
        entry_price = Decimal("50000")
        take_profit = risk_manager.calculate_take_profit(entry_price, Side.BUY)

        # Take profit should be above entry for buy
        assert take_profit > entry_price
        expected = entry_price * (1 + risk_manager.config.default_take_profit_pct)
        assert take_profit == expected

    def test_take_profit_sell_position(self, risk_manager):
        """Test take profit for sell position."""
        entry_price = Decimal("50000")
        take_profit = risk_manager.calculate_take_profit(entry_price, Side.SELL)

        # Take profit should be below entry for sell
        assert take_profit < entry_price
        expected = entry_price * (1 - risk_manager.config.default_take_profit_pct)
        assert take_profit == expected

    def test_custom_stop_loss_pct(self, risk_manager):
        """Test custom stop loss percentage."""
        entry_price = Decimal("50000")
        custom_pct = Decimal("0.01")  # 1%

        stop_loss = risk_manager.calculate_stop_loss(entry_price, Side.BUY, custom_pct)
        expected = entry_price * (1 - custom_pct)
        assert stop_loss == expected

    def test_custom_take_profit_pct(self, risk_manager):
        """Test custom take profit percentage."""
        entry_price = Decimal("50000")
        custom_pct = Decimal("0.02")  # 2%

        take_profit = risk_manager.calculate_take_profit(entry_price, Side.BUY, custom_pct)
        expected = entry_price * (1 + custom_pct)
        assert take_profit == expected


# =============================================================================
# Trade Validation Tests
# =============================================================================

class TestCanTrade:
    """Tests for trade validation."""

    def test_can_trade_default(self, risk_manager, entry_signal):
        """Test can trade in default state."""
        can_trade, reason = risk_manager.can_trade(entry_signal)
        assert can_trade is True
        assert reason == "OK"

    def test_cannot_trade_when_paused(self, risk_manager, entry_signal):
        """Test cannot trade when paused."""
        risk_manager._paused = True

        can_trade, reason = risk_manager.can_trade(entry_signal)
        assert can_trade is False
        assert "paused" in reason.lower()

    def test_pause_expires(self, risk_manager, entry_signal):
        """Test pause expiration."""
        risk_manager._paused = True
        risk_manager._pause_until = datetime.utcnow() - timedelta(seconds=1)

        can_trade, reason = risk_manager.can_trade(entry_signal)
        assert can_trade is True
        assert risk_manager._paused is False

    def test_cannot_trade_max_positions(self, risk_manager, entry_signal, sample_position):
        """Test cannot trade when max positions reached."""
        # Fill all position slots
        for i in range(risk_manager.config.max_positions):
            pos = Position(
                symbol=f"SYMBOL{i}",
                side=Side.BUY,
                size=Decimal("0.01"),
                entry_price=Decimal("50000"),
                mark_price=Decimal("50000"),
                liquidation_price=Decimal("45000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                leverage=10,
                margin_type="CROSSED",
                updated_at=datetime.utcnow(),
            )
            risk_manager.register_position(pos)

        can_trade, reason = risk_manager.can_trade(entry_signal)
        assert can_trade is False
        assert "max positions" in reason.lower()

    def test_cannot_trade_existing_symbol(self, risk_manager, entry_signal, sample_position):
        """Test cannot trade when position exists in symbol."""
        risk_manager.register_position(sample_position)

        can_trade, reason = risk_manager.can_trade(entry_signal)
        assert can_trade is False
        assert "already have position" in reason.lower()

    def test_cannot_trade_daily_limit(self, risk_manager, entry_signal):
        """Test cannot trade when daily trade limit reached."""
        risk_manager._daily_stats.trades = risk_manager.config.max_daily_trades

        can_trade, reason = risk_manager.can_trade(entry_signal)
        assert can_trade is False
        assert "daily trade limit" in reason.lower()

    def test_cannot_trade_daily_loss_limit(self, risk_manager, entry_signal):
        """Test cannot trade when daily loss limit reached."""
        risk_manager._daily_stats.total_pnl = -risk_manager.config.max_daily_loss

        can_trade, reason = risk_manager.can_trade(entry_signal)
        assert can_trade is False
        assert "daily loss limit" in reason.lower()

    def test_cannot_trade_loss_cooldown(self, risk_manager, entry_signal):
        """Test cannot trade during loss cooldown."""
        risk_manager._last_loss_time = datetime.utcnow()

        can_trade, reason = risk_manager.can_trade(entry_signal)
        assert can_trade is False
        assert "cooldown" in reason.lower()

    def test_can_trade_after_cooldown(self, risk_manager, entry_signal):
        """Test can trade after cooldown expires."""
        cooldown = risk_manager.config.cooldown_after_loss
        risk_manager._last_loss_time = datetime.utcnow() - timedelta(seconds=cooldown + 1)

        can_trade, reason = risk_manager.can_trade(entry_signal)
        assert can_trade is True

    def test_cannot_trade_insufficient_capital(self, risk_manager, entry_signal):
        """Test cannot trade with insufficient capital."""
        # Use up almost all capital
        risk_manager.config.total_capital = Decimal("100")
        pos = Position(
            symbol="ETHUSDT",
            side=Side.BUY,
            size=Decimal("0.032"),  # 0.032 * 3000 = 96 notional
            entry_price=Decimal("3000"),
            mark_price=Decimal("3000"),
            liquidation_price=Decimal("2700"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            leverage=10,
            margin_type="CROSSED",
            updated_at=datetime.utcnow(),
        )
        risk_manager.register_position(pos)

        can_trade, reason = risk_manager.can_trade(entry_signal)
        assert can_trade is False
        assert "insufficient" in reason.lower()

    def test_can_trade_exit_signal_no_restrictions(self, risk_manager, exit_signal, sample_position):
        """Test exit signals bypass entry restrictions."""
        risk_manager.register_position(sample_position)
        # Fill positions for other symbols
        risk_manager._positions["ETHUSDT"] = sample_position
        risk_manager._positions["SOLUSDT"] = sample_position

        can_trade, reason = risk_manager.can_trade(exit_signal)
        assert can_trade is True


# =============================================================================
# Position Management Tests
# =============================================================================

class TestPositionManagement:
    """Tests for position management."""

    def test_register_position(self, risk_manager, sample_position):
        """Test registering a position."""
        risk_manager.register_position(sample_position)

        assert sample_position.symbol in risk_manager._positions
        assert risk_manager._daily_stats.trades == 1

    def test_update_position(self, risk_manager, sample_position):
        """Test updating a position."""
        risk_manager.register_position(sample_position)

        # Update mark price
        sample_position.mark_price = Decimal("51000")
        risk_manager.update_position(sample_position)

        updated = risk_manager.get_position(sample_position.symbol)
        assert updated.mark_price == Decimal("51000")

    def test_close_position_profit(self, risk_manager, sample_position):
        """Test closing position with profit."""
        risk_manager.register_position(sample_position)

        pnl = Decimal("50")
        risk_manager.close_position(sample_position.symbol, pnl)

        assert sample_position.symbol not in risk_manager._positions
        assert risk_manager._daily_stats.total_pnl == pnl
        assert risk_manager._daily_stats.wins == 1

    def test_close_position_loss(self, risk_manager, sample_position):
        """Test closing position with loss."""
        risk_manager.register_position(sample_position)

        pnl = Decimal("-30")
        risk_manager.close_position(sample_position.symbol, pnl)

        assert sample_position.symbol not in risk_manager._positions
        assert risk_manager._daily_stats.total_pnl == pnl
        assert risk_manager._daily_stats.losses == 1
        assert risk_manager._last_loss_time is not None

    def test_get_position(self, risk_manager, sample_position):
        """Test getting a position."""
        assert risk_manager.get_position(sample_position.symbol) is None

        risk_manager.register_position(sample_position)
        retrieved = risk_manager.get_position(sample_position.symbol)

        assert retrieved is not None
        assert retrieved.symbol == sample_position.symbol

    def test_get_all_positions(self, risk_manager):
        """Test getting all positions."""
        positions = []
        for i in range(3):
            pos = Position(
                symbol=f"SYMBOL{i}",
                side=Side.BUY,
                size=Decimal("0.01"),
                entry_price=Decimal("50000"),
                mark_price=Decimal("50000"),
                liquidation_price=Decimal("45000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                leverage=10,
                margin_type="CROSSED",
                updated_at=datetime.utcnow(),
            )
            positions.append(pos)
            risk_manager.register_position(pos)

        all_positions = risk_manager.get_all_positions()
        assert len(all_positions) == 3

    def test_available_capital(self, risk_manager, sample_position):
        """Test available capital calculation."""
        initial_available = risk_manager.available_capital
        assert initial_available == risk_manager.config.total_capital

        risk_manager.register_position(sample_position)

        available_after = risk_manager.available_capital
        assert available_after < initial_available
        assert available_after == risk_manager.config.total_capital - sample_position.notional_value


# =============================================================================
# Daily Stats Tests
# =============================================================================

class TestDailyStats:
    """Tests for daily statistics tracking."""

    def test_daily_reset_on_new_day(self, risk_manager):
        """Test daily stats reset on new day."""
        risk_manager._daily_stats.trades = 10
        risk_manager._daily_stats.wins = 5
        risk_manager._daily_stats.date = (datetime.utcnow() - timedelta(days=1)).date()

        risk_manager._check_daily_reset()

        assert risk_manager._daily_stats.trades == 0
        assert risk_manager._daily_stats.wins == 0
        assert risk_manager._daily_stats.date == datetime.utcnow().date()

    def test_drawdown_tracking(self, risk_manager, sample_position):
        """Test max drawdown tracking."""
        risk_manager.register_position(sample_position)

        # Simulate profit then loss
        risk_manager.close_position(sample_position.symbol, Decimal("100"))
        assert risk_manager._daily_stats.peak_pnl == Decimal("100")

        # Register new position
        risk_manager.register_position(sample_position)
        risk_manager.close_position(sample_position.symbol, Decimal("-50"))

        assert risk_manager._daily_stats.max_drawdown == Decimal("50")

    def test_stats_property(self, risk_manager, sample_position):
        """Test stats property returns correct data."""
        risk_manager.register_position(sample_position)
        risk_manager.close_position(sample_position.symbol, Decimal("25"))

        stats = risk_manager.stats

        assert stats["total_capital"] == 1000.0
        assert stats["daily_trades"] == 1
        assert stats["daily_pnl"] == 25.0
        assert stats["daily_wins"] == 1
        assert stats["daily_losses"] == 0


# =============================================================================
# Emergency Controls Tests
# =============================================================================

class TestEmergencyControls:
    """Tests for emergency controls."""

    def test_emergency_stop(self, risk_manager, entry_signal):
        """Test emergency stop functionality."""
        risk_manager.emergency_stop()

        assert risk_manager._paused is True
        assert risk_manager._pause_until is None

        can_trade, _ = risk_manager.can_trade(entry_signal)
        assert can_trade is False

    def test_resume_trading(self, risk_manager, entry_signal):
        """Test resume trading functionality."""
        risk_manager.emergency_stop()
        risk_manager.resume_trading()

        assert risk_manager._paused is False

        can_trade, _ = risk_manager.can_trade(entry_signal)
        assert can_trade is True

    def test_reset_daily_stats(self, risk_manager, sample_position):
        """Test reset daily stats functionality."""
        risk_manager.register_position(sample_position)
        risk_manager.close_position(sample_position.symbol, Decimal("-20"))

        risk_manager.reset_daily_stats()

        assert risk_manager._daily_stats.trades == 0
        assert risk_manager._daily_stats.total_pnl == Decimal("0")
        assert risk_manager._daily_stats.losses == 0

    def test_pause_trading_duration(self, risk_manager):
        """Test timed pause functionality."""
        duration = 300  # 5 minutes
        risk_manager._pause_trading(duration)

        assert risk_manager._paused is True
        assert risk_manager._pause_until is not None

        expected_resume = datetime.utcnow() + timedelta(seconds=duration)
        delta = abs((risk_manager._pause_until - expected_resume).total_seconds())
        assert delta < 2  # Within 2 seconds


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_close_nonexistent_position(self, risk_manager):
        """Test closing position that doesn't exist."""
        # Should not raise an error
        risk_manager.close_position("NONEXISTENT", Decimal("10"))
        assert risk_manager._daily_stats.total_pnl == Decimal("10")

    def test_very_small_capital(self):
        """Test with very small capital."""
        config = {"total_capital": 1}
        rm = RiskManager(config)

        signal = Signal(
            strategy="test",
            signal_type=SignalType.LONG,
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            strength=0.5,
            price=Decimal("50000"),
        )

        size = rm.calculate_position_size(signal)
        # Should return valid (possibly very small) size
        assert size >= 0

    def test_very_large_capital(self):
        """Test with very large capital."""
        config = {"total_capital": 10000000}
        rm = RiskManager(config)

        signal = Signal(
            strategy="test",
            signal_type=SignalType.LONG,
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            strength=0.5,
            price=Decimal("50000"),
        )

        size = rm.calculate_position_size(signal)
        assert size > 0

    def test_multiple_consecutive_losses(self, risk_manager, sample_position):
        """Test behavior with multiple consecutive losses."""
        for i in range(5):
            pos = Position(
                symbol=f"SYMBOL{i}",
                side=Side.BUY,
                size=Decimal("0.01"),
                entry_price=Decimal("50000"),
                mark_price=Decimal("50000"),
                liquidation_price=Decimal("45000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                leverage=10,
                margin_type="CROSSED",
                updated_at=datetime.utcnow(),
            )
            risk_manager.register_position(pos)
            risk_manager.close_position(pos.symbol, Decimal("-5"))

        assert risk_manager._daily_stats.losses == 5
        assert risk_manager._daily_stats.total_pnl == Decimal("-25")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
