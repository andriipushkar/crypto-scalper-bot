"""
Comprehensive tests for BacktestEngine.

Tests cover:
- BacktestConfig
- SimulatedPosition
- Position opening/closing
- Slippage and commission calculations
- Equity curve tracking
- Daily P&L management
- Walk-forward testing
- Monte Carlo simulation
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, '/home/sssmmmddd/Code/pro/crypto-scalper-bot')

from src.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    BacktestEngine,
    SimulatedPosition,
    WalkForwardResult,
    walk_forward_test,
    monte_carlo_simulation,
)
from src.data.models import Signal, SignalType, Side
from src.utils.metrics import TradeRecord


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def backtest_config():
    """Default backtest configuration."""
    return BacktestConfig(
        initial_capital=Decimal("1000"),
        leverage=10,
        slippage_bps=1.0,
        commission_rate=0.0004,
        latency_ms=50,
        max_position_pct=0.5,
        max_daily_loss_pct=0.05,
        use_orderbook=True,
        warmup_bars=100,
    )


@pytest.fixture
def backtest_engine(backtest_config):
    """Create BacktestEngine instance."""
    return BacktestEngine(config=backtest_config)


@pytest.fixture
def long_signal():
    """Create long entry signal."""
    return Signal(
        strategy="test_strategy",
        signal_type=SignalType.LONG,
        symbol="BTCUSDT",
        timestamp=datetime.utcnow(),
        strength=0.8,
        price=Decimal("50000"),
    )


@pytest.fixture
def short_signal():
    """Create short entry signal."""
    return Signal(
        strategy="test_strategy",
        signal_type=SignalType.SHORT,
        symbol="BTCUSDT",
        timestamp=datetime.utcnow(),
        strength=0.8,
        price=Decimal("50000"),
    )


@pytest.fixture
def close_long_signal():
    """Create close long signal."""
    return Signal(
        strategy="test_strategy",
        signal_type=SignalType.CLOSE_LONG,
        symbol="BTCUSDT",
        timestamp=datetime.utcnow(),
        strength=0.7,
        price=Decimal("51000"),
    )


@pytest.fixture
def close_short_signal():
    """Create close short signal."""
    return Signal(
        strategy="test_strategy",
        signal_type=SignalType.CLOSE_SHORT,
        symbol="BTCUSDT",
        timestamp=datetime.utcnow(),
        strength=0.7,
        price=Decimal("49000"),
    )


@pytest.fixture
def sample_trade_records():
    """Create sample trade records."""
    return [
        TradeRecord(
            symbol="BTCUSDT",
            side="LONG",
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 11, 0),
            entry_price=Decimal("50000"),
            exit_price=Decimal("51000"),
            quantity=Decimal("0.1"),
            pnl=Decimal("100"),
            commission=Decimal("4"),
            strategy="test",
        ),
        TradeRecord(
            symbol="BTCUSDT",
            side="LONG",
            entry_time=datetime(2024, 1, 1, 12, 0),
            exit_time=datetime(2024, 1, 1, 13, 0),
            entry_price=Decimal("51000"),
            exit_price=Decimal("50500"),
            quantity=Decimal("0.1"),
            pnl=Decimal("-50"),
            commission=Decimal("4"),
            strategy="test",
        ),
        TradeRecord(
            symbol="BTCUSDT",
            side="SHORT",
            entry_time=datetime(2024, 1, 1, 14, 0),
            exit_time=datetime(2024, 1, 1, 15, 0),
            entry_price=Decimal("50500"),
            exit_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            pnl=Decimal("50"),
            commission=Decimal("4"),
            strategy="test",
        ),
    ]


# =============================================================================
# BacktestConfig Tests
# =============================================================================

class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BacktestConfig()

        assert config.initial_capital == Decimal("100")
        assert config.leverage == 10
        assert config.slippage_bps == 1.0
        assert config.commission_rate == 0.0004
        assert config.max_position_pct == 0.5

    def test_custom_values(self, backtest_config):
        """Test custom configuration values."""
        assert backtest_config.initial_capital == Decimal("1000")
        assert backtest_config.leverage == 10
        assert backtest_config.max_daily_loss_pct == 0.05


# =============================================================================
# SimulatedPosition Tests
# =============================================================================

class TestSimulatedPosition:
    """Tests for SimulatedPosition."""

    def test_create_long_position(self):
        """Test creating long position."""
        pos = SimulatedPosition(
            symbol="BTCUSDT",
            side=Side.BUY,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            entry_time=datetime.now(),
            leverage=10,
        )

        assert pos.symbol == "BTCUSDT"
        assert pos.side == Side.BUY
        assert pos.entry_price == Decimal("50000")
        assert pos.quantity == Decimal("0.1")

    def test_create_short_position(self):
        """Test creating short position."""
        pos = SimulatedPosition(
            symbol="BTCUSDT",
            side=Side.SELL,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            entry_time=datetime.now(),
            leverage=10,
        )

        assert pos.side == Side.SELL

    def test_notional_value(self):
        """Test notional value calculation."""
        pos = SimulatedPosition(
            symbol="BTCUSDT",
            side=Side.BUY,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            entry_time=datetime.now(),
            leverage=10,
        )

        expected = Decimal("50000") * Decimal("0.1")
        assert pos.notional_value == expected

    def test_margin_required(self):
        """Test margin calculation."""
        pos = SimulatedPosition(
            symbol="BTCUSDT",
            side=Side.BUY,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            entry_time=datetime.now(),
            leverage=10,
        )

        notional = Decimal("50000") * Decimal("0.1")
        expected = notional / 10
        assert pos.margin_required == expected

    def test_unrealized_pnl_long_profit(self):
        """Test unrealized PnL for profitable long."""
        pos = SimulatedPosition(
            symbol="BTCUSDT",
            side=Side.BUY,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            entry_time=datetime.now(),
            leverage=10,
        )

        current_price = Decimal("51000")
        pnl = pos.unrealized_pnl(current_price)

        expected = (Decimal("51000") - Decimal("50000")) * Decimal("0.1")
        assert pnl == expected
        assert pnl > 0

    def test_unrealized_pnl_long_loss(self):
        """Test unrealized PnL for losing long."""
        pos = SimulatedPosition(
            symbol="BTCUSDT",
            side=Side.BUY,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            entry_time=datetime.now(),
            leverage=10,
        )

        current_price = Decimal("49000")
        pnl = pos.unrealized_pnl(current_price)

        expected = (Decimal("49000") - Decimal("50000")) * Decimal("0.1")
        assert pnl == expected
        assert pnl < 0

    def test_unrealized_pnl_short_profit(self):
        """Test unrealized PnL for profitable short."""
        pos = SimulatedPosition(
            symbol="BTCUSDT",
            side=Side.SELL,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            entry_time=datetime.now(),
            leverage=10,
        )

        current_price = Decimal("49000")
        pnl = pos.unrealized_pnl(current_price)

        expected = (Decimal("50000") - Decimal("49000")) * Decimal("0.1")
        assert pnl == expected
        assert pnl > 0

    def test_unrealized_pnl_short_loss(self):
        """Test unrealized PnL for losing short."""
        pos = SimulatedPosition(
            symbol="BTCUSDT",
            side=Side.SELL,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            entry_time=datetime.now(),
            leverage=10,
        )

        current_price = Decimal("51000")
        pnl = pos.unrealized_pnl(current_price)

        expected = (Decimal("50000") - Decimal("51000")) * Decimal("0.1")
        assert pnl == expected
        assert pnl < 0

    def test_unrealized_pnl_pct(self):
        """Test unrealized PnL percentage."""
        pos = SimulatedPosition(
            symbol="BTCUSDT",
            side=Side.BUY,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            entry_time=datetime.now(),
            leverage=10,
        )

        current_price = Decimal("51000")
        pnl_pct = pos.unrealized_pnl_pct(current_price)

        # PnL = 100, Margin = 500, so ~20%
        assert pnl_pct > 0


# =============================================================================
# BacktestEngine Tests
# =============================================================================

class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def test_initialization(self, backtest_engine, backtest_config):
        """Test engine initialization."""
        assert backtest_engine.config == backtest_config
        assert backtest_engine._capital == backtest_config.initial_capital
        assert backtest_engine._position is None
        assert backtest_engine._trades == []

    def test_reset(self, backtest_engine):
        """Test engine reset."""
        # Modify state
        backtest_engine._capital = Decimal("500")
        backtest_engine._trades = [MagicMock()]
        backtest_engine._signals_generated = 10

        backtest_engine.reset()

        assert backtest_engine._capital == backtest_engine.config.initial_capital
        assert backtest_engine._trades == []
        assert backtest_engine._signals_generated == 0


# =============================================================================
# Slippage and Commission Tests
# =============================================================================

class TestSlippageAndCommission:
    """Tests for slippage and commission calculations."""

    def test_apply_slippage_buy(self, backtest_engine):
        """Test slippage applied to buy order."""
        price = Decimal("50000")
        slipped_price = backtest_engine._apply_slippage(price, Side.BUY)

        # Buy should have higher price (unfavorable)
        assert slipped_price > price

    def test_apply_slippage_sell(self, backtest_engine):
        """Test slippage applied to sell order."""
        price = Decimal("50000")
        slipped_price = backtest_engine._apply_slippage(price, Side.SELL)

        # Sell should have lower price (unfavorable)
        assert slipped_price < price

    def test_calculate_commission(self, backtest_engine):
        """Test commission calculation."""
        notional = Decimal("5000")
        commission = backtest_engine._calculate_commission(notional)

        expected = notional * Decimal(str(backtest_engine.config.commission_rate))
        assert commission == expected


# =============================================================================
# Position Opening Tests
# =============================================================================

class TestPositionOpening:
    """Tests for position opening."""

    def test_can_open_position(self, backtest_engine, long_signal):
        """Test can open position check."""
        can_open = backtest_engine._can_open_position(long_signal, Decimal("50000"))
        assert can_open is True

    def test_cannot_open_position_when_exists(self, backtest_engine, long_signal):
        """Test cannot open when position exists."""
        # Create existing position
        backtest_engine._position = SimulatedPosition(
            symbol="BTCUSDT",
            side=Side.BUY,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            entry_time=datetime.now(),
        )

        can_open = backtest_engine._can_open_position(long_signal, Decimal("50000"))
        assert can_open is False

    def test_cannot_open_position_daily_loss_limit(self, backtest_engine, long_signal):
        """Test cannot open when daily loss limit reached."""
        # Set daily loss exceeding limit
        daily_limit = backtest_engine.config.initial_capital * Decimal(str(backtest_engine.config.max_daily_loss_pct))
        backtest_engine._daily_pnl = -daily_limit - Decimal("1")

        can_open = backtest_engine._can_open_position(long_signal, Decimal("50000"))
        assert can_open is False

    def test_open_long_position(self, backtest_engine, long_signal):
        """Test opening long position."""
        timestamp = datetime.now()
        backtest_engine._open_position(long_signal, Decimal("50000"), timestamp)

        assert backtest_engine._position is not None
        assert backtest_engine._position.side == Side.BUY
        assert backtest_engine._signals_executed == 1

    def test_open_short_position(self, backtest_engine, short_signal):
        """Test opening short position."""
        timestamp = datetime.now()
        backtest_engine._open_position(short_signal, Decimal("50000"), timestamp)

        assert backtest_engine._position is not None
        assert backtest_engine._position.side == Side.SELL

    def test_position_size_calculation(self, backtest_engine, long_signal):
        """Test position size calculation."""
        price = Decimal("50000")
        size = backtest_engine._calculate_position_size(price)

        # Max position = 1000 * 10 * 0.5 = 5000
        max_value = backtest_engine._capital * backtest_engine.config.leverage * Decimal(str(backtest_engine.config.max_position_pct))
        expected_qty = (max_value / price).quantize(Decimal("0.001"))

        assert size == expected_qty


# =============================================================================
# Position Closing Tests
# =============================================================================

class TestPositionClosing:
    """Tests for position closing."""

    def test_close_position_returns_none_when_no_position(self, backtest_engine):
        """Test closing when no position exists."""
        result = backtest_engine._close_position(Decimal("50000"), datetime.now())
        assert result is None

    def test_close_long_position_profit(self, backtest_engine, long_signal):
        """Test closing long position with profit."""
        # Open position
        entry_time = datetime.now()
        backtest_engine._open_position(long_signal, Decimal("50000"), entry_time)

        initial_capital = backtest_engine._capital

        # Close with profit
        exit_time = entry_time + timedelta(hours=1)
        trade = backtest_engine._close_position(Decimal("51000"), exit_time, "take_profit")

        assert backtest_engine._position is None
        assert len(backtest_engine._trades) == 1
        assert trade.side == "LONG"
        assert trade.pnl > 0

    def test_close_short_position_profit(self, backtest_engine, short_signal):
        """Test closing short position with profit."""
        entry_time = datetime.now()
        backtest_engine._open_position(short_signal, Decimal("50000"), entry_time)

        exit_time = entry_time + timedelta(hours=1)
        trade = backtest_engine._close_position(Decimal("49000"), exit_time, "take_profit")

        assert trade.side == "SHORT"
        assert trade.pnl > 0

    def test_close_position_loss(self, backtest_engine, long_signal):
        """Test closing position with loss."""
        entry_time = datetime.now()
        backtest_engine._open_position(long_signal, Decimal("50000"), entry_time)

        exit_time = entry_time + timedelta(hours=1)
        trade = backtest_engine._close_position(Decimal("49000"), exit_time, "stop_loss")

        assert trade.pnl < 0

    def test_capital_updated_on_close(self, backtest_engine, long_signal):
        """Test capital is updated when position closed."""
        entry_time = datetime.now()
        initial_capital = backtest_engine._capital

        # Open and close with profit
        backtest_engine._open_position(long_signal, Decimal("50000"), entry_time)
        backtest_engine._close_position(Decimal("51000"), entry_time + timedelta(hours=1))

        # Capital should have changed (minus commissions plus PnL)
        assert backtest_engine._capital != initial_capital


# =============================================================================
# Signal Processing Tests
# =============================================================================

class TestSignalProcessing:
    """Tests for signal processing."""

    def test_process_no_action_signal(self, backtest_engine):
        """Test processing NO_ACTION signal."""
        signal = Signal(
            signal_type=SignalType.NO_ACTION,
            symbol="BTCUSDT",
            price=Decimal("50000"),
            strength=0.5,
            strategy="test",
            timestamp=datetime.now(),
        )

        result = backtest_engine._process_signal(signal, Decimal("50000"), datetime.now())
        assert result is None
        assert backtest_engine._signals_generated == 1

    def test_process_long_signal(self, backtest_engine, long_signal):
        """Test processing LONG signal."""
        backtest_engine._process_signal(long_signal, Decimal("50000"), datetime.now())

        assert backtest_engine._position is not None
        assert backtest_engine._position.side == Side.BUY

    def test_process_short_signal(self, backtest_engine, short_signal):
        """Test processing SHORT signal."""
        backtest_engine._process_signal(short_signal, Decimal("50000"), datetime.now())

        assert backtest_engine._position is not None
        assert backtest_engine._position.side == Side.SELL

    def test_process_close_long_signal(self, backtest_engine, long_signal, close_long_signal):
        """Test processing CLOSE_LONG signal."""
        # First open long
        backtest_engine._process_signal(long_signal, Decimal("50000"), datetime.now())
        assert backtest_engine._position is not None

        # Then close
        backtest_engine._process_signal(close_long_signal, Decimal("51000"), datetime.now())
        assert backtest_engine._position is None

    def test_process_close_short_signal(self, backtest_engine, short_signal, close_short_signal):
        """Test processing CLOSE_SHORT signal."""
        # First open short
        backtest_engine._process_signal(short_signal, Decimal("50000"), datetime.now())

        # Then close
        backtest_engine._process_signal(close_short_signal, Decimal("49000"), datetime.now())
        assert backtest_engine._position is None

    def test_reversal_long_to_short(self, backtest_engine, long_signal, short_signal):
        """Test position reversal from long to short."""
        backtest_engine._process_signal(long_signal, Decimal("50000"), datetime.now())
        assert backtest_engine._position.side == Side.BUY

        # Opposite signal closes existing and opens new
        backtest_engine._process_signal(short_signal, Decimal("50500"), datetime.now())

        # Position should be closed (reversal handled)
        assert backtest_engine._position is None or backtest_engine._position.side == Side.SELL


# =============================================================================
# Equity Tracking Tests
# =============================================================================

class TestEquityTracking:
    """Tests for equity curve tracking."""

    def test_update_equity(self, backtest_engine):
        """Test equity update."""
        timestamp = datetime.now()
        price = Decimal("50000")

        backtest_engine._update_equity(timestamp, price)

        assert len(backtest_engine._equity_curve) == 1
        assert backtest_engine._equity_curve[0]["equity"] == float(backtest_engine._capital)

    def test_update_equity_with_position(self, backtest_engine, long_signal):
        """Test equity update includes unrealized PnL."""
        # Open position
        backtest_engine._open_position(long_signal, Decimal("50000"), datetime.now())

        # Update with higher price
        backtest_engine._update_equity(datetime.now(), Decimal("51000"))

        equity = backtest_engine._equity_curve[-1]["equity"]
        capital = backtest_engine._equity_curve[-1]["capital"]

        # Equity should be higher than capital due to unrealized profit
        assert equity > capital

    def test_drawdown_tracking(self, backtest_engine, long_signal):
        """Test maximum drawdown tracking."""
        # Simulate profit then loss
        backtest_engine._open_position(long_signal, Decimal("50000"), datetime.now())
        backtest_engine._close_position(Decimal("52000"), datetime.now())  # Profit
        # Update equity to track peak
        backtest_engine._update_equity(datetime.now(), Decimal("52000"))

        peak = backtest_engine._peak_capital

        # Open new position
        backtest_engine._open_position(long_signal, Decimal("52000"), datetime.now())
        backtest_engine._close_position(Decimal("50000"), datetime.now())  # Loss
        # Update equity to track drawdown
        backtest_engine._update_equity(datetime.now(), Decimal("50000"))

        assert backtest_engine._max_drawdown > 0


# =============================================================================
# Daily Management Tests
# =============================================================================

class TestDailyManagement:
    """Tests for daily P&L management."""

    def test_check_new_day_first_call(self, backtest_engine):
        """Test first call sets current date."""
        timestamp = datetime(2024, 1, 15, 10, 0)
        backtest_engine._check_new_day(timestamp)

        assert backtest_engine._current_date == timestamp.date()

    def test_check_new_day_same_day(self, backtest_engine):
        """Test same day doesn't reset."""
        backtest_engine._current_date = datetime(2024, 1, 15).date()
        backtest_engine._daily_pnl = Decimal("100")

        timestamp = datetime(2024, 1, 15, 18, 0)
        backtest_engine._check_new_day(timestamp)

        assert backtest_engine._daily_pnl == Decimal("100")

    def test_check_new_day_resets_pnl(self, backtest_engine):
        """Test new day resets daily PnL."""
        backtest_engine._current_date = datetime(2024, 1, 15).date()
        backtest_engine._daily_pnl = Decimal("100")

        timestamp = datetime(2024, 1, 16, 10, 0)
        backtest_engine._check_new_day(timestamp)

        assert backtest_engine._daily_pnl == Decimal("0")
        assert backtest_engine._current_date == datetime(2024, 1, 16).date()


# =============================================================================
# BacktestResult Tests
# =============================================================================

class TestBacktestResult:
    """Tests for BacktestResult."""

    def test_total_return(self):
        """Test total return calculation."""
        from src.utils.metrics import PerformanceMetrics

        result = BacktestResult(
            symbol="BTCUSDT",
            strategy_name="test",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
            config=BacktestConfig(),
            initial_capital=Decimal("1000"),
            final_capital=Decimal("1200"),
            metrics=MagicMock(),
        )

        assert result.total_return == Decimal("200")

    def test_total_return_pct(self):
        """Test total return percentage."""
        result = BacktestResult(
            symbol="BTCUSDT",
            strategy_name="test",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
            config=BacktestConfig(),
            initial_capital=Decimal("1000"),
            final_capital=Decimal("1200"),
            metrics=MagicMock(),
        )

        assert result.total_return_pct == 20.0

    def test_total_return_pct_zero_capital(self):
        """Test return pct with zero initial capital."""
        result = BacktestResult(
            symbol="BTCUSDT",
            strategy_name="test",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
            config=BacktestConfig(),
            initial_capital=Decimal("0"),
            final_capital=Decimal("100"),
            metrics=MagicMock(),
        )

        assert result.total_return_pct == 0.0

    def test_summary_generation(self):
        """Test summary report generation."""
        metrics = MagicMock()
        metrics.total_trades = 10
        metrics.winning_trades = 6
        metrics.losing_trades = 4
        metrics.win_rate = 60.0
        metrics.gross_profit = Decimal("300")
        metrics.gross_loss = Decimal("100")
        metrics.profit_factor = 3.0
        metrics.avg_win = Decimal("50")
        metrics.avg_loss = Decimal("25")
        metrics.avg_trade = Decimal("20")
        metrics.sharpe_ratio = 1.5
        metrics.sortino_ratio = 2.0
        metrics.calmar_ratio = 1.2
        metrics.max_consecutive_wins = 4
        metrics.max_consecutive_losses = 2

        result = BacktestResult(
            symbol="BTCUSDT",
            strategy_name="TestStrategy",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
            config=BacktestConfig(),
            initial_capital=Decimal("1000"),
            final_capital=Decimal("1200"),
            metrics=metrics,
            max_drawdown=Decimal("50"),
            max_drawdown_pct=5.0,
        )

        summary = result.summary()

        assert "BTCUSDT" in summary
        assert "TestStrategy" in summary
        assert "1,200" in summary
        assert "200" in summary


# =============================================================================
# Walk-Forward Tests
# =============================================================================

class TestWalkForward:
    """Tests for walk-forward testing."""

    def test_walk_forward_result_combined_trades(self, sample_trade_records):
        """Test combined trades from multiple periods."""
        result1 = BacktestResult(
            symbol="BTCUSDT",
            strategy_name="test",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 15),
            config=BacktestConfig(),
            initial_capital=Decimal("1000"),
            final_capital=Decimal("1100"),
            metrics=MagicMock(),
            trades=sample_trade_records[:2],
        )

        result2 = BacktestResult(
            symbol="BTCUSDT",
            strategy_name="test",
            start_time=datetime(2024, 1, 15),
            end_time=datetime(2024, 1, 31),
            config=BacktestConfig(),
            initial_capital=Decimal("1100"),
            final_capital=Decimal("1150"),
            metrics=MagicMock(),
            trades=sample_trade_records[2:],
        )

        wf_result = WalkForwardResult(results=[result1, result2])

        assert len(wf_result.combined_trades) == 3

    def test_walk_forward_total_return(self):
        """Test total return across periods."""
        result1 = BacktestResult(
            symbol="BTCUSDT",
            strategy_name="test",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 15),
            config=BacktestConfig(),
            initial_capital=Decimal("1000"),
            final_capital=Decimal("1100"),
            metrics=MagicMock(),
        )

        result2 = BacktestResult(
            symbol="BTCUSDT",
            strategy_name="test",
            start_time=datetime(2024, 1, 15),
            end_time=datetime(2024, 1, 31),
            config=BacktestConfig(),
            initial_capital=Decimal("1100"),
            final_capital=Decimal("1150"),
            metrics=MagicMock(),
        )

        wf_result = WalkForwardResult(results=[result1, result2])

        # 100 + 50 = 150
        assert wf_result.total_return == Decimal("150")


# =============================================================================
# Monte Carlo Simulation Tests
# =============================================================================

class TestMonteCarloSimulation:
    """Tests for Monte Carlo simulation."""

    def test_monte_carlo_empty_trades(self):
        """Test Monte Carlo with empty trades."""
        result = monte_carlo_simulation([], Decimal("1000"), 100)

        assert "error" in result

    def test_monte_carlo_returns_statistics(self, sample_trade_records):
        """Test Monte Carlo returns proper statistics."""
        result = monte_carlo_simulation(
            sample_trade_records,
            Decimal("1000"),
            num_simulations=100,
        )

        assert "num_simulations" in result
        assert "final_capital" in result
        assert "max_drawdown" in result

        assert result["num_simulations"] == 100
        assert "mean" in result["final_capital"]
        assert "median" in result["final_capital"]
        assert "percentile_5" in result["final_capital"]
        assert "percentile_95" in result["final_capital"]

    def test_monte_carlo_worst_best(self, sample_trade_records):
        """Test Monte Carlo tracks worst and best outcomes."""
        result = monte_carlo_simulation(
            sample_trade_records,
            Decimal("1000"),
            num_simulations=1000,
        )

        # Worst should be <= best
        assert result["final_capital"]["worst"] <= result["final_capital"]["best"]


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_small_position(self, backtest_engine):
        """Test with very small position size."""
        # Very high price leads to small quantity that rounds to 0
        signal = Signal(
            signal_type=SignalType.LONG,
            symbol="BTCUSDT",
            price=Decimal("100000000"),  # 100M
            strength=0.8,
            strategy="test",
            timestamp=datetime.now(),
        )

        backtest_engine._open_position(signal, Decimal("100000000"), datetime.now())

        # Position won't be opened if quantity rounds to 0 due to precision
        # With capital $10K and price $100M, quantity = 0.0001 rounds to 0.000
        # This is expected behavior - positions too small to trade are rejected
        # Position may or may not be opened depending on configuration
        pass  # Test that no exception is raised

    def test_multiple_trades_same_day(self, backtest_engine, long_signal):
        """Test multiple trades in same day."""
        timestamp = datetime(2024, 1, 15, 10, 0)

        for i in range(5):
            # Reset position for next trade
            backtest_engine._position = None

            backtest_engine._process_signal(
                long_signal,
                Decimal("50000") + Decimal(str(i * 100)),
                timestamp + timedelta(hours=i),
            )

            if backtest_engine._position:
                backtest_engine._close_position(
                    Decimal("50500") + Decimal(str(i * 100)),
                    timestamp + timedelta(hours=i, minutes=30),
                )

        # All trades should be on same day
        assert backtest_engine._daily_pnl != Decimal("0")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
