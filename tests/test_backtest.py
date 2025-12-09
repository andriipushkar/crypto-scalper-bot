"""
Unit tests for backtesting module.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

from src.backtest.engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    SimulatedPosition,
    monte_carlo_simulation,
)
from src.backtest.data_loader import HistoricalDataLoader, OHLCV
from src.data.models import Trade, Signal, SignalType, Side
from src.strategy.base import BaseStrategy
from src.utils.metrics import TradeRecord


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def backtest_config():
    return BacktestConfig(
        initial_capital=Decimal("100"),
        leverage=10,
        slippage_bps=1.0,
        commission_rate=0.0004,
        max_position_pct=0.5,
    )


@pytest.fixture
def backtest_engine(backtest_config):
    return BacktestEngine(backtest_config)


@pytest.fixture
def mock_strategy():
    strategy = Mock(spec=BaseStrategy)
    strategy.__class__.__name__ = "MockStrategy"
    strategy.generate_signal.return_value = None
    strategy.generate_signal_from_bars.return_value = None
    return strategy


@pytest.fixture
def sample_trades():
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    trades = []

    for i in range(100):
        trades.append(Trade(
            symbol="BTCUSDT",
            trade_id=str(i),
            price=Decimal("50000") + Decimal(str(i)),
            quantity=Decimal("0.1"),
            timestamp=base_time + timedelta(seconds=i),
            is_buyer_maker=i % 2 == 0,
        ))

    return trades


@pytest.fixture
def sample_bars():
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    bars = []

    for i in range(100):
        bar = OHLCV(
            timestamp=base_time + timedelta(minutes=i),
            open=Decimal("50000") + Decimal(str(i * 10)),
            high=Decimal("50050") + Decimal(str(i * 10)),
            low=Decimal("49950") + Decimal(str(i * 10)),
            close=Decimal("50020") + Decimal(str(i * 10)),
            volume=Decimal("10"),
            trades=50,
        )
        bars.append(bar)

    return bars


@pytest.fixture
def sample_trade_records():
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    return [
        TradeRecord(
            symbol="BTCUSDT",
            side="LONG",
            entry_time=base_time,
            exit_time=base_time + timedelta(seconds=30),
            entry_price=Decimal("50000"),
            exit_price=Decimal("50100"),
            quantity=Decimal("0.1"),
            pnl=Decimal("10"),
            commission=Decimal("0.5"),
        ),
        TradeRecord(
            symbol="BTCUSDT",
            side="LONG",
            entry_time=base_time + timedelta(minutes=1),
            exit_time=base_time + timedelta(minutes=1, seconds=30),
            entry_price=Decimal("50100"),
            exit_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            pnl=Decimal("-10"),
            commission=Decimal("0.5"),
        ),
        TradeRecord(
            symbol="BTCUSDT",
            side="SHORT",
            entry_time=base_time + timedelta(minutes=2),
            exit_time=base_time + timedelta(minutes=2, seconds=30),
            entry_price=Decimal("50000"),
            exit_price=Decimal("49900"),
            quantity=Decimal("0.1"),
            pnl=Decimal("10"),
            commission=Decimal("0.5"),
        ),
    ]


# =============================================================================
# BacktestConfig Tests
# =============================================================================

class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_default_values(self):
        config = BacktestConfig()

        assert config.initial_capital == Decimal("100")
        assert config.leverage == 10
        assert config.slippage_bps == 1.0
        assert config.commission_rate == 0.0004

    def test_custom_values(self, backtest_config):
        assert backtest_config.initial_capital == Decimal("100")
        assert backtest_config.leverage == 10
        assert backtest_config.max_position_pct == 0.5


# =============================================================================
# SimulatedPosition Tests
# =============================================================================

class TestSimulatedPosition:
    """Tests for SimulatedPosition."""

    @pytest.fixture
    def long_position(self):
        return SimulatedPosition(
            symbol="BTCUSDT",
            side=Side.BUY,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            entry_time=datetime.now(),
            leverage=10,
        )

    @pytest.fixture
    def short_position(self):
        return SimulatedPosition(
            symbol="BTCUSDT",
            side=Side.SELL,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            entry_time=datetime.now(),
            leverage=10,
        )

    def test_notional_value(self, long_position):
        # 50000 * 0.1 = 5000
        assert long_position.notional_value == Decimal("5000")

    def test_margin_required(self, long_position):
        # 5000 / 10 = 500
        assert long_position.margin_required == Decimal("500")

    def test_long_pnl_profit(self, long_position):
        # Price went up
        pnl = long_position.unrealized_pnl(Decimal("51000"))
        # (51000 - 50000) * 0.1 = 100
        assert pnl == Decimal("100")

    def test_long_pnl_loss(self, long_position):
        # Price went down
        pnl = long_position.unrealized_pnl(Decimal("49000"))
        # (49000 - 50000) * 0.1 = -100
        assert pnl == Decimal("-100")

    def test_short_pnl_profit(self, short_position):
        # Price went down = profit for short
        pnl = short_position.unrealized_pnl(Decimal("49000"))
        # (50000 - 49000) * 0.1 = 100
        assert pnl == Decimal("100")

    def test_short_pnl_loss(self, short_position):
        # Price went up = loss for short
        pnl = short_position.unrealized_pnl(Decimal("51000"))
        # (50000 - 51000) * 0.1 = -100
        assert pnl == Decimal("-100")

    def test_unrealized_pnl_pct(self, long_position):
        pct = long_position.unrealized_pnl_pct(Decimal("51000"))
        # 100 / 500 * 100 = 20%
        assert pct == 20.0


# =============================================================================
# BacktestEngine Tests
# =============================================================================

class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def test_initialization(self, backtest_engine, backtest_config):
        assert backtest_engine.config == backtest_config
        assert backtest_engine._capital == Decimal("100")
        assert backtest_engine._position is None

    def test_reset(self, backtest_engine):
        # Modify state
        backtest_engine._capital = Decimal("200")
        backtest_engine._trades = [Mock()]
        backtest_engine._signals_generated = 10

        # Reset
        backtest_engine.reset()

        assert backtest_engine._capital == Decimal("100")
        assert backtest_engine._trades == []
        assert backtest_engine._signals_generated == 0

    def test_apply_slippage_buy(self, backtest_engine):
        price = Decimal("50000")
        slipped = backtest_engine._apply_slippage(price, Side.BUY)

        # Slippage = 1 bps = 0.0001
        # 50000 * (1 + 0.0001) = 50005
        assert slipped == Decimal("50005")

    def test_apply_slippage_sell(self, backtest_engine):
        price = Decimal("50000")
        slipped = backtest_engine._apply_slippage(price, Side.SELL)

        # 50000 * (1 - 0.0001) = 49995
        assert slipped == Decimal("49995")

    def test_calculate_commission(self, backtest_engine):
        notional = Decimal("10000")
        commission = backtest_engine._calculate_commission(notional)

        # 10000 * 0.0004 = 4
        assert commission == Decimal("4")

    def test_calculate_position_size(self, backtest_engine):
        price = Decimal("50000")
        size = backtest_engine._calculate_position_size(price)

        # capital=100, leverage=10, max_position=0.5
        # max_value = 100 * 10 * 0.5 = 500
        # quantity = 500 / 50000 = 0.01
        assert size == Decimal("0.01")

    def test_can_open_position_empty(self, backtest_engine):
        signal = Mock()
        can_open = backtest_engine._can_open_position(signal, Decimal("50000"))
        assert can_open is True

    def test_can_open_position_with_existing(self, backtest_engine):
        backtest_engine._position = Mock()

        signal = Mock()
        can_open = backtest_engine._can_open_position(signal, Decimal("50000"))
        assert can_open is False

    def test_open_long_position(self, backtest_engine):
        signal = Signal(
            strategy="test",
            signal_type=SignalType.LONG,
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            strength=0.8,
            price=Decimal("50000"),
        )

        backtest_engine._open_position(signal, Decimal("50000"), datetime.now())

        assert backtest_engine._position is not None
        assert backtest_engine._position.side == Side.BUY
        assert backtest_engine._signals_executed == 1

    def test_open_short_position(self, backtest_engine):
        signal = Signal(
            strategy="test",
            signal_type=SignalType.SHORT,
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            strength=0.8,
            price=Decimal("50000"),
        )

        backtest_engine._open_position(signal, Decimal("50000"), datetime.now())

        assert backtest_engine._position is not None
        assert backtest_engine._position.side == Side.SELL

    def test_close_position(self, backtest_engine):
        # First open a position
        signal = Signal(
            strategy="test",
            signal_type=SignalType.LONG,
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            strength=0.8,
            price=Decimal("50000"),
        )
        backtest_engine._open_position(signal, Decimal("50000"), datetime.now())

        # Now close it
        trade = backtest_engine._close_position(
            Decimal("50100"),
            datetime.now() + timedelta(seconds=30),
        )

        assert backtest_engine._position is None
        assert trade is not None
        assert len(backtest_engine._trades) == 1

    def test_update_equity(self, backtest_engine):
        timestamp = datetime.now()

        backtest_engine._update_equity(timestamp, Decimal("50000"))

        assert len(backtest_engine._equity_curve) == 1
        assert backtest_engine._equity_curve[0]["equity"] == 100.0

    def test_drawdown_tracking(self, backtest_engine):
        # Simulate profit then loss
        timestamp = datetime.now()

        backtest_engine._capital = Decimal("110")  # 10% profit
        backtest_engine._update_equity(timestamp, Decimal("50000"))

        backtest_engine._capital = Decimal("90")  # Now below initial
        backtest_engine._update_equity(timestamp + timedelta(seconds=1), Decimal("50000"))

        # Drawdown should be from peak (110) to current (90) = 20
        assert backtest_engine._max_drawdown == Decimal("20")


class TestBacktestEngineSignalProcessing:
    """Tests for signal processing in BacktestEngine."""

    def test_process_no_action_signal(self, backtest_engine):
        signal = Signal(
            strategy="test",
            signal_type=SignalType.NO_ACTION,
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            strength=0.5,
            price=Decimal("50000"),
        )

        result = backtest_engine._process_signal(signal, Decimal("50000"), datetime.now())

        assert result is None
        assert backtest_engine._position is None

    def test_process_long_signal(self, backtest_engine):
        signal = Signal(
            strategy="test",
            signal_type=SignalType.LONG,
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            strength=0.8,
            price=Decimal("50000"),
        )

        backtest_engine._process_signal(signal, Decimal("50000"), datetime.now())

        assert backtest_engine._position is not None
        assert backtest_engine._position.side == Side.BUY

    def test_process_close_long_signal(self, backtest_engine):
        # Open long position
        long_signal = Signal(
            strategy="test",
            signal_type=SignalType.LONG,
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            strength=0.8,
            price=Decimal("50000"),
        )
        backtest_engine._process_signal(long_signal, Decimal("50000"), datetime.now())

        # Close it
        close_signal = Signal(
            strategy="test",
            signal_type=SignalType.CLOSE_LONG,
            symbol="BTCUSDT",
            timestamp=datetime.now() + timedelta(seconds=30),
            strength=0.8,
            price=Decimal("50100"),
        )
        backtest_engine._process_signal(
            close_signal,
            Decimal("50100"),
            datetime.now() + timedelta(seconds=30),
        )

        assert backtest_engine._position is None


# =============================================================================
# BacktestResult Tests
# =============================================================================

class TestBacktestResult:
    """Tests for BacktestResult."""

    @pytest.fixture
    def basic_result(self, backtest_config):
        from src.utils.metrics import PerformanceMetrics

        return BacktestResult(
            symbol="BTCUSDT",
            strategy_name="TestStrategy",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
            config=backtest_config,
            initial_capital=Decimal("100"),
            final_capital=Decimal("110"),
            metrics=PerformanceMetrics(
                total_trades=10,
                winning_trades=6,
                losing_trades=4,
                win_rate=60.0,
            ),
        )

    def test_total_return(self, basic_result):
        assert basic_result.total_return == Decimal("10")

    def test_total_return_pct(self, basic_result):
        assert basic_result.total_return_pct == 10.0

    def test_summary_generation(self, basic_result):
        summary = basic_result.summary()

        assert "BTCUSDT" in summary
        assert "TestStrategy" in summary
        assert "10" in summary  # trades
        assert "60" in summary  # win rate


# =============================================================================
# Data Loader Tests
# =============================================================================

class TestOHLCV:
    """Tests for OHLCV dataclass."""

    def test_creation(self):
        bar = OHLCV(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            open=Decimal("50000"),
            high=Decimal("50100"),
            low=Decimal("49900"),
            close=Decimal("50050"),
            volume=Decimal("100"),
            trades=500,
        )

        assert bar.open == Decimal("50000")
        assert bar.high == Decimal("50100")
        assert bar.trades == 500

    def test_default_values(self):
        bar = OHLCV(
            timestamp=datetime.now(),
            open=Decimal("1"),
            high=Decimal("2"),
            low=Decimal("0.5"),
            close=Decimal("1.5"),
            volume=Decimal("10"),
        )

        assert bar.trades == 0
        assert bar.buy_volume == Decimal("0")
        assert bar.sell_volume == Decimal("0")


class TestHistoricalDataLoader:
    """Tests for HistoricalDataLoader."""

    def test_initialization(self):
        loader = HistoricalDataLoader("data/test.db")
        assert loader.data_path.name == "test.db"

    def test_is_configured(self):
        loader = HistoricalDataLoader("data/market_data.db")
        assert loader.data_path.suffix == ".db"


# =============================================================================
# Monte Carlo Tests
# =============================================================================

class TestMonteCarlo:
    """Tests for Monte Carlo simulation."""

    def test_empty_trades(self):
        result = monte_carlo_simulation([], Decimal("100"))
        assert "error" in result

    def test_basic_simulation(self, sample_trade_records):
        result = monte_carlo_simulation(
            trades=sample_trade_records,
            initial_capital=Decimal("100"),
            num_simulations=100,
        )

        assert "num_simulations" in result
        assert result["num_simulations"] == 100
        assert "final_capital" in result
        assert "max_drawdown" in result

    def test_simulation_statistics(self, sample_trade_records):
        result = monte_carlo_simulation(
            trades=sample_trade_records,
            initial_capital=Decimal("100"),
            num_simulations=1000,
        )

        fc = result["final_capital"]
        assert "mean" in fc
        assert "median" in fc
        assert "percentile_5" in fc
        assert "percentile_95" in fc
        assert "worst" in fc
        assert "best" in fc

        # Worst should be <= median <= best
        assert fc["worst"] <= fc["median"] <= fc["best"]


# =============================================================================
# Integration Tests
# =============================================================================

class TestBacktestIntegration:
    """Integration tests for backtest flow."""

    def test_full_backtest_flow(self, backtest_engine, mock_strategy, sample_trades):
        """Test complete backtest with mocked data."""
        # Create a strategy that generates alternating signals
        signal_count = [0]

        def generate_signal(symbol, trades=None, orderbook=None):
            signal_count[0] += 1

            # Generate long on every 20th call
            if signal_count[0] % 20 == 0:
                return Signal(
                    strategy="mock",
                    signal_type=SignalType.LONG,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    strength=0.8,
                    price=Decimal("50000"),
                )

            # Generate close on every 30th call
            if signal_count[0] % 30 == 0:
                return Signal(
                    strategy="mock",
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    strength=0.8,
                    price=Decimal("50050"),
                )

            return None

        mock_strategy.generate_signal.side_effect = generate_signal

        # Create a mock loader
        with patch.object(HistoricalDataLoader, 'load_trades', return_value=sample_trades):
            with patch.object(HistoricalDataLoader, 'connect'):
                with patch.object(HistoricalDataLoader, 'close'):
                    result = backtest_engine.run(
                        strategy=mock_strategy,
                        symbol="BTCUSDT",
                        start=datetime(2024, 1, 1),
                        end=datetime(2024, 1, 2),
                        data_path="test.db",
                    )

        assert result is not None
        assert result.symbol == "BTCUSDT"
        assert result.initial_capital == Decimal("100")

    def test_run_on_bars(self, backtest_engine, mock_strategy, sample_bars):
        """Test backtest using OHLCV bars."""
        # Strategy that generates signals
        signal_count = [0]

        def generate_signal_from_bars(symbol, bars):
            signal_count[0] += 1

            if signal_count[0] == 10:
                return Signal(
                    strategy="mock",
                    signal_type=SignalType.LONG,
                    symbol=symbol,
                    timestamp=bars[-1].timestamp,
                    strength=0.8,
                    price=bars[-1].close,
                )

            if signal_count[0] == 20:
                return Signal(
                    strategy="mock",
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=symbol,
                    timestamp=bars[-1].timestamp,
                    strength=0.8,
                    price=bars[-1].close,
                )

            return None

        mock_strategy.generate_signal_from_bars.side_effect = generate_signal_from_bars

        result = backtest_engine.run_on_bars(
            strategy=mock_strategy,
            symbol="BTCUSDT",
            bars=sample_bars,
        )

        assert result is not None
        assert result.symbol == "BTCUSDT"
        # Should have at least some equity curve entries
        assert len(result.equity_curve) > 0
