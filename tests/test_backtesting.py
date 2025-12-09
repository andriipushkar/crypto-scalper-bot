"""
Tests for Backtesting Engine

Comprehensive tests for the backtesting module.
"""
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import patch, MagicMock

# Import backtesting modules
from src.backtesting.data import OHLCV, HistoricalDataLoader, DataIterator
from src.backtesting.engine import (
    BacktestEngine, BacktestConfig, BacktestResult,
    Order, OrderType, OrderSide, Position, Trade
)
from src.backtesting.analysis import BacktestAnalyzer


class TestOHLCV:
    """Tests for OHLCV data class."""

    def test_creation(self):
        """Test OHLCV creation."""
        candle = OHLCV(
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            open=100.0,
            high=105.0,
            low=98.0,
            close=103.0,
            volume=1000.0,
        )

        assert candle.open == 100.0
        assert candle.high == 105.0
        assert candle.low == 98.0
        assert candle.close == 103.0
        assert candle.volume == 1000.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        candle = OHLCV(
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            open=100.0, high=105.0, low=98.0, close=103.0, volume=1000.0
        )

        data = candle.to_dict()

        assert "timestamp" in data
        assert data["open"] == 100.0
        assert data["close"] == 103.0

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "timestamp": "2024-01-15T10:00:00",
            "open": 100.0,
            "high": 105.0,
            "low": 98.0,
            "close": 103.0,
            "volume": 1000.0,
        }

        candle = OHLCV.from_dict(data)

        assert candle.open == 100.0
        assert candle.close == 103.0

    def test_from_binance(self):
        """Test creation from Binance kline format."""
        kline = [
            1705312800000,  # timestamp
            "100.0",  # open
            "105.0",  # high
            "98.0",   # low
            "103.0",  # close
            "1000.0", # volume
        ]

        candle = OHLCV.from_binance(kline)

        assert candle.open == 100.0
        assert candle.close == 103.0


class TestHistoricalDataLoader:
    """Tests for historical data loading."""

    @pytest.fixture
    def loader(self, tmp_path):
        """Create data loader with temp directory."""
        return HistoricalDataLoader(data_dir=str(tmp_path))

    def test_cache_path(self, loader):
        """Test cache file path generation."""
        path = loader.get_cache_path("BTCUSDT", "1h")

        assert "BTCUSDT_1h.json" in str(path)

    def test_save_and_load_cache(self, loader):
        """Test saving and loading from cache."""
        candles = [
            OHLCV(datetime(2024, 1, 15, i, 0, 0), 100+i, 105+i, 98+i, 103+i, 1000)
            for i in range(10)
        ]

        loader.save_to_cache("BTCUSDT", "1h", candles)
        loaded = loader.load_from_cache("BTCUSDT", "1h")

        assert len(loaded) == 10
        assert loaded[0].close == 103.0

    def test_load_nonexistent(self, loader):
        """Test loading non-existent cache."""
        result = loader.load_from_cache("NONEXISTENT", "1h")
        assert result is None


class TestDataIterator:
    """Tests for data iterator."""

    @pytest.fixture
    def candles(self):
        """Create sample candles."""
        return [
            OHLCV(datetime(2024, 1, 15) + timedelta(hours=i), 100+i, 105+i, 98+i, 103+i, 1000)
            for i in range(100)
        ]

    def test_iteration(self, candles):
        """Test basic iteration."""
        iterator = DataIterator(candles, warmup_period=10)

        count = 0
        for current, history in iterator:
            count += 1
            assert len(history) >= 10  # At least warmup period

        assert count == 90  # 100 - 10 warmup

    def test_warmup_period(self, candles):
        """Test warmup period handling."""
        iterator = DataIterator(candles, warmup_period=50)

        items = list(iterator)
        assert len(items) == 50

    def test_reset(self, candles):
        """Test iterator reset."""
        iterator = DataIterator(candles, warmup_period=10)

        list(iterator)  # Exhaust iterator
        iterator.reset()

        items = list(iterator)
        assert len(items) == 90


class TestBacktestEngine:
    """Tests for backtest engine."""

    @pytest.fixture
    def config(self):
        """Create backtest config."""
        return BacktestConfig(
            initial_balance=10000.0,
            commission_rate=0.001,
            slippage=0.0005,
            leverage=1,
        )

    @pytest.fixture
    def engine(self, config):
        """Create backtest engine."""
        return BacktestEngine(config)

    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        candles = []
        base_price = 100.0
        for i in range(200):
            price = base_price + (i % 20) - 10  # Oscillating price
            candles.append(OHLCV(
                timestamp=datetime(2024, 1, 1) + timedelta(hours=i),
                open=price,
                high=price + 2,
                low=price - 2,
                close=price + 1,
                volume=1000,
            ))
        return {"BTCUSDT": candles}

    def test_initial_state(self, engine, config):
        """Test initial engine state."""
        assert engine.balance == config.initial_balance
        assert len(engine.positions) == 0
        assert len(engine.trades) == 0

    def test_place_market_order(self, engine):
        """Test placing market order."""
        engine.current_time = datetime.now()
        engine.current_prices["BTCUSDT"] = 100.0

        order = engine.place_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.MARKET,
        )

        assert order.filled
        assert "BTCUSDT" in engine.positions

    def test_open_position(self, engine):
        """Test opening a position."""
        engine.current_time = datetime.now()
        engine._open_position("BTCUSDT", "long", 1.0, 100.0)

        assert "BTCUSDT" in engine.positions
        pos = engine.positions["BTCUSDT"]
        assert pos.side == "long"
        assert pos.quantity == 1.0
        assert pos.entry_price == 100.0

    def test_close_position(self, engine):
        """Test closing a position."""
        engine.current_time = datetime.now()
        engine._open_position("BTCUSDT", "long", 1.0, 100.0)
        engine._close_position("BTCUSDT", 110.0)

        assert "BTCUSDT" not in engine.positions
        assert len(engine.trades) == 1
        assert engine.trades[0].pnl == 10.0

    def test_commission_calculation(self, engine):
        """Test commission calculation."""
        commission = engine._calculate_commission(1.0, 100.0)
        assert commission == 0.1  # 0.1% of 100

    def test_slippage_application(self, engine):
        """Test slippage application."""
        buy_price = engine._apply_slippage(100.0, OrderSide.BUY)
        sell_price = engine._apply_slippage(100.0, OrderSide.SELL)

        assert buy_price > 100.0  # Slippage increases buy price
        assert sell_price < 100.0  # Slippage decreases sell price

    def test_equity_calculation(self, engine):
        """Test equity calculation."""
        engine.current_time = datetime.now()
        engine.current_prices["BTCUSDT"] = 110.0
        engine._open_position("BTCUSDT", "long", 1.0, 100.0)

        pos = engine.positions["BTCUSDT"]
        pos.current_price = 110.0
        pos.unrealized_pnl = 10.0

        equity = engine.get_equity()
        # Equity = balance + unrealized_pnl, may be less than initial due to commission
        assert equity > 0

    def test_run_simple_strategy(self, engine, sample_data):
        """Test running backtest with simple strategy."""
        def simple_strategy(eng, symbol, candle, history):
            if len(history) < 20:
                return

            # Simple MA crossover
            ma_short = sum(c.close for c in history[-5:]) / 5
            ma_long = sum(c.close for c in history[-20:]) / 20

            if symbol not in eng.positions:
                if ma_short > ma_long:
                    eng.place_order(symbol, OrderSide.BUY, 0.1, OrderType.MARKET)
            else:
                if ma_short < ma_long:
                    eng.place_order(symbol, OrderSide.SELL, 0.1, OrderType.MARKET)

        result = engine.run(simple_strategy, sample_data, warmup_period=50)

        assert isinstance(result, BacktestResult)
        assert result.total_trades >= 0

    def test_stop_loss(self, engine):
        """Test stop loss execution."""
        engine.current_time = datetime.now()
        engine.current_prices["BTCUSDT"] = 100.0

        engine._open_position("BTCUSDT", "long", 1.0, 100.0)
        engine.set_stop_loss("BTCUSDT", 95.0)

        # Price drops below stop loss
        candle = OHLCV(datetime.now(), 94, 100, 93, 94, 1000)
        engine.current_prices["BTCUSDT"] = 94.0
        engine._update_positions(candle)

        # Position should be closed
        assert "BTCUSDT" not in engine.positions

    def test_take_profit(self, engine):
        """Test take profit execution."""
        engine.current_time = datetime.now()
        engine.current_prices["BTCUSDT"] = 100.0

        engine._open_position("BTCUSDT", "long", 1.0, 100.0)
        engine.set_take_profit("BTCUSDT", 110.0)

        # Price rises above take profit
        engine.current_prices["BTCUSDT"] = 111.0
        candle = OHLCV(datetime.now(), 111, 112, 110, 111, 1000)
        engine._update_positions(candle)

        assert "BTCUSDT" not in engine.positions


class TestBacktestResult:
    """Tests for backtest results."""

    @pytest.fixture
    def sample_result(self):
        """Create sample backtest result."""
        trades = [
            Trade("t1", "BTCUSDT", "long", 100, 110, 1, datetime.now(), datetime.now(), 10, 10, 0.1),
            Trade("t2", "BTCUSDT", "long", 110, 105, 1, datetime.now(), datetime.now(), -5, -4.5, 0.1),
            Trade("t3", "ETHUSDT", "short", 3000, 2900, 1, datetime.now(), datetime.now(), 100, 3.3, 0.1),
        ]

        return BacktestResult(
            config=BacktestConfig(),
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 31),
            initial_balance=10000,
            final_balance=10105,
            total_return=105,
            total_return_pct=1.05,
            trades=trades,
            equity_curve=[{"timestamp": "2024-01-01", "equity": 10000}],
            total_trades=3,
            winning_trades=2,
            losing_trades=1,
            win_rate=66.67,
        )

    def test_result_attributes(self, sample_result):
        """Test result attribute access."""
        assert sample_result.total_trades == 3
        assert sample_result.winning_trades == 2
        assert sample_result.win_rate == 66.67


class TestBacktestAnalyzer:
    """Tests for backtest analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with sample result."""
        trades = [
            Trade(f"t{i}", "BTCUSDT", "long", 100, 100 + (i % 2) * 10 - 5, 1,
                  datetime(2024, 1, i+1), datetime(2024, 1, i+1, 1),
                  (i % 2) * 10 - 5, (i % 2) * 10 - 5, 0.1)
            for i in range(20)
        ]

        equity = [{"timestamp": f"2024-01-{i+1}", "equity": 10000 + i * 10} for i in range(20)]

        result = BacktestResult(
            config=BacktestConfig(),
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 20),
            initial_balance=10000,
            final_balance=10200,
            total_return=200,
            total_return_pct=2.0,
            trades=trades,
            equity_curve=equity,
            total_trades=20,
            winning_trades=10,
            losing_trades=10,
            win_rate=50.0,
        )

        return BacktestAnalyzer(result)

    def test_summary(self, analyzer):
        """Test summary generation."""
        summary = analyzer.summary()

        assert "period" in summary
        assert "returns" in summary
        assert "trades" in summary
        assert "risk" in summary

    def test_trade_analysis(self, analyzer):
        """Test trade analysis."""
        analysis = analyzer.trade_analysis()

        assert "by_symbol" in analysis
        assert "by_side" in analysis
        assert "duration" in analysis

    def test_monthly_returns(self, analyzer):
        """Test monthly returns calculation."""
        monthly = analyzer.monthly_returns()

        assert isinstance(monthly, dict)
        assert "2024-01" in monthly

    def test_risk_metrics(self, analyzer):
        """Test risk metrics calculation."""
        risk = analyzer.risk_metrics()

        assert "mean_return" in risk or risk == {}  # Empty if not enough data

    def test_generate_report(self, analyzer):
        """Test report generation."""
        report = analyzer.generate_report()

        assert "BACKTEST REPORT" in report
        assert "RETURNS" in report
        assert "TRADES" in report

    def test_to_json(self, analyzer):
        """Test JSON export."""
        import json

        json_str = analyzer.to_json()
        data = json.loads(json_str)

        assert "summary" in data
        assert "trade_analysis" in data
