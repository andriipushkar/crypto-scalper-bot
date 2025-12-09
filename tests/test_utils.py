"""
Unit tests for utility modules.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import tempfile

from src.utils.config import (
    load_config,
    validate_config,
    merge_configs,
)
from src.utils.metrics import (
    TradeRecord,
    PerformanceMetrics,
    calculate_metrics,
    format_metrics_report,
)


# =============================================================================
# Config Tests
# =============================================================================

class TestLoadConfig:
    """Tests for config loading."""

    def test_load_existing_config(self):
        # Uses actual config file
        config = load_config("config/settings.yaml")

        assert "exchange" in config
        assert "trading" in config

    def test_load_missing_config(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


class TestValidateConfig:
    """Tests for config validation."""

    def test_valid_config(self):
        config = {
            "exchange": {"testnet": True},
            "trading": {
                "symbols": ["BTCUSDT"],
                "leverage": 10,
                "margin_type": "CROSSED",
            },
            "strategies": {
                "orderbook_imbalance": {"enabled": True},
            },
        }

        errors = validate_config(config)
        assert len(errors) == 0

    def test_missing_exchange(self):
        config = {
            "trading": {"symbols": ["BTCUSDT"]},
        }

        errors = validate_config(config)
        assert any("exchange" in e for e in errors)

    def test_missing_symbols(self):
        config = {
            "exchange": {"testnet": True},
            "trading": {"symbols": []},
        }

        errors = validate_config(config)
        assert any("symbols" in e for e in errors)

    def test_invalid_leverage(self):
        config = {
            "exchange": {"testnet": True},
            "trading": {
                "symbols": ["BTCUSDT"],
                "leverage": 200,  # Too high
            },
            "strategies": {"test": {"enabled": True}},
        }

        errors = validate_config(config)
        assert any("leverage" in e for e in errors)

    def test_invalid_margin_type(self):
        config = {
            "exchange": {"testnet": True},
            "trading": {
                "symbols": ["BTCUSDT"],
                "margin_type": "INVALID",
            },
            "strategies": {"test": {"enabled": True}},
        }

        errors = validate_config(config)
        assert any("margin_type" in e for e in errors)

    def test_no_strategies_enabled(self):
        config = {
            "exchange": {"testnet": True},
            "trading": {"symbols": ["BTCUSDT"]},
            "strategies": {
                "test": {"enabled": False},
            },
        }

        errors = validate_config(config)
        assert any("strategy" in e.lower() for e in errors)


class TestMergeConfigs:
    """Tests for config merging."""

    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        result = merge_configs(base, override)

        assert result["a"] == 1
        assert result["b"] == 3
        assert result["c"] == 4

    def test_deep_merge(self):
        base = {
            "exchange": {"testnet": True, "timeout": 30},
            "trading": {"symbols": ["BTCUSDT"]},
        }
        override = {
            "exchange": {"testnet": False},
        }

        result = merge_configs(base, override)

        assert result["exchange"]["testnet"] == False
        assert result["exchange"]["timeout"] == 30
        assert result["trading"]["symbols"] == ["BTCUSDT"]

    def test_multiple_configs(self):
        a = {"x": 1}
        b = {"y": 2}
        c = {"z": 3}

        result = merge_configs(a, b, c)

        assert result == {"x": 1, "y": 2, "z": 3}


# =============================================================================
# Metrics Tests
# =============================================================================

class TestTradeRecord:
    """Tests for TradeRecord class."""

    @pytest.fixture
    def winning_trade(self):
        return TradeRecord(
            symbol="BTCUSDT",
            side="LONG",
            entry_time=datetime(2024, 1, 1, 12, 0, 0),
            exit_time=datetime(2024, 1, 1, 12, 0, 30),
            entry_price=Decimal("50000"),
            exit_price=Decimal("50100"),
            quantity=Decimal("0.1"),
            pnl=Decimal("10"),
            commission=Decimal("0.5"),
            strategy="test",
        )

    @pytest.fixture
    def losing_trade(self):
        return TradeRecord(
            symbol="BTCUSDT",
            side="LONG",
            entry_time=datetime(2024, 1, 1, 12, 0, 0),
            exit_time=datetime(2024, 1, 1, 12, 1, 0),
            entry_price=Decimal("50000"),
            exit_price=Decimal("49900"),
            quantity=Decimal("0.1"),
            pnl=Decimal("-10"),
            commission=Decimal("0.5"),
            strategy="test",
        )

    def test_duration(self, winning_trade):
        assert winning_trade.duration == timedelta(seconds=30)
        assert winning_trade.duration_seconds == 30

    def test_return_pct_long_win(self, winning_trade):
        # (50100 - 50000) / 50000 * 100 = 0.2%
        expected = Decimal("0.2")
        assert winning_trade.return_pct == expected

    def test_return_pct_long_loss(self, losing_trade):
        # (49900 - 50000) / 50000 * 100 = -0.2%
        expected = Decimal("-0.2")
        assert losing_trade.return_pct == expected

    def test_return_pct_short(self):
        trade = TradeRecord(
            symbol="BTCUSDT",
            side="SHORT",
            entry_time=datetime(2024, 1, 1, 12, 0, 0),
            exit_time=datetime(2024, 1, 1, 12, 0, 30),
            entry_price=Decimal("50000"),
            exit_price=Decimal("49900"),  # Price went down = win for short
            quantity=Decimal("0.1"),
            pnl=Decimal("10"),
            commission=Decimal("0.5"),
        )
        # (50000 - 49900) / 50000 * 100 = 0.2%
        expected = Decimal("0.2")
        assert trade.return_pct == expected

    def test_is_winner(self, winning_trade, losing_trade):
        assert winning_trade.is_winner == True
        assert losing_trade.is_winner == False

    def test_net_pnl(self, winning_trade):
        # 10 - 0.5 = 9.5
        assert winning_trade.net_pnl == Decimal("9.5")


class TestCalculateMetrics:
    """Tests for metrics calculation."""

    @pytest.fixture
    def sample_trades(self):
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        return [
            # Win
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
                strategy="imbalance",
            ),
            # Loss
            TradeRecord(
                symbol="BTCUSDT",
                side="LONG",
                entry_time=base_time + timedelta(minutes=1),
                exit_time=base_time + timedelta(minutes=1, seconds=45),
                entry_price=Decimal("50100"),
                exit_price=Decimal("50000"),
                quantity=Decimal("0.1"),
                pnl=Decimal("-10"),
                commission=Decimal("0.5"),
                strategy="imbalance",
            ),
            # Win
            TradeRecord(
                symbol="ETHUSDT",
                side="SHORT",
                entry_time=base_time + timedelta(minutes=2),
                exit_time=base_time + timedelta(minutes=2, seconds=20),
                entry_price=Decimal("3000"),
                exit_price=Decimal("2990"),
                quantity=Decimal("1"),
                pnl=Decimal("10"),
                commission=Decimal("0.5"),
                strategy="volume",
            ),
        ]

    def test_empty_trades(self):
        metrics = calculate_metrics([])
        assert metrics.total_trades == 0
        assert metrics.total_pnl == Decimal("0")

    def test_basic_counts(self, sample_trades):
        metrics = calculate_metrics(sample_trades)

        assert metrics.total_trades == 3
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 1

    def test_pnl_calculation(self, sample_trades):
        metrics = calculate_metrics(sample_trades)

        assert metrics.total_pnl == Decimal("10")  # 10 - 10 + 10
        assert metrics.gross_profit == Decimal("20")  # 10 + 10
        assert metrics.gross_loss == Decimal("10")  # abs(-10)
        assert metrics.total_commission == Decimal("1.5")  # 0.5 * 3

    def test_win_rate(self, sample_trades):
        metrics = calculate_metrics(sample_trades)

        # 2 wins / 3 trades = 66.67%
        assert abs(metrics.win_rate - 66.67) < 0.1

    def test_profit_factor(self, sample_trades):
        metrics = calculate_metrics(sample_trades)

        # gross_profit / gross_loss = 20 / 10 = 2.0
        assert metrics.profit_factor == 2.0

    def test_averages(self, sample_trades):
        metrics = calculate_metrics(sample_trades)

        assert metrics.avg_win == Decimal("10")  # 20 / 2
        assert metrics.avg_loss == Decimal("10")  # 10 / 1
        assert metrics.avg_trade == Decimal("10") / 3  # 10 / 3

    def test_time_metrics(self, sample_trades):
        metrics = calculate_metrics(sample_trades)

        # Trades: 30s, 45s, 20s = avg 31.67s
        assert abs(metrics.avg_trade_duration - 31.67) < 0.1

    def test_by_symbol(self, sample_trades):
        metrics = calculate_metrics(sample_trades)

        assert "BTCUSDT" in metrics.by_symbol
        assert "ETHUSDT" in metrics.by_symbol

        btc = metrics.by_symbol["BTCUSDT"]
        assert btc["trades"] == 2
        assert btc["wins"] == 1

        eth = metrics.by_symbol["ETHUSDT"]
        assert eth["trades"] == 1
        assert eth["wins"] == 1

    def test_by_strategy(self, sample_trades):
        metrics = calculate_metrics(sample_trades)

        assert "imbalance" in metrics.by_strategy
        assert "volume" in metrics.by_strategy

        imbalance = metrics.by_strategy["imbalance"]
        assert imbalance["trades"] == 2

    def test_streaks(self, sample_trades):
        metrics = calculate_metrics(sample_trades)

        # W, L, W
        assert metrics.max_consecutive_wins == 1
        assert metrics.max_consecutive_losses == 1
        assert metrics.current_streak == 1  # Last trade was a win

    def test_consecutive_wins(self):
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        trades = [
            TradeRecord(
                symbol="BTCUSDT", side="LONG",
                entry_time=base_time + timedelta(minutes=i),
                exit_time=base_time + timedelta(minutes=i, seconds=30),
                entry_price=Decimal("50000"), exit_price=Decimal("50100"),
                quantity=Decimal("0.1"), pnl=Decimal("10"), commission=Decimal("0.5"),
            )
            for i in range(5)  # 5 consecutive wins
        ]

        metrics = calculate_metrics(trades)

        assert metrics.max_consecutive_wins == 5
        assert metrics.current_streak == 5


class TestFormatMetricsReport:
    """Tests for report formatting."""

    def test_format_empty_metrics(self):
        metrics = PerformanceMetrics()
        report = format_metrics_report(metrics)

        assert "PERFORMANCE REPORT" in report
        assert "Total Trades:" in report

    def test_format_with_data(self):
        metrics = PerformanceMetrics(
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            total_pnl=Decimal("50"),
            win_rate=60.0,
            profit_factor=1.5,
        )

        report = format_metrics_report(metrics)

        assert "10" in report  # total trades
        assert "60.0%" in report  # win rate
        assert "1.50" in report  # profit factor
