"""
Tests for Walk-Forward Optimization module.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any

from src.optimization.walk_forward import (
    WalkForwardOptimizer,
    WalkForwardConfig,
    WalkForwardWindow,
    WalkForwardResult,
    AutoRetrainer,
)
from src.optimization.optuna_optimizer import (
    StrategyOptimizer,
    OptimizationConfig,
    STRATEGY_PARAM_FUNCTIONS,
    get_orderbook_imbalance_params,
    get_volume_spike_params,
    get_grid_trading_params,
    get_mean_reversion_params,
)


# =============================================================================
# Walk-Forward Config Tests
# =============================================================================

class TestWalkForwardConfig:
    """Tests for WalkForwardConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WalkForwardConfig()
        assert config.train_period_days == 30
        assert config.test_period_days == 7
        assert config.step_days == 7
        assert config.optimization_trials == 50
        assert config.symbol == "BTCUSDT"

    def test_custom_config(self):
        """Test custom configuration."""
        config = WalkForwardConfig(
            train_period_days=14,
            test_period_days=3,
            optimization_trials=100,
            symbol="ETHUSDT",
        )
        assert config.train_period_days == 14
        assert config.test_period_days == 3
        assert config.optimization_trials == 100
        assert config.symbol == "ETHUSDT"


# =============================================================================
# Walk-Forward Window Tests
# =============================================================================

class TestWalkForwardWindow:
    """Tests for WalkForwardWindow."""

    def test_window_creation(self):
        """Test window creation."""
        start = datetime(2024, 1, 1)
        window = WalkForwardWindow(
            window_id=0,
            train_start=start,
            train_end=start + timedelta(days=30),
            test_start=start + timedelta(days=30),
            test_end=start + timedelta(days=37),
        )
        assert window.window_id == 0
        assert window.best_params is None
        assert window.test_trades == 0

    def test_window_with_results(self):
        """Test window with results."""
        start = datetime(2024, 1, 1)
        window = WalkForwardWindow(
            window_id=1,
            train_start=start,
            train_end=start + timedelta(days=30),
            test_start=start + timedelta(days=30),
            test_end=start + timedelta(days=37),
            best_params={"threshold": 1.5},
            test_metrics={"sharpe_ratio": 1.5, "total_return": 5.0},
            test_trades=25,
        )
        assert window.best_params == {"threshold": 1.5}
        assert window.test_trades == 25


# =============================================================================
# Walk-Forward Result Tests
# =============================================================================

class TestWalkForwardResult:
    """Tests for WalkForwardResult."""

    def test_result_creation(self):
        """Test result creation."""
        result = WalkForwardResult(
            strategy_name="test_strategy",
            config=WalkForwardConfig(),
            windows=[],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
        )
        assert result.strategy_name == "test_strategy"
        assert result.total_return == 0.0
        assert result.total_trades == 0

    def test_result_to_dict(self):
        """Test result serialization."""
        result = WalkForwardResult(
            strategy_name="test_strategy",
            config=WalkForwardConfig(),
            windows=[],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            total_return=10.5,
            avg_sharpe=1.2,
            windows_profitable=5,
        )
        data = result.to_dict()

        assert data["strategy"] == "test_strategy"
        assert data["total_return"] == 10.5
        assert data["avg_sharpe"] == 1.2
        assert data["windows_profitable"] == 5
        assert "start_date" in data
        assert "end_date" in data


# =============================================================================
# Walk-Forward Optimizer Tests
# =============================================================================

class TestWalkForwardOptimizer:
    """Tests for WalkForwardOptimizer."""

    @pytest.fixture
    def mock_strategy_class(self):
        """Create mock strategy class."""
        return MagicMock()

    @pytest.fixture
    def mock_param_fn(self):
        """Create mock parameter function."""
        def param_fn(trial):
            return {"threshold": 1.5}
        return param_fn

    @pytest.fixture
    def optimizer(self, mock_strategy_class, mock_param_fn):
        """Create optimizer instance."""
        config = WalkForwardConfig(
            train_period_days=7,
            test_period_days=3,
            step_days=3,
        )
        return WalkForwardOptimizer(
            strategy_name="test_strategy",
            strategy_class=mock_strategy_class,
            param_space_fn=mock_param_fn,
            config=config,
        )

    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.strategy_name == "test_strategy"
        assert optimizer.config.train_period_days == 7
        assert optimizer.config.test_period_days == 3

    def test_create_windows(self, optimizer):
        """Test window creation."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 2, 1)

        windows = optimizer._create_windows(start, end)

        assert len(windows) > 0
        for window in windows:
            assert window.train_end == window.test_start
            assert (window.train_end - window.train_start).days == 7
            assert (window.test_end - window.test_start).days == 3

    def test_create_windows_not_enough_data(self, optimizer):
        """Test window creation with insufficient data."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)  # Only 4 days

        windows = optimizer._create_windows(start, end)
        assert len(windows) == 0

    def test_get_latest_params(self, optimizer, mock_strategy_class, mock_param_fn):
        """Test getting latest parameters."""
        # Initially no params
        assert optimizer.get_latest_params() is None

        # Add window with params
        window = WalkForwardWindow(
            window_id=0,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 1, 8),
            test_start=datetime(2024, 1, 8),
            test_end=datetime(2024, 1, 11),
            best_params={"threshold": 2.0},
        )
        optimizer._windows = [window]

        assert optimizer.get_latest_params() == {"threshold": 2.0}

    def test_param_stability_empty(self, optimizer):
        """Test parameter stability with no results."""
        stability = optimizer.get_param_stability()
        assert stability == {}


# =============================================================================
# Auto Retrainer Tests
# =============================================================================

class TestAutoRetrainer:
    """Tests for AutoRetrainer."""

    @pytest.fixture
    def mock_optimizer(self):
        """Create mock optimizer."""
        optimizer = MagicMock()
        optimizer.get_latest_params.return_value = {"threshold": 1.5}
        return optimizer

    @pytest.fixture
    def retrainer(self, mock_optimizer):
        """Create retrainer instance."""
        return AutoRetrainer(
            optimizer=mock_optimizer,
            retrain_interval_days=7,
            performance_threshold=-0.05,
        )

    def test_should_retrain_first_time(self, retrainer):
        """Test retraining needed on first run."""
        assert retrainer.should_retrain(0.0) is True

    def test_should_retrain_after_interval(self, retrainer):
        """Test retraining after interval."""
        retrainer._last_retrain = datetime.now() - timedelta(days=8)
        assert retrainer.should_retrain(0.0) is True

    def test_should_not_retrain_within_interval(self, retrainer):
        """Test no retraining within interval."""
        retrainer._last_retrain = datetime.now() - timedelta(days=3)
        assert retrainer.should_retrain(0.0) is False

    def test_should_retrain_poor_performance(self, retrainer):
        """Test retraining on poor performance."""
        retrainer._last_retrain = datetime.now() - timedelta(days=1)
        assert retrainer.should_retrain(-0.10) is True  # -10% < -5% threshold

    def test_update_performance(self, retrainer):
        """Test performance tracking."""
        retrainer._performance_since_retrain = 0.0
        retrainer.update_performance(0.02)
        assert retrainer._performance_since_retrain == 0.02
        retrainer.update_performance(0.03)
        assert retrainer._performance_since_retrain == 0.05

    def test_days_since_retrain(self, retrainer):
        """Test days since retrain calculation."""
        assert retrainer.days_since_retrain == -1  # Never trained

        retrainer._last_retrain = datetime.now() - timedelta(days=5)
        assert retrainer.days_since_retrain == 5


# =============================================================================
# Optuna Optimizer Tests
# =============================================================================

class TestOptunaOptimizer:
    """Tests for StrategyOptimizer."""

    def test_param_functions_exist(self):
        """Test all param functions are defined."""
        assert "orderbook_imbalance" in STRATEGY_PARAM_FUNCTIONS
        assert "volume_spike" in STRATEGY_PARAM_FUNCTIONS
        assert "grid_trading" in STRATEGY_PARAM_FUNCTIONS
        assert "mean_reversion" in STRATEGY_PARAM_FUNCTIONS
        assert "combined" in STRATEGY_PARAM_FUNCTIONS

    def test_orderbook_imbalance_params(self):
        """Test orderbook imbalance parameter space."""
        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 1.5
        mock_trial.suggest_int.return_value = 5

        params = get_orderbook_imbalance_params(mock_trial)

        assert "imbalance_threshold" in params
        assert "max_spread" in params
        assert "levels" in params
        assert "signal_cooldown" in params
        assert params["enabled"] is True

    def test_volume_spike_params(self):
        """Test volume spike parameter space."""
        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 3.0
        mock_trial.suggest_int.return_value = 60

        params = get_volume_spike_params(mock_trial)

        assert "volume_multiplier" in params
        assert "lookback_seconds" in params
        assert "min_volume_usd" in params

    def test_grid_trading_params(self):
        """Test grid trading parameter space."""
        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 0.05
        mock_trial.suggest_int.return_value = 10
        mock_trial.suggest_categorical.return_value = "neutral"

        params = get_grid_trading_params(mock_trial)

        assert "range_percent" in params
        assert "grid_levels" in params
        assert "direction" in params
        assert "trailing_up" in params
        assert params["auto_range"] is True

    def test_mean_reversion_params(self):
        """Test mean reversion parameter space."""
        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 2.0
        mock_trial.suggest_int.return_value = 20
        mock_trial.suggest_categorical.return_value = True

        params = get_mean_reversion_params(mock_trial)

        assert "lookback_period" in params
        assert "std_dev_multiplier" in params
        assert "entry_z_score" in params
        assert "exit_z_score" in params
        assert "rsi_period" in params
        assert "use_rsi_confirmation" in params


class TestOptimizationConfig:
    """Tests for OptimizationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = OptimizationConfig()
        assert config.n_trials == 100
        assert config.objective_metric == "sharpe_ratio"
        assert config.symbol == "BTCUSDT"
        assert config.enable_pruning is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = OptimizationConfig(
            n_trials=50,
            objective_metric="profit_factor",
            symbol="ETHUSDT",
            train_days=7,
        )
        assert config.n_trials == 50
        assert config.objective_metric == "profit_factor"
        assert config.symbol == "ETHUSDT"
        assert config.train_days == 7


# =============================================================================
# Integration Tests
# =============================================================================

class TestOptimizationIntegration:
    """Integration tests for optimization module."""

    def test_window_dates_continuity(self):
        """Test that windows are continuous without gaps."""
        config = WalkForwardConfig(
            train_period_days=14,
            test_period_days=7,
            step_days=7,
        )

        optimizer = WalkForwardOptimizer(
            strategy_name="test",
            strategy_class=MagicMock(),
            param_space_fn=lambda t: {},
            config=config,
        )

        start = datetime(2024, 1, 1)
        end = datetime(2024, 3, 1)
        windows = optimizer._create_windows(start, end)

        # Check windows are sequential
        for i in range(len(windows) - 1):
            # Test end should be close to next train start (may overlap)
            diff = abs((windows[i+1].train_start - windows[i].train_start).days)
            assert diff == config.step_days

    def test_result_aggregation(self):
        """Test result aggregation."""
        windows = [
            WalkForwardWindow(
                window_id=0,
                train_start=datetime(2024, 1, 1),
                train_end=datetime(2024, 1, 15),
                test_start=datetime(2024, 1, 15),
                test_end=datetime(2024, 1, 22),
                test_metrics={"total_return": 5.0, "sharpe_ratio": 1.2},
                test_trades=10,
            ),
            WalkForwardWindow(
                window_id=1,
                train_start=datetime(2024, 1, 8),
                train_end=datetime(2024, 1, 22),
                test_start=datetime(2024, 1, 22),
                test_end=datetime(2024, 1, 29),
                test_metrics={"total_return": -2.0, "sharpe_ratio": 0.5},
                test_trades=8,
            ),
        ]

        result = WalkForwardResult(
            strategy_name="test",
            config=WalkForwardConfig(),
            windows=windows,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 2, 1),
            total_return=3.0,  # 5 - 2
            avg_sharpe=0.85,  # (1.2 + 0.5) / 2
            total_trades=18,  # 10 + 8
            windows_profitable=1,
        )

        data = result.to_dict()
        assert data["total_return"] == 3.0
        assert data["total_trades"] == 18
        assert data["windows_profitable"] == 1
        assert data["windows_total"] == 2
