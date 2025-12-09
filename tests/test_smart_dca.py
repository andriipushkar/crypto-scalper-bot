"""
Tests for Smart DCA Strategy module.

100% coverage for:
- MarketMetrics and data classes
- SmartDCAStrategy entry evaluation
- ML model training and scoring
- Position management
- Fear & Greed Index fetcher
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from src.strategy.smart_dca import (
    SmartDCAStrategy,
    SmartDCAConfig,
    SmartDCAState,
    MarketMetrics,
    DCAEntry,
    MarketCondition,
    EntryQuality,
    DCAMode,
    FearGreedFetcher,
)


# =============================================================================
# MarketMetrics Tests
# =============================================================================

class TestMarketMetrics:
    """Tests for MarketMetrics dataclass."""

    def test_metrics_creation_defaults(self):
        """Test metrics creation with defaults."""
        metrics = MarketMetrics()

        assert metrics.price == Decimal("0")
        assert metrics.rsi_14 == 50.0
        assert metrics.fear_greed_index == 50
        assert metrics.bb_position == 0.5
        assert metrics.timestamp is not None

    def test_metrics_creation_full(self):
        """Test metrics creation with all values."""
        metrics = MarketMetrics(
            price=Decimal("50000"),
            price_change_1h=-2.5,
            price_change_24h=-5.0,
            price_change_7d=-10.0,
            rsi_14=25.0,
            rsi_7=22.0,
            macd=-100,
            macd_signal=-50,
            macd_histogram=-50,
            bb_position=0.1,
            ema_9_distance=-1.5,
            ema_21_distance=-2.5,
            sma_50_distance=-5.0,
            sma_200_distance=-10.0,
            volume_24h=Decimal("1000000000"),
            volume_change=50.0,
            volume_sma_ratio=2.5,
            atr_14=1500,
            volatility_24h=3.5,
            fear_greed_index=15,
            fear_greed_change=-10,
            funding_rate=-0.02,
            exchange_inflow=1000,
            exchange_outflow=5000,
            whale_transactions=25,
        )

        assert metrics.price == Decimal("50000")
        assert metrics.rsi_14 == 25.0
        assert metrics.fear_greed_index == 15
        assert metrics.funding_rate == -0.02

    def test_to_feature_vector(self):
        """Test feature vector generation."""
        metrics = MarketMetrics(
            price_change_1h=-2.5,
            price_change_24h=-5.0,
            price_change_7d=-10.0,
            rsi_14=25.0,
            rsi_7=22.0,
            macd=-100,
            macd_signal=-50,
            macd_histogram=-50,
            bb_position=0.1,
            ema_9_distance=-1.5,
            ema_21_distance=-2.5,
            sma_50_distance=-5.0,
            sma_200_distance=-10.0,
            volume_change=50.0,
            volume_sma_ratio=2.5,
            atr_14=1500,
            volatility_24h=3.5,
            fear_greed_index=15,
            funding_rate=-0.02,
            exchange_inflow=1000,
            exchange_outflow=5000,
        )

        features = metrics.to_feature_vector()

        assert len(features) == 20
        assert features[0] == -2.5  # price_change_1h
        assert features[3] == 25.0  # rsi_14
        assert features[17] == 0.15  # fear_greed normalized
        assert features[18] == pytest.approx(-2.0)  # funding_rate scaled
        assert features[19] == -4000  # net flow


# =============================================================================
# SmartDCAConfig Tests
# =============================================================================

class TestSmartDCAConfig:
    """Tests for SmartDCAConfig."""

    def test_config_defaults(self):
        """Test default configuration."""
        config = SmartDCAConfig()

        assert config.base_amount == Decimal("100")
        assert config.max_entries == 10
        assert config.min_interval_hours == 4
        assert config.min_entry_score == 0.4
        assert config.mode == DCAMode.BALANCED

    def test_config_custom(self):
        """Test custom configuration."""
        config = SmartDCAConfig(
            base_amount=Decimal("500"),
            max_entries=20,
            min_entry_score=0.6,
            mode=DCAMode.CONSERVATIVE,
            take_profit_pct=0.20,
            stop_loss_pct=0.15,
        )

        assert config.base_amount == Decimal("500")
        assert config.max_entries == 20
        assert config.mode == DCAMode.CONSERVATIVE


# =============================================================================
# SmartDCAStrategy Tests
# =============================================================================

class TestSmartDCAStrategy:
    """Tests for SmartDCAStrategy class."""

    @pytest.fixture
    def strategy(self):
        """Create a SmartDCAStrategy instance."""
        config = SmartDCAConfig(
            base_amount=Decimal("100"),
            min_entry_score=0.4,
            mode=DCAMode.BALANCED,
        )
        return SmartDCAStrategy(symbol="BTCUSDT", config=config)

    @pytest.fixture
    def conservative_strategy(self):
        """Create a conservative strategy."""
        config = SmartDCAConfig(mode=DCAMode.CONSERVATIVE)
        return SmartDCAStrategy(symbol="BTCUSDT", config=config)

    @pytest.fixture
    def aggressive_strategy(self):
        """Create an aggressive strategy."""
        config = SmartDCAConfig(mode=DCAMode.AGGRESSIVE)
        return SmartDCAStrategy(symbol="BTCUSDT", config=config)

    # -------------------------------------------------------------------------
    # Market Condition Tests
    # -------------------------------------------------------------------------

    def test_get_market_condition_extreme_fear(self, strategy):
        """Test extreme fear classification."""
        metrics = MarketMetrics(fear_greed_index=10)
        condition = strategy.get_market_condition(metrics)
        assert condition == MarketCondition.EXTREME_FEAR

    def test_get_market_condition_fear(self, strategy):
        """Test fear classification."""
        metrics = MarketMetrics(fear_greed_index=30)
        condition = strategy.get_market_condition(metrics)
        assert condition == MarketCondition.FEAR

    def test_get_market_condition_neutral(self, strategy):
        """Test neutral classification."""
        metrics = MarketMetrics(fear_greed_index=50)
        condition = strategy.get_market_condition(metrics)
        assert condition == MarketCondition.NEUTRAL

    def test_get_market_condition_greed(self, strategy):
        """Test greed classification."""
        metrics = MarketMetrics(fear_greed_index=70)
        condition = strategy.get_market_condition(metrics)
        assert condition == MarketCondition.GREED

    def test_get_market_condition_extreme_greed(self, strategy):
        """Test extreme greed classification."""
        metrics = MarketMetrics(fear_greed_index=90)
        condition = strategy.get_market_condition(metrics)
        assert condition == MarketCondition.EXTREME_GREED

    # -------------------------------------------------------------------------
    # Rule-Based Scoring Tests
    # -------------------------------------------------------------------------

    def test_rule_based_score_oversold(self, strategy):
        """Test scoring for oversold conditions."""
        metrics = MarketMetrics(
            rsi_14=25,
            fear_greed_index=15,
            bb_position=0.1,
            price_change_24h=-12,  # Major dip
            volume_sma_ratio=2.5,
            funding_rate=-0.02,
            macd=-100,
            macd_histogram=10,  # Bullish divergence
        )

        score = strategy._rule_based_score(metrics)

        # Should have high score due to multiple bullish signals
        assert score >= 0.7

    def test_rule_based_score_overbought(self, strategy):
        """Test scoring for overbought conditions."""
        metrics = MarketMetrics(
            rsi_14=75,
            fear_greed_index=85,
            bb_position=0.9,
            price_change_24h=5,
            funding_rate=0.02,
            macd=100,
            macd_histogram=-10,  # Bearish divergence
        )

        score = strategy._rule_based_score(metrics)

        # Should have low score due to bearish signals
        assert score <= 0.4

    def test_rule_based_score_neutral(self, strategy):
        """Test scoring for neutral conditions."""
        metrics = MarketMetrics(
            rsi_14=50,
            fear_greed_index=50,
            bb_position=0.5,
        )

        score = strategy._rule_based_score(metrics)

        # Should be around 0.5
        assert 0.4 <= score <= 0.6

    def test_rule_based_score_clamps_to_range(self, strategy):
        """Test that score is clamped to 0-1 range."""
        # Extremely bullish
        metrics_bullish = MarketMetrics(
            rsi_14=10,
            fear_greed_index=5,
            bb_position=0.0,
            price_change_24h=-20,
            volume_sma_ratio=5.0,
            funding_rate=-0.1,
            macd=-500,
            macd_histogram=100,
        )

        score = strategy._rule_based_score(metrics_bullish)
        assert 0.0 <= score <= 1.0

        # Extremely bearish
        metrics_bearish = MarketMetrics(
            rsi_14=90,
            fear_greed_index=95,
            bb_position=1.0,
            funding_rate=0.1,
        )

        score = strategy._rule_based_score(metrics_bearish)
        assert 0.0 <= score <= 1.0

    # -------------------------------------------------------------------------
    # Entry Evaluation Tests
    # -------------------------------------------------------------------------

    def test_evaluate_entry_excellent(self, strategy):
        """Test excellent entry evaluation."""
        metrics = MarketMetrics(
            rsi_14=20,
            fear_greed_index=10,
            bb_position=0.05,
            price_change_24h=-15,
        )

        score, quality = strategy.evaluate_entry(metrics)

        assert score >= 0.8
        assert quality == EntryQuality.EXCELLENT

    def test_evaluate_entry_good(self, strategy):
        """Test good entry evaluation."""
        metrics = MarketMetrics(
            rsi_14=35,
            fear_greed_index=25,
            bb_position=0.15,
        )

        score, quality = strategy.evaluate_entry(metrics)

        assert 0.6 <= score < 0.8
        assert quality == EntryQuality.GOOD

    def test_evaluate_entry_average(self, strategy):
        """Test average entry evaluation."""
        metrics = MarketMetrics(
            rsi_14=50,
            fear_greed_index=45,
        )

        score, quality = strategy.evaluate_entry(metrics)

        assert 0.4 <= score < 0.6
        assert quality == EntryQuality.AVERAGE

    def test_evaluate_entry_poor(self, strategy):
        """Test poor entry evaluation."""
        metrics = MarketMetrics(
            rsi_14=65,
            fear_greed_index=70,
        )

        score, quality = strategy.evaluate_entry(metrics)

        assert 0.2 <= score < 0.4
        assert quality == EntryQuality.POOR

    def test_evaluate_entry_bad(self, strategy):
        """Test bad entry evaluation."""
        metrics = MarketMetrics(
            rsi_14=80,
            fear_greed_index=90,
            bb_position=0.95,
        )

        score, quality = strategy.evaluate_entry(metrics)

        assert score < 0.2
        assert quality == EntryQuality.BAD

    def test_evaluate_entry_conservative_mode(self, conservative_strategy):
        """Test that conservative mode reduces score."""
        metrics = MarketMetrics(
            rsi_14=40,
            fear_greed_index=40,
        )

        score_conservative, _ = conservative_strategy.evaluate_entry(metrics)

        # Create balanced strategy for comparison
        balanced = SmartDCAStrategy(
            symbol="BTCUSDT",
            config=SmartDCAConfig(mode=DCAMode.BALANCED),
        )
        score_balanced, _ = balanced.evaluate_entry(metrics)

        assert score_conservative < score_balanced

    def test_evaluate_entry_aggressive_mode(self, aggressive_strategy):
        """Test that aggressive mode increases score."""
        metrics = MarketMetrics(
            rsi_14=40,
            fear_greed_index=40,
        )

        score_aggressive, _ = aggressive_strategy.evaluate_entry(metrics)

        balanced = SmartDCAStrategy(
            symbol="BTCUSDT",
            config=SmartDCAConfig(mode=DCAMode.BALANCED),
        )
        score_balanced, _ = balanced.evaluate_entry(metrics)

        assert score_aggressive >= score_balanced

    # -------------------------------------------------------------------------
    # Entry Amount Calculation Tests
    # -------------------------------------------------------------------------

    def test_calculate_entry_amount_excellent(self, strategy):
        """Test entry amount for excellent quality."""
        amount = strategy.calculate_entry_amount(0.9, EntryQuality.EXCELLENT)

        # Base 100 * excellent multiplier 2.0 * (0.8 + 0.9 * 0.4) = 100 * 2.0 * 1.16 = 232
        assert amount > strategy.config.base_amount * Decimal("1.5")

    def test_calculate_entry_amount_good(self, strategy):
        """Test entry amount for good quality."""
        amount = strategy.calculate_entry_amount(0.7, EntryQuality.GOOD)

        # Base 100 * good multiplier 1.5 * (0.8 + 0.7 * 0.4)
        assert amount > strategy.config.base_amount

    def test_calculate_entry_amount_average(self, strategy):
        """Test entry amount for average quality."""
        amount = strategy.calculate_entry_amount(0.5, EntryQuality.AVERAGE)

        # Base 100 * average multiplier 1.0 * (0.8 + 0.5 * 0.4)
        assert amount == pytest.approx(Decimal("100"), abs=Decimal("20"))

    # -------------------------------------------------------------------------
    # Should Enter Tests
    # -------------------------------------------------------------------------

    def test_should_enter_success(self, strategy):
        """Test successful entry decision."""
        metrics = MarketMetrics(fear_greed_index=30)

        should, reason = strategy.should_enter(0.6, EntryQuality.GOOD, metrics)

        assert should is True
        assert reason == "Entry approved"

    def test_should_enter_max_entries_reached(self, strategy):
        """Test rejection when max entries reached."""
        strategy.state.entries = [MagicMock()] * 10  # Max entries
        metrics = MarketMetrics()

        should, reason = strategy.should_enter(0.8, EntryQuality.EXCELLENT, metrics)

        assert should is False
        assert "Max entries reached" in reason

    def test_should_enter_score_below_threshold(self, strategy):
        """Test rejection when score is too low."""
        metrics = MarketMetrics()

        should, reason = strategy.should_enter(0.3, EntryQuality.POOR, metrics)

        assert should is False
        assert "below threshold" in reason

    def test_should_enter_too_soon(self, strategy):
        """Test rejection when entry is too soon."""
        strategy.state.last_entry_time = datetime.utcnow() - timedelta(hours=1)
        metrics = MarketMetrics()

        should, reason = strategy.should_enter(0.6, EntryQuality.GOOD, metrics)

        assert should is False
        assert "Too soon" in reason

    def test_should_enter_conservative_rejects_average(self, conservative_strategy):
        """Test conservative mode rejects average quality."""
        metrics = MarketMetrics(fear_greed_index=50)

        should, reason = conservative_strategy.should_enter(
            0.5, EntryQuality.AVERAGE, metrics
        )

        assert should is False
        assert "Conservative mode" in reason

    def test_should_enter_rejects_in_greed(self, strategy):
        """Test rejection during greedy market for non-excellent entry."""
        metrics = MarketMetrics(fear_greed_index=75)

        should, reason = strategy.should_enter(0.6, EntryQuality.GOOD, metrics)

        assert should is False
        assert "greedy" in reason

    # -------------------------------------------------------------------------
    # Position Management Tests
    # -------------------------------------------------------------------------

    def test_record_entry(self, strategy):
        """Test recording a DCA entry."""
        metrics = MarketMetrics(fear_greed_index=30)

        entry = strategy.record_entry(
            price=Decimal("50000"),
            quantity=Decimal("0.002"),
            score=0.7,
            metrics=metrics,
        )

        assert entry.price == Decimal("50000")
        assert entry.quantity == Decimal("0.002")
        assert entry.entry_score == 0.7
        assert entry.market_condition == MarketCondition.FEAR

        assert strategy.state.is_active is True
        assert strategy.state.total_invested == Decimal("100")  # 50000 * 0.002
        assert strategy.state.total_quantity == Decimal("0.002")
        assert strategy.state.average_entry_price == Decimal("50000")
        assert len(strategy.state.entries) == 1

    def test_record_multiple_entries(self, strategy):
        """Test recording multiple DCA entries."""
        metrics = MarketMetrics()

        # First entry at 50000
        strategy.record_entry(
            price=Decimal("50000"),
            quantity=Decimal("0.002"),
            score=0.7,
            metrics=metrics,
        )

        # Second entry at 48000
        strategy.record_entry(
            price=Decimal("48000"),
            quantity=Decimal("0.002"),
            score=0.8,
            metrics=metrics,
        )

        assert strategy.state.total_invested == Decimal("196")  # 100 + 96
        assert strategy.state.total_quantity == Decimal("0.004")
        # Average = 196 / 0.004 = 49000
        assert strategy.state.average_entry_price == Decimal("49000")
        assert len(strategy.state.entries) == 2

    def test_update_position(self, strategy):
        """Test position update with current price."""
        metrics = MarketMetrics()
        strategy.record_entry(
            price=Decimal("50000"),
            quantity=Decimal("0.002"),
            score=0.7,
            metrics=metrics,
        )

        # Price goes up
        strategy.update_position(Decimal("52000"))

        # PnL = 52000 * 0.002 - 100 = 104 - 100 = 4
        assert strategy.state.unrealized_pnl == Decimal("4")
        assert strategy.state.unrealized_pnl_pct == pytest.approx(4.0)
        assert strategy.state.highest_price_since_entry == Decimal("52000")

    def test_update_position_inactive(self, strategy):
        """Test update position when not active."""
        strategy.update_position(Decimal("50000"))
        # Should not raise, just return
        assert strategy.state.unrealized_pnl == Decimal("0")

    # -------------------------------------------------------------------------
    # Take Profit Tests
    # -------------------------------------------------------------------------

    def test_should_take_profit_full(self, strategy):
        """Test full take profit trigger."""
        metrics = MarketMetrics()
        strategy.record_entry(
            price=Decimal("50000"),
            quantity=Decimal("0.002"),
            score=0.7,
            metrics=metrics,
        )

        # Price up 15% - config default TP
        should_tp, amount = strategy.should_take_profit(Decimal("57500"))

        assert should_tp is True
        assert amount == 1.0

    def test_should_take_profit_partial(self, strategy):
        """Test partial take profit trigger."""
        metrics = MarketMetrics()
        strategy.record_entry(
            price=Decimal("50000"),
            quantity=Decimal("0.002"),
            score=0.7,
            metrics=metrics,
        )

        # Price up 10% - config default partial TP
        should_tp, amount = strategy.should_take_profit(Decimal("55000"))

        assert should_tp is True
        assert amount == 0.5  # Default partial TP amount

    def test_should_take_profit_not_triggered(self, strategy):
        """Test take profit not triggered."""
        metrics = MarketMetrics()
        strategy.record_entry(
            price=Decimal("50000"),
            quantity=Decimal("0.002"),
            score=0.7,
            metrics=metrics,
        )

        # Price up only 5%
        should_tp, amount = strategy.should_take_profit(Decimal("52500"))

        assert should_tp is False
        assert amount == 0.0

    def test_should_take_profit_inactive(self, strategy):
        """Test take profit when inactive."""
        should_tp, amount = strategy.should_take_profit(Decimal("50000"))

        assert should_tp is False
        assert amount == 0.0

    # -------------------------------------------------------------------------
    # Stop Loss Tests
    # -------------------------------------------------------------------------

    def test_should_stop_loss_triggered(self, strategy):
        """Test stop loss trigger."""
        metrics = MarketMetrics()
        strategy.record_entry(
            price=Decimal("50000"),
            quantity=Decimal("0.002"),
            score=0.7,
            metrics=metrics,
        )

        # Price down 20% - config default SL
        should_sl = strategy.should_stop_loss(Decimal("40000"))

        assert should_sl is True

    def test_should_stop_loss_not_triggered(self, strategy):
        """Test stop loss not triggered."""
        metrics = MarketMetrics()
        strategy.record_entry(
            price=Decimal("50000"),
            quantity=Decimal("0.002"),
            score=0.7,
            metrics=metrics,
        )

        # Price down only 10%
        should_sl = strategy.should_stop_loss(Decimal("45000"))

        assert should_sl is False

    def test_should_stop_loss_inactive(self, strategy):
        """Test stop loss when inactive."""
        should_sl = strategy.should_stop_loss(Decimal("50000"))
        assert should_sl is False

    # -------------------------------------------------------------------------
    # Close Position Tests
    # -------------------------------------------------------------------------

    def test_close_position_full(self, strategy):
        """Test full position close."""
        metrics = MarketMetrics()
        strategy.record_entry(
            price=Decimal("50000"),
            quantity=Decimal("0.002"),
            score=0.7,
            metrics=metrics,
        )

        result = strategy.close_position(Decimal("55000"), amount_pct=1.0)

        assert result["quantity_closed"] == pytest.approx(0.002)
        assert result["exit_price"] == 55000.0
        assert result["pnl_pct"] == pytest.approx(10.0)
        assert result["entries_count"] == 1

        assert strategy.state.is_active is False
        assert strategy.state.total_quantity == Decimal("0")

    def test_close_position_partial(self, strategy):
        """Test partial position close."""
        metrics = MarketMetrics()
        strategy.record_entry(
            price=Decimal("50000"),
            quantity=Decimal("0.004"),
            score=0.7,
            metrics=metrics,
        )

        result = strategy.close_position(Decimal("55000"), amount_pct=0.5)

        assert result["quantity_closed"] == pytest.approx(0.002)
        assert strategy.state.is_active is True
        assert strategy.state.total_quantity == Decimal("0.002")

    def test_close_position_inactive(self, strategy):
        """Test close when inactive."""
        result = strategy.close_position(Decimal("50000"))
        assert result == {}

    # -------------------------------------------------------------------------
    # Signal Generation Tests
    # -------------------------------------------------------------------------

    def test_generate_entry_signal_success(self, strategy):
        """Test successful signal generation."""
        metrics = MarketMetrics(
            price=Decimal("50000"),
            fear_greed_index=25,
        )

        signal = strategy.generate_entry_signal(
            metrics=metrics,
            score=0.7,
            quality=EntryQuality.GOOD,
        )

        assert signal is not None
        assert signal.symbol == "BTCUSDT"
        assert signal.price == Decimal("50000")
        assert signal.strength == 0.7
        assert "smart_dca" in signal.strategy

    def test_generate_entry_signal_rejected(self, strategy):
        """Test signal rejection."""
        strategy.state.entries = [MagicMock()] * 10  # Max entries
        metrics = MarketMetrics(price=Decimal("50000"))

        signal = strategy.generate_entry_signal(
            metrics=metrics,
            score=0.7,
            quality=EntryQuality.GOOD,
        )

        assert signal is None

    # -------------------------------------------------------------------------
    # Statistics Tests
    # -------------------------------------------------------------------------

    def test_get_statistics(self, strategy):
        """Test statistics retrieval."""
        metrics = MarketMetrics()
        strategy.record_entry(
            price=Decimal("50000"),
            quantity=Decimal("0.002"),
            score=0.7,
            metrics=metrics,
        )

        stats = strategy.get_statistics()

        assert stats["symbol"] == "BTCUSDT"
        assert stats["mode"] == "balanced"
        assert stats["is_active"] is True
        assert stats["entries_count"] == 1
        assert stats["total_invested"] == 100.0
        assert stats["average_entry_price"] == 50000.0


# =============================================================================
# ML Model Tests (with sklearn available)
# =============================================================================

class TestSmartDCAML:
    """Tests for ML functionality."""

    @pytest.fixture
    def strategy(self):
        """Create strategy for ML tests."""
        return SmartDCAStrategy(
            symbol="BTCUSDT",
            config=SmartDCAConfig(),
        )

    def test_train_model_insufficient_data(self, strategy):
        """Test training with insufficient data."""
        features = [[0.1] * 20 for _ in range(50)]
        labels = [1] * 50

        accuracy = strategy.train_model(features, labels)

        assert accuracy == 0.0
        assert strategy.state.model_trained is False

    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="sklearn not installed"),
        reason="sklearn not installed"
    )
    def test_train_model_success(self, strategy):
        """Test successful model training."""
        # Generate enough training data
        np.random.seed(42)
        features = np.random.randn(200, 20).tolist()
        labels = [1 if sum(f) > 0 else 0 for f in features]

        accuracy = strategy.train_model(features, labels)

        assert accuracy > 0.0
        assert strategy.state.model_trained is True
        assert strategy._model is not None
        assert strategy._scaler is not None

    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="sklearn not installed"),
        reason="sklearn not installed"
    )
    def test_ml_score(self, strategy):
        """Test ML-based scoring."""
        # Train model first
        np.random.seed(42)
        features = np.random.randn(200, 20).tolist()
        labels = [1 if f[0] > 0 else 0 for f in features]
        strategy.train_model(features, labels)

        # Test scoring
        metrics = MarketMetrics(
            price_change_1h=1.0,
            rsi_14=30.0,
            fear_greed_index=25,
        )

        score = strategy._ml_score(metrics)

        assert 0.0 <= score <= 1.0


# =============================================================================
# Fear & Greed Fetcher Tests
# =============================================================================

class TestFearGreedFetcher:
    """Tests for FearGreedFetcher."""

    @pytest.fixture
    def fetcher(self):
        """Create a fetcher instance."""
        return FearGreedFetcher()

    @pytest.mark.asyncio
    async def test_fetch_success(self, fetcher):
        """Test successful fetch."""
        mock_response_data = {
            "data": [{"value": "25", "value_classification": "Extreme Fear"}]
        }

        with patch("aiohttp.ClientSession") as mock_session_class:
            # Create mock response
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)

            # Create context manager for response
            response_cm = MagicMock()
            response_cm.__aenter__ = AsyncMock(return_value=mock_response)
            response_cm.__aexit__ = AsyncMock(return_value=None)

            # Create mock session
            mock_session = MagicMock()
            mock_session.get = MagicMock(return_value=response_cm)

            # Create context manager for session
            session_cm = MagicMock()
            session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            session_cm.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = session_cm

            result = await fetcher.fetch()

            assert result["value"] == 25
            assert result["classification"] == "Extreme Fear"

    @pytest.mark.asyncio
    async def test_fetch_uses_cache(self, fetcher):
        """Test that fetcher uses cache."""
        fetcher._cache = {
            "value": 50,
            "classification": "Neutral",
            "timestamp": datetime.utcnow(),
        }
        fetcher._cache_time = datetime.utcnow()

        result = await fetcher.fetch()

        assert result["value"] == 50

    @pytest.mark.asyncio
    async def test_fetch_error_returns_default(self, fetcher):
        """Test fetch error returns default value."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            # Create mock session that raises error on get
            mock_session = MagicMock()
            mock_session.get = MagicMock(side_effect=Exception("Network error"))

            session_cm = MagicMock()
            session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            session_cm.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = session_cm

            result = await fetcher.fetch()

            assert result["value"] == 50  # Default

    def test_get_cached_value(self, fetcher):
        """Test getting cached value."""
        assert fetcher.get_cached_value() == 50  # Default when no cache

        fetcher._cache = {"value": 25}
        assert fetcher.get_cached_value() == 25


# =============================================================================
# DCAEntry Tests
# =============================================================================

class TestDCAEntry:
    """Tests for DCAEntry dataclass."""

    def test_entry_creation(self):
        """Test entry creation."""
        entry = DCAEntry(
            entry_id="test123",
            timestamp=datetime.utcnow(),
            price=Decimal("50000"),
            quantity=Decimal("0.002"),
            entry_score=0.75,
            market_condition=MarketCondition.FEAR,
            notes="Test entry",
        )

        assert entry.entry_id == "test123"
        assert entry.price == Decimal("50000")
        assert entry.entry_score == 0.75
        assert entry.market_condition == MarketCondition.FEAR


# =============================================================================
# Orderbook Handler Tests
# =============================================================================

class TestOrderbookHandler:
    """Tests for orderbook handling."""

    def test_on_orderbook_returns_none(self):
        """Test that on_orderbook returns None (not used by Smart DCA)."""
        strategy = SmartDCAStrategy(symbol="BTCUSDT")

        result = strategy.on_orderbook(MagicMock())

        assert result is None
