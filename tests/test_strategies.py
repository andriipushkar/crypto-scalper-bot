"""
Unit tests for trading strategies.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from src.strategy.base import BaseStrategy, StrategyState
from src.strategy.orderbook_imbalance import OrderBookImbalanceStrategy
from src.strategy.volume_spike import VolumeSpikeStrategy, VolumeBar
from src.data.models import (
    OrderBookSnapshot,
    OrderBookLevel,
    Trade,
    Signal,
    SignalType,
    Side,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def balanced_orderbook():
    """Order book with equal bid/ask volumes."""
    bids = [
        OrderBookLevel(Decimal("50000"), Decimal("1.0")),
        OrderBookLevel(Decimal("49999"), Decimal("1.0")),
        OrderBookLevel(Decimal("49998"), Decimal("1.0")),
        OrderBookLevel(Decimal("49997"), Decimal("1.0")),
        OrderBookLevel(Decimal("49996"), Decimal("1.0")),
    ]
    asks = [
        OrderBookLevel(Decimal("50001"), Decimal("1.0")),
        OrderBookLevel(Decimal("50002"), Decimal("1.0")),
        OrderBookLevel(Decimal("50003"), Decimal("1.0")),
        OrderBookLevel(Decimal("50004"), Decimal("1.0")),
        OrderBookLevel(Decimal("50005"), Decimal("1.0")),
    ]
    return OrderBookSnapshot(
        symbol="BTCUSDT",
        timestamp=datetime.utcnow(),
        bids=bids,
        asks=asks,
        last_update_id=12345,
    )


@pytest.fixture
def bullish_orderbook():
    """Order book with more bids than asks (bullish)."""
    bids = [
        OrderBookLevel(Decimal("50000"), Decimal("5.0")),  # 5x more
        OrderBookLevel(Decimal("49999"), Decimal("5.0")),
        OrderBookLevel(Decimal("49998"), Decimal("5.0")),
        OrderBookLevel(Decimal("49997"), Decimal("5.0")),
        OrderBookLevel(Decimal("49996"), Decimal("5.0")),
    ]
    asks = [
        OrderBookLevel(Decimal("50001"), Decimal("1.0")),
        OrderBookLevel(Decimal("50002"), Decimal("1.0")),
        OrderBookLevel(Decimal("50003"), Decimal("1.0")),
        OrderBookLevel(Decimal("50004"), Decimal("1.0")),
        OrderBookLevel(Decimal("50005"), Decimal("1.0")),
    ]
    return OrderBookSnapshot(
        symbol="BTCUSDT",
        timestamp=datetime.utcnow(),
        bids=bids,
        asks=asks,
        last_update_id=12345,
    )


@pytest.fixture
def bearish_orderbook():
    """Order book with more asks than bids (bearish)."""
    bids = [
        OrderBookLevel(Decimal("50000"), Decimal("1.0")),
        OrderBookLevel(Decimal("49999"), Decimal("1.0")),
        OrderBookLevel(Decimal("49998"), Decimal("1.0")),
        OrderBookLevel(Decimal("49997"), Decimal("1.0")),
        OrderBookLevel(Decimal("49996"), Decimal("1.0")),
    ]
    asks = [
        OrderBookLevel(Decimal("50001"), Decimal("5.0")),  # 5x more
        OrderBookLevel(Decimal("50002"), Decimal("5.0")),
        OrderBookLevel(Decimal("50003"), Decimal("5.0")),
        OrderBookLevel(Decimal("50004"), Decimal("5.0")),
        OrderBookLevel(Decimal("50005"), Decimal("5.0")),
    ]
    return OrderBookSnapshot(
        symbol="BTCUSDT",
        timestamp=datetime.utcnow(),
        bids=bids,
        asks=asks,
        last_update_id=12345,
    )


@pytest.fixture
def wide_spread_orderbook():
    """Order book with wide spread."""
    bids = [
        OrderBookLevel(Decimal("49900"), Decimal("5.0")),  # Wide spread
    ]
    asks = [
        OrderBookLevel(Decimal("50100"), Decimal("5.0")),  # 0.4% spread
    ]
    return OrderBookSnapshot(
        symbol="BTCUSDT",
        timestamp=datetime.utcnow(),
        bids=bids,
        asks=asks,
        last_update_id=12345,
    )


# =============================================================================
# OrderBookImbalanceStrategy Tests
# =============================================================================

class TestOrderBookImbalanceStrategy:
    """Tests for OrderBookImbalanceStrategy."""

    @pytest.fixture
    def strategy(self):
        config = {
            "enabled": True,
            "imbalance_threshold": 1.5,
            "max_spread": 0.001,  # 0.1%
            "levels": 5,
            "min_volume_usd": 1000,
            "signal_cooldown": 0,  # No cooldown for tests
            "min_strength": 0.0,
        }
        return OrderBookImbalanceStrategy(config)

    def test_init(self, strategy):
        assert strategy.enabled == True
        assert strategy.imbalance_threshold == Decimal("1.5")
        assert strategy.levels == 5

    def test_no_signal_on_balanced_book(self, strategy, balanced_orderbook):
        signal = strategy.on_orderbook(balanced_orderbook)
        assert signal is None

    def test_long_signal_on_bullish_book(self, strategy, bullish_orderbook):
        signal = strategy.on_orderbook(bullish_orderbook)

        assert signal is not None
        assert signal.signal_type == SignalType.LONG
        assert signal.symbol == "BTCUSDT"
        assert signal.strength > 0

    def test_short_signal_on_bearish_book(self, strategy, bearish_orderbook):
        signal = strategy.on_orderbook(bearish_orderbook)

        assert signal is not None
        assert signal.signal_type == SignalType.SHORT
        assert signal.symbol == "BTCUSDT"

    def test_no_signal_on_wide_spread(self, strategy, wide_spread_orderbook):
        signal = strategy.on_orderbook(wide_spread_orderbook)
        assert signal is None

    def test_cooldown_enforcement(self, bullish_orderbook):
        config = {
            "enabled": True,
            "imbalance_threshold": 1.5,
            "max_spread": 0.001,
            "levels": 5,
            "min_volume_usd": 1000,
            "signal_cooldown": 60,  # 60 second cooldown
            "min_strength": 0.0,
        }
        strategy = OrderBookImbalanceStrategy(config)

        # First signal should work
        signal1 = strategy.on_orderbook(bullish_orderbook)
        assert signal1 is not None

        # Second signal should be blocked by cooldown
        signal2 = strategy.on_orderbook(bullish_orderbook)
        assert signal2 is None

    def test_signal_metadata(self, strategy, bullish_orderbook):
        signal = strategy.on_orderbook(bullish_orderbook)

        assert signal is not None
        assert "imbalance" in signal.metadata
        assert "bid_volume" in signal.metadata
        assert "ask_volume" in signal.metadata
        assert "spread_bps" in signal.metadata

    def test_disabled_strategy(self, bullish_orderbook):
        config = {
            "enabled": False,
            "imbalance_threshold": 1.5,
        }
        strategy = OrderBookImbalanceStrategy(config)

        signal = strategy.on_orderbook(bullish_orderbook)
        assert signal is None

    def test_history_tracking(self, strategy, balanced_orderbook):
        # Process multiple order books
        for _ in range(5):
            strategy.on_orderbook(balanced_orderbook)

        assert len(strategy._history) == 5

    def test_get_current_imbalance(self, strategy, bullish_orderbook):
        strategy.on_orderbook(bullish_orderbook)

        imbalance = strategy.get_current_imbalance()
        assert imbalance is not None
        assert imbalance > Decimal("1")  # Bullish imbalance

    def test_reset(self, strategy, bullish_orderbook):
        strategy.on_orderbook(bullish_orderbook)
        assert len(strategy._history) > 0

        strategy.reset()

        assert len(strategy._history) == 0
        assert strategy._state.signals_generated == 0


# =============================================================================
# VolumeSpikeStrategy Tests
# =============================================================================

class TestVolumeSpikeStrategy:
    """Tests for VolumeSpikeStrategy."""

    @pytest.fixture
    def strategy(self):
        config = {
            "enabled": True,
            "volume_multiplier": 3.0,
            "lookback_seconds": 60,
            "min_volume_usd": 100,
            "flow_threshold": 1.5,
            "signal_cooldown": 0,
            "min_strength": 0.0,
        }
        return VolumeSpikeStrategy(config)

    def test_init(self, strategy):
        assert strategy.enabled == True
        assert strategy.volume_multiplier == 3.0
        assert strategy.lookback_seconds == 60

    def test_no_signal_without_history(self, strategy, balanced_orderbook):
        signal = strategy.on_orderbook(balanced_orderbook)
        assert signal is None

    def test_trade_accumulation(self, strategy):
        # Create a buy trade
        trade = Trade(
            symbol="BTCUSDT",
            trade_id=1,
            price=Decimal("50000"),
            quantity=Decimal("1.0"),
            timestamp=datetime.utcnow(),
            is_buyer_maker=False,  # Buy
        )

        strategy.on_trade(trade)

        assert strategy._current_bar is not None
        assert strategy._current_bar.buy_volume == Decimal("1.0")
        assert strategy._current_bar.sell_volume == Decimal("0")

    def test_volume_bar_properties(self):
        bar = VolumeBar(
            timestamp=datetime.utcnow(),
            buy_volume=Decimal("10"),
            sell_volume=Decimal("5"),
            buy_value=Decimal("500000"),
            sell_value=Decimal("250000"),
            trade_count=100,
        )

        assert bar.total_volume == Decimal("15")
        assert bar.total_value == Decimal("750000")
        assert bar.net_flow == Decimal("250000")
        assert bar.flow_ratio == Decimal("2")

    def test_spike_detection(self, strategy):
        # Build up history with small volume
        for i in range(20):
            bar = VolumeBar(
                timestamp=datetime.utcnow() - timedelta(seconds=60-i),
                buy_value=Decimal("1000"),
                sell_value=Decimal("1000"),
            )
            strategy._volume_history.append(bar)

        # Create a spike (10x average with strong buy flow)
        strategy._current_bar = VolumeBar(
            timestamp=datetime.utcnow(),
            buy_value=Decimal("15000"),  # 7.5x
            sell_value=Decimal("5000"),   # Total 10x
        )
        strategy._last_mid_price = Decimal("50000")
        strategy._last_symbol = "BTCUSDT"

        signal = strategy._check_volume_spike()

        assert signal is not None
        assert signal.signal_type == SignalType.LONG  # Buy pressure

    def test_no_spike_normal_volume(self, strategy):
        # Build up history
        for i in range(20):
            bar = VolumeBar(
                timestamp=datetime.utcnow() - timedelta(seconds=60-i),
                buy_value=Decimal("1000"),
                sell_value=Decimal("1000"),
            )
            strategy._volume_history.append(bar)

        # Normal volume (not a spike)
        strategy._current_bar = VolumeBar(
            timestamp=datetime.utcnow(),
            buy_value=Decimal("1200"),
            sell_value=Decimal("800"),
        )
        strategy._last_mid_price = Decimal("50000")

        signal = strategy._check_volume_spike()
        assert signal is None

    def test_get_average_volume(self, strategy):
        for i in range(10):
            bar = VolumeBar(
                timestamp=datetime.utcnow() - timedelta(seconds=i),
                buy_value=Decimal("1000"),
                sell_value=Decimal("1000"),
            )
            strategy._volume_history.append(bar)

        avg = strategy.get_average_volume()
        assert avg == Decimal("2000")

    def test_reset(self, strategy):
        strategy._volume_history.append(VolumeBar(timestamp=datetime.utcnow()))
        strategy._current_bar = VolumeBar(timestamp=datetime.utcnow())

        strategy.reset()

        assert len(strategy._volume_history) == 0
        assert strategy._current_bar is None


# =============================================================================
# BaseStrategy Tests
# =============================================================================

class TestBaseStrategy:
    """Tests for BaseStrategy base class functionality."""

    def test_strategy_state(self):
        state = StrategyState()

        assert state.last_signal_time is None
        assert state.signals_generated == 0
        assert state.errors == 0

    def test_can_signal_first_time(self):
        config = {"enabled": True, "signal_cooldown": 10}
        strategy = OrderBookImbalanceStrategy(config)

        assert strategy.can_signal() == True

    def test_can_signal_cooldown(self):
        config = {"enabled": True, "signal_cooldown": 10}
        strategy = OrderBookImbalanceStrategy(config)

        # Simulate a signal
        strategy._state.last_signal_time = datetime.utcnow()

        assert strategy.can_signal() == False

    def test_can_signal_after_cooldown(self):
        config = {"enabled": True, "signal_cooldown": 1}
        strategy = OrderBookImbalanceStrategy(config)

        # Signal 2 seconds ago
        strategy._state.last_signal_time = datetime.utcnow() - timedelta(seconds=2)

        assert strategy.can_signal() == True

    def test_stats(self):
        config = {"enabled": True}
        strategy = OrderBookImbalanceStrategy(config)

        stats = strategy.stats

        assert "name" in stats
        assert "enabled" in stats
        assert "signals_generated" in stats
        assert stats["name"] == "OrderBookImbalanceStrategy"


# =============================================================================
# Arbitrage Strategy Tests
# =============================================================================

class TestArbitrageDetector:
    """Tests for arbitrage detection."""

    @pytest.fixture
    def detector(self):
        """Create arbitrage detector."""
        from src.strategies.arbitrage import ArbitrageDetector
        return ArbitrageDetector(
            exchanges=["binance", "bybit"],
            min_profit_pct=Decimal("0.1"),
            fee_rate=Decimal("0.001"),
        )

    def test_update_price(self, detector):
        """Test price update."""
        detector.update_price(
            exchange="binance",
            symbol="BTCUSDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            volume=Decimal("100"),
        )

        assert "binance" in detector.prices
        assert "BTCUSDT" in detector.prices["binance"]
        assert detector.prices["binance"]["BTCUSDT"]["bid"] == Decimal("50000")

    def test_scan_cross_exchange_no_opportunity(self, detector):
        """Test scan with no arbitrage opportunity."""
        # Same prices on both exchanges
        detector.update_price("binance", "BTCUSDT", Decimal("50000"), Decimal("50010"), Decimal("100"))
        detector.update_price("bybit", "BTCUSDT", Decimal("50000"), Decimal("50010"), Decimal("100"))

        opportunities = detector.scan_cross_exchange("BTCUSDT")

        # Should be empty or unprofitable
        profitable = [o for o in opportunities if o.is_profitable()]
        assert len(profitable) == 0

    def test_scan_cross_exchange_with_opportunity(self, detector):
        """Test scan with arbitrage opportunity."""
        # Buy low on binance, sell high on bybit
        # Need spread_pct > fee_pct(0.2%) + slippage(0.1%) + min_profit(0.1%) = 0.4%
        # spread >= 50010 * 0.004 = 200
        detector.update_price("binance", "BTCUSDT", Decimal("50000"), Decimal("50010"), Decimal("100"))
        detector.update_price("bybit", "BTCUSDT", Decimal("50250"), Decimal("50260"), Decimal("100"))

        opportunities = detector.scan_cross_exchange("BTCUSDT")

        # Should find opportunity (spread = 240, spread_pct = 0.48%)
        assert len(opportunities) > 0

    def test_opportunity_profitability_check(self, detector):
        """Test opportunity profitability calculation."""
        from src.strategies.arbitrage import ArbitrageOpportunity, ArbitrageType

        opp = ArbitrageOpportunity(
            id="test_1",
            arb_type=ArbitrageType.CROSS_EXCHANGE,
            symbol="BTCUSDT",
            exchanges=["binance", "bybit"],
            buy_exchange="binance",
            sell_exchange="bybit",
            buy_price=Decimal("50000"),
            sell_price=Decimal("50100"),
            spread=Decimal("100"),
            spread_pct=Decimal("0.2"),
            estimated_profit=Decimal("10"),
            estimated_profit_pct=Decimal("0.15"),
            volume_available=Decimal("1"),
        )

        assert opp.is_profitable(Decimal("0.1"))
        assert not opp.is_profitable(Decimal("0.2"))

    def test_opportunity_to_dict(self, detector):
        """Test opportunity serialization."""
        from src.strategies.arbitrage import ArbitrageOpportunity, ArbitrageType

        opp = ArbitrageOpportunity(
            id="test_1",
            arb_type=ArbitrageType.CROSS_EXCHANGE,
            symbol="BTCUSDT",
            exchanges=["binance", "bybit"],
            buy_exchange="binance",
            sell_exchange="bybit",
            buy_price=Decimal("50000"),
            sell_price=Decimal("50100"),
            spread=Decimal("100"),
            spread_pct=Decimal("0.2"),
            estimated_profit=Decimal("10"),
            estimated_profit_pct=Decimal("0.15"),
            volume_available=Decimal("1"),
        )

        data = opp.to_dict()

        assert data["id"] == "test_1"
        assert data["type"] == "cross_exchange"
        assert "detected_at" in data


# =============================================================================
# ML Signals Tests
# =============================================================================

class TestFeatureExtractor:
    """Tests for ML feature extraction."""

    @pytest.fixture
    def extractor(self):
        """Create feature extractor."""
        pytest.importorskip("numpy")
        from src.strategies.ml_signals import FeatureExtractor
        return FeatureExtractor()

    def test_extract_features(self, extractor):
        """Test feature extraction."""
        prices = [100 + i * 0.1 for i in range(100)]
        volumes = [1000 + i * 10 for i in range(100)]

        features = extractor.extract_features(prices, volumes)

        assert "price_current" in features
        assert "return_mean" in features
        assert "rsi_14" in features
        assert "ma_20" in features
        assert "volume_current" in features

    def test_rsi_calculation(self, extractor):
        """Test RSI calculation."""
        import numpy as np

        # Uptrend
        prices = np.array([100 + i for i in range(20)])
        rsi = extractor._calculate_rsi(prices, 14)

        assert rsi > 50  # Should be bullish

        # Downtrend
        prices = np.array([100 - i for i in range(20)])
        rsi = extractor._calculate_rsi(prices, 14)

        assert rsi < 50  # Should be bearish


class TestMLModel:
    """Tests for ML models."""

    @pytest.fixture
    def model(self):
        """Create Random Forest model."""
        pytest.importorskip("sklearn")
        from src.strategies.ml_signals import RandomForestModel
        return RandomForestModel(n_estimators=10, max_depth=5)

    def test_train_model(self, model):
        """Test model training."""
        import numpy as np

        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)
        feature_names = [f"feature_{i}" for i in range(10)]

        accuracy = model.train(X, y, feature_names)

        assert model.is_trained
        assert 0 <= accuracy <= 1

    def test_predict(self, model):
        """Test model prediction."""
        import numpy as np
        from src.strategies.ml_signals import SignalType

        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)
        feature_names = [f"feature_{i}" for i in range(10)]
        model.train(X, y, feature_names)

        features = {f"feature_{i}": np.random.randn() for i in range(10)}
        signal_type, confidence = model.predict(features)

        assert isinstance(signal_type, SignalType)
        assert 0 <= confidence <= 1


# =============================================================================
# Sentiment Analysis Tests
# =============================================================================

class TestTextSentimentAnalyzer:
    """Tests for text sentiment analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create sentiment analyzer."""
        from src.strategies.sentiment import TextSentimentAnalyzer
        return TextSentimentAnalyzer(use_transformer=False)

    def test_clean_text(self, analyzer):
        """Test text cleaning."""
        text = "Check out https://example.com @user #crypto $BTC!"
        cleaned = analyzer._clean_text(text)

        assert "https" not in cleaned
        assert "@user" not in cleaned
        assert "crypto" in cleaned

    def test_analyze_bullish(self, analyzer):
        """Test bullish sentiment analysis."""
        text = "BTC to the moon! Bullish pump coming, time to buy!"
        sentiment, confidence = analyzer.analyze(text)

        assert sentiment > 0  # Bullish

    def test_analyze_bearish(self, analyzer):
        """Test bearish sentiment analysis."""
        text = "Crash incoming, sell everything! This is a scam, going to dump."
        sentiment, confidence = analyzer.analyze(text)

        assert sentiment < 0  # Bearish


class TestSentimentAggregator:
    """Tests for sentiment aggregation."""

    @pytest.fixture
    def aggregator(self):
        """Create sentiment aggregator."""
        from src.strategies.sentiment import SentimentAggregator
        return SentimentAggregator()

    def test_add_data(self, aggregator):
        """Test adding sentiment data."""
        from src.strategies.sentiment import SentimentData, DataSource

        data = SentimentData(
            source=DataSource.TWITTER,
            symbol="BTCUSDT",
            text="test",
            sentiment_score=0.5,
            confidence=0.8,
        )

        aggregator.add_data(data)

        assert "BTCUSDT" in aggregator.history
        assert len(aggregator.history["BTCUSDT"]) == 1

    def test_get_aggregate(self, aggregator):
        """Test getting aggregate sentiment."""
        from src.strategies.sentiment import SentimentData, DataSource

        for i in range(20):
            aggregator.add_data(SentimentData(
                source=DataSource.TWITTER,
                symbol="BTCUSDT",
                text=f"test {i}",
                sentiment_score=0.3 + i * 0.01,
                confidence=0.8,
            ))

        aggregate = aggregator.get_aggregate("BTCUSDT")

        assert aggregate.symbol == "BTCUSDT"
        assert aggregate.sample_size == 20
        assert aggregate.overall_score > 0
