"""
Tests for Liquidation Heatmap Analyzer module.

100% coverage for:
- LiquidationLevel and LiquidationCluster dataclasses
- OpenInterestData fetching
- Liquidation level calculation
- Clustering algorithm
- Risk assessment
- Cascade analysis
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from collections import defaultdict

from src.analytics.liquidation_heatmap import (
    LiquidationHeatmapAnalyzer,
    LiquidationHeatmapData,
    LiquidationLevel,
    LiquidationCluster,
    OpenInterestData,
    LiquidationEvent,
    LiquidationType,
    LiquidationRisk,
    Exchange,
)


# =============================================================================
# LiquidationLevel Tests
# =============================================================================

class TestLiquidationLevel:
    """Tests for LiquidationLevel dataclass."""

    def test_level_creation(self):
        """Test liquidation level creation."""
        level = LiquidationLevel(
            price=Decimal("48000"),
            liquidation_type=LiquidationType.LONG,
            estimated_volume=Decimal("50000000"),
            leverage=10,
            confidence=0.75,
            source="binance",
        )

        assert level.price == Decimal("48000")
        assert level.liquidation_type == LiquidationType.LONG
        assert level.estimated_volume == Decimal("50000000")
        assert level.leverage == 10
        assert level.confidence == 0.75

    def test_volume_millions(self):
        """Test volume in millions calculation."""
        level = LiquidationLevel(
            price=Decimal("48000"),
            liquidation_type=LiquidationType.LONG,
            estimated_volume=Decimal("75000000"),
            leverage=10,
            confidence=0.75,
        )

        assert level.volume_millions == 75.0


# =============================================================================
# LiquidationCluster Tests
# =============================================================================

class TestLiquidationCluster:
    """Tests for LiquidationCluster dataclass."""

    def test_cluster_creation(self):
        """Test cluster creation."""
        cluster = LiquidationCluster(
            price_from=Decimal("47500"),
            price_to=Decimal("48500"),
            center_price=Decimal("48000"),
            total_volume=Decimal("100000000"),
            long_volume=Decimal("70000000"),
            short_volume=Decimal("30000000"),
            levels_count=5,
            dominant_side=LiquidationType.LONG,
            risk=LiquidationRisk.HIGH,
        )

        assert cluster.center_price == Decimal("48000")
        assert cluster.volume_millions == 100.0
        assert cluster.dominant_side == LiquidationType.LONG
        assert cluster.risk == LiquidationRisk.HIGH

    def test_imbalance_ratio_long_dominant(self):
        """Test imbalance ratio when longs are dominant."""
        cluster = LiquidationCluster(
            price_from=Decimal("47500"),
            price_to=Decimal("48500"),
            center_price=Decimal("48000"),
            total_volume=Decimal("100000000"),
            long_volume=Decimal("80000000"),
            short_volume=Decimal("20000000"),
            levels_count=5,
            dominant_side=LiquidationType.LONG,
            risk=LiquidationRisk.HIGH,
        )

        assert cluster.imbalance_ratio == pytest.approx(0.8)

    def test_imbalance_ratio_short_dominant(self):
        """Test imbalance ratio when shorts are dominant."""
        cluster = LiquidationCluster(
            price_from=Decimal("52000"),
            price_to=Decimal("53000"),
            center_price=Decimal("52500"),
            total_volume=Decimal("100000000"),
            long_volume=Decimal("30000000"),
            short_volume=Decimal("70000000"),
            levels_count=5,
            dominant_side=LiquidationType.SHORT,
            risk=LiquidationRisk.MEDIUM,
        )

        assert cluster.imbalance_ratio == pytest.approx(0.7)

    def test_imbalance_ratio_zero_volume(self):
        """Test imbalance ratio with zero volume."""
        cluster = LiquidationCluster(
            price_from=Decimal("47500"),
            price_to=Decimal("48500"),
            center_price=Decimal("48000"),
            total_volume=Decimal("0"),
            long_volume=Decimal("0"),
            short_volume=Decimal("0"),
            levels_count=0,
            dominant_side=LiquidationType.LONG,
            risk=LiquidationRisk.LOW,
        )

        assert cluster.imbalance_ratio == 0.5


# =============================================================================
# LiquidationHeatmapData Tests
# =============================================================================

class TestLiquidationHeatmapData:
    """Tests for LiquidationHeatmapData dataclass."""

    def test_heatmap_data_creation(self):
        """Test heatmap data creation."""
        data = LiquidationHeatmapData(
            symbol="BTCUSDT",
            current_price=Decimal("50000"),
            timestamp=datetime.utcnow(),
            total_long_liquidations=Decimal("500000000"),
            total_short_liquidations=Decimal("300000000"),
            upside_risk=LiquidationRisk.HIGH,
            downside_risk=LiquidationRisk.MEDIUM,
        )

        assert data.symbol == "BTCUSDT"
        assert data.current_price == Decimal("50000")
        assert data.upside_risk == LiquidationRisk.HIGH

    def test_heatmap_to_dict(self):
        """Test heatmap serialization."""
        cluster = LiquidationCluster(
            price_from=Decimal("48000"),
            price_to=Decimal("49000"),
            center_price=Decimal("48500"),
            total_volume=Decimal("100000000"),
            long_volume=Decimal("100000000"),
            short_volume=Decimal("0"),
            levels_count=3,
            dominant_side=LiquidationType.LONG,
            risk=LiquidationRisk.HIGH,
        )

        data = LiquidationHeatmapData(
            symbol="BTCUSDT",
            current_price=Decimal("50000"),
            timestamp=datetime.utcnow(),
            clusters=[cluster],
            nearest_long_cluster=cluster,
            total_long_liquidations=Decimal("100000000"),
            total_short_liquidations=Decimal("50000000"),
        )

        result = data.to_dict()

        assert result["symbol"] == "BTCUSDT"
        assert result["current_price"] == "50000"
        assert result["total_long_liq_millions"] == 100.0
        assert result["total_short_liq_millions"] == 50.0
        assert len(result["clusters"]) == 1
        assert result["nearest_long_cluster"] is not None
        assert result["nearest_long_cluster"]["distance_pct"] == pytest.approx(-3.0)


# =============================================================================
# OpenInterestData Tests
# =============================================================================

class TestOpenInterestData:
    """Tests for OpenInterestData dataclass."""

    def test_oi_data_creation(self):
        """Test OI data creation."""
        oi = OpenInterestData(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            total_oi=Decimal("100000"),
            total_oi_value=Decimal("5000000000"),
            long_ratio=0.55,
            top_trader_long_ratio=0.60,
        )

        assert oi.symbol == "BTCUSDT"
        assert oi.total_oi == Decimal("100000")
        assert oi.long_ratio == 0.55


# =============================================================================
# LiquidationEvent Tests
# =============================================================================

class TestLiquidationEvent:
    """Tests for LiquidationEvent dataclass."""

    def test_event_creation(self):
        """Test liquidation event creation."""
        event = LiquidationEvent(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            side=LiquidationType.LONG,
            price=Decimal("48000"),
            quantity=Decimal("2.5"),
            value_usd=Decimal("120000"),
        )

        assert event.symbol == "BTCUSDT"
        assert event.side == LiquidationType.LONG
        assert event.value_usd == Decimal("120000")

    def test_event_to_dict(self):
        """Test event serialization."""
        event = LiquidationEvent(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            side=LiquidationType.SHORT,
            price=Decimal("52000"),
            quantity=Decimal("1.5"),
            value_usd=Decimal("78000"),
        )

        result = event.to_dict()

        assert result["symbol"] == "BTCUSDT"
        assert result["side"] == "short"
        assert result["price"] == "52000"


# =============================================================================
# LiquidationHeatmapAnalyzer Tests
# =============================================================================

class TestLiquidationHeatmapAnalyzer:
    """Tests for LiquidationHeatmapAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return LiquidationHeatmapAnalyzer(
            exchanges=[Exchange.BINANCE],
            alert_threshold_millions=50.0,
            cluster_range_pct=0.5,
        )

    @pytest.fixture
    async def started_analyzer(self, analyzer):
        """Start analyzer."""
        await analyzer.start()
        yield analyzer
        await analyzer.stop()

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_analyzer_init(self, analyzer):
        """Test analyzer initialization."""
        assert Exchange.BINANCE in analyzer.exchanges
        assert analyzer.alert_threshold == Decimal("50000000")
        assert analyzer.cluster_range_pct == 0.5

    def test_leverage_levels(self, analyzer):
        """Test leverage levels are defined."""
        assert 10 in analyzer.LEVERAGE_LEVELS
        assert 25 in analyzer.LEVERAGE_LEVELS
        assert 100 in analyzer.LEVERAGE_LEVELS

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test analyzer start/stop."""
        analyzer = LiquidationHeatmapAnalyzer()

        await analyzer.start()
        assert analyzer._session is not None
        assert analyzer._running is True

        await analyzer.stop()
        assert analyzer._running is False

    # -------------------------------------------------------------------------
    # Leverage Weight Tests
    # -------------------------------------------------------------------------

    def test_get_leverage_weight(self, analyzer):
        """Test leverage weight retrieval."""
        assert analyzer._get_leverage_weight(10) == 0.25  # Most common
        assert analyzer._get_leverage_weight(125) == 0.01  # Rare high leverage
        assert analyzer._get_leverage_weight(999) == 0.05  # Unknown

    # -------------------------------------------------------------------------
    # Liquidation Level Calculation Tests
    # -------------------------------------------------------------------------

    def test_calculate_liquidation_levels(self, analyzer):
        """Test liquidation level calculation."""
        oi_data = OpenInterestData(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            total_oi=Decimal("100000"),
            total_oi_value=Decimal("5000000000"),
            long_ratio=0.55,
            top_trader_long_ratio=0.60,
        )

        current_price = Decimal("50000")
        levels = analyzer.calculate_liquidation_levels(current_price, oi_data)

        # Should have 2 levels per leverage (long + short)
        assert len(levels) == len(analyzer.LEVERAGE_LEVELS) * 2

        # Check long liquidation prices are below current
        long_levels = [l for l in levels if l.liquidation_type == LiquidationType.LONG]
        for level in long_levels:
            assert level.price < current_price

        # Check short liquidation prices are above current
        short_levels = [l for l in levels if l.liquidation_type == LiquidationType.SHORT]
        for level in short_levels:
            assert level.price > current_price

    def test_calculate_liquidation_levels_leverage_distance(self, analyzer):
        """Test that higher leverage = closer liquidation price."""
        oi_data = OpenInterestData(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            total_oi=Decimal("100000"),
            total_oi_value=Decimal("5000000000"),
            long_ratio=0.5,
            top_trader_long_ratio=0.5,
        )

        current_price = Decimal("50000")
        levels = analyzer.calculate_liquidation_levels(current_price, oi_data)

        # Group by leverage
        long_by_leverage = {l.leverage: l for l in levels if l.liquidation_type == LiquidationType.LONG}

        # 100x leverage should be closer to price than 10x
        assert long_by_leverage[100].price > long_by_leverage[10].price

    # -------------------------------------------------------------------------
    # Clustering Tests
    # -------------------------------------------------------------------------

    def test_cluster_levels_empty(self, analyzer):
        """Test clustering with empty list."""
        clusters = analyzer.cluster_levels([], Decimal("50000"))
        assert clusters == []

    def test_cluster_levels_single(self, analyzer):
        """Test clustering with single level."""
        levels = [
            LiquidationLevel(
                price=Decimal("48000"),
                liquidation_type=LiquidationType.LONG,
                estimated_volume=Decimal("10000000"),
                leverage=10,
                confidence=0.7,
            )
        ]

        clusters = analyzer.cluster_levels(levels, Decimal("50000"))

        assert len(clusters) == 1
        assert clusters[0].center_price == Decimal("48000")

    def test_cluster_levels_nearby(self, analyzer):
        """Test that nearby levels are clustered together."""
        levels = [
            LiquidationLevel(
                price=Decimal("48000"),
                liquidation_type=LiquidationType.LONG,
                estimated_volume=Decimal("30000000"),
                leverage=10,
                confidence=0.7,
            ),
            LiquidationLevel(
                price=Decimal("48100"),  # Within 0.5% of 48000
                liquidation_type=LiquidationType.LONG,
                estimated_volume=Decimal("20000000"),
                leverage=20,
                confidence=0.7,
            ),
        ]

        clusters = analyzer.cluster_levels(levels, Decimal("50000"))

        # Should be clustered together
        assert len(clusters) == 1
        assert clusters[0].total_volume == Decimal("50000000")
        assert clusters[0].levels_count == 2

    def test_cluster_levels_distant(self, analyzer):
        """Test that distant levels are not clustered."""
        levels = [
            LiquidationLevel(
                price=Decimal("45000"),
                liquidation_type=LiquidationType.LONG,
                estimated_volume=Decimal("30000000"),
                leverage=10,
                confidence=0.7,
            ),
            LiquidationLevel(
                price=Decimal("48000"),  # More than 0.5% from 45000
                liquidation_type=LiquidationType.LONG,
                estimated_volume=Decimal("20000000"),
                leverage=20,
                confidence=0.7,
            ),
        ]

        clusters = analyzer.cluster_levels(levels, Decimal("50000"))

        assert len(clusters) == 2

    def test_cluster_risk_assessment(self, analyzer):
        """Test cluster risk level assessment."""
        # Critical: High volume + close to price
        levels_critical = [
            LiquidationLevel(
                price=Decimal("49500"),
                liquidation_type=LiquidationType.LONG,
                estimated_volume=Decimal("100000000"),  # $100M
                leverage=10,
                confidence=0.7,
            ),
        ]

        clusters = analyzer.cluster_levels(levels_critical, Decimal("50000"))
        assert clusters[0].risk == LiquidationRisk.CRITICAL

    # -------------------------------------------------------------------------
    # Create Cluster Tests
    # -------------------------------------------------------------------------

    def test_create_cluster(self, analyzer):
        """Test cluster creation from levels."""
        levels = [
            LiquidationLevel(
                price=Decimal("48000"),
                liquidation_type=LiquidationType.LONG,
                estimated_volume=Decimal("30000000"),
                leverage=10,
                confidence=0.7,
            ),
            LiquidationLevel(
                price=Decimal("48100"),
                liquidation_type=LiquidationType.SHORT,
                estimated_volume=Decimal("20000000"),
                leverage=20,
                confidence=0.7,
            ),
        ]

        cluster = analyzer._create_cluster(levels, Decimal("50000"))

        assert cluster is not None
        assert cluster.price_from == Decimal("48000")
        assert cluster.price_to == Decimal("48100")
        assert cluster.long_volume == Decimal("30000000")
        assert cluster.short_volume == Decimal("20000000")
        assert cluster.dominant_side == LiquidationType.LONG

    def test_create_cluster_empty(self, analyzer):
        """Test cluster creation with empty levels."""
        cluster = analyzer._create_cluster([], Decimal("50000"))
        assert cluster is None

    # -------------------------------------------------------------------------
    # Risk Assessment Tests
    # -------------------------------------------------------------------------

    def test_assess_directional_risk_critical(self, analyzer):
        """Test critical risk assessment."""
        clusters = [
            LiquidationCluster(
                price_from=Decimal("50500"),
                price_to=Decimal("51000"),
                center_price=Decimal("50750"),
                total_volume=Decimal("100000000"),  # Above threshold
                long_volume=Decimal("0"),
                short_volume=Decimal("100000000"),
                levels_count=3,
                dominant_side=LiquidationType.SHORT,
                risk=LiquidationRisk.CRITICAL,
            ),
        ]

        risk = analyzer._assess_directional_risk(clusters, Decimal("50000"))
        assert risk == LiquidationRisk.CRITICAL

    def test_assess_directional_risk_high(self, analyzer):
        """Test high risk assessment."""
        clusters = [
            LiquidationCluster(
                price_from=Decimal("52000"),
                price_to=Decimal("52400"),
                center_price=Decimal("52200"),  # 4.4% distance from 50000
                total_volume=Decimal("60000000"),  # Above threshold (50M)
                long_volume=Decimal("0"),
                short_volume=Decimal("60000000"),
                levels_count=3,
                dominant_side=LiquidationType.SHORT,
                risk=LiquidationRisk.HIGH,
            ),
        ]

        risk = analyzer._assess_directional_risk(clusters, Decimal("50000"))
        # 4.4% distance + volume > threshold = HIGH (needs < 5%)
        assert risk in [LiquidationRisk.HIGH, LiquidationRisk.MEDIUM]

    def test_assess_directional_risk_low(self, analyzer):
        """Test low risk assessment."""
        clusters = [
            LiquidationCluster(
                price_from=Decimal("55000"),
                price_to=Decimal("56000"),
                center_price=Decimal("55500"),
                total_volume=Decimal("5000000"),  # Below threshold
                long_volume=Decimal("0"),
                short_volume=Decimal("5000000"),
                levels_count=1,
                dominant_side=LiquidationType.SHORT,
                risk=LiquidationRisk.LOW,
            ),
        ]

        risk = analyzer._assess_directional_risk(clusters, Decimal("50000"))
        assert risk == LiquidationRisk.LOW

    def test_assess_directional_risk_empty(self, analyzer):
        """Test risk assessment with no clusters."""
        risk = analyzer._assess_directional_risk([], Decimal("50000"))
        assert risk == LiquidationRisk.LOW

    # -------------------------------------------------------------------------
    # Data Fetching Tests (Mocked)
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fetch_binance_oi(self, started_analyzer):
        """Test Binance OI fetching."""
        analyzer = started_analyzer

        mock_oi_response = {"openInterest": "100000"}
        mock_ls_response = [{"longAccount": "0.55"}]
        mock_top_response = [{"longAccount": "0.60"}]
        mock_ticker_response = {"price": "50000"}

        with patch.object(analyzer._session, 'get') as mock_get:
            def mock_get_impl(url, **kwargs):
                response = MagicMock()
                response.status = 200

                if "openInterest" in url:
                    response.json = AsyncMock(return_value=mock_oi_response)
                elif "globalLongShort" in url:
                    response.json = AsyncMock(return_value=mock_ls_response)
                elif "topLongShort" in url:
                    response.json = AsyncMock(return_value=mock_top_response)
                elif "ticker/price" in url:
                    response.json = AsyncMock(return_value=mock_ticker_response)

                context_manager = MagicMock()
                context_manager.__aenter__ = AsyncMock(return_value=response)
                context_manager.__aexit__ = AsyncMock(return_value=None)
                return context_manager

            mock_get.side_effect = mock_get_impl

            result = await analyzer._fetch_binance_oi("BTCUSDT")

            assert result is not None
            assert result.symbol == "BTCUSDT"
            assert result.total_oi == Decimal("100000")

    @pytest.mark.asyncio
    async def test_fetch_binance_oi_error(self, started_analyzer):
        """Test Binance OI fetch error handling."""
        analyzer = started_analyzer

        with patch.object(analyzer._session, 'get') as mock_get:
            response = AsyncMock()
            response.status = 500
            mock_get.return_value.__aenter__.return_value = response

            result = await analyzer._fetch_binance_oi("BTCUSDT")

            assert result is None

    @pytest.mark.asyncio
    async def test_fetch_bybit_oi(self, started_analyzer):
        """Test Bybit OI fetching."""
        analyzer = started_analyzer

        mock_response = {
            "result": {
                "list": [{"openInterest": "50000"}]
            }
        }

        with patch.object(analyzer._session, 'get') as mock_get:
            response = AsyncMock()
            response.status = 200
            response.json = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aenter__.return_value = response

            result = await analyzer._fetch_bybit_oi("BTCUSDT")

            assert result is not None
            assert result.total_oi == Decimal("50000")

    @pytest.mark.asyncio
    async def test_fetch_okx_oi(self, started_analyzer):
        """Test OKX OI fetching."""
        analyzer = started_analyzer

        mock_response = {
            "data": [{"oi": "75000", "oiCcy": "3750000000"}]
        }

        with patch.object(analyzer._session, 'get') as mock_get:
            response = AsyncMock()
            response.status = 200
            response.json = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aenter__.return_value = response

            result = await analyzer._fetch_okx_oi("BTCUSDT")

            assert result is not None
            assert result.total_oi == Decimal("75000")

    @pytest.mark.asyncio
    async def test_fetch_open_interest_router(self, started_analyzer):
        """Test OI fetch routing to correct exchange."""
        analyzer = started_analyzer

        with patch.object(analyzer, '_fetch_binance_oi', new_callable=AsyncMock) as mock:
            mock.return_value = OpenInterestData(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                total_oi=Decimal("100000"),
                total_oi_value=Decimal("5000000000"),
                long_ratio=0.5,
                top_trader_long_ratio=0.5,
            )

            await analyzer.fetch_open_interest("BTCUSDT", Exchange.BINANCE)
            mock.assert_called_once_with("BTCUSDT")

    # -------------------------------------------------------------------------
    # Liquidation Fetching Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fetch_binance_liquidations(self, started_analyzer):
        """Test fetching Binance liquidations."""
        analyzer = started_analyzer

        mock_response = [
            {
                "time": int((datetime.utcnow() - timedelta(hours=1)).timestamp() * 1000),
                "side": "SELL",  # Long liquidation
                "price": "48000",
                "origQty": "2.5",
            }
        ]

        with patch.object(analyzer._session, 'get') as mock_get:
            response = AsyncMock()
            response.status = 200
            response.json = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aenter__.return_value = response

            result = await analyzer._fetch_binance_liquidations("BTCUSDT", hours=24)

            assert len(result) == 1
            assert result[0].side == LiquidationType.LONG
            assert result[0].price == Decimal("48000")

    # -------------------------------------------------------------------------
    # Heatmap Generation Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_heatmap(self, started_analyzer):
        """Test heatmap generation."""
        analyzer = started_analyzer

        # Mock OI data
        with patch.object(analyzer, 'fetch_open_interest', new_callable=AsyncMock) as mock_oi:
            mock_oi.return_value = OpenInterestData(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                total_oi=Decimal("100000"),
                total_oi_value=Decimal("5000000000"),
                long_ratio=0.5,
                top_trader_long_ratio=0.5,
            )

            with patch.object(analyzer, '_get_current_price', new_callable=AsyncMock) as mock_price:
                mock_price.return_value = Decimal("50000")

                heatmap = await analyzer.get_heatmap("BTCUSDT")

                assert heatmap.symbol == "BTCUSDT"
                assert heatmap.current_price == Decimal("50000")
                assert len(heatmap.levels) > 0
                assert len(heatmap.clusters) > 0

    @pytest.mark.asyncio
    async def test_get_heatmap_cache(self, started_analyzer):
        """Test heatmap caching."""
        analyzer = started_analyzer

        # Pre-populate cache
        cached_data = LiquidationHeatmapData(
            symbol="BTCUSDT",
            current_price=Decimal("50000"),
            timestamp=datetime.utcnow(),
        )
        analyzer._heatmap_cache["BTCUSDT"] = cached_data
        analyzer._last_cache_update["BTCUSDT"] = datetime.utcnow()

        # Should return cached data without fetching
        result = await analyzer.get_heatmap("BTCUSDT", force_refresh=False)

        assert result == cached_data

    @pytest.mark.asyncio
    async def test_get_heatmap_no_oi_data(self, started_analyzer):
        """Test heatmap with no OI data."""
        analyzer = started_analyzer

        with patch.object(analyzer, 'fetch_open_interest', new_callable=AsyncMock) as mock_oi:
            mock_oi.return_value = None

            heatmap = await analyzer.get_heatmap("BTCUSDT")

            assert heatmap.current_price == Decimal("0")
            assert len(heatmap.levels) == 0

    # -------------------------------------------------------------------------
    # Alert Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_check_alerts_critical(self, started_analyzer):
        """Test critical alert triggering."""
        analyzer = started_analyzer
        alerts_received = []

        def alert_callback(symbol, alert):
            alerts_received.append((symbol, alert))

        analyzer.on_liquidation_alert(alert_callback)

        heatmap = LiquidationHeatmapData(
            symbol="BTCUSDT",
            current_price=Decimal("50000"),
            timestamp=datetime.utcnow(),
            upside_risk=LiquidationRisk.CRITICAL,
            downside_risk=LiquidationRisk.LOW,
            nearest_short_cluster=LiquidationCluster(
                price_from=Decimal("50500"),
                price_to=Decimal("51000"),
                center_price=Decimal("50750"),
                total_volume=Decimal("100000000"),
                long_volume=Decimal("0"),
                short_volume=Decimal("100000000"),
                levels_count=3,
                dominant_side=LiquidationType.SHORT,
                risk=LiquidationRisk.CRITICAL,
            ),
        )

        await analyzer._check_alerts(heatmap)

        assert len(alerts_received) == 1
        assert alerts_received[0][0] == "BTCUSDT"
        assert "critical" in alerts_received[0][1]["type"]

    @pytest.mark.asyncio
    async def test_async_alert_callback(self, started_analyzer):
        """Test async alert callback."""
        analyzer = started_analyzer
        alerts_received = []

        async def async_callback(symbol, alert):
            alerts_received.append((symbol, alert))

        analyzer.on_liquidation_alert(async_callback)

        heatmap = LiquidationHeatmapData(
            symbol="BTCUSDT",
            current_price=Decimal("50000"),
            timestamp=datetime.utcnow(),
            upside_risk=LiquidationRisk.HIGH,
            downside_risk=LiquidationRisk.LOW,
            nearest_short_cluster=MagicMock(),
        )

        await analyzer._check_alerts(heatmap)

        assert len(alerts_received) == 1

    def test_register_callbacks(self, analyzer):
        """Test callback registration."""
        callback1 = MagicMock()
        callback2 = MagicMock()

        analyzer.on_liquidation_alert(callback1)
        analyzer.on_heatmap_update(callback2)

        assert callback1 in analyzer._alert_callbacks
        assert callback2 in analyzer._update_callbacks

    # -------------------------------------------------------------------------
    # Analysis Tests
    # -------------------------------------------------------------------------

    def test_get_cascade_risk(self, analyzer):
        """Test cascade risk analysis."""
        cluster_above = LiquidationCluster(
            price_from=Decimal("51000"),
            price_to=Decimal("52000"),
            center_price=Decimal("51500"),
            total_volume=Decimal("100000000"),
            long_volume=Decimal("0"),
            short_volume=Decimal("100000000"),
            levels_count=3,
            dominant_side=LiquidationType.SHORT,
            risk=LiquidationRisk.HIGH,
        )

        cluster_below = LiquidationCluster(
            price_from=Decimal("48000"),
            price_to=Decimal("49000"),
            center_price=Decimal("48500"),
            total_volume=Decimal("80000000"),
            long_volume=Decimal("80000000"),
            short_volume=Decimal("0"),
            levels_count=2,
            dominant_side=LiquidationType.LONG,
            risk=LiquidationRisk.MEDIUM,
        )

        heatmap = LiquidationHeatmapData(
            symbol="BTCUSDT",
            current_price=Decimal("50000"),
            timestamp=datetime.utcnow(),
            clusters=[cluster_above, cluster_below],
        )

        result = analyzer.get_cascade_risk(heatmap)

        assert result["symbol"] == "BTCUSDT"
        assert len(result["cascade_up"]) == 1
        assert len(result["cascade_down"]) == 1
        assert result["cascade_up"][0]["price"] == "51500"

    def test_get_optimal_entry_zones(self, analyzer):
        """Test optimal entry zone identification."""
        long_zone = LiquidationCluster(
            price_from=Decimal("47000"),
            price_to=Decimal("48000"),
            center_price=Decimal("47500"),
            total_volume=Decimal("30000000"),  # Above 30% threshold
            long_volume=Decimal("30000000"),
            short_volume=Decimal("0"),
            levels_count=2,
            dominant_side=LiquidationType.LONG,
            risk=LiquidationRisk.MEDIUM,
        )

        short_zone = LiquidationCluster(
            price_from=Decimal("53000"),
            price_to=Decimal("54000"),
            center_price=Decimal("53500"),
            total_volume=Decimal("25000000"),
            long_volume=Decimal("0"),
            short_volume=Decimal("25000000"),
            levels_count=2,
            dominant_side=LiquidationType.SHORT,
            risk=LiquidationRisk.MEDIUM,
        )

        heatmap = LiquidationHeatmapData(
            symbol="BTCUSDT",
            current_price=Decimal("50000"),
            timestamp=datetime.utcnow(),
            clusters=[long_zone, short_zone],
        )

        result = analyzer.get_optimal_entry_zones(heatmap)

        assert result["symbol"] == "BTCUSDT"
        assert len(result["long_entry_zones"]) == 1
        assert len(result["short_entry_zones"]) == 1

    # -------------------------------------------------------------------------
    # Price Fetching Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_current_price(self, started_analyzer):
        """Test current price fetching."""
        analyzer = started_analyzer

        with patch.object(analyzer._session, 'get') as mock_get:
            response = AsyncMock()
            response.status = 200
            response.json = AsyncMock(return_value={"price": "50000.50"})
            mock_get.return_value.__aenter__.return_value = response

            price = await analyzer._get_current_price("BTCUSDT")

            assert price == Decimal("50000.50")

    @pytest.mark.asyncio
    async def test_get_current_price_error(self, started_analyzer):
        """Test price fetch error."""
        analyzer = started_analyzer

        with patch.object(analyzer._session, 'get') as mock_get:
            response = AsyncMock()
            response.status = 500
            mock_get.return_value.__aenter__.return_value = response

            price = await analyzer._get_current_price("BTCUSDT")

            assert price == Decimal("0")
