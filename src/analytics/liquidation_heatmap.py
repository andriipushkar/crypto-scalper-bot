"""
Liquidation Heatmap Analyzer.

Моніторинг та аналіз рівнів ліквідацій для прогнозування волатильних рухів:
- Розрахунок рівнів ліквідацій на основі Open Interest
- Heatmap візуалізація концентрації ліквідацій
- Алерти при наближенні ціни до великих рівнів
- Історичний аналіз каскадних ліквідацій

Features:
- Підтримка різних бірж (Binance, Bybit, OKX)
- Розрахунок для різних рівнів leverage
- Інтеграція з WebSocket для real-time оновлень
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Callable
from collections import defaultdict
import aiohttp
from loguru import logger


# =============================================================================
# Enums and Types
# =============================================================================

class LiquidationType(Enum):
    """Type of liquidation."""
    LONG = "long"
    SHORT = "short"


class LiquidationRisk(Enum):
    """Liquidation risk level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Exchange(Enum):
    """Supported exchanges."""
    BINANCE = "binance"
    BYBIT = "bybit"
    OKX = "okx"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LiquidationLevel:
    """A single liquidation level."""
    price: Decimal
    liquidation_type: LiquidationType
    estimated_volume: Decimal  # Estimated liquidation volume in USD
    leverage: int
    confidence: float  # 0-1 confidence in the estimate
    source: str = ""  # Data source

    @property
    def volume_millions(self) -> float:
        """Volume in millions USD."""
        return float(self.estimated_volume / Decimal("1000000"))


@dataclass
class LiquidationCluster:
    """Cluster of liquidation levels."""
    price_from: Decimal
    price_to: Decimal
    center_price: Decimal
    total_volume: Decimal
    long_volume: Decimal
    short_volume: Decimal
    levels_count: int
    dominant_side: LiquidationType
    risk: LiquidationRisk

    @property
    def volume_millions(self) -> float:
        """Total volume in millions."""
        return float(self.total_volume / Decimal("1000000"))

    @property
    def imbalance_ratio(self) -> float:
        """Ratio of dominant side volume to total."""
        if self.total_volume == 0:
            return 0.5
        if self.dominant_side == LiquidationType.LONG:
            return float(self.long_volume / self.total_volume)
        return float(self.short_volume / self.total_volume)


@dataclass
class LiquidationHeatmapData:
    """Heatmap data for visualization."""
    symbol: str
    current_price: Decimal
    timestamp: datetime

    # Levels
    levels: List[LiquidationLevel] = field(default_factory=list)
    clusters: List[LiquidationCluster] = field(default_factory=list)

    # Summary
    total_long_liquidations: Decimal = Decimal("0")
    total_short_liquidations: Decimal = Decimal("0")
    nearest_long_cluster: Optional[LiquidationCluster] = None
    nearest_short_cluster: Optional[LiquidationCluster] = None

    # Risk assessment
    upside_risk: LiquidationRisk = LiquidationRisk.LOW
    downside_risk: LiquidationRisk = LiquidationRisk.LOW

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API/visualization."""
        return {
            "symbol": self.symbol,
            "current_price": str(self.current_price),
            "timestamp": self.timestamp.isoformat(),
            "total_long_liq_millions": float(self.total_long_liquidations / Decimal("1000000")),
            "total_short_liq_millions": float(self.total_short_liquidations / Decimal("1000000")),
            "upside_risk": self.upside_risk.value,
            "downside_risk": self.downside_risk.value,
            "nearest_long_cluster": {
                "price": str(self.nearest_long_cluster.center_price),
                "volume_millions": self.nearest_long_cluster.volume_millions,
                "distance_pct": float((self.nearest_long_cluster.center_price - self.current_price) / self.current_price * 100),
            } if self.nearest_long_cluster else None,
            "nearest_short_cluster": {
                "price": str(self.nearest_short_cluster.center_price),
                "volume_millions": self.nearest_short_cluster.volume_millions,
                "distance_pct": float((self.current_price - self.nearest_short_cluster.center_price) / self.current_price * 100),
            } if self.nearest_short_cluster else None,
            "clusters": [
                {
                    "price_from": str(c.price_from),
                    "price_to": str(c.price_to),
                    "center_price": str(c.center_price),
                    "volume_millions": c.volume_millions,
                    "dominant_side": c.dominant_side.value,
                    "risk": c.risk.value,
                }
                for c in self.clusters
            ],
        }


@dataclass
class OpenInterestData:
    """Open Interest data from exchange."""
    symbol: str
    timestamp: datetime
    total_oi: Decimal  # Total OI in contracts/coins
    total_oi_value: Decimal  # Total OI in USD
    long_ratio: float  # Long/Short ratio (>0.5 = more longs)
    top_trader_long_ratio: float  # Top traders L/S ratio


@dataclass
class LiquidationEvent:
    """Historical liquidation event."""
    symbol: str
    timestamp: datetime
    side: LiquidationType
    price: Decimal
    quantity: Decimal
    value_usd: Decimal

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "side": self.side.value,
            "price": str(self.price),
            "quantity": str(self.quantity),
            "value_usd": str(self.value_usd),
        }


# =============================================================================
# Liquidation Heatmap Analyzer
# =============================================================================

class LiquidationHeatmapAnalyzer:
    """
    Liquidation Heatmap Analyzer.

    Analyzes open interest and price data to estimate liquidation levels
    and generate heatmaps for trading decisions.

    Usage:
        analyzer = LiquidationHeatmapAnalyzer()
        await analyzer.start()

        # Get heatmap data
        heatmap = await analyzer.get_heatmap("BTCUSDT")

        # Subscribe to alerts
        analyzer.on_liquidation_alert(callback)
    """

    # Common leverage levels
    LEVERAGE_LEVELS = [2, 3, 5, 10, 20, 25, 50, 75, 100, 125]

    def __init__(
        self,
        exchanges: List[Exchange] = None,
        alert_threshold_millions: float = 50.0,
        cluster_range_pct: float = 0.5,  # 0.5% price range for clustering
    ):
        """
        Initialize Liquidation Heatmap Analyzer.

        Args:
            exchanges: List of exchanges to monitor
            alert_threshold_millions: Alert when liquidation volume > threshold (in millions USD)
            cluster_range_pct: Price range percentage for clustering levels
        """
        self.exchanges = exchanges or [Exchange.BINANCE]
        self.alert_threshold = Decimal(str(alert_threshold_millions * 1_000_000))
        self.cluster_range_pct = cluster_range_pct

        # Data storage
        self._open_interest: Dict[str, OpenInterestData] = {}
        self._liquidation_history: Dict[str, List[LiquidationEvent]] = defaultdict(list)
        self._heatmap_cache: Dict[str, LiquidationHeatmapData] = {}
        self._cache_duration = timedelta(minutes=5)
        self._last_cache_update: Dict[str, datetime] = {}

        # Callbacks
        self._alert_callbacks: List[Callable] = []
        self._update_callbacks: List[Callable] = []

        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None

        # State
        self._running = False

        logger.info(f"Liquidation Heatmap Analyzer initialized for {[e.value for e in self.exchanges]}")

    async def start(self) -> None:
        """Start the analyzer."""
        self._session = aiohttp.ClientSession()
        self._running = True
        logger.info("Liquidation Heatmap Analyzer started")

    async def stop(self) -> None:
        """Stop the analyzer."""
        self._running = False
        if self._session:
            await self._session.close()
        logger.info("Liquidation Heatmap Analyzer stopped")

    # =========================================================================
    # Data Fetching
    # =========================================================================

    async def fetch_open_interest(self, symbol: str, exchange: Exchange = Exchange.BINANCE) -> Optional[OpenInterestData]:
        """Fetch Open Interest data from exchange."""
        try:
            if exchange == Exchange.BINANCE:
                return await self._fetch_binance_oi(symbol)
            elif exchange == Exchange.BYBIT:
                return await self._fetch_bybit_oi(symbol)
            elif exchange == Exchange.OKX:
                return await self._fetch_okx_oi(symbol)
        except Exception as e:
            logger.error(f"Failed to fetch OI for {symbol} from {exchange.value}: {e}")
        return None

    async def _fetch_binance_oi(self, symbol: str) -> Optional[OpenInterestData]:
        """Fetch OI from Binance."""
        base_url = "https://fapi.binance.com"

        # Get OI
        oi_url = f"{base_url}/fapi/v1/openInterest?symbol={symbol}"
        async with self._session.get(oi_url) as resp:
            if resp.status != 200:
                return None
            oi_data = await resp.json()

        # Get Long/Short ratio
        ls_url = f"{base_url}/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=5m&limit=1"
        async with self._session.get(ls_url) as resp:
            if resp.status == 200:
                ls_data = await resp.json()
                long_ratio = float(ls_data[0]["longAccount"]) if ls_data else 0.5
            else:
                long_ratio = 0.5

        # Get top trader ratio
        top_url = f"{base_url}/futures/data/topLongShortAccountRatio?symbol={symbol}&period=5m&limit=1"
        async with self._session.get(top_url) as resp:
            if resp.status == 200:
                top_data = await resp.json()
                top_long_ratio = float(top_data[0]["longAccount"]) if top_data else 0.5
            else:
                top_long_ratio = 0.5

        # Get current price for value calculation
        ticker_url = f"{base_url}/fapi/v1/ticker/price?symbol={symbol}"
        async with self._session.get(ticker_url) as resp:
            if resp.status == 200:
                ticker = await resp.json()
                price = Decimal(ticker["price"])
            else:
                price = Decimal("0")

        oi = Decimal(oi_data["openInterest"])
        oi_value = oi * price

        data = OpenInterestData(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            total_oi=oi,
            total_oi_value=oi_value,
            long_ratio=long_ratio,
            top_trader_long_ratio=top_long_ratio,
        )

        self._open_interest[symbol] = data
        return data

    async def _fetch_bybit_oi(self, symbol: str) -> Optional[OpenInterestData]:
        """Fetch OI from Bybit."""
        base_url = "https://api.bybit.com"

        url = f"{base_url}/v5/market/open-interest?category=linear&symbol={symbol}&intervalTime=5min&limit=1"
        async with self._session.get(url) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()

        if data.get("result", {}).get("list"):
            oi_info = data["result"]["list"][0]
            return OpenInterestData(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                total_oi=Decimal(oi_info["openInterest"]),
                total_oi_value=Decimal("0"),  # Need price for value
                long_ratio=0.5,
                top_trader_long_ratio=0.5,
            )
        return None

    async def _fetch_okx_oi(self, symbol: str) -> Optional[OpenInterestData]:
        """Fetch OI from OKX."""
        # Convert symbol format (BTCUSDT -> BTC-USDT-SWAP)
        base = symbol[:-4]
        quote = symbol[-4:]
        okx_symbol = f"{base}-{quote}-SWAP"

        base_url = "https://www.okx.com"
        url = f"{base_url}/api/v5/public/open-interest?instType=SWAP&instId={okx_symbol}"

        async with self._session.get(url) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()

        if data.get("data"):
            oi_info = data["data"][0]
            return OpenInterestData(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                total_oi=Decimal(oi_info["oi"]),
                total_oi_value=Decimal(oi_info.get("oiCcy", "0")),
                long_ratio=0.5,
                top_trader_long_ratio=0.5,
            )
        return None

    async def fetch_recent_liquidations(
        self,
        symbol: str,
        exchange: Exchange = Exchange.BINANCE,
        hours: int = 24,
    ) -> List[LiquidationEvent]:
        """Fetch recent liquidation events."""
        try:
            if exchange == Exchange.BINANCE:
                return await self._fetch_binance_liquidations(symbol, hours)
        except Exception as e:
            logger.error(f"Failed to fetch liquidations for {symbol}: {e}")
        return []

    async def _fetch_binance_liquidations(self, symbol: str, hours: int) -> List[LiquidationEvent]:
        """Fetch liquidations from Binance."""
        # Binance doesn't have a direct liquidation API, but we can use forceOrders
        base_url = "https://fapi.binance.com"
        url = f"{base_url}/fapi/v1/forceOrders?symbol={symbol}&limit=100"

        async with self._session.get(url) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()

        events = []
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        for item in data:
            timestamp = datetime.fromtimestamp(item["time"] / 1000)
            if timestamp < cutoff:
                continue

            event = LiquidationEvent(
                symbol=symbol,
                timestamp=timestamp,
                side=LiquidationType.LONG if item["side"] == "SELL" else LiquidationType.SHORT,
                price=Decimal(item["price"]),
                quantity=Decimal(item["origQty"]),
                value_usd=Decimal(item["price"]) * Decimal(item["origQty"]),
            )
            events.append(event)

        # Store in history
        self._liquidation_history[symbol].extend(events)
        self._liquidation_history[symbol] = [
            e for e in self._liquidation_history[symbol]
            if e.timestamp > datetime.utcnow() - timedelta(hours=48)
        ][-1000:]  # Keep last 1000

        return events

    # =========================================================================
    # Liquidation Level Calculation
    # =========================================================================

    def calculate_liquidation_levels(
        self,
        current_price: Decimal,
        oi_data: OpenInterestData,
    ) -> List[LiquidationLevel]:
        """
        Calculate potential liquidation levels based on Open Interest.

        Uses leverage levels to estimate where positions would be liquidated.
        """
        levels = []

        # Estimate OI distribution by leverage
        # Higher leverage = closer liquidation price
        for leverage in self.LEVERAGE_LEVELS:
            # Calculate liquidation prices
            # For longs: liq_price = entry * (1 - 1/leverage * maintenance_margin_rate)
            # Simplified: liq_price ≈ entry * (1 - 0.8/leverage)
            maintenance_factor = Decimal("0.8")

            # Long liquidation (below current price)
            long_liq_price = current_price * (1 - maintenance_factor / leverage)

            # Short liquidation (above current price)
            short_liq_price = current_price * (1 + maintenance_factor / leverage)

            # Estimate volume at this level based on OI and leverage distribution
            # Higher leverage typically has less OI
            leverage_weight = self._get_leverage_weight(leverage)
            base_volume = oi_data.total_oi_value * Decimal(str(leverage_weight))

            # Adjust by long/short ratio
            long_volume = base_volume * Decimal(str(oi_data.long_ratio))
            short_volume = base_volume * Decimal(str(1 - oi_data.long_ratio))

            levels.append(LiquidationLevel(
                price=long_liq_price,
                liquidation_type=LiquidationType.LONG,
                estimated_volume=long_volume,
                leverage=leverage,
                confidence=0.6 + leverage_weight * 0.3,
                source=f"estimated_{leverage}x",
            ))

            levels.append(LiquidationLevel(
                price=short_liq_price,
                liquidation_type=LiquidationType.SHORT,
                estimated_volume=short_volume,
                leverage=leverage,
                confidence=0.6 + leverage_weight * 0.3,
                source=f"estimated_{leverage}x",
            ))

        return levels

    def _get_leverage_weight(self, leverage: int) -> float:
        """Get weight for leverage level (based on typical distribution)."""
        # Most positions are at lower leverage
        weights = {
            2: 0.05,
            3: 0.08,
            5: 0.15,
            10: 0.25,
            20: 0.20,
            25: 0.10,
            50: 0.08,
            75: 0.05,
            100: 0.03,
            125: 0.01,
        }
        return weights.get(leverage, 0.05)

    def cluster_levels(
        self,
        levels: List[LiquidationLevel],
        current_price: Decimal,
    ) -> List[LiquidationCluster]:
        """Cluster nearby liquidation levels."""
        if not levels:
            return []

        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x.price)

        clusters = []
        current_cluster_levels = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            # Check if level is within cluster range of current cluster
            cluster_center = sum(l.price for l in current_cluster_levels) / len(current_cluster_levels)
            distance_pct = abs(float((level.price - cluster_center) / cluster_center * 100))

            if distance_pct <= self.cluster_range_pct:
                current_cluster_levels.append(level)
            else:
                # Finalize current cluster
                if current_cluster_levels:
                    cluster = self._create_cluster(current_cluster_levels, current_price)
                    if cluster:
                        clusters.append(cluster)
                current_cluster_levels = [level]

        # Finalize last cluster
        if current_cluster_levels:
            cluster = self._create_cluster(current_cluster_levels, current_price)
            if cluster:
                clusters.append(cluster)

        return clusters

    def _create_cluster(
        self,
        levels: List[LiquidationLevel],
        current_price: Decimal,
    ) -> Optional[LiquidationCluster]:
        """Create a cluster from levels."""
        if not levels:
            return None

        prices = [l.price for l in levels]
        long_vol = sum(l.estimated_volume for l in levels if l.liquidation_type == LiquidationType.LONG)
        short_vol = sum(l.estimated_volume for l in levels if l.liquidation_type == LiquidationType.SHORT)
        total_vol = long_vol + short_vol

        center = sum(prices) / len(prices)
        distance_pct = abs(float((center - current_price) / current_price * 100))

        # Determine risk level
        if total_vol > self.alert_threshold and distance_pct < 2:
            risk = LiquidationRisk.CRITICAL
        elif total_vol > self.alert_threshold * Decimal("0.5") and distance_pct < 3:
            risk = LiquidationRisk.HIGH
        elif total_vol > self.alert_threshold * Decimal("0.2"):
            risk = LiquidationRisk.MEDIUM
        else:
            risk = LiquidationRisk.LOW

        return LiquidationCluster(
            price_from=min(prices),
            price_to=max(prices),
            center_price=center,
            total_volume=total_vol,
            long_volume=long_vol,
            short_volume=short_vol,
            levels_count=len(levels),
            dominant_side=LiquidationType.LONG if long_vol > short_vol else LiquidationType.SHORT,
            risk=risk,
        )

    # =========================================================================
    # Heatmap Generation
    # =========================================================================

    async def get_heatmap(
        self,
        symbol: str,
        force_refresh: bool = False,
    ) -> LiquidationHeatmapData:
        """
        Get liquidation heatmap for a symbol.

        Args:
            symbol: Trading pair symbol
            force_refresh: Force refresh cache

        Returns:
            LiquidationHeatmapData with levels and clusters
        """
        # Check cache
        if not force_refresh and symbol in self._heatmap_cache:
            last_update = self._last_cache_update.get(symbol)
            if last_update and datetime.utcnow() - last_update < self._cache_duration:
                return self._heatmap_cache[symbol]

        # Fetch OI data
        oi_data = await self.fetch_open_interest(symbol)
        if not oi_data:
            # Return empty heatmap
            return LiquidationHeatmapData(
                symbol=symbol,
                current_price=Decimal("0"),
                timestamp=datetime.utcnow(),
            )

        # Get current price
        current_price = await self._get_current_price(symbol)

        # Calculate levels
        levels = self.calculate_liquidation_levels(current_price, oi_data)

        # Cluster levels
        clusters = self.cluster_levels(levels, current_price)

        # Calculate totals
        total_long = sum(l.estimated_volume for l in levels if l.liquidation_type == LiquidationType.LONG)
        total_short = sum(l.estimated_volume for l in levels if l.liquidation_type == LiquidationType.SHORT)

        # Find nearest clusters
        long_clusters = [c for c in clusters if c.center_price < current_price]
        short_clusters = [c for c in clusters if c.center_price > current_price]

        nearest_long = max(long_clusters, key=lambda c: c.center_price) if long_clusters else None
        nearest_short = min(short_clusters, key=lambda c: c.center_price) if short_clusters else None

        # Assess risk
        upside_risk = self._assess_directional_risk(short_clusters, current_price)
        downside_risk = self._assess_directional_risk(long_clusters, current_price)

        heatmap = LiquidationHeatmapData(
            symbol=symbol,
            current_price=current_price,
            timestamp=datetime.utcnow(),
            levels=levels,
            clusters=clusters,
            total_long_liquidations=total_long,
            total_short_liquidations=total_short,
            nearest_long_cluster=nearest_long,
            nearest_short_cluster=nearest_short,
            upside_risk=upside_risk,
            downside_risk=downside_risk,
        )

        # Cache
        self._heatmap_cache[symbol] = heatmap
        self._last_cache_update[symbol] = datetime.utcnow()

        # Check alerts
        await self._check_alerts(heatmap)

        return heatmap

    def _assess_directional_risk(
        self,
        clusters: List[LiquidationCluster],
        current_price: Decimal,
    ) -> LiquidationRisk:
        """Assess risk in a direction based on clusters."""
        if not clusters:
            return LiquidationRisk.LOW

        # Sort by distance from current price
        for cluster in clusters:
            distance_pct = abs(float((cluster.center_price - current_price) / current_price * 100))

            if cluster.total_volume > self.alert_threshold:
                if distance_pct < 2:
                    return LiquidationRisk.CRITICAL
                elif distance_pct < 5:
                    return LiquidationRisk.HIGH

            if cluster.total_volume > self.alert_threshold * Decimal("0.5"):
                if distance_pct < 3:
                    return LiquidationRisk.HIGH
                elif distance_pct < 5:
                    return LiquidationRisk.MEDIUM

        return LiquidationRisk.LOW

    async def _get_current_price(self, symbol: str) -> Decimal:
        """Get current price for symbol."""
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
        async with self._session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                return Decimal(data["price"])
        return Decimal("0")

    # =========================================================================
    # Alerts
    # =========================================================================

    async def _check_alerts(self, heatmap: LiquidationHeatmapData) -> None:
        """Check if any alert conditions are met."""
        alerts = []

        # Check critical risk levels
        if heatmap.upside_risk == LiquidationRisk.CRITICAL:
            alerts.append({
                "type": "critical_short_liquidations",
                "message": f"Critical short liquidation levels near price for {heatmap.symbol}",
                "cluster": heatmap.nearest_short_cluster,
            })

        if heatmap.downside_risk == LiquidationRisk.CRITICAL:
            alerts.append({
                "type": "critical_long_liquidations",
                "message": f"Critical long liquidation levels near price for {heatmap.symbol}",
                "cluster": heatmap.nearest_long_cluster,
            })

        # High risk alerts
        if heatmap.upside_risk == LiquidationRisk.HIGH:
            alerts.append({
                "type": "high_short_liquidations",
                "message": f"High short liquidation concentration for {heatmap.symbol}",
                "cluster": heatmap.nearest_short_cluster,
            })

        if heatmap.downside_risk == LiquidationRisk.HIGH:
            alerts.append({
                "type": "high_long_liquidations",
                "message": f"High long liquidation concentration for {heatmap.symbol}",
                "cluster": heatmap.nearest_long_cluster,
            })

        # Trigger callbacks
        for alert in alerts:
            for callback in self._alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(heatmap.symbol, alert)
                    else:
                        callback(heatmap.symbol, alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

    def on_liquidation_alert(self, callback: Callable) -> None:
        """Register callback for liquidation alerts."""
        self._alert_callbacks.append(callback)

    def on_heatmap_update(self, callback: Callable) -> None:
        """Register callback for heatmap updates."""
        self._update_callbacks.append(callback)

    # =========================================================================
    # Analysis
    # =========================================================================

    def get_cascade_risk(self, heatmap: LiquidationHeatmapData) -> Dict[str, Any]:
        """
        Analyze risk of cascading liquidations.

        Returns analysis of potential cascade scenarios.
        """
        clusters = heatmap.clusters
        current_price = heatmap.current_price

        cascade_up = []
        cascade_down = []

        # Analyze upward cascade (short liquidations)
        short_clusters = sorted(
            [c for c in clusters if c.center_price > current_price],
            key=lambda c: c.center_price
        )

        cumulative = Decimal("0")
        for cluster in short_clusters:
            cumulative += cluster.short_volume
            distance_pct = float((cluster.center_price - current_price) / current_price * 100)
            cascade_up.append({
                "price": str(cluster.center_price),
                "distance_pct": distance_pct,
                "volume_at_level": cluster.volume_millions,
                "cumulative_millions": float(cumulative / Decimal("1000000")),
            })

        # Analyze downward cascade (long liquidations)
        long_clusters = sorted(
            [c for c in clusters if c.center_price < current_price],
            key=lambda c: c.center_price,
            reverse=True
        )

        cumulative = Decimal("0")
        for cluster in long_clusters:
            cumulative += cluster.long_volume
            distance_pct = float((current_price - cluster.center_price) / current_price * 100)
            cascade_down.append({
                "price": str(cluster.center_price),
                "distance_pct": distance_pct,
                "volume_at_level": cluster.volume_millions,
                "cumulative_millions": float(cumulative / Decimal("1000000")),
            })

        return {
            "symbol": heatmap.symbol,
            "current_price": str(current_price),
            "cascade_up": cascade_up,
            "cascade_down": cascade_down,
            "upside_cascade_risk": "HIGH" if len(cascade_up) > 3 and cascade_up[0]["distance_pct"] < 3 else "LOW",
            "downside_cascade_risk": "HIGH" if len(cascade_down) > 3 and cascade_down[0]["distance_pct"] < 3 else "LOW",
        }

    def get_optimal_entry_zones(self, heatmap: LiquidationHeatmapData) -> Dict[str, Any]:
        """
        Identify optimal entry zones based on liquidation levels.

        Areas with high liquidation concentration can act as magnets/support/resistance.
        """
        clusters = heatmap.clusters
        current_price = heatmap.current_price

        # Long entry zones (below current price, where liquidations would cause bounce)
        long_zones = []
        for cluster in clusters:
            if cluster.center_price < current_price and cluster.dominant_side == LiquidationType.LONG:
                # High long liquidations = potential bounce zone
                if cluster.total_volume > self.alert_threshold * Decimal("0.3"):
                    distance_pct = float((current_price - cluster.center_price) / current_price * 100)
                    long_zones.append({
                        "price_zone": f"{cluster.price_from} - {cluster.price_to}",
                        "center": str(cluster.center_price),
                        "distance_pct": distance_pct,
                        "liquidation_volume_millions": cluster.volume_millions,
                        "strength": "strong" if cluster.total_volume > self.alert_threshold else "moderate",
                    })

        # Short entry zones (above current price)
        short_zones = []
        for cluster in clusters:
            if cluster.center_price > current_price and cluster.dominant_side == LiquidationType.SHORT:
                if cluster.total_volume > self.alert_threshold * Decimal("0.3"):
                    distance_pct = float((cluster.center_price - current_price) / current_price * 100)
                    short_zones.append({
                        "price_zone": f"{cluster.price_from} - {cluster.price_to}",
                        "center": str(cluster.center_price),
                        "distance_pct": distance_pct,
                        "liquidation_volume_millions": cluster.volume_millions,
                        "strength": "strong" if cluster.total_volume > self.alert_threshold else "moderate",
                    })

        return {
            "symbol": heatmap.symbol,
            "long_entry_zones": sorted(long_zones, key=lambda x: x["distance_pct"]),
            "short_entry_zones": sorted(short_zones, key=lambda x: x["distance_pct"]),
        }
