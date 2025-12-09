"""
Cluster Analysis for Scalping (Кластерний аналіз).

Об'ємний аналіз свічок на основі кластерів - ключовий інструмент скальпера.

Концепція:
"Кластери показують розподіл обсягу всередині свічки,
виявляючи зони підтримки/опору та точки накопичення/розподілу"

Особливості:
1. Footprint Chart - візуалізація bid/ask обсягу по ціновим рівнях
2. POC (Point of Control) - ціна з максимальним обсягом
3. Value Area - зона де пройшло 70% обсягу
4. Delta - різниця між aggressive buyers та sellers
5. Volume Profile - розподіл обсягу по цінах
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

from loguru import logger


class ClusterType(Enum):
    """Тип кластера."""
    ACCUMULATION = "accumulation"      # Накопичення (покупки домінують)
    DISTRIBUTION = "distribution"      # Розподіл (продажі домінують)
    NEUTRAL = "neutral"                # Нейтральний
    INITIATIVE_BUY = "initiative_buy"  # Ініціативні покупки
    INITIATIVE_SELL = "initiative_sell"  # Ініціативні продажі


class ImbalanceType(Enum):
    """Тип дисбалансу на рівні."""
    STACKED_BID = "stacked_bid"        # Стек бідів
    STACKED_ASK = "stacked_ask"        # Стек асків
    NONE = "none"


@dataclass
class PriceLevel:
    """Ціновий рівень всередині кластера."""
    price: Decimal
    bid_volume: Decimal = Decimal("0")      # Обсяг покупок (aggressive)
    ask_volume: Decimal = Decimal("0")      # Обсяг продажів (aggressive)
    total_volume: Decimal = Decimal("0")
    trades_count: int = 0
    delta: Decimal = Decimal("0")           # bid - ask

    @property
    def imbalance_ratio(self) -> float:
        """Співвідношення bid/ask."""
        if self.ask_volume == 0:
            return float("inf") if self.bid_volume > 0 else 1.0
        return float(self.bid_volume / self.ask_volume)

    @property
    def is_bid_imbalance(self) -> bool:
        """Чи є перевага покупців (300%+)."""
        return self.imbalance_ratio >= 3.0 and float(self.bid_volume) > 1000

    @property
    def is_ask_imbalance(self) -> bool:
        """Чи є перевага продавців."""
        if self.bid_volume == 0:
            return float(self.ask_volume) > 1000
        return (1 / self.imbalance_ratio) >= 3.0 and float(self.ask_volume) > 1000


@dataclass
class VolumeCluster:
    """Кластер обсягу (одна свічка з деталями)."""
    symbol: str
    timestamp: datetime
    timeframe: int                          # секунди

    # OHLCV
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    total_volume: Decimal

    # Кластерні дані
    levels: Dict[Decimal, PriceLevel] = field(default_factory=dict)

    # Розраховані метрики
    poc_price: Optional[Decimal] = None     # Point of Control
    poc_volume: Decimal = Decimal("0")
    value_area_high: Optional[Decimal] = None
    value_area_low: Optional[Decimal] = None
    total_delta: Decimal = Decimal("0")
    delta_percent: float = 0.0

    # Класифікація
    cluster_type: ClusterType = ClusterType.NEUTRAL

    # Виявлені дисбаланси
    bid_imbalances: List[Decimal] = field(default_factory=list)
    ask_imbalances: List[Decimal] = field(default_factory=list)

    @property
    def is_bullish(self) -> bool:
        """Чи бичачий кластер."""
        return self.total_delta > 0 and self.close_price > self.open_price

    @property
    def is_bearish(self) -> bool:
        """Чи ведмежий кластер."""
        return self.total_delta < 0 and self.close_price < self.open_price

    @property
    def has_strong_imbalance(self) -> bool:
        """Чи є сильний дисбаланс."""
        return len(self.bid_imbalances) >= 3 or len(self.ask_imbalances) >= 3


@dataclass
class VolumeProfile:
    """Профіль обсягу за період."""
    symbol: str
    start_time: datetime
    end_time: datetime

    levels: Dict[Decimal, PriceLevel] = field(default_factory=dict)

    poc_price: Optional[Decimal] = None
    poc_volume: Decimal = Decimal("0")
    value_area_high: Optional[Decimal] = None
    value_area_low: Optional[Decimal] = None

    high_volume_nodes: List[Decimal] = field(default_factory=list)  # HVN
    low_volume_nodes: List[Decimal] = field(default_factory=list)   # LVN


@dataclass
class ClusterSignal:
    """Сигнал від кластерного аналізу."""
    timestamp: datetime
    signal_type: str
    direction: str  # "bullish", "bearish"
    strength: float
    price: Decimal
    description: str
    cluster: Optional[VolumeCluster] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterAnalysisConfig:
    """Конфігурація кластерного аналізу."""
    # Таймфрейми
    cluster_timeframe: int = 60             # секунди (1 хвилина)
    profile_timeframe: int = 3600           # секунди (1 година)

    # Value Area
    value_area_percent: float = 70.0        # % обсягу для VA

    # Дисбаланси
    imbalance_ratio: float = 3.0            # 300% = дисбаланс
    min_imbalance_volume: float = 1000      # Мін. обсяг для дисбалансу

    # Stacked imbalances
    stacked_imbalance_count: int = 3        # Мін. послідовних дисбалансів

    # Delta
    strong_delta_percent: float = 30.0      # Сильна дельта

    # POC
    poc_touch_threshold: float = 0.1        # % від POC для "торкання"

    # Сигнали
    min_signal_strength: float = 0.6

    # Ціновий крок
    price_step: Optional[Decimal] = None    # Авто або фіксований


class ClusterAnalyzer:
    """
    Аналізатор кластерів для скальпінгу.

    Аналізує:
    1. Розподіл обсягу всередині свічок
    2. Дельту (aggressive buyers vs sellers)
    3. Point of Control (POC)
    4. Value Area (VA)
    5. Дисбаланси на рівнях
    6. Stacked imbalances
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Ініціалізація аналізатора."""
        self._config = self._parse_config(config or {})

        # Поточні кластери по символах
        self._current_clusters: Dict[str, VolumeCluster] = {}

        # Історія кластерів
        self._cluster_history: Dict[str, List[VolumeCluster]] = {}

        # Volume Profile
        self._volume_profiles: Dict[str, VolumeProfile] = {}

        # Статистика
        self._clusters_created = 0
        self._signals_generated = 0

        logger.info(
            f"[ClusterAnalyzer] Initialized with timeframe={self._config.cluster_timeframe}s"
        )

    def _parse_config(self, config: Dict[str, Any]) -> ClusterAnalysisConfig:
        """Парсинг конфігурації."""
        return ClusterAnalysisConfig(
            cluster_timeframe=config.get("cluster_timeframe", 60),
            profile_timeframe=config.get("profile_timeframe", 3600),
            value_area_percent=config.get("value_area_percent", 70.0),
            imbalance_ratio=config.get("imbalance_ratio", 3.0),
            min_imbalance_volume=config.get("min_imbalance_volume", 1000),
            stacked_imbalance_count=config.get("stacked_imbalance_count", 3),
            strong_delta_percent=config.get("strong_delta_percent", 30.0),
            poc_touch_threshold=config.get("poc_touch_threshold", 0.1),
            min_signal_strength=config.get("min_signal_strength", 0.6),
            price_step=Decimal(str(config["price_step"])) if config.get("price_step") else None
        )

    def process_trade(
        self,
        symbol: str,
        price: Decimal,
        quantity: Decimal,
        is_buyer_maker: bool,
        timestamp: Optional[datetime] = None
    ) -> Optional[ClusterSignal]:
        """
        Обробити угоду та додати до кластера.

        Args:
            symbol: Символ
            price: Ціна
            quantity: Обсяг
            is_buyer_maker: True якщо покупець maker (продавець aggressor)
            timestamp: Час угоди

        Returns:
            Сигнал якщо кластер закрився
        """
        ts = timestamp or datetime.utcnow()

        # Ініціалізація
        if symbol not in self._cluster_history:
            self._cluster_history[symbol] = []

        # Визначити ціновий рівень
        level_price = self._get_price_level(symbol, price)

        # Отримати/створити поточний кластер
        cluster = self._get_or_create_cluster(symbol, ts, price)

        # Перевірити чи потрібно закрити поточний кластер
        signal = None
        cluster_start = cluster.timestamp
        cluster_end = cluster_start + timedelta(seconds=self._config.cluster_timeframe)

        if ts >= cluster_end:
            # Закрити поточний кластер
            signal = self._finalize_cluster(symbol, cluster)

            # Створити новий кластер
            cluster = self._create_new_cluster(symbol, ts, price)
            self._current_clusters[symbol] = cluster

        # Додати угоду до кластера
        self._add_trade_to_cluster(cluster, level_price, quantity, is_buyer_maker)

        # Оновити Volume Profile
        self._update_volume_profile(symbol, level_price, quantity, is_buyer_maker, ts)

        return signal

    def _get_price_level(self, symbol: str, price: Decimal) -> Decimal:
        """Визначити ціновий рівень (округлення)."""
        if self._config.price_step:
            # Використати заданий крок
            step = self._config.price_step
        else:
            # Автоматично визначити крок на основі ціни
            if price >= 10000:
                step = Decimal("1")       # BTC - $1
            elif price >= 100:
                step = Decimal("0.1")     # ETH - $0.1
            else:
                step = Decimal("0.01")    # Altcoins - $0.01

        return (price / step).quantize(Decimal("1")) * step

    def _get_or_create_cluster(
        self,
        symbol: str,
        timestamp: datetime,
        price: Decimal
    ) -> VolumeCluster:
        """Отримати або створити кластер."""
        if symbol in self._current_clusters:
            return self._current_clusters[symbol]

        cluster = self._create_new_cluster(symbol, timestamp, price)
        self._current_clusters[symbol] = cluster
        return cluster

    def _create_new_cluster(
        self,
        symbol: str,
        timestamp: datetime,
        price: Decimal
    ) -> VolumeCluster:
        """Створити новий кластер."""
        self._clusters_created += 1

        return VolumeCluster(
            symbol=symbol,
            timestamp=timestamp,
            timeframe=self._config.cluster_timeframe,
            open_price=price,
            high_price=price,
            low_price=price,
            close_price=price,
            total_volume=Decimal("0"),
            levels={}
        )

    def _add_trade_to_cluster(
        self,
        cluster: VolumeCluster,
        price: Decimal,
        quantity: Decimal,
        is_buyer_maker: bool
    ) -> None:
        """Додати угоду до кластера."""
        # Оновити OHLC
        if price > cluster.high_price:
            cluster.high_price = price
        if price < cluster.low_price:
            cluster.low_price = price
        cluster.close_price = price
        cluster.total_volume += quantity

        # Оновити рівень
        if price not in cluster.levels:
            cluster.levels[price] = PriceLevel(price=price)

        level = cluster.levels[price]

        if is_buyer_maker:
            # Продавець aggressor (market sell)
            level.ask_volume += quantity
        else:
            # Покупець aggressor (market buy)
            level.bid_volume += quantity

        level.total_volume += quantity
        level.trades_count += 1
        level.delta = level.bid_volume - level.ask_volume

    def _finalize_cluster(
        self,
        symbol: str,
        cluster: VolumeCluster
    ) -> Optional[ClusterSignal]:
        """Завершити кластер та генерувати сигнал."""
        if not cluster.levels:
            return None

        # Розрахувати метрики
        self._calculate_cluster_metrics(cluster)

        # Зберегти в історію
        self._cluster_history[symbol].append(cluster)

        # Обмежити розмір історії
        if len(self._cluster_history[symbol]) > 1000:
            self._cluster_history[symbol] = self._cluster_history[symbol][-500:]

        # Генерувати сигнал
        signal = self._generate_cluster_signal(cluster)

        if signal:
            self._signals_generated += 1

        return signal

    def _calculate_cluster_metrics(self, cluster: VolumeCluster) -> None:
        """Розрахувати метрики кластера."""
        if not cluster.levels:
            return

        # POC (Point of Control) - рівень з максимальним обсягом
        max_volume = Decimal("0")
        poc_price = None

        for price, level in cluster.levels.items():
            if level.total_volume > max_volume:
                max_volume = level.total_volume
                poc_price = price

        cluster.poc_price = poc_price
        cluster.poc_volume = max_volume

        # Total Delta
        total_bid = sum(l.bid_volume for l in cluster.levels.values())
        total_ask = sum(l.ask_volume for l in cluster.levels.values())
        cluster.total_delta = total_bid - total_ask

        if cluster.total_volume > 0:
            cluster.delta_percent = float(
                cluster.total_delta / cluster.total_volume * 100
            )

        # Value Area (70% обсягу навколо POC)
        if poc_price:
            self._calculate_value_area(cluster)

        # Виявити дисбаланси
        self._detect_imbalances(cluster)

        # Класифікувати кластер
        self._classify_cluster(cluster)

    def _calculate_value_area(self, cluster: VolumeCluster) -> None:
        """Розрахувати Value Area."""
        if not cluster.poc_price:
            return

        target_volume = cluster.total_volume * Decimal(
            str(self._config.value_area_percent / 100)
        )

        # Сортувати рівні за ціною
        sorted_levels = sorted(cluster.levels.items(), key=lambda x: x[0])

        # Знайти індекс POC
        poc_index = None
        for i, (price, _) in enumerate(sorted_levels):
            if price == cluster.poc_price:
                poc_index = i
                break

        if poc_index is None:
            return

        # Розширювати від POC доки не наберемо 70%
        accumulated = cluster.levels[cluster.poc_price].total_volume
        low_idx = poc_index
        high_idx = poc_index

        while accumulated < target_volume:
            expand_up = high_idx < len(sorted_levels) - 1
            expand_down = low_idx > 0

            if not expand_up and not expand_down:
                break

            up_volume = Decimal("0")
            down_volume = Decimal("0")

            if expand_up:
                up_volume = sorted_levels[high_idx + 1][1].total_volume
            if expand_down:
                down_volume = sorted_levels[low_idx - 1][1].total_volume

            if up_volume >= down_volume and expand_up:
                high_idx += 1
                accumulated += up_volume
            elif expand_down:
                low_idx -= 1
                accumulated += down_volume
            elif expand_up:
                high_idx += 1
                accumulated += up_volume

        cluster.value_area_low = sorted_levels[low_idx][0]
        cluster.value_area_high = sorted_levels[high_idx][0]

    def _detect_imbalances(self, cluster: VolumeCluster) -> None:
        """Виявити дисбаланси на рівнях."""
        bid_imbalances = []
        ask_imbalances = []

        for price, level in cluster.levels.items():
            if level.is_bid_imbalance:
                bid_imbalances.append(price)
            elif level.is_ask_imbalance:
                ask_imbalances.append(price)

        cluster.bid_imbalances = sorted(bid_imbalances)
        cluster.ask_imbalances = sorted(ask_imbalances)

    def _classify_cluster(self, cluster: VolumeCluster) -> None:
        """Класифікувати кластер."""
        delta_pct = abs(cluster.delta_percent)

        if delta_pct < 10:
            cluster.cluster_type = ClusterType.NEUTRAL
        elif cluster.total_delta > 0:
            if len(cluster.bid_imbalances) >= self._config.stacked_imbalance_count:
                cluster.cluster_type = ClusterType.INITIATIVE_BUY
            else:
                cluster.cluster_type = ClusterType.ACCUMULATION
        else:
            if len(cluster.ask_imbalances) >= self._config.stacked_imbalance_count:
                cluster.cluster_type = ClusterType.INITIATIVE_SELL
            else:
                cluster.cluster_type = ClusterType.DISTRIBUTION

    def _generate_cluster_signal(self, cluster: VolumeCluster) -> Optional[ClusterSignal]:
        """Генерувати сигнал на основі кластера."""
        strength = 0.0
        signal_type = ""
        direction = ""
        description = ""

        # Сигнал 1: Сильна дельта
        if abs(cluster.delta_percent) >= self._config.strong_delta_percent:
            strength = min(abs(cluster.delta_percent) / 50, 1.0)
            direction = "bullish" if cluster.total_delta > 0 else "bearish"
            signal_type = "strong_delta"
            description = f"Strong {'buying' if direction == 'bullish' else 'selling'} pressure: {cluster.delta_percent:.1f}% delta"

        # Сигнал 2: Stacked imbalances
        elif cluster.has_strong_imbalance:
            if len(cluster.bid_imbalances) >= self._config.stacked_imbalance_count:
                strength = min(len(cluster.bid_imbalances) / 5, 1.0)
                direction = "bullish"
                signal_type = "stacked_bid_imbalance"
                description = f"Stacked bid imbalances: {len(cluster.bid_imbalances)} levels"
            elif len(cluster.ask_imbalances) >= self._config.stacked_imbalance_count:
                strength = min(len(cluster.ask_imbalances) / 5, 1.0)
                direction = "bearish"
                signal_type = "stacked_ask_imbalance"
                description = f"Stacked ask imbalances: {len(cluster.ask_imbalances)} levels"

        # Сигнал 3: Initiative buying/selling
        elif cluster.cluster_type in (ClusterType.INITIATIVE_BUY, ClusterType.INITIATIVE_SELL):
            strength = 0.7
            direction = "bullish" if cluster.cluster_type == ClusterType.INITIATIVE_BUY else "bearish"
            signal_type = "initiative"
            description = f"Initiative {'buying' if direction == 'bullish' else 'selling'} detected"

        if strength < self._config.min_signal_strength:
            return None

        return ClusterSignal(
            timestamp=datetime.utcnow(),
            signal_type=signal_type,
            direction=direction,
            strength=strength,
            price=cluster.close_price,
            description=description,
            cluster=cluster,
            metadata={
                "delta_percent": cluster.delta_percent,
                "poc_price": float(cluster.poc_price) if cluster.poc_price else None,
                "value_area_high": float(cluster.value_area_high) if cluster.value_area_high else None,
                "value_area_low": float(cluster.value_area_low) if cluster.value_area_low else None,
                "cluster_type": cluster.cluster_type.value,
                "bid_imbalances": len(cluster.bid_imbalances),
                "ask_imbalances": len(cluster.ask_imbalances),
            }
        )

    def _update_volume_profile(
        self,
        symbol: str,
        price: Decimal,
        quantity: Decimal,
        is_buyer_maker: bool,
        timestamp: datetime
    ) -> None:
        """Оновити Volume Profile."""
        # Ініціалізація
        if symbol not in self._volume_profiles:
            self._volume_profiles[symbol] = VolumeProfile(
                symbol=symbol,
                start_time=timestamp,
                end_time=timestamp + timedelta(seconds=self._config.profile_timeframe),
                levels={}
            )

        profile = self._volume_profiles[symbol]

        # Перевірити чи потрібно новий профіль
        if timestamp >= profile.end_time:
            # Фіналізувати старий профіль
            self._finalize_volume_profile(profile)

            # Створити новий
            self._volume_profiles[symbol] = VolumeProfile(
                symbol=symbol,
                start_time=timestamp,
                end_time=timestamp + timedelta(seconds=self._config.profile_timeframe),
                levels={}
            )
            profile = self._volume_profiles[symbol]

        # Додати до профілю
        if price not in profile.levels:
            profile.levels[price] = PriceLevel(price=price)

        level = profile.levels[price]

        if is_buyer_maker:
            level.ask_volume += quantity
        else:
            level.bid_volume += quantity

        level.total_volume += quantity
        level.trades_count += 1
        level.delta = level.bid_volume - level.ask_volume

    def _finalize_volume_profile(self, profile: VolumeProfile) -> None:
        """Фіналізувати Volume Profile."""
        if not profile.levels:
            return

        # POC
        max_volume = Decimal("0")
        for price, level in profile.levels.items():
            if level.total_volume > max_volume:
                max_volume = level.total_volume
                profile.poc_price = price
                profile.poc_volume = max_volume

        # Value Area (аналогічно до кластера)
        # ... (спрощено)

        # HVN/LVN
        avg_volume = sum(l.total_volume for l in profile.levels.values()) / len(profile.levels)

        for price, level in profile.levels.items():
            if level.total_volume > avg_volume * Decimal("1.5"):
                profile.high_volume_nodes.append(price)
            elif level.total_volume < avg_volume * Decimal("0.5"):
                profile.low_volume_nodes.append(price)

    def get_current_cluster(self, symbol: str) -> Optional[VolumeCluster]:
        """Отримати поточний кластер."""
        return self._current_clusters.get(symbol)

    def get_cluster_history(
        self,
        symbol: str,
        count: int = 10
    ) -> List[VolumeCluster]:
        """Отримати історію кластерів."""
        history = self._cluster_history.get(symbol, [])
        return history[-count:]

    def get_volume_profile(self, symbol: str) -> Optional[VolumeProfile]:
        """Отримати поточний Volume Profile."""
        return self._volume_profiles.get(symbol)

    def get_poc(self, symbol: str) -> Optional[Decimal]:
        """Отримати поточний POC."""
        profile = self._volume_profiles.get(symbol)
        return profile.poc_price if profile else None

    def get_value_area(
        self,
        symbol: str
    ) -> Optional[Tuple[Decimal, Decimal]]:
        """Отримати Value Area (VAL, VAH)."""
        profile = self._volume_profiles.get(symbol)
        if profile and profile.value_area_low and profile.value_area_high:
            return (profile.value_area_low, profile.value_area_high)
        return None

    @property
    def stats(self) -> Dict[str, Any]:
        """Статистика аналізатора."""
        return {
            "clusters_created": self._clusters_created,
            "signals_generated": self._signals_generated,
            "symbols_tracked": len(self._current_clusters),
        }

    def reset(self, symbol: Optional[str] = None) -> None:
        """Скинути стан."""
        if symbol:
            self._current_clusters.pop(symbol, None)
            self._cluster_history.pop(symbol, None)
            self._volume_profiles.pop(symbol, None)
        else:
            self._current_clusters.clear()
            self._cluster_history.clear()
            self._volume_profiles.clear()
