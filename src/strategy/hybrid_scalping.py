"""
Hybrid Scalping Strategy (Гібридна стратегія скальпінгу).

Комбінує три підходи до скальпінгу:
1. Класичний (стакан) - дисбаланс ордербуку, стіни, фронтраннінг
2. Імпульсний - слідування за індексами-поводирями
3. Кластерний - аналіз обсягів, дельта, POC

Концепція:
"Гібридна стратегія комбінує оцінку моментальної ліквідності
зі спостереженням за глобальними індексами та об'ємним аналізом"
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

from loguru import logger

from src.data.models import OrderBookSnapshot, Signal, SignalType, Trade
from src.strategy.base import BaseStrategy
from src.strategy.advanced_orderbook import AdvancedOrderBookStrategy, OrderBookWall
from src.strategy.impulse_scalping import ImpulseScalpingStrategy, LeaderAsset, ImpulseEvent
from src.analytics.print_tape import PrintTapeAnalyzer, TapeSignal, OrderFlowMetrics
from src.analytics.cluster_analysis import ClusterAnalyzer, ClusterSignal, VolumeCluster


class SignalSource(Enum):
    """Джерело сигналу."""
    ORDERBOOK = "orderbook"
    IMPULSE = "impulse"
    TAPE = "tape"
    CLUSTER = "cluster"
    COMPOSITE = "composite"


@dataclass
class HybridSignal:
    """Гібридний сигнал з усіх джерел."""
    timestamp: datetime
    direction: str  # "long" або "short"
    strength: float
    sources: List[SignalSource]
    entry_price: Decimal
    stop_loss: Decimal
    take_profit: Decimal

    # Деталі від кожного джерела
    orderbook_signal: Optional[Signal] = None
    impulse_events: List[ImpulseEvent] = field(default_factory=list)
    tape_signal: Optional[TapeSignal] = None
    cluster_signal: Optional[ClusterSignal] = None

    # Метадані
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        """Впевненість на основі кількості підтверджень."""
        return min(len(self.sources) / 3, 1.0)


@dataclass
class HybridScalpingConfig:
    """Конфігурація гібридної стратегії."""
    enabled: bool = True
    signal_cooldown: float = 3.0
    min_strength: float = 0.6

    # Ваги джерел
    weights: Dict[str, float] = field(default_factory=lambda: {
        "orderbook": 0.35,
        "impulse": 0.25,
        "tape": 0.25,
        "cluster": 0.15,
    })

    # Мінімальна кількість підтверджень
    min_confirmations: int = 2

    # Режим агрегації
    aggregation_mode: str = "weighted"  # "weighted", "unanimous", "majority"

    # Пороги
    composite_threshold: float = 0.6

    # Risk/Reward
    default_sl_pct: float = 0.3         # 0.3%
    default_tp_pct: float = 0.5         # 0.5%
    dynamic_sl_tp: bool = True          # Адаптивні SL/TP

    # Фільтри
    max_spread_bps: float = 10.0
    min_volume_usd: float = 10000

    # Мультитаймфрейм
    use_mtf_confirmation: bool = True
    mtf_timeframes: List[int] = field(default_factory=lambda: [60, 300, 900])


class HybridScalpingStrategy(BaseStrategy):
    """
    Гібридна стратегія скальпінгу.

    Об'єднує:
    1. AdvancedOrderBookStrategy - стакан, стіни, фронтраннінг
    2. ImpulseScalpingStrategy - індекси-поводирі
    3. PrintTapeAnalyzer - лента принтів
    4. ClusterAnalyzer - кластерний аналіз

    Генерує сигнал тільки при підтвердженні від кількох джерел.
    """

    def __init__(self, config: Dict[str, Any]):
        """Ініціалізація стратегії."""
        super().__init__(config)

        self._config = self._parse_config(config)

        # Ініціалізація компонентів
        self._orderbook_strategy = AdvancedOrderBookStrategy(
            config.get("orderbook", {})
        )

        self._impulse_strategy = ImpulseScalpingStrategy(
            config.get("impulse", {})
        )

        self._tape_analyzer = PrintTapeAnalyzer(
            config.get("tape", {})
        )

        self._cluster_analyzer = ClusterAnalyzer(
            config.get("cluster", {})
        )

        # Останні сигнали від кожного компонента
        self._last_signals: Dict[SignalSource, Any] = {}

        # Історія гібридних сигналів
        self._signal_history: List[HybridSignal] = []

        # Статистика
        self._total_signals = 0
        self._confirmed_signals = 0
        self._sources_agreement = {
            1: 0,  # 1 джерело
            2: 0,  # 2 джерела
            3: 0,  # 3 джерела
            4: 0,  # 4 джерела
        }

        # Підписатися на сигнали від tape analyzer
        self._tape_analyzer.on_signal(self._on_tape_signal)

        logger.info(
            f"[{self.name}] Initialized with weights: {self._config.weights}, "
            f"min_confirmations={self._config.min_confirmations}"
        )

    def _parse_config(self, config: Dict[str, Any]) -> HybridScalpingConfig:
        """Парсинг конфігурації."""
        weights = config.get("weights", {})
        return HybridScalpingConfig(
            enabled=config.get("enabled", True),
            signal_cooldown=config.get("signal_cooldown", 3.0),
            min_strength=config.get("min_strength", 0.6),
            weights={
                "orderbook": weights.get("orderbook", 0.35),
                "impulse": weights.get("impulse", 0.25),
                "tape": weights.get("tape", 0.25),
                "cluster": weights.get("cluster", 0.15),
            },
            min_confirmations=config.get("min_confirmations", 2),
            aggregation_mode=config.get("aggregation_mode", "weighted"),
            composite_threshold=config.get("composite_threshold", 0.6),
            default_sl_pct=config.get("default_sl_pct", 0.3),
            default_tp_pct=config.get("default_tp_pct", 0.5),
            dynamic_sl_tp=config.get("dynamic_sl_tp", True),
            max_spread_bps=config.get("max_spread_bps", 10.0),
            min_volume_usd=config.get("min_volume_usd", 10000)
        )

    async def start(self) -> None:
        """Запуск стратегії."""
        await self._impulse_strategy.start()
        logger.info(f"[{self.name}] Started")

    async def stop(self) -> None:
        """Зупинка стратегії."""
        await self._impulse_strategy.stop()
        logger.info(f"[{self.name}] Stopped")

    def on_orderbook(self, snapshot: OrderBookSnapshot) -> Optional[Signal]:
        """
        Обробка оновлення ордербуку.

        Args:
            snapshot: Знімок ордербуку

        Returns:
            Сигнал якщо умови виконані
        """
        symbol = snapshot.symbol

        # Базова валідація
        spread_bps = float(snapshot.spread_bps or 0)
        if spread_bps > self._config.max_spread_bps:
            return None

        # 1. Отримати сигнал від orderbook strategy
        orderbook_signal = self._orderbook_strategy.on_orderbook(snapshot)
        if orderbook_signal:
            self._last_signals[SignalSource.ORDERBOOK] = orderbook_signal

        # 2. Отримати сигнал від impulse strategy
        impulse_signal = self._impulse_strategy.on_orderbook(snapshot)
        if impulse_signal:
            self._last_signals[SignalSource.IMPULSE] = impulse_signal

        # 3. Перевірити cooldown
        if not self.can_signal():
            return None

        # 4. Агрегувати сигнали
        hybrid_signal = self._aggregate_signals(symbol, snapshot)

        if hybrid_signal:
            self._total_signals += 1
            self._confirmed_signals += 1
            self._sources_agreement[len(hybrid_signal.sources)] = \
                self._sources_agreement.get(len(hybrid_signal.sources), 0) + 1

            self._signal_history.append(hybrid_signal)

            # Конвертувати в Signal
            return self._to_signal(hybrid_signal, snapshot)

        return None

    def on_trade(self, trade: Trade) -> Optional[Signal]:
        """
        Обробка угоди.

        Args:
            trade: Торгова угода
        """
        symbol = trade.symbol

        # 1. Оновити tape analyzer
        tape_signal = self._tape_analyzer.process_trade(
            symbol=symbol,
            price=trade.price,
            quantity=trade.quantity,
            is_buyer_maker=trade.is_buyer_maker,
            timestamp=trade.timestamp
        )

        # 2. Оновити cluster analyzer
        cluster_signal = self._cluster_analyzer.process_trade(
            symbol=symbol,
            price=trade.price,
            quantity=trade.quantity,
            is_buyer_maker=trade.is_buyer_maker,
            timestamp=trade.timestamp
        )

        if cluster_signal:
            self._last_signals[SignalSource.CLUSTER] = cluster_signal

        # 3. Оновити orderbook strategy (для ікеберг-детекції)
        self._orderbook_strategy.on_trade(trade)

        return None

    def _on_tape_signal(self, signal: TapeSignal) -> None:
        """Callback для сигналів від tape analyzer."""
        self._last_signals[SignalSource.TAPE] = signal

    def update_leader_price(
        self,
        asset: LeaderAsset,
        price: Decimal,
        volume: Decimal = Decimal("0")
    ) -> None:
        """
        Оновити ціну індексу-поводиря.

        Args:
            asset: Індекс
            price: Ціна
            volume: Обсяг
        """
        self._impulse_strategy.update_leader_price(asset, price, volume)

    def _aggregate_signals(
        self,
        symbol: str,
        snapshot: OrderBookSnapshot
    ) -> Optional[HybridSignal]:
        """
        Агрегувати сигнали від усіх джерел.

        Args:
            symbol: Символ
            snapshot: Знімок ордербуку

        Returns:
            HybridSignal якщо підтверджено
        """
        now = datetime.utcnow()
        signal_window = timedelta(seconds=5)  # Сигнали актуальні 5 секунд

        # Зібрати актуальні сигнали
        signals: Dict[SignalSource, Tuple[str, float]] = {}

        # Orderbook
        ob_signal = self._last_signals.get(SignalSource.ORDERBOOK)
        if ob_signal and (now - ob_signal.timestamp).total_seconds() < 5:
            direction = "long" if ob_signal.signal_type == SignalType.LONG else "short"
            signals[SignalSource.ORDERBOOK] = (direction, ob_signal.strength)

        # Impulse
        imp_signal = self._last_signals.get(SignalSource.IMPULSE)
        if imp_signal and (now - imp_signal.timestamp).total_seconds() < 5:
            direction = "long" if imp_signal.signal_type == SignalType.LONG else "short"
            signals[SignalSource.IMPULSE] = (direction, imp_signal.strength)

        # Tape
        tape_signal = self._last_signals.get(SignalSource.TAPE)
        if tape_signal and (now - tape_signal.timestamp).total_seconds() < 5:
            direction = "long" if tape_signal.direction == "bullish" else "short"
            signals[SignalSource.TAPE] = (direction, tape_signal.strength)

        # Cluster
        cluster_signal = self._last_signals.get(SignalSource.CLUSTER)
        if cluster_signal and (now - cluster_signal.timestamp).total_seconds() < 5:
            direction = "long" if cluster_signal.direction == "bullish" else "short"
            signals[SignalSource.CLUSTER] = (direction, cluster_signal.strength)

        # Перевірити мінімальну кількість підтверджень
        if len(signals) < self._config.min_confirmations:
            return None

        # Визначити напрямок консенсусу
        long_score = 0.0
        short_score = 0.0

        for source, (direction, strength) in signals.items():
            weight = self._config.weights.get(source.value, 0.25)
            if direction == "long":
                long_score += strength * weight
            else:
                short_score += strength * weight

        # Перевірити поріг
        if max(long_score, short_score) < self._config.composite_threshold:
            return None

        # Визначити фінальний напрямок
        if long_score > short_score:
            final_direction = "long"
            final_strength = long_score
        else:
            final_direction = "short"
            final_strength = short_score

        # Перевірити консенсус напрямків
        directions = [d for d, _ in signals.values()]
        if self._config.aggregation_mode == "unanimous":
            if not all(d == final_direction for d in directions):
                return None

        # Розрахувати SL/TP
        entry_price = snapshot.mid_price
        sl, tp = self._calculate_sl_tp(
            symbol, entry_price, final_direction, snapshot
        )

        # Створити гібридний сигнал
        return HybridSignal(
            timestamp=now,
            direction=final_direction,
            strength=final_strength,
            sources=list(signals.keys()),
            entry_price=entry_price,
            stop_loss=sl,
            take_profit=tp,
            orderbook_signal=ob_signal,
            impulse_events=self._impulse_strategy.get_active_impulses(),
            tape_signal=tape_signal,
            cluster_signal=cluster_signal,
            metadata={
                "long_score": long_score,
                "short_score": short_score,
                "confirmations": len(signals),
                "spread_bps": float(snapshot.spread_bps or 0),
            }
        )

    def _calculate_sl_tp(
        self,
        symbol: str,
        entry_price: Decimal,
        direction: str,
        snapshot: OrderBookSnapshot
    ) -> Tuple[Decimal, Decimal]:
        """
        Розрахувати Stop Loss та Take Profit.

        Args:
            symbol: Символ
            entry_price: Ціна входу
            direction: Напрямок
            snapshot: Знімок ордербуку

        Returns:
            (stop_loss, take_profit)
        """
        if self._config.dynamic_sl_tp:
            # Динамічний розрахунок на основі стін та VA
            walls = self._orderbook_strategy.get_walls(symbol)
            supports, resistances = self._orderbook_strategy.get_confirmed_support_resistance(symbol)

            if direction == "long":
                # SL нижче найближчої підтримки
                if supports:
                    sl = supports[0] * Decimal("0.999")
                else:
                    sl = entry_price * (1 - Decimal(str(self._config.default_sl_pct / 100)))

                # TP на найближчому опорі
                if resistances:
                    tp = resistances[0] * Decimal("0.999")
                else:
                    tp = entry_price * (1 + Decimal(str(self._config.default_tp_pct / 100)))
            else:
                # Short
                if resistances:
                    sl = resistances[0] * Decimal("1.001")
                else:
                    sl = entry_price * (1 + Decimal(str(self._config.default_sl_pct / 100)))

                if supports:
                    tp = supports[0] * Decimal("1.001")
                else:
                    tp = entry_price * (1 - Decimal(str(self._config.default_tp_pct / 100)))
        else:
            # Фіксовані SL/TP
            sl_pct = Decimal(str(self._config.default_sl_pct / 100))
            tp_pct = Decimal(str(self._config.default_tp_pct / 100))

            if direction == "long":
                sl = entry_price * (1 - sl_pct)
                tp = entry_price * (1 + tp_pct)
            else:
                sl = entry_price * (1 + sl_pct)
                tp = entry_price * (1 - tp_pct)

        return sl, tp

    def _to_signal(
        self,
        hybrid: HybridSignal,
        snapshot: OrderBookSnapshot
    ) -> Signal:
        """Конвертувати HybridSignal в Signal."""
        signal_type = SignalType.LONG if hybrid.direction == "long" else SignalType.SHORT

        return self.create_signal(
            signal_type=signal_type,
            symbol=snapshot.symbol,
            price=snapshot.mid_price,
            strength=hybrid.strength,
            metadata={
                "signal_source": "hybrid",
                "sources": [s.value for s in hybrid.sources],
                "confirmations": len(hybrid.sources),
                "stop_loss": float(hybrid.stop_loss),
                "take_profit": float(hybrid.take_profit),
                "long_score": hybrid.metadata.get("long_score"),
                "short_score": hybrid.metadata.get("short_score"),
            }
        )

    def get_component_status(self) -> Dict[str, Any]:
        """Отримати статус всіх компонентів."""
        return {
            "orderbook": self._orderbook_strategy.stats,
            "impulse": self._impulse_strategy.stats,
            "tape": self._tape_analyzer.stats,
            "cluster": self._cluster_analyzer.stats,
        }

    def get_order_flow_metrics(
        self,
        symbol: str
    ) -> Optional[OrderFlowMetrics]:
        """Отримати метрики потоку ордерів."""
        return self._tape_analyzer.get_metrics(symbol, "short")

    def get_current_cluster(
        self,
        symbol: str
    ) -> Optional[VolumeCluster]:
        """Отримати поточний кластер."""
        return self._cluster_analyzer.get_current_cluster(symbol)

    def get_walls(self, symbol: str) -> List[OrderBookWall]:
        """Отримати стіни ордербуку."""
        return self._orderbook_strategy.get_walls(symbol)

    def get_active_impulses(self) -> List[ImpulseEvent]:
        """Отримати активні імпульси."""
        return self._impulse_strategy.get_active_impulses()

    @property
    def stats(self) -> Dict[str, Any]:
        """Статистика стратегії."""
        base = super().stats
        base.update({
            "total_signals": self._total_signals,
            "confirmed_signals": self._confirmed_signals,
            "confirmation_rate": (
                self._confirmed_signals / self._total_signals * 100
                if self._total_signals > 0 else 0
            ),
            "sources_agreement": self._sources_agreement,
            "components": self.get_component_status(),
        })
        return base

    def reset(self) -> None:
        """Скинути стан."""
        super().reset()
        self._orderbook_strategy.reset()
        self._impulse_strategy.reset()
        self._tape_analyzer.reset()
        self._cluster_analyzer.reset()
        self._last_signals.clear()
        self._signal_history.clear()
        self._total_signals = 0
        self._confirmed_signals = 0
        self._sources_agreement = {1: 0, 2: 0, 3: 0, 4: 0}
