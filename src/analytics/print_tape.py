"""
Print Tape Analyzer (Лента принтів / Time & Sales).

Аналізатор стрічки угод для скальпінгу.
Відстежує:
- Агресивні покупці/продавці (market orders)
- Великі угоди (whale trades)
- Кластери угод
- Швидкість потоку ордерів
- Дисбаланс агресорів

Концепція з класичного скальпінгу:
"Лента принтів показує реальні угоди та волатильність ринку в реальному часі"
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, Deque, Tuple, Callable
from enum import Enum

from loguru import logger


class TradeAggressor(Enum):
    """Хто ініціював угоду."""
    BUYER = "buyer"    # Покупець (купив по ask)
    SELLER = "seller"  # Продавець (продав по bid)
    UNKNOWN = "unknown"


class TradeSize(Enum):
    """Класифікація розміру угоди."""
    SMALL = "small"         # Роздрібна
    MEDIUM = "medium"       # Середня
    LARGE = "large"         # Велика
    WHALE = "whale"         # Кит
    MEGA_WHALE = "mega"     # Мега-кит


@dataclass
class TapeEntry:
    """Запис в стрічці угод."""
    timestamp: datetime
    symbol: str
    price: Decimal
    quantity: Decimal
    value_usd: Decimal
    aggressor: TradeAggressor
    size_class: TradeSize
    is_liquidation: bool = False
    trade_id: Optional[str] = None


@dataclass
class TapeCluster:
    """Кластер угод (серія угод в одному напрямку)."""
    start_time: datetime
    end_time: datetime
    direction: TradeAggressor
    trades_count: int
    total_volume: Decimal
    total_value_usd: Decimal
    price_start: Decimal
    price_end: Decimal
    avg_trade_size: Decimal
    has_whale: bool = False


@dataclass
class OrderFlowMetrics:
    """
    Метрики потоку ордерів.

    Розширено концепціями з книги "Скальпинг":
    - Velocity (швидкість) - поточна швидкість потоку
    - Acceleration (прискорення) - зміна швидкості
    """
    timestamp: datetime

    # Обсяги
    buy_volume: Decimal = Decimal("0")
    sell_volume: Decimal = Decimal("0")
    total_volume: Decimal = Decimal("0")

    # Кількість угод
    buy_trades: int = 0
    sell_trades: int = 0
    total_trades: int = 0

    # Великі угоди
    whale_buys: int = 0
    whale_sells: int = 0
    whale_volume: Decimal = Decimal("0")

    # Дельта (дисбаланс)
    volume_delta: Decimal = Decimal("0")  # buy - sell
    delta_percent: float = 0.0

    # Швидкість (Velocity)
    trades_per_second: float = 0.0
    volume_per_second: Decimal = Decimal("0")

    # Прискорення (Acceleration) - з книги "Скальпинг"
    trade_acceleration: float = 0.0      # Зміна trades_per_second
    volume_acceleration: float = 0.0      # Зміна volume_per_second
    delta_acceleration: float = 0.0       # Зміна delta (напрямок тиску)

    # CVD (Cumulative Volume Delta)
    cvd: Decimal = Decimal("0")

    # Momentum індикатори
    buy_pressure: float = 0.0    # 0-1, частка покупок у потоці
    sell_pressure: float = 0.0   # 0-1, частка продажів у потоці
    pressure_change: float = 0.0  # Зміна тиску (-1 до 1)

    @property
    def is_bullish(self) -> bool:
        """Чи бичачий потік."""
        return self.volume_delta > 0 and self.delta_percent > 10

    @property
    def is_bearish(self) -> bool:
        """Чи ведмежий потік."""
        return self.volume_delta < 0 and self.delta_percent < -10

    @property
    def is_accelerating_bullish(self) -> bool:
        """Чи прискорюється бичачий потік."""
        return self.is_bullish and self.delta_acceleration > 0

    @property
    def is_accelerating_bearish(self) -> bool:
        """Чи прискорюється ведмежий потік."""
        return self.is_bearish and self.delta_acceleration < 0

    @property
    def flow_intensity(self) -> float:
        """Інтенсивність потоку (0-1 нормалізована)."""
        if self.trades_per_second == 0:
            return 0
        # Нормалізуємо по типовому діапазону 0-10 trades/sec
        return min(self.trades_per_second / 10, 1.0)


@dataclass
class TapeSignal:
    """Сигнал від аналізу стрічки."""
    timestamp: datetime
    signal_type: str  # "aggressive_buying", "aggressive_selling", "whale_alert", etc.
    direction: str    # "bullish", "bearish"
    strength: float   # 0.0 - 1.0
    description: str
    metrics: OrderFlowMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrintTapeConfig:
    """Конфігурація аналізатора стрічки."""
    # Розміри угод (в USD)
    small_threshold: float = 1000
    medium_threshold: float = 10000
    large_threshold: float = 50000
    whale_threshold: float = 100000
    mega_whale_threshold: float = 500000

    # Вікна аналізу
    short_window_seconds: int = 10
    medium_window_seconds: int = 60
    long_window_seconds: int = 300

    # Пороги сигналів
    delta_threshold_percent: float = 20.0  # Дельта > 20% для сигналу
    whale_cluster_threshold: int = 3       # 3+ китів = кластер
    flow_spike_multiplier: float = 3.0     # 3x від середнього = спайк

    # Кластеризація
    cluster_time_gap_ms: int = 500         # Макс. проміжок між угодами в кластері
    min_cluster_trades: int = 5            # Мін. угод для кластера

    # Буфер
    max_tape_entries: int = 10000          # Макс. записів в буфері
    max_clusters: int = 100                # Макс. кластерів


class PrintTapeAnalyzer:
    """
    Аналізатор стрічки угод (Print Tape / Time & Sales).

    Використовує концепції класичного скальпінгу:
    1. Відстеження агресорів (хто ініціює угоди)
    2. Виявлення великих гравців (китів)
    3. Кластерний аналіз потоку угод
    4. CVD (Cumulative Volume Delta)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Ініціалізація аналізатора.

        Args:
            config: Конфігурація
        """
        self._config = self._parse_config(config or {})

        # Буфери для кожного символу
        self._tape: Dict[str, Deque[TapeEntry]] = {}
        self._clusters: Dict[str, Deque[TapeCluster]] = {}

        # CVD (Cumulative Volume Delta) для кожного символу
        self._cvd: Dict[str, Decimal] = {}

        # Метрики по вікнах
        self._short_metrics: Dict[str, OrderFlowMetrics] = {}
        self._medium_metrics: Dict[str, OrderFlowMetrics] = {}
        self._long_metrics: Dict[str, OrderFlowMetrics] = {}

        # Середні значення для визначення спайків
        self._avg_volume_per_sec: Dict[str, Decimal] = {}
        self._avg_trades_per_sec: Dict[str, float] = {}

        # Історія метрик для розрахунку прискорення
        self._metrics_history: Dict[str, Deque[OrderFlowMetrics]] = {}

        # Callbacks для сигналів
        self._signal_callbacks: List[Callable[[TapeSignal], None]] = []

        # Статистика
        self._total_trades_processed = 0
        self._signals_generated = 0
        self._acceleration_signals = 0

        logger.info(
            f"[PrintTapeAnalyzer] Initialized with whale threshold: "
            f"${self._config.whale_threshold:,.0f}"
        )

    def _parse_config(self, config: Dict[str, Any]) -> PrintTapeConfig:
        """Парсинг конфігурації."""
        return PrintTapeConfig(
            small_threshold=config.get("small_threshold", 1000),
            medium_threshold=config.get("medium_threshold", 10000),
            large_threshold=config.get("large_threshold", 50000),
            whale_threshold=config.get("whale_threshold", 100000),
            mega_whale_threshold=config.get("mega_whale_threshold", 500000),
            short_window_seconds=config.get("short_window_seconds", 10),
            medium_window_seconds=config.get("medium_window_seconds", 60),
            long_window_seconds=config.get("long_window_seconds", 300),
            delta_threshold_percent=config.get("delta_threshold_percent", 20.0),
            whale_cluster_threshold=config.get("whale_cluster_threshold", 3),
            flow_spike_multiplier=config.get("flow_spike_multiplier", 3.0),
            cluster_time_gap_ms=config.get("cluster_time_gap_ms", 500),
            min_cluster_trades=config.get("min_cluster_trades", 5),
            max_tape_entries=config.get("max_tape_entries", 10000),
            max_clusters=config.get("max_clusters", 100)
        )

    def on_signal(self, callback: Callable[[TapeSignal], None]) -> None:
        """
        Зареєструвати callback для сигналів.

        Args:
            callback: Функція для виклику при сигналі
        """
        self._signal_callbacks.append(callback)

    def process_trade(
        self,
        symbol: str,
        price: Decimal,
        quantity: Decimal,
        is_buyer_maker: bool,
        timestamp: Optional[datetime] = None,
        trade_id: Optional[str] = None,
        is_liquidation: bool = False
    ) -> Optional[TapeSignal]:
        """
        Обробити нову угоду.

        Args:
            symbol: Символ
            price: Ціна
            quantity: Обсяг
            is_buyer_maker: True якщо покупець був maker (продавець - aggressor)
            timestamp: Час угоди
            trade_id: ID угоди
            is_liquidation: Чи це ліквідація

        Returns:
            TapeSignal якщо умови виконані
        """
        ts = timestamp or datetime.utcnow()

        # Ініціалізація буферів для символу
        if symbol not in self._tape:
            self._tape[symbol] = deque(maxlen=self._config.max_tape_entries)
            self._clusters[symbol] = deque(maxlen=self._config.max_clusters)
            self._cvd[symbol] = Decimal("0")

        # Визначити агресора
        aggressor = TradeAggressor.SELLER if is_buyer_maker else TradeAggressor.BUYER

        # Розрахувати вартість
        value_usd = price * quantity

        # Класифікувати розмір
        size_class = self._classify_size(float(value_usd))

        # Створити запис
        entry = TapeEntry(
            timestamp=ts,
            symbol=symbol,
            price=price,
            quantity=quantity,
            value_usd=value_usd,
            aggressor=aggressor,
            size_class=size_class,
            is_liquidation=is_liquidation,
            trade_id=trade_id
        )

        # Додати до буферу
        self._tape[symbol].append(entry)

        # Оновити CVD
        if aggressor == TradeAggressor.BUYER:
            self._cvd[symbol] += value_usd
        else:
            self._cvd[symbol] -= value_usd

        # Оновити метрики
        self._update_metrics(symbol)

        # Оновити кластери
        self._update_clusters(symbol, entry)

        # Статистика
        self._total_trades_processed += 1

        # Перевірити сигнали
        signal = self._check_signals(symbol)

        if signal:
            self._signals_generated += 1
            for callback in self._signal_callbacks:
                try:
                    callback(signal)
                except Exception as e:
                    logger.error(f"[PrintTapeAnalyzer] Callback error: {e}")

        return signal

    def _classify_size(self, value_usd: float) -> TradeSize:
        """Класифікувати розмір угоди."""
        if value_usd >= self._config.mega_whale_threshold:
            return TradeSize.MEGA_WHALE
        elif value_usd >= self._config.whale_threshold:
            return TradeSize.WHALE
        elif value_usd >= self._config.large_threshold:
            return TradeSize.LARGE
        elif value_usd >= self._config.medium_threshold:
            return TradeSize.MEDIUM
        else:
            return TradeSize.SMALL

    def _update_metrics(self, symbol: str) -> None:
        """Оновити метрики потоку для символу."""
        tape = self._tape.get(symbol, deque())
        if not tape:
            return

        now = datetime.utcnow()

        # Оновити метрики для кожного вікна
        for window, metrics_dict, name in [
            (self._config.short_window_seconds, self._short_metrics, "short"),
            (self._config.medium_window_seconds, self._medium_metrics, "medium"),
            (self._config.long_window_seconds, self._long_metrics, "long"),
        ]:
            cutoff = now - timedelta(seconds=window)
            window_trades = [t for t in tape if t.timestamp >= cutoff]

            if not window_trades:
                continue

            metrics = self._calculate_metrics(window_trades, window, symbol)
            metrics_dict[symbol] = metrics

    def _calculate_metrics(
        self,
        trades: List[TapeEntry],
        window_seconds: int,
        symbol: str
    ) -> OrderFlowMetrics:
        """
        Розрахувати метрики для набору угод.

        Включає розрахунок velocity та acceleration з книги "Скальпинг".
        """
        buy_volume = Decimal("0")
        sell_volume = Decimal("0")
        buy_trades = 0
        sell_trades = 0
        whale_buys = 0
        whale_sells = 0
        whale_volume = Decimal("0")

        for trade in trades:
            if trade.aggressor == TradeAggressor.BUYER:
                buy_volume += trade.value_usd
                buy_trades += 1
                if trade.size_class in (TradeSize.WHALE, TradeSize.MEGA_WHALE):
                    whale_buys += 1
                    whale_volume += trade.value_usd
            else:
                sell_volume += trade.value_usd
                sell_trades += 1
                if trade.size_class in (TradeSize.WHALE, TradeSize.MEGA_WHALE):
                    whale_sells += 1
                    whale_volume += trade.value_usd

        total_volume = buy_volume + sell_volume
        volume_delta = buy_volume - sell_volume

        delta_percent = 0.0
        if total_volume > 0:
            delta_percent = float(volume_delta / total_volume * 100)

        trades_per_second = len(trades) / window_seconds if window_seconds > 0 else 0
        volume_per_second = total_volume / window_seconds if window_seconds > 0 else Decimal("0")

        # Розрахунок buy/sell pressure
        total_trades = len(trades)
        buy_pressure = buy_trades / total_trades if total_trades > 0 else 0.5
        sell_pressure = sell_trades / total_trades if total_trades > 0 else 0.5

        # Розрахунок acceleration (з книги "Скальпинг")
        trade_acceleration = 0.0
        volume_acceleration = 0.0
        delta_acceleration = 0.0
        pressure_change = 0.0

        # Ініціалізувати історію якщо потрібно
        if symbol not in self._metrics_history:
            self._metrics_history[symbol] = deque(maxlen=20)

        history = self._metrics_history[symbol]
        if history:
            prev_metrics = history[-1]

            # Розрахунок прискорення
            time_diff = (datetime.utcnow() - prev_metrics.timestamp).total_seconds()
            if time_diff > 0:
                # Trade acceleration (зміна швидкості угод)
                trade_acceleration = (trades_per_second - prev_metrics.trades_per_second) / time_diff

                # Volume acceleration (зміна швидкості обсягу)
                prev_vol_per_sec = float(prev_metrics.volume_per_second)
                curr_vol_per_sec = float(volume_per_second)
                volume_acceleration = (curr_vol_per_sec - prev_vol_per_sec) / time_diff

                # Delta acceleration (зміна напрямку тиску)
                delta_acceleration = (delta_percent - prev_metrics.delta_percent) / time_diff

                # Pressure change (зміна тиску buy/sell)
                pressure_change = buy_pressure - prev_metrics.buy_pressure

        metrics = OrderFlowMetrics(
            timestamp=datetime.utcnow(),
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            total_volume=total_volume,
            buy_trades=buy_trades,
            sell_trades=sell_trades,
            total_trades=total_trades,
            whale_buys=whale_buys,
            whale_sells=whale_sells,
            whale_volume=whale_volume,
            volume_delta=volume_delta,
            delta_percent=delta_percent,
            trades_per_second=trades_per_second,
            volume_per_second=volume_per_second,
            trade_acceleration=trade_acceleration,
            volume_acceleration=volume_acceleration,
            delta_acceleration=delta_acceleration,
            cvd=self._cvd.get(symbol, Decimal("0")),
            buy_pressure=buy_pressure,
            sell_pressure=sell_pressure,
            pressure_change=pressure_change
        )

        # Зберегти в історію
        self._metrics_history[symbol].append(metrics)

        return metrics

    def _update_clusters(self, symbol: str, new_entry: TapeEntry) -> None:
        """Оновити кластери угод."""
        clusters = self._clusters[symbol]

        # Спробувати додати до існуючого кластера
        if clusters:
            last_cluster = clusters[-1]
            time_gap = (new_entry.timestamp - last_cluster.end_time).total_seconds() * 1000

            # Якщо угода в тому ж напрямку і не великий проміжок
            if (new_entry.aggressor.value == last_cluster.direction.value and
                time_gap <= self._config.cluster_time_gap_ms):
                # Оновити кластер
                clusters[-1] = TapeCluster(
                    start_time=last_cluster.start_time,
                    end_time=new_entry.timestamp,
                    direction=last_cluster.direction,
                    trades_count=last_cluster.trades_count + 1,
                    total_volume=last_cluster.total_volume + new_entry.quantity,
                    total_value_usd=last_cluster.total_value_usd + new_entry.value_usd,
                    price_start=last_cluster.price_start,
                    price_end=new_entry.price,
                    avg_trade_size=(last_cluster.total_value_usd + new_entry.value_usd) / (last_cluster.trades_count + 1),
                    has_whale=last_cluster.has_whale or new_entry.size_class in (TradeSize.WHALE, TradeSize.MEGA_WHALE)
                )
                return

        # Створити новий кластер
        clusters.append(TapeCluster(
            start_time=new_entry.timestamp,
            end_time=new_entry.timestamp,
            direction=new_entry.aggressor,
            trades_count=1,
            total_volume=new_entry.quantity,
            total_value_usd=new_entry.value_usd,
            price_start=new_entry.price,
            price_end=new_entry.price,
            avg_trade_size=new_entry.value_usd,
            has_whale=new_entry.size_class in (TradeSize.WHALE, TradeSize.MEGA_WHALE)
        ))

    def _check_signals(self, symbol: str) -> Optional[TapeSignal]:
        """Перевірити умови для сигналів."""
        short_metrics = self._short_metrics.get(symbol)
        medium_metrics = self._medium_metrics.get(symbol)

        if not short_metrics:
            return None

        # Сигнал 1: Агресивні покупки/продажі
        if abs(short_metrics.delta_percent) > self._config.delta_threshold_percent:
            direction = "bullish" if short_metrics.delta_percent > 0 else "bearish"
            signal_type = "aggressive_buying" if direction == "bullish" else "aggressive_selling"

            strength = min(abs(short_metrics.delta_percent) / 50, 1.0)

            return TapeSignal(
                timestamp=datetime.utcnow(),
                signal_type=signal_type,
                direction=direction,
                strength=strength,
                description=f"Strong {direction} order flow: {short_metrics.delta_percent:.1f}% delta",
                metrics=short_metrics,
                metadata={
                    "delta_percent": short_metrics.delta_percent,
                    "buy_volume": float(short_metrics.buy_volume),
                    "sell_volume": float(short_metrics.sell_volume),
                }
            )

        # Сигнал 2: Кластер китів
        whale_count = short_metrics.whale_buys + short_metrics.whale_sells
        if whale_count >= self._config.whale_cluster_threshold:
            direction = "bullish" if short_metrics.whale_buys > short_metrics.whale_sells else "bearish"

            return TapeSignal(
                timestamp=datetime.utcnow(),
                signal_type="whale_cluster",
                direction=direction,
                strength=min(whale_count / 5, 1.0),
                description=f"Whale cluster detected: {whale_count} large trades",
                metrics=short_metrics,
                metadata={
                    "whale_buys": short_metrics.whale_buys,
                    "whale_sells": short_metrics.whale_sells,
                    "whale_volume": float(short_metrics.whale_volume),
                }
            )

        # Сигнал 3: Спайк обсягу
        avg_vol = self._avg_volume_per_sec.get(symbol, Decimal("0"))
        if avg_vol > 0 and short_metrics.volume_per_second > avg_vol * Decimal(str(self._config.flow_spike_multiplier)):
            direction = "bullish" if short_metrics.delta_percent > 0 else "bearish"

            spike_ratio = float(short_metrics.volume_per_second / avg_vol)

            return TapeSignal(
                timestamp=datetime.utcnow(),
                signal_type="volume_spike",
                direction=direction,
                strength=min(spike_ratio / 5, 1.0),
                description=f"Volume spike: {spike_ratio:.1f}x average",
                metrics=short_metrics,
                metadata={
                    "spike_ratio": spike_ratio,
                    "current_volume_per_sec": float(short_metrics.volume_per_second),
                    "avg_volume_per_sec": float(avg_vol),
                }
            )

        # Сигнал 4: Прискорення потоку (з книги "Скальпинг")
        if abs(short_metrics.trade_acceleration) > 0.5:  # Сильне прискорення
            if short_metrics.is_accelerating_bullish:
                self._acceleration_signals += 1
                return TapeSignal(
                    timestamp=datetime.utcnow(),
                    signal_type="acceleration_bullish",
                    direction="bullish",
                    strength=min(abs(short_metrics.trade_acceleration) / 2, 1.0),
                    description=f"Accelerating bullish flow: {short_metrics.trade_acceleration:.2f} trades/sec²",
                    metrics=short_metrics,
                    metadata={
                        "trade_acceleration": short_metrics.trade_acceleration,
                        "volume_acceleration": short_metrics.volume_acceleration,
                        "delta_acceleration": short_metrics.delta_acceleration,
                        "buy_pressure": short_metrics.buy_pressure,
                    }
                )
            elif short_metrics.is_accelerating_bearish:
                self._acceleration_signals += 1
                return TapeSignal(
                    timestamp=datetime.utcnow(),
                    signal_type="acceleration_bearish",
                    direction="bearish",
                    strength=min(abs(short_metrics.trade_acceleration) / 2, 1.0),
                    description=f"Accelerating bearish flow: {short_metrics.trade_acceleration:.2f} trades/sec²",
                    metrics=short_metrics,
                    metadata={
                        "trade_acceleration": short_metrics.trade_acceleration,
                        "volume_acceleration": short_metrics.volume_acceleration,
                        "delta_acceleration": short_metrics.delta_acceleration,
                        "sell_pressure": short_metrics.sell_pressure,
                    }
                )

        # Сигнал 5: Різка зміна тиску (pressure reversal)
        if abs(short_metrics.pressure_change) > 0.3:  # 30%+ зміна
            direction = "bullish" if short_metrics.pressure_change > 0 else "bearish"
            return TapeSignal(
                timestamp=datetime.utcnow(),
                signal_type="pressure_reversal",
                direction=direction,
                strength=min(abs(short_metrics.pressure_change) * 2, 1.0),
                description=f"Pressure reversal to {direction}: {short_metrics.pressure_change:.1%}",
                metrics=short_metrics,
                metadata={
                    "pressure_change": short_metrics.pressure_change,
                    "buy_pressure": short_metrics.buy_pressure,
                    "sell_pressure": short_metrics.sell_pressure,
                }
            )

        # Оновити середні значення (для майбутніх спайків)
        if medium_metrics and medium_metrics.volume_per_second > 0:
            if symbol not in self._avg_volume_per_sec:
                self._avg_volume_per_sec[symbol] = medium_metrics.volume_per_second
            else:
                # EMA з коефіцієнтом 0.1
                self._avg_volume_per_sec[symbol] = (
                    self._avg_volume_per_sec[symbol] * Decimal("0.9") +
                    medium_metrics.volume_per_second * Decimal("0.1")
                )

        return None

    def get_metrics(
        self,
        symbol: str,
        window: str = "short"
    ) -> Optional[OrderFlowMetrics]:
        """
        Отримати метрики для символу.

        Args:
            symbol: Символ
            window: "short", "medium", або "long"

        Returns:
            Метрики або None
        """
        if window == "short":
            return self._short_metrics.get(symbol)
        elif window == "medium":
            return self._medium_metrics.get(symbol)
        elif window == "long":
            return self._long_metrics.get(symbol)
        return None

    def get_recent_clusters(
        self,
        symbol: str,
        count: int = 10
    ) -> List[TapeCluster]:
        """
        Отримати останні кластери угод.

        Args:
            symbol: Символ
            count: Кількість

        Returns:
            Список кластерів
        """
        clusters = self._clusters.get(symbol, deque())
        return list(clusters)[-count:]

    def get_cvd(self, symbol: str) -> Decimal:
        """
        Отримати Cumulative Volume Delta.

        Args:
            symbol: Символ

        Returns:
            CVD значення
        """
        return self._cvd.get(symbol, Decimal("0"))

    def get_tape(
        self,
        symbol: str,
        count: int = 100,
        min_size: Optional[TradeSize] = None
    ) -> List[TapeEntry]:
        """
        Отримати останні записи стрічки.

        Args:
            symbol: Символ
            count: Кількість записів
            min_size: Мінімальний розмір угоди для фільтрації

        Returns:
            Список записів
        """
        tape = self._tape.get(symbol, deque())

        if min_size:
            size_order = [
                TradeSize.SMALL, TradeSize.MEDIUM, TradeSize.LARGE,
                TradeSize.WHALE, TradeSize.MEGA_WHALE
            ]
            min_index = size_order.index(min_size)
            filtered = [
                t for t in tape
                if size_order.index(t.size_class) >= min_index
            ]
            return filtered[-count:]

        return list(tape)[-count:]

    def get_whale_trades(
        self,
        symbol: str,
        seconds: int = 60
    ) -> List[TapeEntry]:
        """
        Отримати угоди китів за період.

        Args:
            symbol: Символ
            seconds: Період в секундах

        Returns:
            Список угод китів
        """
        tape = self._tape.get(symbol, deque())
        cutoff = datetime.utcnow() - timedelta(seconds=seconds)

        return [
            t for t in tape
            if t.timestamp >= cutoff and
            t.size_class in (TradeSize.WHALE, TradeSize.MEGA_WHALE)
        ]

    @property
    def stats(self) -> Dict[str, Any]:
        """Отримати статистику аналізатора."""
        return {
            "total_trades_processed": self._total_trades_processed,
            "signals_generated": self._signals_generated,
            "acceleration_signals": self._acceleration_signals,
            "symbols_tracked": len(self._tape),
            "config": {
                "whale_threshold": self._config.whale_threshold,
                "delta_threshold": self._config.delta_threshold_percent,
            }
        }

    def reset(self, symbol: Optional[str] = None) -> None:
        """
        Скинути стан аналізатора.

        Args:
            symbol: Символ для скидання (або всі якщо None)
        """
        if symbol:
            if symbol in self._tape:
                self._tape[symbol].clear()
            if symbol in self._clusters:
                self._clusters[symbol].clear()
            if symbol in self._cvd:
                self._cvd[symbol] = Decimal("0")
            if symbol in self._metrics_history:
                self._metrics_history[symbol].clear()
            self._short_metrics.pop(symbol, None)
            self._medium_metrics.pop(symbol, None)
            self._long_metrics.pop(symbol, None)
        else:
            self._tape.clear()
            self._clusters.clear()
            self._cvd.clear()
            self._metrics_history.clear()
            self._short_metrics.clear()
            self._medium_metrics.clear()
            self._long_metrics.clear()
            self._avg_volume_per_sec.clear()
            self._avg_trades_per_sec.clear()
            self._acceleration_signals = 0

        logger.info(f"[PrintTapeAnalyzer] Reset {'symbol ' + symbol if symbol else 'all'}")
