"""
Trendline Breakout Strategy (Стратегія пробою трендової лінії).

Реалізує стратегію #3 з книги "Скальпинг: практическое руководство трейдера".

Концепція:
"Будуємо трендову лінію по локальних мінімумах (для висхідного тренду)
або максимумах (для низхідного тренду). Пробій лінії з обсягом - сигнал
для входу в протилежному напрямку."

Логіка:
1. Автоматично визначаємо pivot points (локальні екстремуми)
2. Будуємо трендову лінію по 2+ точках
3. Чекаємо на пробій лінії
4. Підтверджуємо пробій обсягом
5. Входимо в напрямку пробою
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, Deque, Tuple
from enum import Enum
import math

from loguru import logger

from src.data.models import OrderBookSnapshot, Signal, SignalType, Trade
from src.strategy.base import BaseStrategy


class TrendDirection(Enum):
    """Напрямок тренду."""
    UP = "up"
    DOWN = "down"


class PivotType(Enum):
    """Тип pivot point."""
    HIGH = "high"   # Локальний максимум
    LOW = "low"     # Локальний мінімум


@dataclass
class PivotPoint:
    """Pivot point (локальний екстремум)."""
    timestamp: datetime
    price: Decimal
    pivot_type: PivotType
    index: int  # Індекс в історії
    strength: int = 1  # Кількість барів зліва/справа


@dataclass
class Trendline:
    """Трендова лінія."""
    direction: TrendDirection
    start_point: PivotPoint
    end_point: PivotPoint
    slope: float  # Нахил (ціна за секунду)
    intercept: float  # Перетин
    touches: int = 2  # Кількість дотиків
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_valid: bool = True
    last_touch: Optional[datetime] = None

    def price_at_time(self, timestamp: datetime) -> Decimal:
        """Розрахувати ціну на лінії для даного часу."""
        time_diff = (timestamp - self.start_point.timestamp).total_seconds()
        price = float(self.start_point.price) + self.slope * time_diff
        return Decimal(str(price))

    @property
    def age_seconds(self) -> float:
        """Вік лінії в секундах."""
        return (datetime.utcnow() - self.created_at).total_seconds()


@dataclass
class BreakoutEvent:
    """Подія пробою трендової лінії."""
    timestamp: datetime
    trendline: Trendline
    breakout_price: Decimal
    trendline_price: Decimal  # Ціна на лінії в момент пробою
    breakout_distance_pct: float
    volume_at_breakout: Decimal
    avg_volume: Decimal
    is_confirmed: bool = False


@dataclass
class TrendlineBreakoutConfig:
    """Конфігурація стратегії пробою трендових ліній."""
    enabled: bool = True
    signal_cooldown: float = 15.0
    min_strength: float = 0.6

    # Pivot detection
    pivot_lookback: int = 5           # Барів зліва/справа для pivot
    min_pivot_distance: int = 10      # Мін. відстань між pivots (барів)
    max_pivots_to_track: int = 20     # Макс. pivots для зберігання

    # Trendline creation
    min_trendline_touches: int = 2    # Мін. дотиків для валідної лінії
    max_trendline_age_sec: int = 3600  # Макс. вік лінії (1 година)
    min_trendline_slope: float = 0.00001  # Мін. нахил
    trendline_touch_tolerance_pct: float = 0.1  # Толерантність дотику (%)

    # Breakout detection
    breakout_threshold_pct: float = 0.15  # Поріг пробою (% від ціни)
    breakout_confirmation_bars: int = 2   # Барів для підтвердження
    volume_confirmation_multiplier: float = 1.3  # Обсяг для підтвердження

    # Trading
    entry_after_pullback: bool = True   # Входити після відкату
    pullback_threshold_pct: float = 0.05  # Поріг відкату
    stop_beyond_trendline_pct: float = 0.3  # Стоп за лінією
    target_extension_pct: float = 0.5    # Ціль = пробій * extension

    # Filters
    max_spread_bps: float = 15.0
    min_trend_angle_deg: float = 5.0    # Мін. кут тренду (градуси)
    max_trend_angle_deg: float = 75.0   # Макс. кут тренду

    # History
    price_history_size: int = 500


@dataclass
class PriceBar:
    """Цінова свічка/бар."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


class TrendlineBreakoutStrategy(BaseStrategy):
    """
    Стратегія пробою трендових ліній.

    Особливості:
    1. Автоматичне виявлення pivot points
    2. Побудова трендових ліній
    3. Детекція пробоїв з підтвердженням обсягом
    4. Вхід після pullback до лінії
    """

    def __init__(self, config: Dict[str, Any]):
        """Ініціалізація стратегії."""
        super().__init__(config)

        self._config = self._parse_config(config)

        # Історія цін (агрегована в бари)
        self._price_bars: Dict[str, Deque[PriceBar]] = {}

        # Поточний бар що формується
        self._current_bars: Dict[str, PriceBar] = {}
        self._bar_start_times: Dict[str, datetime] = {}

        # Pivot points
        self._pivots: Dict[str, List[PivotPoint]] = {}

        # Активні трендові лінії
        self._trendlines: Dict[str, List[Trendline]] = {}

        # Активні пробої
        self._active_breakouts: Dict[str, List[BreakoutEvent]] = {}

        # Середній обсяг
        self._avg_volume: Dict[str, Decimal] = {}

        # Статистика
        self._pivots_detected = 0
        self._trendlines_created = 0
        self._breakouts_detected = 0
        self._signals_generated = 0

        logger.info(
            f"[{self.name}] Initialized with pivot_lookback={self._config.pivot_lookback}, "
            f"breakout_threshold={self._config.breakout_threshold_pct}%"
        )

    def _parse_config(self, config: Dict[str, Any]) -> TrendlineBreakoutConfig:
        """Парсинг конфігурації."""
        return TrendlineBreakoutConfig(
            enabled=config.get("enabled", True),
            signal_cooldown=config.get("signal_cooldown", 15.0),
            min_strength=config.get("min_strength", 0.6),
            pivot_lookback=config.get("pivot_lookback", 5),
            min_pivot_distance=config.get("min_pivot_distance", 10),
            max_pivots_to_track=config.get("max_pivots_to_track", 20),
            min_trendline_touches=config.get("min_trendline_touches", 2),
            max_trendline_age_sec=config.get("max_trendline_age_sec", 3600),
            min_trendline_slope=config.get("min_trendline_slope", 0.00001),
            trendline_touch_tolerance_pct=config.get("trendline_touch_tolerance_pct", 0.1),
            breakout_threshold_pct=config.get("breakout_threshold_pct", 0.15),
            breakout_confirmation_bars=config.get("breakout_confirmation_bars", 2),
            volume_confirmation_multiplier=config.get("volume_confirmation_multiplier", 1.3),
            entry_after_pullback=config.get("entry_after_pullback", True),
            pullback_threshold_pct=config.get("pullback_threshold_pct", 0.05),
            stop_beyond_trendline_pct=config.get("stop_beyond_trendline_pct", 0.3),
            target_extension_pct=config.get("target_extension_pct", 0.5),
            max_spread_bps=config.get("max_spread_bps", 15.0),
            min_trend_angle_deg=config.get("min_trend_angle_deg", 5.0),
            max_trend_angle_deg=config.get("max_trend_angle_deg", 75.0),
            price_history_size=config.get("price_history_size", 500)
        )

    def on_orderbook(self, snapshot: OrderBookSnapshot) -> Optional[Signal]:
        """Обробка оновлення ордербуку."""
        symbol = snapshot.symbol

        # Валідація
        if not snapshot.best_bid or not snapshot.best_ask:
            return None

        spread_bps = float(snapshot.spread_bps or 0)
        if spread_bps > self._config.max_spread_bps:
            return None

        mid_price = snapshot.mid_price
        volume = snapshot.bid_volume(5) + snapshot.ask_volume(5)
        timestamp = snapshot.timestamp

        # Ініціалізація
        self._init_buffers(symbol)

        # Оновити бар
        bar_completed = self._update_bar(symbol, mid_price, volume, timestamp)

        if bar_completed:
            # Шукати нові pivots
            self._detect_pivots(symbol)

            # Оновити/створити трендові лінії
            self._update_trendlines(symbol)

        # Перевірити пробої
        self._check_breakouts(symbol, mid_price, volume, timestamp)

        # Генерувати сигнали
        if self.can_signal():
            signal = self._check_breakout_signal(symbol, snapshot)
            if signal:
                return signal

        return None

    def on_trade(self, trade: Trade) -> Optional[Signal]:
        """Обробка угоди."""
        symbol = trade.symbol
        self._init_buffers(symbol)
        self._update_bar(symbol, trade.price, trade.quantity, trade.timestamp)
        return None

    def _init_buffers(self, symbol: str) -> None:
        """Ініціалізувати буфери."""
        if symbol not in self._price_bars:
            self._price_bars[symbol] = deque(maxlen=self._config.price_history_size)
        if symbol not in self._pivots:
            self._pivots[symbol] = []
        if symbol not in self._trendlines:
            self._trendlines[symbol] = []
        if symbol not in self._active_breakouts:
            self._active_breakouts[symbol] = []

    def _update_bar(
        self,
        symbol: str,
        price: Decimal,
        volume: Decimal,
        timestamp: datetime
    ) -> bool:
        """
        Оновити поточний бар.

        Returns:
            True якщо бар завершено
        """
        bar_duration = 10  # 10 секунд на бар (для скальпінгу)

        # Перевірити чи потрібно закрити поточний бар
        bar_completed = False

        if symbol not in self._current_bars:
            # Новий бар
            self._current_bars[symbol] = PriceBar(
                timestamp=timestamp,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume
            )
            self._bar_start_times[symbol] = timestamp
        else:
            current_bar = self._current_bars[symbol]
            bar_age = (timestamp - self._bar_start_times[symbol]).total_seconds()

            if bar_age >= bar_duration:
                # Закрити бар
                current_bar.close = price
                self._price_bars[symbol].append(current_bar)
                bar_completed = True

                # Оновити середній обсяг
                self._update_avg_volume(symbol)

                # Почати новий бар
                self._current_bars[symbol] = PriceBar(
                    timestamp=timestamp,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=volume
                )
                self._bar_start_times[symbol] = timestamp
            else:
                # Оновити поточний бар
                current_bar.high = max(current_bar.high, price)
                current_bar.low = min(current_bar.low, price)
                current_bar.close = price
                current_bar.volume += volume

        return bar_completed

    def _update_avg_volume(self, symbol: str) -> None:
        """Оновити середній обсяг."""
        bars = self._price_bars.get(symbol)
        if bars and len(bars) >= 10:
            volumes = [b.volume for b in list(bars)[-50:]]
            self._avg_volume[symbol] = sum(volumes) / len(volumes)

    def _detect_pivots(self, symbol: str) -> None:
        """Виявити pivot points."""
        bars = list(self._price_bars.get(symbol, []))
        lookback = self._config.pivot_lookback

        if len(bars) < lookback * 2 + 1:
            return

        # Перевірити останній потенційний pivot (з затримкою lookback)
        check_index = len(bars) - lookback - 1

        if check_index < lookback:
            return

        bar = bars[check_index]

        # Перевірити чи це локальний максимум
        is_high = all(
            bar.high >= bars[i].high
            for i in range(check_index - lookback, check_index + lookback + 1)
            if i != check_index
        )

        # Перевірити чи це локальний мінімум
        is_low = all(
            bar.low <= bars[i].low
            for i in range(check_index - lookback, check_index + lookback + 1)
            if i != check_index
        )

        if is_high:
            # Перевірити мінімальну відстань від попереднього pivot
            if self._pivots[symbol]:
                last_pivot = self._pivots[symbol][-1]
                if check_index - last_pivot.index < self._config.min_pivot_distance:
                    return

            pivot = PivotPoint(
                timestamp=bar.timestamp,
                price=bar.high,
                pivot_type=PivotType.HIGH,
                index=check_index,
                strength=lookback
            )
            self._pivots[symbol].append(pivot)
            self._pivots_detected += 1

            # Обмежити кількість
            if len(self._pivots[symbol]) > self._config.max_pivots_to_track:
                self._pivots[symbol] = self._pivots[symbol][-self._config.max_pivots_to_track:]

            logger.debug(f"[{self.name}] Pivot HIGH at {bar.high}")

        elif is_low:
            if self._pivots[symbol]:
                last_pivot = self._pivots[symbol][-1]
                if check_index - last_pivot.index < self._config.min_pivot_distance:
                    return

            pivot = PivotPoint(
                timestamp=bar.timestamp,
                price=bar.low,
                pivot_type=PivotType.LOW,
                index=check_index,
                strength=lookback
            )
            self._pivots[symbol].append(pivot)
            self._pivots_detected += 1

            if len(self._pivots[symbol]) > self._config.max_pivots_to_track:
                self._pivots[symbol] = self._pivots[symbol][-self._config.max_pivots_to_track:]

            logger.debug(f"[{self.name}] Pivot LOW at {bar.low}")

    def _update_trendlines(self, symbol: str) -> None:
        """Оновити та створити трендові лінії."""
        pivots = self._pivots.get(symbol, [])

        # Видалити старі лінії
        self._trendlines[symbol] = [
            tl for tl in self._trendlines[symbol]
            if tl.age_seconds < self._config.max_trendline_age_sec and tl.is_valid
        ]

        if len(pivots) < 2:
            return

        # Спробувати створити нові лінії
        # Лінія підтримки (по LOW pivots)
        low_pivots = [p for p in pivots if p.pivot_type == PivotType.LOW][-5:]
        if len(low_pivots) >= 2:
            self._try_create_trendline(symbol, low_pivots, TrendDirection.UP)

        # Лінія опору (по HIGH pivots)
        high_pivots = [p for p in pivots if p.pivot_type == PivotType.HIGH][-5:]
        if len(high_pivots) >= 2:
            self._try_create_trendline(symbol, high_pivots, TrendDirection.DOWN)

    def _try_create_trendline(
        self,
        symbol: str,
        pivots: List[PivotPoint],
        direction: TrendDirection
    ) -> None:
        """Спробувати створити трендову лінію."""
        if len(pivots) < 2:
            return

        # Взяти два останні pivots
        p1, p2 = pivots[-2], pivots[-1]

        # Розрахувати нахил
        time_diff = (p2.timestamp - p1.timestamp).total_seconds()
        if time_diff == 0:
            return

        slope = float(p2.price - p1.price) / time_diff

        # Перевірити кут
        angle_rad = math.atan(slope * 3600)  # Нормалізувати до години
        angle_deg = abs(math.degrees(angle_rad))

        if angle_deg < self._config.min_trend_angle_deg:
            return
        if angle_deg > self._config.max_trend_angle_deg:
            return

        # Перевірити напрямок
        if direction == TrendDirection.UP and slope <= 0:
            return
        if direction == TrendDirection.DOWN and slope >= 0:
            return

        # Перевірити чи така лінія вже існує
        for existing in self._trendlines[symbol]:
            if existing.start_point.index == p1.index and existing.end_point.index == p2.index:
                return

        # Створити лінію
        trendline = Trendline(
            direction=direction,
            start_point=p1,
            end_point=p2,
            slope=slope,
            intercept=float(p1.price),
            touches=2
        )

        self._trendlines[symbol].append(trendline)
        self._trendlines_created += 1

        logger.debug(
            f"[{self.name}] Trendline {direction.value} created: "
            f"{float(p1.price):.4f} -> {float(p2.price):.4f}"
        )

    def _check_breakouts(
        self,
        symbol: str,
        current_price: Decimal,
        volume: Decimal,
        timestamp: datetime
    ) -> None:
        """Перевірити пробої трендових ліній."""
        trendlines = self._trendlines.get(symbol, [])
        avg_vol = self._avg_volume.get(symbol, Decimal("1"))

        for trendline in trendlines:
            if not trendline.is_valid:
                continue

            # Розрахувати ціну на лінії
            line_price = trendline.price_at_time(timestamp)
            distance_pct = float((current_price - line_price) / line_price * 100)

            threshold = self._config.breakout_threshold_pct

            # Перевірити пробій
            is_breakout = False
            if trendline.direction == TrendDirection.UP:
                # Лінія підтримки - пробій вниз
                if distance_pct < -threshold:
                    is_breakout = True
            else:
                # Лінія опору - пробій вгору
                if distance_pct > threshold:
                    is_breakout = True

            if is_breakout:
                # Перевірити обсяг
                volume_confirmed = volume >= avg_vol * Decimal(
                    str(self._config.volume_confirmation_multiplier)
                )

                breakout = BreakoutEvent(
                    timestamp=timestamp,
                    trendline=trendline,
                    breakout_price=current_price,
                    trendline_price=line_price,
                    breakout_distance_pct=distance_pct,
                    volume_at_breakout=volume,
                    avg_volume=avg_vol,
                    is_confirmed=volume_confirmed
                )

                self._active_breakouts[symbol].append(breakout)
                trendline.is_valid = False  # Лінія пробита
                self._breakouts_detected += 1

                logger.info(
                    f"[{self.name}] Breakout {trendline.direction.value} trendline: "
                    f"{float(current_price):.4f} vs line {float(line_price):.4f} "
                    f"({distance_pct:.2f}%)"
                )

    def _check_breakout_signal(
        self,
        symbol: str,
        snapshot: OrderBookSnapshot
    ) -> Optional[Signal]:
        """Перевірити сигнал на основі пробою."""
        breakouts = self._active_breakouts.get(symbol, [])
        if not breakouts:
            return None

        mid_price = snapshot.mid_price
        now = datetime.utcnow()

        # Знайти підтверджений пробій
        for breakout in breakouts:
            age = (now - breakout.timestamp).total_seconds()

            # Пробій актуальний 30 секунд
            if age > 30:
                continue

            if not breakout.is_confirmed:
                continue

            trendline = breakout.trendline

            # Визначити напрямок сигналу
            if trendline.direction == TrendDirection.UP:
                # Пробій лінії підтримки = SHORT
                signal_type = SignalType.SHORT
                target = mid_price * (1 - Decimal(str(self._config.target_extension_pct / 100)))
                stop = breakout.trendline_price * (
                    1 + Decimal(str(self._config.stop_beyond_trendline_pct / 100))
                )
            else:
                # Пробій лінії опору = LONG
                signal_type = SignalType.LONG
                target = mid_price * (1 + Decimal(str(self._config.target_extension_pct / 100)))
                stop = breakout.trendline_price * (
                    1 - Decimal(str(self._config.stop_beyond_trendline_pct / 100))
                )

            # Розрахувати силу сигналу
            strength = 0.6
            if breakout.is_confirmed:
                strength += 0.2
            if abs(breakout.breakout_distance_pct) > self._config.breakout_threshold_pct * 1.5:
                strength += 0.1

            # Видалити оброблений пробій
            self._active_breakouts[symbol] = [
                b for b in breakouts if b != breakout
            ]
            self._signals_generated += 1

            return self.create_signal(
                signal_type=signal_type,
                symbol=symbol,
                price=mid_price,
                strength=min(strength, 1.0),
                metadata={
                    "signal_source": "trendline_breakout",
                    "trendline_direction": trendline.direction.value,
                    "breakout_price": float(breakout.breakout_price),
                    "trendline_price": float(breakout.trendline_price),
                    "breakout_distance_pct": breakout.breakout_distance_pct,
                    "volume_confirmed": breakout.is_confirmed,
                    "target_price": float(target),
                    "stop_price": float(stop),
                    "trendline_age_sec": trendline.age_seconds,
                    "trendline_touches": trendline.touches,
                }
            )

        return None

    def get_trendlines(self, symbol: str) -> List[Dict[str, Any]]:
        """Отримати активні трендові лінії."""
        result = []
        for tl in self._trendlines.get(symbol, []):
            if tl.is_valid:
                result.append({
                    "direction": tl.direction.value,
                    "start_price": float(tl.start_point.price),
                    "end_price": float(tl.end_point.price),
                    "current_price": float(tl.price_at_time(datetime.utcnow())),
                    "slope": tl.slope,
                    "touches": tl.touches,
                    "age_seconds": tl.age_seconds,
                })
        return result

    def get_pivots(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Отримати pivot points."""
        result = []
        for pivot in self._pivots.get(symbol, [])[-limit:]:
            result.append({
                "type": pivot.pivot_type.value,
                "price": float(pivot.price),
                "timestamp": pivot.timestamp.isoformat(),
                "strength": pivot.strength,
            })
        return result

    @property
    def stats(self) -> Dict[str, Any]:
        """Статистика стратегії."""
        base = super().stats
        base.update({
            "pivots_detected": self._pivots_detected,
            "trendlines_created": self._trendlines_created,
            "breakouts_detected": self._breakouts_detected,
            "signals_generated": self._signals_generated,
            "active_trendlines": sum(
                len([t for t in tls if t.is_valid])
                for tls in self._trendlines.values()
            ),
        })
        return base

    def reset(self) -> None:
        """Скинути стан."""
        super().reset()
        self._price_bars.clear()
        self._current_bars.clear()
        self._bar_start_times.clear()
        self._pivots.clear()
        self._trendlines.clear()
        self._active_breakouts.clear()
        self._avg_volume.clear()
        self._pivots_detected = 0
        self._trendlines_created = 0
        self._breakouts_detected = 0
        self._signals_generated = 0
