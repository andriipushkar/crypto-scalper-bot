"""
Range Trading Strategy (Стратегія торгівлі в рейнджі).

Реалізує концепції з книги "Скальпинг: практическое руководство трейдера":
- Стратегія #1: Повернення в рейндж після хибного пробою
- Стратегія #2: Торгівля від меж рейнджу

Концепція:
"Ціна більшу частину часу рухається в рейнджі. Хибні пробої
рейнджу часто призводять до швидкого повернення ціни назад."

Логіка:
1. Визначаємо рейндж (high/low за період)
2. Чекаємо на пробій
3. Якщо пробій не підтверджується обсягом - входимо в зворотному напрямку
4. Ціль - протилежна межа рейнджу
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, Deque, Tuple
from enum import Enum

from loguru import logger

from src.data.models import OrderBookSnapshot, Signal, SignalType, Trade
from src.strategy.base import BaseStrategy


class RangeState(Enum):
    """Стан рейнджу."""
    FORMING = "forming"           # Рейндж формується
    ESTABLISHED = "established"   # Рейндж встановлено
    BREAKOUT_UP = "breakout_up"   # Пробій вгору
    BREAKOUT_DOWN = "breakout_down"  # Пробій вниз
    FALSE_BREAKOUT_UP = "false_breakout_up"    # Хибний пробій вгору
    FALSE_BREAKOUT_DOWN = "false_breakout_down"  # Хибний пробій вниз


class BreakoutType(Enum):
    """Тип пробою."""
    TRUE_BREAKOUT = "true"        # Справжній пробій
    FALSE_BREAKOUT = "false"      # Хибний пробій
    PENDING = "pending"           # Очікує підтвердження


@dataclass
class PriceLevel:
    """Ціновий рівень з метаданими."""
    price: Decimal
    timestamp: datetime
    touches: int = 1
    volume: Decimal = Decimal("0")


@dataclass
class Range:
    """Визначений торговий рейндж."""
    high: Decimal
    low: Decimal
    mid: Decimal
    width_pct: float
    formed_at: datetime
    touches_high: int = 0
    touches_low: int = 0
    volume_in_range: Decimal = Decimal("0")
    is_valid: bool = True

    @property
    def width(self) -> Decimal:
        """Ширина рейнджу."""
        return self.high - self.low


@dataclass
class BreakoutEvent:
    """Подія пробою рейнджу."""
    timestamp: datetime
    direction: str  # "up" або "down"
    breakout_price: Decimal
    range_boundary: Decimal
    volume_at_breakout: Decimal
    avg_volume: Decimal
    is_confirmed: bool = False
    confirmation_time: Optional[datetime] = None
    result: BreakoutType = BreakoutType.PENDING


@dataclass
class RangeTradingConfig:
    """Конфігурація стратегії торгівлі в рейнджі."""
    enabled: bool = True
    signal_cooldown: float = 10.0
    min_strength: float = 0.6

    # Визначення рейнджу
    range_lookback_periods: int = 50      # Періодів для визначення рейнджу
    min_range_width_pct: float = 0.2      # Мін. ширина рейнджу (%)
    max_range_width_pct: float = 2.0      # Макс. ширина рейнджу (%)
    min_touches: int = 2                   # Мін. дотиків до межі
    range_touch_tolerance_pct: float = 0.05  # Толерантність дотику (%)

    # Пробій
    breakout_threshold_pct: float = 0.1   # Поріг пробою (% за межу)
    breakout_confirmation_bars: int = 3    # Барів для підтвердження
    breakout_confirmation_time_sec: int = 30  # Секунд для підтвердження
    volume_confirmation_multiplier: float = 1.5  # Обсяг для підтвердження (x avg)

    # Хибний пробій
    false_breakout_return_pct: float = 0.05  # Повернення для хибного пробою
    false_breakout_time_sec: int = 60     # Час для класифікації як хибний

    # Торгівля
    entry_from_boundary_pct: float = 0.1  # Вхід від межі (%)
    target_opposite_boundary: bool = True  # Ціль - протилежна межа
    stop_beyond_breakout_pct: float = 0.2  # Стоп за пробоєм (%)

    # Фільтри
    max_spread_bps: float = 10.0
    min_volume_multiplier: float = 0.8    # Мін. обсяг відносно середнього

    # Буфери
    price_history_size: int = 500


@dataclass
class PricePoint:
    """Точка ціни."""
    timestamp: datetime
    price: Decimal
    volume: Decimal = Decimal("0")


class RangeTradingStrategy(BaseStrategy):
    """
    Стратегія торгівлі в рейнджі.

    Особливості:
    1. Автоматичне визначення рейнджу
    2. Детекція хибних пробоїв
    3. Торгівля на поверненні в рейндж
    4. Торгівля від меж рейнджу
    """

    def __init__(self, config: Dict[str, Any]):
        """Ініціалізація стратегії."""
        super().__init__(config)

        self._config = self._parse_config(config)

        # Історія цін
        self._price_history: Dict[str, Deque[PricePoint]] = {}

        # Поточні рейнджі
        self._current_ranges: Dict[str, Range] = {}

        # Активні пробої
        self._active_breakouts: Dict[str, BreakoutEvent] = {}

        # Історія пробоїв
        self._breakout_history: Dict[str, List[BreakoutEvent]] = {}

        # Середній обсяг
        self._avg_volume: Dict[str, Decimal] = {}

        # Статистика
        self._ranges_detected = 0
        self._true_breakouts = 0
        self._false_breakouts = 0
        self._signals_from_false_breakout = 0
        self._signals_from_boundary = 0

        logger.info(
            f"[{self.name}] Initialized with range_width: "
            f"{self._config.min_range_width_pct}-{self._config.max_range_width_pct}%"
        )

    def _parse_config(self, config: Dict[str, Any]) -> RangeTradingConfig:
        """Парсинг конфігурації."""
        return RangeTradingConfig(
            enabled=config.get("enabled", True),
            signal_cooldown=config.get("signal_cooldown", 10.0),
            min_strength=config.get("min_strength", 0.6),
            range_lookback_periods=config.get("range_lookback_periods", 50),
            min_range_width_pct=config.get("min_range_width_pct", 0.2),
            max_range_width_pct=config.get("max_range_width_pct", 2.0),
            min_touches=config.get("min_touches", 2),
            range_touch_tolerance_pct=config.get("range_touch_tolerance_pct", 0.05),
            breakout_threshold_pct=config.get("breakout_threshold_pct", 0.1),
            breakout_confirmation_bars=config.get("breakout_confirmation_bars", 3),
            breakout_confirmation_time_sec=config.get("breakout_confirmation_time_sec", 30),
            volume_confirmation_multiplier=config.get("volume_confirmation_multiplier", 1.5),
            false_breakout_return_pct=config.get("false_breakout_return_pct", 0.05),
            false_breakout_time_sec=config.get("false_breakout_time_sec", 60),
            entry_from_boundary_pct=config.get("entry_from_boundary_pct", 0.1),
            target_opposite_boundary=config.get("target_opposite_boundary", True),
            stop_beyond_breakout_pct=config.get("stop_beyond_breakout_pct", 0.2),
            max_spread_bps=config.get("max_spread_bps", 10.0),
            min_volume_multiplier=config.get("min_volume_multiplier", 0.8),
            price_history_size=config.get("price_history_size", 500)
        )

    def on_orderbook(self, snapshot: OrderBookSnapshot) -> Optional[Signal]:
        """
        Обробка оновлення ордербуку.

        Args:
            snapshot: Знімок ордербуку

        Returns:
            Сигнал якщо умови виконані
        """
        symbol = snapshot.symbol

        # Ініціалізація
        self._init_buffers(symbol)

        # Валідація
        if not snapshot.best_bid or not snapshot.best_ask:
            return None

        spread_bps = float(snapshot.spread_bps or 0)
        if spread_bps > self._config.max_spread_bps:
            return None

        mid_price = snapshot.mid_price
        volume = snapshot.bid_volume(5) + snapshot.ask_volume(5)

        # Оновити історію
        self._update_history(symbol, mid_price, volume, snapshot.timestamp)

        # Оновити середній обсяг
        self._update_avg_volume(symbol)

        # Визначити або оновити рейндж
        self._update_range(symbol)

        # Перевірити пробої
        self._check_breakouts(symbol, mid_price, volume, snapshot.timestamp)

        # Перевірити сигнали
        if self.can_signal():
            # 1. Сигнал на хибному пробої
            signal = self._check_false_breakout_signal(symbol, snapshot)
            if signal:
                return signal

            # 2. Сигнал від межі рейнджу
            signal = self._check_boundary_signal(symbol, snapshot)
            if signal:
                return signal

        return None

    def on_trade(self, trade: Trade) -> Optional[Signal]:
        """Обробка угоди."""
        symbol = trade.symbol
        self._init_buffers(symbol)

        # Оновити історію
        self._update_history(symbol, trade.price, trade.quantity, trade.timestamp)

        return None

    def _init_buffers(self, symbol: str) -> None:
        """Ініціалізувати буфери для символу."""
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self._config.price_history_size)
        if symbol not in self._breakout_history:
            self._breakout_history[symbol] = []

    def _update_history(
        self,
        symbol: str,
        price: Decimal,
        volume: Decimal,
        timestamp: datetime
    ) -> None:
        """Оновити історію цін."""
        self._price_history[symbol].append(PricePoint(
            timestamp=timestamp,
            price=price,
            volume=volume
        ))

    def _update_avg_volume(self, symbol: str) -> None:
        """Оновити середній обсяг."""
        history = self._price_history.get(symbol)
        if not history or len(history) < 10:
            return

        volumes = [p.volume for p in history if p.volume > 0]
        if volumes:
            self._avg_volume[symbol] = sum(volumes) / len(volumes)

    def _update_range(self, symbol: str) -> None:
        """Визначити або оновити рейндж."""
        history = self._price_history.get(symbol)
        if not history or len(history) < self._config.range_lookback_periods:
            return

        # Отримати останні N цін
        recent = list(history)[-self._config.range_lookback_periods:]
        prices = [p.price for p in recent]

        # Знайти high/low
        high = max(prices)
        low = min(prices)
        mid = (high + low) / 2

        # Розрахувати ширину
        width_pct = float((high - low) / mid * 100) if mid > 0 else 0

        # Перевірити чи ширина в допустимих межах
        if not (self._config.min_range_width_pct <= width_pct <= self._config.max_range_width_pct):
            self._current_ranges.pop(symbol, None)
            return

        # Порахувати дотики до меж
        tolerance = mid * Decimal(str(self._config.range_touch_tolerance_pct / 100))
        touches_high = sum(1 for p in prices if abs(p - high) <= tolerance)
        touches_low = sum(1 for p in prices if abs(p - low) <= tolerance)

        # Перевірити мінімальну кількість дотиків
        if touches_high < self._config.min_touches or touches_low < self._config.min_touches:
            self._current_ranges.pop(symbol, None)
            return

        # Порахувати обсяг в рейнджі
        volume_in_range = sum(p.volume for p in recent)

        # Створити або оновити рейндж
        if symbol not in self._current_ranges:
            self._ranges_detected += 1
            logger.debug(
                f"[{self.name}] Range detected for {symbol}: "
                f"{float(low):.4f} - {float(high):.4f} ({width_pct:.2f}%)"
            )

        self._current_ranges[symbol] = Range(
            high=high,
            low=low,
            mid=mid,
            width_pct=width_pct,
            formed_at=datetime.utcnow(),
            touches_high=touches_high,
            touches_low=touches_low,
            volume_in_range=volume_in_range,
            is_valid=True
        )

    def _check_breakouts(
        self,
        symbol: str,
        current_price: Decimal,
        volume: Decimal,
        timestamp: datetime
    ) -> None:
        """Перевірити та класифікувати пробої."""
        range_obj = self._current_ranges.get(symbol)
        if not range_obj or not range_obj.is_valid:
            return

        avg_vol = self._avg_volume.get(symbol, Decimal("1"))
        breakout_threshold = range_obj.mid * Decimal(str(self._config.breakout_threshold_pct / 100))

        # Перевірити чи є активний пробій
        active_breakout = self._active_breakouts.get(symbol)

        if active_breakout:
            # Класифікувати активний пробій
            self._classify_breakout(symbol, current_price, timestamp, active_breakout, range_obj)
        else:
            # Перевірити на новий пробій
            if current_price > range_obj.high + breakout_threshold:
                # Пробій вгору
                self._active_breakouts[symbol] = BreakoutEvent(
                    timestamp=timestamp,
                    direction="up",
                    breakout_price=current_price,
                    range_boundary=range_obj.high,
                    volume_at_breakout=volume,
                    avg_volume=avg_vol,
                    is_confirmed=False
                )
                logger.debug(f"[{self.name}] Breakout UP detected for {symbol} at {current_price}")

            elif current_price < range_obj.low - breakout_threshold:
                # Пробій вниз
                self._active_breakouts[symbol] = BreakoutEvent(
                    timestamp=timestamp,
                    direction="down",
                    breakout_price=current_price,
                    range_boundary=range_obj.low,
                    volume_at_breakout=volume,
                    avg_volume=avg_vol,
                    is_confirmed=False
                )
                logger.debug(f"[{self.name}] Breakout DOWN detected for {symbol} at {current_price}")

    def _classify_breakout(
        self,
        symbol: str,
        current_price: Decimal,
        timestamp: datetime,
        breakout: BreakoutEvent,
        range_obj: Range
    ) -> None:
        """Класифікувати пробій як справжній або хибний."""
        time_since_breakout = (timestamp - breakout.timestamp).total_seconds()

        # Перевірити чи пробій вже класифіковано
        if breakout.result != BreakoutType.PENDING:
            return

        # Перевірити підтвердження обсягом
        volume_confirmed = breakout.volume_at_breakout >= (
            breakout.avg_volume * Decimal(str(self._config.volume_confirmation_multiplier))
        )

        return_threshold = range_obj.mid * Decimal(str(self._config.false_breakout_return_pct / 100))

        if breakout.direction == "up":
            # Перевірити повернення в рейндж
            if current_price < range_obj.high - return_threshold:
                # Хибний пробій - ціна повернулася
                breakout.result = BreakoutType.FALSE_BREAKOUT
                breakout.confirmation_time = timestamp
                self._false_breakouts += 1
                self._breakout_history[symbol].append(breakout)
                logger.info(f"[{self.name}] FALSE breakout UP for {symbol}")

            elif time_since_breakout > self._config.breakout_confirmation_time_sec and volume_confirmed:
                # Справжній пробій - час та обсяг підтвердили
                breakout.result = BreakoutType.TRUE_BREAKOUT
                breakout.is_confirmed = True
                breakout.confirmation_time = timestamp
                self._true_breakouts += 1
                self._breakout_history[symbol].append(breakout)
                self._current_ranges.pop(symbol, None)  # Рейндж зламано
                logger.info(f"[{self.name}] TRUE breakout UP for {symbol}")

        else:  # direction == "down"
            if current_price > range_obj.low + return_threshold:
                breakout.result = BreakoutType.FALSE_BREAKOUT
                breakout.confirmation_time = timestamp
                self._false_breakouts += 1
                self._breakout_history[symbol].append(breakout)
                logger.info(f"[{self.name}] FALSE breakout DOWN for {symbol}")

            elif time_since_breakout > self._config.breakout_confirmation_time_sec and volume_confirmed:
                breakout.result = BreakoutType.TRUE_BREAKOUT
                breakout.is_confirmed = True
                breakout.confirmation_time = timestamp
                self._true_breakouts += 1
                self._breakout_history[symbol].append(breakout)
                self._current_ranges.pop(symbol, None)
                logger.info(f"[{self.name}] TRUE breakout DOWN for {symbol}")

        # Очистити якщо класифіковано
        if breakout.result != BreakoutType.PENDING:
            self._active_breakouts.pop(symbol, None)

        # Таймаут без класифікації
        if time_since_breakout > self._config.false_breakout_time_sec:
            breakout.result = BreakoutType.FALSE_BREAKOUT
            self._active_breakouts.pop(symbol, None)

    def _check_false_breakout_signal(
        self,
        symbol: str,
        snapshot: OrderBookSnapshot
    ) -> Optional[Signal]:
        """Перевірити сигнал на хибному пробої."""
        breakout_history = self._breakout_history.get(symbol, [])
        if not breakout_history:
            return None

        # Знайти нещодавній хибний пробій
        now = datetime.utcnow()
        recent_false_breakouts = [
            b for b in breakout_history
            if b.result == BreakoutType.FALSE_BREAKOUT and
            b.confirmation_time and
            (now - b.confirmation_time).total_seconds() < 30  # 30 сек вікно
        ]

        if not recent_false_breakouts:
            return None

        latest_breakout = recent_false_breakouts[-1]
        range_obj = self._current_ranges.get(symbol)

        if not range_obj:
            return None

        mid_price = snapshot.mid_price

        # Визначити напрямок сигналу (протилежний до пробою)
        if latest_breakout.direction == "up":
            # Хибний пробій вгору -> SHORT
            signal_type = SignalType.SHORT
            target = range_obj.low
            stop = range_obj.high * (1 + Decimal(str(self._config.stop_beyond_breakout_pct / 100)))
        else:
            # Хибний пробій вниз -> LONG
            signal_type = SignalType.LONG
            target = range_obj.high
            stop = range_obj.low * (1 - Decimal(str(self._config.stop_beyond_breakout_pct / 100)))

        # Розрахувати силу сигналу
        # Чим більше дотиків до меж - тим сильніший сигнал
        touch_factor = min((range_obj.touches_high + range_obj.touches_low) / 10, 1.0)
        strength = 0.6 + touch_factor * 0.4

        self._signals_from_false_breakout += 1

        # Видалити використаний пробій
        self._breakout_history[symbol] = [
            b for b in breakout_history if b != latest_breakout
        ]

        return self.create_signal(
            signal_type=signal_type,
            symbol=symbol,
            price=mid_price,
            strength=strength,
            metadata={
                "signal_source": "false_breakout",
                "breakout_direction": latest_breakout.direction,
                "range_high": float(range_obj.high),
                "range_low": float(range_obj.low),
                "range_width_pct": range_obj.width_pct,
                "target_price": float(target),
                "stop_price": float(stop),
                "touches_high": range_obj.touches_high,
                "touches_low": range_obj.touches_low,
            }
        )

    def _check_boundary_signal(
        self,
        symbol: str,
        snapshot: OrderBookSnapshot
    ) -> Optional[Signal]:
        """Перевірити сигнал від межі рейнджу."""
        range_obj = self._current_ranges.get(symbol)
        if not range_obj or not range_obj.is_valid:
            return None

        mid_price = snapshot.mid_price
        entry_threshold = range_obj.mid * Decimal(str(self._config.entry_from_boundary_pct / 100))

        # Перевірити близькість до меж
        near_high = abs(mid_price - range_obj.high) <= entry_threshold
        near_low = abs(mid_price - range_obj.low) <= entry_threshold

        if not near_high and not near_low:
            return None

        # Перевірити що немає активного пробою
        if symbol in self._active_breakouts:
            return None

        # Перевірити імбаланс ордербуку для підтвердження
        imbalance = float(snapshot.imbalance(5))

        if near_high and imbalance < 0.9:
            # Біля верхньої межі + більше продавців -> SHORT
            signal_type = SignalType.SHORT
            target = range_obj.low if self._config.target_opposite_boundary else range_obj.mid
            stop = range_obj.high * (1 + Decimal(str(self._config.stop_beyond_breakout_pct / 100)))

        elif near_low and imbalance > 1.1:
            # Біля нижньої межі + більше покупців -> LONG
            signal_type = SignalType.LONG
            target = range_obj.high if self._config.target_opposite_boundary else range_obj.mid
            stop = range_obj.low * (1 - Decimal(str(self._config.stop_beyond_breakout_pct / 100)))

        else:
            return None

        # Розрахувати силу
        strength = 0.5 + min(abs(imbalance - 1.0) / 2, 0.5)

        self._signals_from_boundary += 1

        return self.create_signal(
            signal_type=signal_type,
            symbol=symbol,
            price=mid_price,
            strength=strength,
            metadata={
                "signal_source": "range_boundary",
                "boundary": "high" if near_high else "low",
                "range_high": float(range_obj.high),
                "range_low": float(range_obj.low),
                "range_width_pct": range_obj.width_pct,
                "target_price": float(target),
                "stop_price": float(stop),
                "imbalance": imbalance,
            }
        )

    def get_current_range(self, symbol: str) -> Optional[Range]:
        """Отримати поточний рейндж для символу."""
        return self._current_ranges.get(symbol)

    def get_active_breakout(self, symbol: str) -> Optional[BreakoutEvent]:
        """Отримати активний пробій."""
        return self._active_breakouts.get(symbol)

    def get_breakout_history(self, symbol: str, limit: int = 10) -> List[BreakoutEvent]:
        """Отримати історію пробоїв."""
        history = self._breakout_history.get(symbol, [])
        return history[-limit:]

    def get_range_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Отримати позицію ціни відносно рейнджу."""
        range_obj = self._current_ranges.get(symbol)
        history = self._price_history.get(symbol)

        if not range_obj or not history:
            return None

        current_price = history[-1].price

        # Позиція: 0 = low, 0.5 = mid, 1 = high
        if range_obj.width > 0:
            position = float((current_price - range_obj.low) / range_obj.width)
        else:
            position = 0.5

        return {
            "current_price": float(current_price),
            "range_high": float(range_obj.high),
            "range_low": float(range_obj.low),
            "range_mid": float(range_obj.mid),
            "position_in_range": position,
            "distance_to_high_pct": float((range_obj.high - current_price) / current_price * 100),
            "distance_to_low_pct": float((current_price - range_obj.low) / current_price * 100),
        }

    @property
    def stats(self) -> Dict[str, Any]:
        """Статистика стратегії."""
        base = super().stats
        base.update({
            "ranges_detected": self._ranges_detected,
            "true_breakouts": self._true_breakouts,
            "false_breakouts": self._false_breakouts,
            "false_breakout_rate": (
                self._false_breakouts / (self._true_breakouts + self._false_breakouts) * 100
                if (self._true_breakouts + self._false_breakouts) > 0 else 0
            ),
            "signals_from_false_breakout": self._signals_from_false_breakout,
            "signals_from_boundary": self._signals_from_boundary,
            "active_ranges": len(self._current_ranges),
        })
        return base

    def reset(self) -> None:
        """Скинути стан."""
        super().reset()
        self._price_history.clear()
        self._current_ranges.clear()
        self._active_breakouts.clear()
        self._breakout_history.clear()
        self._avg_volume.clear()
        self._ranges_detected = 0
        self._true_breakouts = 0
        self._false_breakouts = 0
        self._signals_from_false_breakout = 0
        self._signals_from_boundary = 0
