"""
Session-Based Trading Strategy (Сесійна стратегія).

Адаптація концепції "Morning Return" з книги для 24/7 крипто-ринку.

Концепція:
"На початку торгової сесії ціна часто повертається до закриття
попередньої сесії або тестує ключові рівні сесії"

Крипто-адаптація:
- Азіатська сесія: 00:00-08:00 UTC (Токіо, Гонконг, Сінгапур)
- Європейська сесія: 07:00-16:00 UTC (Лондон, Франкфурт)
- Американська сесія: 13:00-22:00 UTC (Нью-Йорк)

Логіка:
1. Відстежуємо рівні кожної сесії (open, high, low, close)
2. На відкритті нової сесії аналізуємо gap
3. Торгуємо повернення до рівнів попередньої сесії
4. Використовуємо overlap сесій для підвищеної волатильності
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time, timezone
from decimal import Decimal
from typing import Optional, Dict, Any, List, Deque, Tuple
from enum import Enum

from loguru import logger

from src.data.models import OrderBookSnapshot, Signal, SignalType, Trade
from src.strategy.base import BaseStrategy


class TradingSession(Enum):
    """Торгові сесії."""
    ASIA = "asia"
    EUROPE = "europe"
    US = "us"
    OVERLAP_ASIA_EUROPE = "asia_europe_overlap"
    OVERLAP_EUROPE_US = "europe_us_overlap"


@dataclass
class SessionConfig:
    """Конфігурація сесії."""
    name: TradingSession
    start_hour: int  # UTC
    end_hour: int    # UTC
    typical_volatility: float  # Типова волатильність (%)
    weight: float = 1.0


@dataclass
class SessionLevels:
    """Рівні сесії."""
    session: TradingSession
    date: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Optional[Decimal] = None
    vwap: Decimal = Decimal("0")
    total_volume: Decimal = Decimal("0")
    trade_count: int = 0
    is_complete: bool = False

    @property
    def range(self) -> Decimal:
        """Діапазон сесії."""
        return self.high_price - self.low_price

    @property
    def range_pct(self) -> float:
        """Діапазон у відсотках."""
        if self.open_price > 0:
            return float(self.range / self.open_price * 100)
        return 0.0

    @property
    def mid_price(self) -> Decimal:
        """Середня ціна сесії."""
        return (self.high_price + self.low_price) / 2


@dataclass
class SessionGap:
    """Gap між сесіями."""
    from_session: TradingSession
    to_session: TradingSession
    gap_price: Decimal  # Різниця цін
    gap_pct: float      # Gap у відсотках
    direction: str      # "up" або "down"
    timestamp: datetime


@dataclass
class SessionTradingConfig:
    """Конфігурація сесійної стратегії."""
    enabled: bool = True
    signal_cooldown: float = 30.0
    min_strength: float = 0.6

    # Сесії
    sessions: List[SessionConfig] = field(default_factory=lambda: [
        SessionConfig(
            name=TradingSession.ASIA,
            start_hour=0,
            end_hour=8,
            typical_volatility=1.5,
            weight=0.8
        ),
        SessionConfig(
            name=TradingSession.EUROPE,
            start_hour=7,
            end_hour=16,
            typical_volatility=2.0,
            weight=1.0
        ),
        SessionConfig(
            name=TradingSession.US,
            start_hour=13,
            end_hour=22,
            typical_volatility=2.5,
            weight=1.2
        ),
    ])

    # Gap trading
    min_gap_pct: float = 0.3           # Мін. gap для сигналу (%)
    max_gap_pct: float = 3.0           # Макс. gap (більше = тренд)
    gap_fill_probability: float = 0.7  # Ймовірність заповнення gap

    # Session return
    session_return_threshold_pct: float = 0.5  # Відхилення від close для сигналу
    session_return_target_pct: float = 0.8     # Цільове повернення (% від gap)

    # Overlap trading
    trade_overlaps: bool = True
    overlap_volatility_boost: float = 1.5

    # Previous session levels
    use_prev_session_levels: bool = True
    prev_session_level_tolerance_pct: float = 0.1

    # Risk
    stop_beyond_session_extreme_pct: float = 0.2

    # Фільтри
    max_spread_bps: float = 15.0
    min_volume_for_signal: float = 1000  # USD

    # Історія
    sessions_history_days: int = 7


class SessionTradingStrategy(BaseStrategy):
    """
    Сесійна стратегія торгівлі.

    Особливості:
    1. Відстеження рівнів кожної сесії
    2. Gap trading на відкритті сесій
    3. Торгівля повернення до рівнів попередньої сесії
    4. Підвищена увага до overlap сесій
    """

    def __init__(self, config: Dict[str, Any]):
        """Ініціалізація стратегії."""
        super().__init__(config)

        self._config = self._parse_config(config)

        # Поточна сесія
        self._current_session: Optional[TradingSession] = None

        # Рівні поточних сесій
        self._current_levels: Dict[str, Dict[TradingSession, SessionLevels]] = {}

        # Історія сесій
        self._session_history: Dict[str, List[SessionLevels]] = {}

        # Виявлені gaps
        self._active_gaps: Dict[str, SessionGap] = {}

        # VWAP компоненти
        self._vwap_sum: Dict[str, Dict[TradingSession, Decimal]] = {}
        self._volume_sum: Dict[str, Dict[TradingSession, Decimal]] = {}

        # Статистика
        self._session_changes = 0
        self._gaps_detected = 0
        self._gap_fills = 0
        self._signals_from_gap = 0
        self._signals_from_level = 0

        logger.info(
            f"[{self.name}] Initialized with {len(self._config.sessions)} sessions"
        )

    def _parse_config(self, config: Dict[str, Any]) -> SessionTradingConfig:
        """Парсинг конфігурації."""
        sessions = []
        for sess_cfg in config.get("sessions", []):
            sessions.append(SessionConfig(
                name=TradingSession(sess_cfg.get("name", "us")),
                start_hour=sess_cfg.get("start_hour", 13),
                end_hour=sess_cfg.get("end_hour", 22),
                typical_volatility=sess_cfg.get("typical_volatility", 2.0),
                weight=sess_cfg.get("weight", 1.0)
            ))

        return SessionTradingConfig(
            enabled=config.get("enabled", True),
            signal_cooldown=config.get("signal_cooldown", 30.0),
            min_strength=config.get("min_strength", 0.6),
            sessions=sessions if sessions else SessionTradingConfig().sessions,
            min_gap_pct=config.get("min_gap_pct", 0.3),
            max_gap_pct=config.get("max_gap_pct", 3.0),
            gap_fill_probability=config.get("gap_fill_probability", 0.7),
            session_return_threshold_pct=config.get("session_return_threshold_pct", 0.5),
            session_return_target_pct=config.get("session_return_target_pct", 0.8),
            trade_overlaps=config.get("trade_overlaps", True),
            overlap_volatility_boost=config.get("overlap_volatility_boost", 1.5),
            use_prev_session_levels=config.get("use_prev_session_levels", True),
            prev_session_level_tolerance_pct=config.get("prev_session_level_tolerance_pct", 0.1),
            stop_beyond_session_extreme_pct=config.get("stop_beyond_session_extreme_pct", 0.2),
            max_spread_bps=config.get("max_spread_bps", 15.0),
            min_volume_for_signal=config.get("min_volume_for_signal", 1000)
        )

    def _get_current_session(self, timestamp: datetime) -> Tuple[TradingSession, bool]:
        """
        Визначити поточну сесію та чи це overlap.

        Returns:
            (session, is_overlap)
        """
        # Конвертувати в UTC якщо потрібно
        if timestamp.tzinfo is None:
            utc_time = timestamp
        else:
            utc_time = timestamp.astimezone(timezone.utc).replace(tzinfo=None)

        hour = utc_time.hour
        active_sessions = []

        for sess_cfg in self._config.sessions:
            if sess_cfg.start_hour <= sess_cfg.end_hour:
                # Звичайна сесія (не перетинає північ)
                if sess_cfg.start_hour <= hour < sess_cfg.end_hour:
                    active_sessions.append(sess_cfg.name)
            else:
                # Сесія перетинає північ
                if hour >= sess_cfg.start_hour or hour < sess_cfg.end_hour:
                    active_sessions.append(sess_cfg.name)

        is_overlap = len(active_sessions) > 1

        if is_overlap:
            # Визначити тип overlap
            if TradingSession.ASIA in active_sessions and TradingSession.EUROPE in active_sessions:
                return TradingSession.OVERLAP_ASIA_EUROPE, True
            elif TradingSession.EUROPE in active_sessions and TradingSession.US in active_sessions:
                return TradingSession.OVERLAP_EUROPE_US, True

        # Повернути головну сесію (з найбільшою вагою)
        if active_sessions:
            main_session = max(
                active_sessions,
                key=lambda s: next(
                    (cfg.weight for cfg in self._config.sessions if cfg.name == s),
                    0
                )
            )
            return main_session, is_overlap

        # За замовчуванням - азіатська
        return TradingSession.ASIA, False

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

        # Визначити поточну сесію
        current_session, is_overlap = self._get_current_session(timestamp)

        # Перевірити зміну сесії
        session_changed = self._check_session_change(symbol, current_session, mid_price, timestamp)

        # Оновити рівні сесії
        self._update_session_levels(symbol, current_session, mid_price, volume, timestamp)

        # Перевірити сигнали
        if self.can_signal():
            # 1. Gap fill signal
            signal = self._check_gap_fill_signal(symbol, snapshot)
            if signal:
                return signal

            # 2. Session level signal
            signal = self._check_session_level_signal(symbol, snapshot, current_session)
            if signal:
                return signal

            # 3. Overlap volatility signal
            if is_overlap and self._config.trade_overlaps:
                signal = self._check_overlap_signal(symbol, snapshot, current_session)
                if signal:
                    return signal

        return None

    def on_trade(self, trade: Trade) -> Optional[Signal]:
        """Обробка угоди."""
        symbol = trade.symbol
        self._init_buffers(symbol)

        current_session, _ = self._get_current_session(trade.timestamp)
        self._update_session_levels(
            symbol, current_session, trade.price, trade.quantity, trade.timestamp
        )

        return None

    def _init_buffers(self, symbol: str) -> None:
        """Ініціалізувати буфери."""
        if symbol not in self._current_levels:
            self._current_levels[symbol] = {}
        if symbol not in self._session_history:
            self._session_history[symbol] = []
        if symbol not in self._vwap_sum:
            self._vwap_sum[symbol] = {}
        if symbol not in self._volume_sum:
            self._volume_sum[symbol] = {}

    def _check_session_change(
        self,
        symbol: str,
        new_session: TradingSession,
        price: Decimal,
        timestamp: datetime
    ) -> bool:
        """Перевірити та обробити зміну сесії."""
        if self._current_session == new_session:
            return False

        old_session = self._current_session
        self._current_session = new_session
        self._session_changes += 1

        # Завершити стару сесію
        if old_session and symbol in self._current_levels:
            old_levels = self._current_levels[symbol].get(old_session)
            if old_levels and not old_levels.is_complete:
                old_levels.close_price = price
                old_levels.is_complete = True
                self._session_history[symbol].append(old_levels)

                # Обмежити історію
                max_history = self._config.sessions_history_days * 3  # 3 сесії на день
                if len(self._session_history[symbol]) > max_history:
                    self._session_history[symbol] = self._session_history[symbol][-max_history:]

        # Перевірити gap
        if old_session and symbol in self._current_levels:
            old_levels = self._current_levels[symbol].get(old_session)
            if old_levels and old_levels.close_price:
                gap = price - old_levels.close_price
                gap_pct = float(gap / old_levels.close_price * 100)

                if abs(gap_pct) >= self._config.min_gap_pct:
                    self._active_gaps[symbol] = SessionGap(
                        from_session=old_session,
                        to_session=new_session,
                        gap_price=gap,
                        gap_pct=gap_pct,
                        direction="up" if gap > 0 else "down",
                        timestamp=timestamp
                    )
                    self._gaps_detected += 1
                    logger.info(
                        f"[{self.name}] Gap detected for {symbol}: {gap_pct:.2f}% "
                        f"({old_session.value} -> {new_session.value})"
                    )

        # Ініціалізувати нову сесію
        self._current_levels[symbol][new_session] = SessionLevels(
            session=new_session,
            date=timestamp,
            open_price=price,
            high_price=price,
            low_price=price
        )
        self._vwap_sum[symbol][new_session] = Decimal("0")
        self._volume_sum[symbol][new_session] = Decimal("0")

        logger.debug(f"[{self.name}] Session changed to {new_session.value} for {symbol}")

        return True

    def _update_session_levels(
        self,
        symbol: str,
        session: TradingSession,
        price: Decimal,
        volume: Decimal,
        timestamp: datetime
    ) -> None:
        """Оновити рівні сесії."""
        if session not in self._current_levels.get(symbol, {}):
            self._current_levels[symbol][session] = SessionLevels(
                session=session,
                date=timestamp,
                open_price=price,
                high_price=price,
                low_price=price
            )
            self._vwap_sum[symbol][session] = Decimal("0")
            self._volume_sum[symbol][session] = Decimal("0")

        levels = self._current_levels[symbol][session]

        # Оновити high/low
        levels.high_price = max(levels.high_price, price)
        levels.low_price = min(levels.low_price, price)

        # Оновити VWAP
        if volume > 0:
            self._vwap_sum[symbol][session] += price * volume
            self._volume_sum[symbol][session] += volume
            levels.total_volume += volume
            levels.trade_count += 1

            if self._volume_sum[symbol][session] > 0:
                levels.vwap = self._vwap_sum[symbol][session] / self._volume_sum[symbol][session]

    def _check_gap_fill_signal(
        self,
        symbol: str,
        snapshot: OrderBookSnapshot
    ) -> Optional[Signal]:
        """Перевірити сигнал на заповнення gap."""
        gap = self._active_gaps.get(symbol)
        if not gap:
            return None

        # Gap старіший за годину - не торгуємо
        if (datetime.utcnow() - gap.timestamp).total_seconds() > 3600:
            self._active_gaps.pop(symbol, None)
            return None

        # Перевірити розмір gap
        if abs(gap.gap_pct) > self._config.max_gap_pct:
            # Занадто великий gap - це тренд
            self._active_gaps.pop(symbol, None)
            return None

        mid_price = snapshot.mid_price

        # Отримати рівень закриття попередньої сесії
        prev_levels = self._get_previous_session_levels(symbol, gap.from_session)
        if not prev_levels or not prev_levels.close_price:
            return None

        prev_close = prev_levels.close_price
        current_distance_pct = float((mid_price - prev_close) / prev_close * 100)

        # Визначити напрямок сигналу
        if gap.direction == "up":
            # Gap вгору -> очікуємо повернення вниз (SHORT)
            # Сигнал коли ціна почала повертатися
            if current_distance_pct < gap.gap_pct * 0.7:  # Вже заповнено 30%
                signal_type = SignalType.SHORT
                target = prev_close
                stop = prev_levels.high_price * (
                    1 + Decimal(str(self._config.stop_beyond_session_extreme_pct / 100))
                )
            else:
                return None
        else:
            # Gap вниз -> очікуємо повернення вгору (LONG)
            if current_distance_pct > gap.gap_pct * 0.7:
                signal_type = SignalType.LONG
                target = prev_close
                stop = prev_levels.low_price * (
                    1 - Decimal(str(self._config.stop_beyond_session_extreme_pct / 100))
                )
            else:
                return None

        # Розрахувати силу (чим більший gap, тим більша ймовірність fill)
        strength = min(abs(gap.gap_pct) / self._config.max_gap_pct * 0.5 + 0.5, 1.0)

        self._signals_from_gap += 1

        # Видалити gap (сигнал вже видано)
        self._active_gaps.pop(symbol, None)
        self._gap_fills += 1

        return self.create_signal(
            signal_type=signal_type,
            symbol=symbol,
            price=mid_price,
            strength=strength,
            metadata={
                "signal_source": "gap_fill",
                "gap_pct": gap.gap_pct,
                "gap_direction": gap.direction,
                "from_session": gap.from_session.value,
                "to_session": gap.to_session.value,
                "prev_close": float(prev_close),
                "target_price": float(target),
                "stop_price": float(stop),
            }
        )

    def _check_session_level_signal(
        self,
        symbol: str,
        snapshot: OrderBookSnapshot,
        current_session: TradingSession
    ) -> Optional[Signal]:
        """Перевірити сигнал від рівнів попередньої сесії."""
        if not self._config.use_prev_session_levels:
            return None

        # Отримати рівні попередньої сесії
        prev_levels = self._get_previous_session_levels(symbol)
        if not prev_levels:
            return None

        mid_price = snapshot.mid_price
        tolerance = mid_price * Decimal(str(self._config.prev_session_level_tolerance_pct / 100))

        # Перевірити близькість до ключових рівнів
        signal_type = None
        level_name = None
        level_price = None

        # Перевірка high попередньої сесії (опір)
        if abs(mid_price - prev_levels.high_price) <= tolerance:
            # Біля high - потенційний опір
            imbalance = float(snapshot.imbalance(5))
            if imbalance < 0.8:  # Продавців більше
                signal_type = SignalType.SHORT
                level_name = "prev_session_high"
                level_price = prev_levels.high_price

        # Перевірка low попередньої сесії (підтримка)
        elif abs(mid_price - prev_levels.low_price) <= tolerance:
            imbalance = float(snapshot.imbalance(5))
            if imbalance > 1.2:  # Покупців більше
                signal_type = SignalType.LONG
                level_name = "prev_session_low"
                level_price = prev_levels.low_price

        # Перевірка VWAP попередньої сесії
        elif prev_levels.vwap > 0 and abs(mid_price - prev_levels.vwap) <= tolerance:
            # VWAP - магнітний рівень
            imbalance = float(snapshot.imbalance(5))
            if mid_price > prev_levels.vwap and imbalance < 0.9:
                signal_type = SignalType.SHORT
                level_name = "prev_session_vwap"
                level_price = prev_levels.vwap
            elif mid_price < prev_levels.vwap and imbalance > 1.1:
                signal_type = SignalType.LONG
                level_name = "prev_session_vwap"
                level_price = prev_levels.vwap

        if signal_type is None:
            return None

        # Визначити target та stop
        if signal_type == SignalType.LONG:
            target = prev_levels.high_price
            stop = level_price * (1 - Decimal(str(self._config.stop_beyond_session_extreme_pct / 100)))
        else:
            target = prev_levels.low_price
            stop = level_price * (1 + Decimal(str(self._config.stop_beyond_session_extreme_pct / 100)))

        strength = 0.65

        self._signals_from_level += 1

        return self.create_signal(
            signal_type=signal_type,
            symbol=symbol,
            price=mid_price,
            strength=strength,
            metadata={
                "signal_source": "session_level",
                "level_name": level_name,
                "level_price": float(level_price),
                "prev_session": prev_levels.session.value,
                "prev_session_high": float(prev_levels.high_price),
                "prev_session_low": float(prev_levels.low_price),
                "prev_session_vwap": float(prev_levels.vwap),
                "target_price": float(target),
                "stop_price": float(stop),
            }
        )

    def _check_overlap_signal(
        self,
        symbol: str,
        snapshot: OrderBookSnapshot,
        overlap_session: TradingSession
    ) -> Optional[Signal]:
        """Перевірити сигнал під час overlap сесій."""
        # Під час overlap - підвищена волатильність
        # Шукаємо momentum сигнали

        current_levels = self._current_levels.get(symbol, {}).get(overlap_session)
        if not current_levels:
            return None

        mid_price = snapshot.mid_price
        session_range = current_levels.range

        if session_range == 0:
            return None

        # Позиція в діапазоні сесії
        position_in_range = float((mid_price - current_levels.low_price) / session_range)

        # Імбаланс ордербуку
        imbalance = float(snapshot.imbalance(10))

        # Сигнал на momentum
        signal_type = None

        # Ціна в верхній частині + сильний імбаланс покупців = продовження вгору
        if position_in_range > 0.8 and imbalance > 1.5:
            signal_type = SignalType.LONG

        # Ціна в нижній частині + сильний імбаланс продавців = продовження вниз
        elif position_in_range < 0.2 and imbalance < 0.67:
            signal_type = SignalType.SHORT

        if signal_type is None:
            return None

        # Підсилений сигнал для overlap
        strength = min(0.6 + abs(imbalance - 1) * 0.2, 1.0)
        strength *= self._config.overlap_volatility_boost

        return self.create_signal(
            signal_type=signal_type,
            symbol=symbol,
            price=mid_price,
            strength=min(strength, 1.0),
            metadata={
                "signal_source": "overlap_momentum",
                "overlap_session": overlap_session.value,
                "position_in_range": position_in_range,
                "session_range_pct": current_levels.range_pct,
                "imbalance": imbalance,
            }
        )

    def _get_previous_session_levels(
        self,
        symbol: str,
        specific_session: Optional[TradingSession] = None
    ) -> Optional[SessionLevels]:
        """Отримати рівні попередньої сесії."""
        history = self._session_history.get(symbol, [])

        if not history:
            return None

        if specific_session:
            # Знайти конкретну сесію
            for levels in reversed(history):
                if levels.session == specific_session and levels.is_complete:
                    return levels
        else:
            # Остання завершена сесія
            for levels in reversed(history):
                if levels.is_complete:
                    return levels

        return None

    def get_current_session(self) -> Optional[TradingSession]:
        """Отримати поточну сесію."""
        return self._current_session

    def get_session_levels(self, symbol: str) -> Dict[str, Any]:
        """Отримати рівні всіх активних сесій."""
        result = {}
        for session, levels in self._current_levels.get(symbol, {}).items():
            result[session.value] = {
                "open": float(levels.open_price),
                "high": float(levels.high_price),
                "low": float(levels.low_price),
                "close": float(levels.close_price) if levels.close_price else None,
                "vwap": float(levels.vwap),
                "range_pct": levels.range_pct,
                "is_complete": levels.is_complete,
            }
        return result

    def get_active_gap(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Отримати активний gap."""
        gap = self._active_gaps.get(symbol)
        if not gap:
            return None

        return {
            "from_session": gap.from_session.value,
            "to_session": gap.to_session.value,
            "gap_pct": gap.gap_pct,
            "direction": gap.direction,
            "timestamp": gap.timestamp.isoformat(),
        }

    @property
    def stats(self) -> Dict[str, Any]:
        """Статистика стратегії."""
        base = super().stats
        base.update({
            "session_changes": self._session_changes,
            "gaps_detected": self._gaps_detected,
            "gap_fills": self._gap_fills,
            "gap_fill_rate": (
                self._gap_fills / self._gaps_detected * 100
                if self._gaps_detected > 0 else 0
            ),
            "signals_from_gap": self._signals_from_gap,
            "signals_from_level": self._signals_from_level,
            "current_session": self._current_session.value if self._current_session else None,
        })
        return base

    def reset(self) -> None:
        """Скинути стан."""
        super().reset()
        self._current_session = None
        self._current_levels.clear()
        self._session_history.clear()
        self._active_gaps.clear()
        self._vwap_sum.clear()
        self._volume_sum.clear()
        self._session_changes = 0
        self._gaps_detected = 0
        self._gap_fills = 0
        self._signals_from_gap = 0
        self._signals_from_level = 0
