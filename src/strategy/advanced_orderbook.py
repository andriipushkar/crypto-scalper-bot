"""
Advanced Order Book Strategy with Front-Running Detection.

Розширена стратегія аналізу ордербуку для скальпінгу.

Включає:
1. Класичний дисбаланс bid/ask
2. Фронтраннінг - виявлення великих ордерів та торгівля "перед" ними
3. Спуфінг-детекція - виявлення фейкових ордерів
4. Ікеберг-детекція - пошук прихованих великих ордерів
5. Аналіз глибини книги
6. Динамічні рівні підтримки/опору
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


class WallType(Enum):
    """Тип стіни ордерів."""
    BID_WALL = "bid_wall"      # Стіна покупців
    ASK_WALL = "ask_wall"      # Стіна продавців


class WallReaction(Enum):
    """Реакція ціни на стіну (з книги "Скальпинг")."""
    BOUNCE = "bounce"          # Відскок від стіни
    BREAKOUT = "breakout"      # Пробій стіни
    ABSORPTION = "absorption"  # Поглинання стіни
    PENDING = "pending"        # Очікує визначення


class BidAskFlipType(Enum):
    """Тип flip bid/ask."""
    BID_TO_ASK = "bid_to_ask"  # Великий bid став ask (ведмежий)
    ASK_TO_BID = "ask_to_bid"  # Великий ask став bid (бичачий)


class OrderType(Enum):
    """Класифікація ордерів."""
    NORMAL = "normal"
    LARGE = "large"           # Великий ордер
    WALL = "wall"             # Стіна (дуже великий)
    ICEBERG = "iceberg"       # Ікеберг (прихований)
    SPOOF = "spoof"           # Спуфінг (фейковий)


@dataclass
class OrderLevel:
    """Рівень ордербуку з аналізом."""
    price: Decimal
    quantity: Decimal
    value_usd: Decimal
    side: str  # "bid" або "ask"
    order_type: OrderType
    distance_from_mid_pct: float
    cumulative_volume: Decimal
    timestamp: datetime


@dataclass
class OrderBookWall:
    """Виявлена стіна ордерів."""
    wall_type: WallType
    price: Decimal
    volume: Decimal
    value_usd: Decimal
    distance_from_mid_pct: float
    first_seen: datetime
    last_seen: datetime
    times_seen: int
    is_moving: bool = False        # Чи рухається стіна
    is_potential_spoof: bool = False


@dataclass
class IcebergOrder:
    """Виявлений ікеберг-ордер."""
    price: Decimal
    visible_volume: Decimal
    estimated_hidden_volume: Decimal
    side: str
    detection_confidence: float
    first_detected: datetime


@dataclass
class FrontRunOpportunity:
    """Можливість для фронтраннінгу."""
    timestamp: datetime
    direction: str  # "long" або "short"
    trigger_wall: OrderBookWall
    expected_price_move_pct: float
    confidence: float
    entry_price: Decimal
    target_price: Decimal
    stop_price: Decimal


@dataclass
class WallReactionEvent:
    """
    Подія реакції на стіну (з книги "Скальпинг").

    Стратегії:
    - Size Breakout: пробій стіни → торгуємо в напрямку пробою
    - Size Bounce: відскок від стіни → торгуємо відскок
    """
    timestamp: datetime
    wall: OrderBookWall
    reaction: WallReaction
    price_at_reaction: Decimal
    volume_at_reaction: Decimal
    price_after: Optional[Decimal] = None
    confirmed: bool = False
    signal_generated: bool = False


@dataclass
class BidAskFlipEvent:
    """
    Подія flip великого ордера з bid на ask або навпаки.

    Концепція з книги:
    "Коли великий bid раптово зникає і з'являється великий ask на тій же ціні
    або вище - це сигнал що великий гравець змінив позицію"
    """
    timestamp: datetime
    flip_type: BidAskFlipType
    price_level: Decimal
    original_volume: Decimal
    new_volume: Decimal
    price_change_pct: float
    is_same_price: bool  # Чи на тій самій ціні


@dataclass
class AdvancedOrderBookConfig:
    """Конфігурація розширеної стратегії ордербуку."""
    enabled: bool = True
    signal_cooldown: float = 5.0
    min_strength: float = 0.6

    # Аналіз книги
    levels_to_analyze: int = 20
    max_spread_bps: float = 10.0

    # Виявлення стін
    wall_threshold_usd: float = 100000      # Мін. розмір стіни ($)
    wall_min_times_seen: int = 3            # Мін. разів для підтвердження
    wall_max_distance_pct: float = 1.0      # Макс. відстань від mid (%)

    # Фронтраннінг
    frontrun_enabled: bool = True
    frontrun_min_wall_size_usd: float = 200000
    frontrun_max_distance_pct: float = 0.5
    frontrun_confidence_threshold: float = 0.7

    # Спуфінг-детекція
    spoof_detection_enabled: bool = True
    spoof_pull_time_ms: int = 500           # Якщо зникає < 500мс = спуф
    spoof_min_size_usd: float = 50000

    # Ікеберг-детекція
    iceberg_detection_enabled: bool = True
    iceberg_min_fills: int = 3              # Мін. виконань на одній ціні
    iceberg_refill_time_ms: int = 1000      # Час поповнення

    # Дисбаланс
    imbalance_threshold: float = 1.5
    imbalance_levels: int = 5

    # Size Bounce/Breakout (з книги "Скальпинг")
    size_reaction_enabled: bool = True
    bounce_confirmation_pct: float = 0.1       # Відскок > 0.1% = bounce
    breakout_confirmation_pct: float = 0.15    # Пробій > 0.15% = breakout
    reaction_confirmation_time_sec: int = 10   # Час для підтвердження
    absorption_volume_threshold: float = 0.5   # Якщо стіна < 50% = absorption

    # Bid/Ask Flip Detection
    flip_detection_enabled: bool = True
    flip_min_volume_usd: float = 50000         # Мін. розмір для flip
    flip_price_tolerance_pct: float = 0.1      # Толерантність ціни для "same price"
    flip_time_window_sec: int = 5              # Вікно часу для виявлення flip

    # Буфери
    history_size: int = 100
    wall_history_size: int = 50


@dataclass
class BookSnapshot:
    """Знімок стану книги для історії."""
    timestamp: datetime
    mid_price: Decimal
    spread_bps: Decimal
    imbalance: Decimal
    bid_depth: Decimal
    ask_depth: Decimal
    walls: List[OrderBookWall]


class AdvancedOrderBookStrategy(BaseStrategy):
    """
    Розширена стратегія аналізу ордербуку.

    Особливості:
    1. Фронтраннінг великих ордерів
    2. Детекція спуфінгу
    3. Пошук ікебергів
    4. Динамічні рівні
    """

    def __init__(self, config: Dict[str, Any]):
        """Ініціалізація стратегії."""
        super().__init__(config)

        self._config = self._parse_config(config)

        # Історія книги
        self._book_history: Dict[str, Deque[BookSnapshot]] = {}

        # Виявлені стіни
        self._walls: Dict[str, Dict[Decimal, OrderBookWall]] = {}

        # Виявлені ікеберги
        self._icebergs: Dict[str, List[IcebergOrder]] = {}

        # Історія рівнів для детекції спуфінгу
        self._level_history: Dict[str, Dict[Decimal, List[Tuple[datetime, Decimal]]]] = {}

        # Статистика виконань на рівнях (для ікебергів)
        self._level_fills: Dict[str, Dict[Decimal, List[datetime]]] = {}

        # Активні можливості фронтраннінгу
        self._frontrun_opportunities: List[FrontRunOpportunity] = []

        # Wall reactions (Size Bounce/Breakout)
        self._wall_reactions: Dict[str, List[WallReactionEvent]] = {}
        self._pending_reactions: Dict[str, Dict[Decimal, WallReactionEvent]] = {}

        # Bid/Ask Flip tracking
        self._large_orders_history: Dict[str, Dict[str, List[Tuple[datetime, Decimal, Decimal]]]] = {}
        self._flip_events: Dict[str, List[BidAskFlipEvent]] = {}

        # Статистика
        self._walls_detected = 0
        self._spoofs_detected = 0
        self._icebergs_detected = 0
        self._frontrun_signals = 0
        self._bounce_signals = 0
        self._breakout_signals = 0
        self._flip_signals = 0

        logger.info(
            f"[{self.name}] Initialized with wall_threshold=${self._config.wall_threshold_usd:,.0f}, "
            f"frontrun={self._config.frontrun_enabled}"
        )

    def _parse_config(self, config: Dict[str, Any]) -> AdvancedOrderBookConfig:
        """Парсинг конфігурації."""
        return AdvancedOrderBookConfig(
            enabled=config.get("enabled", True),
            signal_cooldown=config.get("signal_cooldown", 5.0),
            min_strength=config.get("min_strength", 0.6),
            levels_to_analyze=config.get("levels_to_analyze", 20),
            max_spread_bps=config.get("max_spread_bps", 10.0),
            wall_threshold_usd=config.get("wall_threshold_usd", 100000),
            wall_min_times_seen=config.get("wall_min_times_seen", 3),
            wall_max_distance_pct=config.get("wall_max_distance_pct", 1.0),
            frontrun_enabled=config.get("frontrun_enabled", True),
            frontrun_min_wall_size_usd=config.get("frontrun_min_wall_size_usd", 200000),
            frontrun_max_distance_pct=config.get("frontrun_max_distance_pct", 0.5),
            frontrun_confidence_threshold=config.get("frontrun_confidence_threshold", 0.7),
            spoof_detection_enabled=config.get("spoof_detection_enabled", True),
            spoof_pull_time_ms=config.get("spoof_pull_time_ms", 500),
            spoof_min_size_usd=config.get("spoof_min_size_usd", 50000),
            iceberg_detection_enabled=config.get("iceberg_detection_enabled", True),
            iceberg_min_fills=config.get("iceberg_min_fills", 3),
            iceberg_refill_time_ms=config.get("iceberg_refill_time_ms", 1000),
            imbalance_threshold=config.get("imbalance_threshold", 1.5),
            imbalance_levels=config.get("imbalance_levels", 5)
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

        # Ініціалізація буферів
        self._init_buffers(symbol)

        # Базова валідація
        if not snapshot.best_bid or not snapshot.best_ask:
            return None

        spread_bps = float(snapshot.spread_bps or 0)
        if spread_bps > self._config.max_spread_bps:
            return None

        mid_price = snapshot.mid_price

        # 1. Аналіз стін
        walls = self._detect_walls(snapshot)

        # 2. Оновити історію стін (для спуф-детекції)
        self._update_wall_history(symbol, walls, snapshot.timestamp)

        # 3. Детекція спуфінгу
        if self._config.spoof_detection_enabled:
            self._detect_spoofing(symbol)

        # 4. Зберегти знімок
        self._save_snapshot(symbol, snapshot, walls)

        # 5. Перевірити можливості фронтраннінгу
        if self._config.frontrun_enabled and self.can_signal():
            frontrun_signal = self._check_frontrun(symbol, snapshot, walls)
            if frontrun_signal:
                return frontrun_signal

        # 6. Size Bounce/Breakout (з книги "Скальпинг")
        if self._config.size_reaction_enabled and self.can_signal():
            self._track_wall_reactions(symbol, snapshot, walls)
            reaction_signal = self._check_wall_reaction_signal(symbol, snapshot)
            if reaction_signal:
                return reaction_signal

        # 7. Bid/Ask Flip Detection
        if self._config.flip_detection_enabled and self.can_signal():
            self._track_large_orders(symbol, snapshot)
            flip_signal = self._check_flip_signal(symbol, snapshot)
            if flip_signal:
                return flip_signal

        # 8. Класичний сигнал на дисбалансі
        if self.can_signal():
            imbalance_signal = self._check_imbalance(snapshot)
            if imbalance_signal:
                return imbalance_signal

        return None

    def on_trade(self, trade: Trade) -> Optional[Signal]:
        """
        Обробка угоди для детекції ікебергів.

        Args:
            trade: Торгова угода
        """
        if not self._config.iceberg_detection_enabled:
            return None

        symbol = trade.symbol
        price = trade.price

        # Ініціалізація
        if symbol not in self._level_fills:
            self._level_fills[symbol] = {}

        # Записати виконання на цьому рівні
        if price not in self._level_fills[symbol]:
            self._level_fills[symbol][price] = []

        self._level_fills[symbol][price].append(trade.timestamp)

        # Очистити старі записи
        cutoff = datetime.utcnow() - timedelta(minutes=5)
        self._level_fills[symbol][price] = [
            t for t in self._level_fills[symbol][price]
            if t >= cutoff
        ]

        # Перевірити на ікеберг
        self._check_iceberg(symbol, price, trade)

        return None

    def _init_buffers(self, symbol: str) -> None:
        """Ініціалізувати буфери для символу."""
        if symbol not in self._book_history:
            self._book_history[symbol] = deque(maxlen=self._config.history_size)
        if symbol not in self._walls:
            self._walls[symbol] = {}
        if symbol not in self._icebergs:
            self._icebergs[symbol] = []
        if symbol not in self._level_history:
            self._level_history[symbol] = {}

    def _detect_walls(self, snapshot: OrderBookSnapshot) -> List[OrderBookWall]:
        """
        Виявити стіни в ордербуку.

        Args:
            snapshot: Знімок ордербуку

        Returns:
            Список виявлених стін
        """
        walls = []
        mid_price = snapshot.mid_price

        if mid_price == 0:
            return walls

        # Аналіз бідів
        cumulative = Decimal("0")
        for i, (price, qty) in enumerate(snapshot.bids[:self._config.levels_to_analyze]):
            value_usd = price * qty
            cumulative += qty
            distance_pct = float((mid_price - price) / mid_price * 100)

            if distance_pct > self._config.wall_max_distance_pct:
                break

            if float(value_usd) >= self._config.wall_threshold_usd:
                walls.append(OrderBookWall(
                    wall_type=WallType.BID_WALL,
                    price=price,
                    volume=qty,
                    value_usd=value_usd,
                    distance_from_mid_pct=distance_pct,
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    times_seen=1
                ))

        # Аналіз асків
        cumulative = Decimal("0")
        for i, (price, qty) in enumerate(snapshot.asks[:self._config.levels_to_analyze]):
            value_usd = price * qty
            cumulative += qty
            distance_pct = float((price - mid_price) / mid_price * 100)

            if distance_pct > self._config.wall_max_distance_pct:
                break

            if float(value_usd) >= self._config.wall_threshold_usd:
                walls.append(OrderBookWall(
                    wall_type=WallType.ASK_WALL,
                    price=price,
                    volume=qty,
                    value_usd=value_usd,
                    distance_from_mid_pct=distance_pct,
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    times_seen=1
                ))

        return walls

    def _update_wall_history(
        self,
        symbol: str,
        current_walls: List[OrderBookWall],
        timestamp: datetime
    ) -> None:
        """Оновити історію стін для трекінгу."""
        existing_walls = self._walls[symbol]

        # Оновити існуючі стіни
        current_prices = {w.price for w in current_walls}

        for wall in current_walls:
            if wall.price in existing_walls:
                # Оновити існуючу
                old_wall = existing_walls[wall.price]
                existing_walls[wall.price] = OrderBookWall(
                    wall_type=wall.wall_type,
                    price=wall.price,
                    volume=wall.volume,
                    value_usd=wall.value_usd,
                    distance_from_mid_pct=wall.distance_from_mid_pct,
                    first_seen=old_wall.first_seen,
                    last_seen=timestamp,
                    times_seen=old_wall.times_seen + 1,
                    is_moving=wall.price != old_wall.price,
                    is_potential_spoof=old_wall.is_potential_spoof
                )
            else:
                # Нова стіна
                existing_walls[wall.price] = wall
                self._walls_detected += 1

        # Відмітити зниклі стіни для спуф-детекції
        for price in list(existing_walls.keys()):
            if price not in current_prices:
                wall = existing_walls[price]
                lifetime_ms = (timestamp - wall.last_seen).total_seconds() * 1000

                if lifetime_ms < self._config.spoof_pull_time_ms:
                    # Потенційний спуф
                    wall.is_potential_spoof = True
                    self._spoofs_detected += 1
                    logger.debug(
                        f"[{self.name}] Potential spoof detected at {price} "
                        f"(lifetime={lifetime_ms:.0f}ms)"
                    )

                # Видалити якщо давно не бачили
                if (timestamp - wall.last_seen).total_seconds() > 60:
                    del existing_walls[price]

    def _detect_spoofing(self, symbol: str) -> None:
        """Виявити спуфінг на основі історії."""
        # Спуфінг визначається як:
        # 1. Великий ордер з'являється
        # 2. Швидко зникає (< 500мс)
        # 3. Не виконується
        pass  # Логіка вже в _update_wall_history

    def _check_iceberg(self, symbol: str, price: Decimal, trade: Trade) -> None:
        """
        Перевірити на ікеберг-ордер.

        Ікеберг виявляється коли:
        1. На одній ціні багато виконань
        2. Ордер "поповнюється" після виконання
        """
        fills = self._level_fills[symbol].get(price, [])

        if len(fills) < self._config.iceberg_min_fills:
            return

        # Перевірити часові інтервали між виконаннями
        time_gaps = []
        for i in range(1, len(fills)):
            gap = (fills[i] - fills[i-1]).total_seconds() * 1000
            time_gaps.append(gap)

        if not time_gaps:
            return

        avg_gap = sum(time_gaps) / len(time_gaps)

        # Якщо середній інтервал близький до часу поповнення - це ікеберг
        if avg_gap <= self._config.iceberg_refill_time_ms:
            # Оцінити прихований обсяг
            visible_volume = trade.quantity
            estimated_hidden = visible_volume * Decimal(str(len(fills)))

            # Перевірити чи вже не виявлено
            existing = [
                ice for ice in self._icebergs[symbol]
                if ice.price == price
            ]

            if not existing:
                iceberg = IcebergOrder(
                    price=price,
                    visible_volume=visible_volume,
                    estimated_hidden_volume=estimated_hidden,
                    side="bid" if trade.is_buyer_maker else "ask",
                    detection_confidence=min(len(fills) / 10, 1.0),
                    first_detected=datetime.utcnow()
                )
                self._icebergs[symbol].append(iceberg)
                self._icebergs_detected += 1

                logger.info(
                    f"[{self.name}] Iceberg detected at {price}: "
                    f"~{estimated_hidden} estimated volume"
                )

    def _save_snapshot(
        self,
        symbol: str,
        snapshot: OrderBookSnapshot,
        walls: List[OrderBookWall]
    ) -> None:
        """Зберегти знімок книги."""
        self._book_history[symbol].append(BookSnapshot(
            timestamp=snapshot.timestamp,
            mid_price=snapshot.mid_price,
            spread_bps=snapshot.spread_bps or Decimal("0"),
            imbalance=snapshot.imbalance(self._config.imbalance_levels),
            bid_depth=snapshot.bid_volume(self._config.levels_to_analyze),
            ask_depth=snapshot.ask_volume(self._config.levels_to_analyze),
            walls=walls
        ))

    def _check_frontrun(
        self,
        symbol: str,
        snapshot: OrderBookSnapshot,
        walls: List[OrderBookWall]
    ) -> Optional[Signal]:
        """
        Перевірити можливість фронтраннінгу.

        Логіка:
        1. Знайти велику стіну (bid або ask)
        2. Якщо стіна близько до ціни - ринок може "відштовхнутися"
        3. Торгувати в напрямку від стіни
        """
        mid_price = snapshot.mid_price

        # Знайти підтверджені стіни
        confirmed_walls = [
            w for w in self._walls[symbol].values()
            if (w.times_seen >= self._config.wall_min_times_seen and
                float(w.value_usd) >= self._config.frontrun_min_wall_size_usd and
                w.distance_from_mid_pct <= self._config.frontrun_max_distance_pct and
                not w.is_potential_spoof)
        ]

        if not confirmed_walls:
            return None

        # Знайти найближчу стіну
        closest_wall = min(confirmed_walls, key=lambda w: w.distance_from_mid_pct)

        # Визначити напрямок
        if closest_wall.wall_type == WallType.BID_WALL:
            # Велика bid-стіна = підтримка = ціна може піти вгору
            direction = SignalType.LONG
            expected_move = closest_wall.distance_from_mid_pct * 0.5  # Консервативно
            target_price = mid_price * (1 + Decimal(str(expected_move / 100)))
            stop_price = closest_wall.price * Decimal("0.999")  # Трохи нижче стіни
        else:
            # Велика ask-стіна = опір = ціна може піти вниз
            direction = SignalType.SHORT
            expected_move = closest_wall.distance_from_mid_pct * 0.5
            target_price = mid_price * (1 - Decimal(str(expected_move / 100)))
            stop_price = closest_wall.price * Decimal("1.001")

        # Розрахувати впевненість
        confidence = min(
            closest_wall.times_seen / 10 *
            float(closest_wall.value_usd) / self._config.frontrun_min_wall_size_usd,
            1.0
        )

        if confidence < self._config.frontrun_confidence_threshold:
            return None

        # Перевірити чи ордербук підтверджує
        imbalance = float(snapshot.imbalance(self._config.imbalance_levels))
        if direction == SignalType.LONG and imbalance < 1.0:
            return None
        if direction == SignalType.SHORT and imbalance > 1.0:
            return None

        self._frontrun_signals += 1

        return self.create_signal(
            signal_type=direction,
            symbol=symbol,
            price=mid_price,
            strength=confidence,
            metadata={
                "signal_source": "frontrun",
                "wall_price": float(closest_wall.price),
                "wall_value_usd": float(closest_wall.value_usd),
                "wall_type": closest_wall.wall_type.value,
                "wall_distance_pct": closest_wall.distance_from_mid_pct,
                "wall_times_seen": closest_wall.times_seen,
                "target_price": float(target_price),
                "stop_price": float(stop_price),
                "imbalance": imbalance,
            }
        )

    def _check_imbalance(self, snapshot: OrderBookSnapshot) -> Optional[Signal]:
        """
        Перевірити класичний дисбаланс.

        Args:
            snapshot: Знімок ордербуку

        Returns:
            Сигнал якщо дисбаланс перевищує поріг
        """
        imbalance = float(snapshot.imbalance(self._config.imbalance_levels))
        threshold = self._config.imbalance_threshold

        if imbalance >= threshold:
            # Бичачий дисбаланс
            excess = imbalance - threshold
            strength = min(0.5 + (excess / threshold) * 0.5, 1.0)

            return self.create_signal(
                signal_type=SignalType.LONG,
                symbol=snapshot.symbol,
                price=snapshot.mid_price,
                strength=strength,
                metadata={
                    "signal_source": "imbalance",
                    "imbalance": imbalance,
                    "bid_volume": float(snapshot.bid_volume(self._config.imbalance_levels)),
                    "ask_volume": float(snapshot.ask_volume(self._config.imbalance_levels)),
                }
            )

        elif imbalance <= 1 / threshold:
            # Ведмежий дисбаланс
            inverse = 1 / imbalance if imbalance > 0 else 999
            excess = inverse - threshold
            strength = min(0.5 + (excess / threshold) * 0.5, 1.0)

            return self.create_signal(
                signal_type=SignalType.SHORT,
                symbol=snapshot.symbol,
                price=snapshot.mid_price,
                strength=strength,
                metadata={
                    "signal_source": "imbalance",
                    "imbalance": imbalance,
                    "bid_volume": float(snapshot.bid_volume(self._config.imbalance_levels)),
                    "ask_volume": float(snapshot.ask_volume(self._config.imbalance_levels)),
                }
            )

        return None

    def _track_wall_reactions(
        self,
        symbol: str,
        snapshot: OrderBookSnapshot,
        current_walls: List[OrderBookWall]
    ) -> None:
        """
        Відстежувати реакцію ціни на стіни (Size Bounce/Breakout).

        Концепція з книги "Скальпинг":
        - Size Breakout: ціна пробиває великий ордер → торгуємо в напрямку пробою
        - Size Bounce: ціна відскакує від великого ордера → торгуємо відскок
        """
        if symbol not in self._pending_reactions:
            self._pending_reactions[symbol] = {}
        if symbol not in self._wall_reactions:
            self._wall_reactions[symbol] = []

        mid_price = snapshot.mid_price
        confirmed_walls = self._walls.get(symbol, {})

        # Перевірити реакцію на кожну підтверджену стіну
        for price, wall in list(confirmed_walls.items()):
            if wall.times_seen < self._config.wall_min_times_seen:
                continue

            # Перевірити чи ціна наближається до стіни
            distance_pct = abs(float((mid_price - price) / mid_price * 100))

            # Якщо ціна біля стіни і ще немає pending reaction
            if distance_pct < 0.2 and price not in self._pending_reactions[symbol]:
                self._pending_reactions[symbol][price] = WallReactionEvent(
                    timestamp=datetime.utcnow(),
                    wall=wall,
                    reaction=WallReaction.PENDING,
                    price_at_reaction=mid_price,
                    volume_at_reaction=wall.volume
                )

        # Класифікувати pending reactions
        for price, reaction in list(self._pending_reactions[symbol].items()):
            if reaction.confirmed:
                continue

            time_elapsed = (datetime.utcnow() - reaction.timestamp).total_seconds()

            # Перевірити тайм-аут
            if time_elapsed > self._config.reaction_confirmation_time_sec:
                self._pending_reactions[symbol].pop(price, None)
                continue

            wall = reaction.wall
            price_change_pct = float((mid_price - reaction.price_at_reaction) / reaction.price_at_reaction * 100)

            # Перевірити чи стіна ще існує
            wall_still_exists = price in confirmed_walls
            wall_absorbed = False
            if wall_still_exists:
                current_volume = confirmed_walls[price].volume
                if current_volume < wall.volume * Decimal(str(self._config.absorption_volume_threshold)):
                    wall_absorbed = True

            # Класифікувати реакцію
            if wall.wall_type == WallType.BID_WALL:
                # Bid wall - очікуємо bounce вгору або breakout вниз
                if price_change_pct > self._config.bounce_confirmation_pct and wall_still_exists:
                    # Bounce вгору від bid wall
                    reaction.reaction = WallReaction.BOUNCE
                    reaction.price_after = mid_price
                    reaction.confirmed = True
                    self._wall_reactions[symbol].append(reaction)
                    logger.debug(f"[{self.name}] Bounce UP from bid wall at {price}")

                elif price_change_pct < -self._config.breakout_confirmation_pct:
                    if wall_absorbed or not wall_still_exists:
                        # Breakout вниз через bid wall
                        reaction.reaction = WallReaction.BREAKOUT
                        reaction.price_after = mid_price
                        reaction.confirmed = True
                        self._wall_reactions[symbol].append(reaction)
                        logger.debug(f"[{self.name}] Breakout DOWN through bid wall at {price}")
                    elif wall_absorbed:
                        reaction.reaction = WallReaction.ABSORPTION
                        reaction.price_after = mid_price
                        reaction.confirmed = True
                        self._wall_reactions[symbol].append(reaction)

            else:  # ASK_WALL
                # Ask wall - очікуємо bounce вниз або breakout вгору
                if price_change_pct < -self._config.bounce_confirmation_pct and wall_still_exists:
                    # Bounce вниз від ask wall
                    reaction.reaction = WallReaction.BOUNCE
                    reaction.price_after = mid_price
                    reaction.confirmed = True
                    self._wall_reactions[symbol].append(reaction)
                    logger.debug(f"[{self.name}] Bounce DOWN from ask wall at {price}")

                elif price_change_pct > self._config.breakout_confirmation_pct:
                    if wall_absorbed or not wall_still_exists:
                        # Breakout вгору через ask wall
                        reaction.reaction = WallReaction.BREAKOUT
                        reaction.price_after = mid_price
                        reaction.confirmed = True
                        self._wall_reactions[symbol].append(reaction)
                        logger.debug(f"[{self.name}] Breakout UP through ask wall at {price}")

            # Очистити підтверджену reaction
            if reaction.confirmed:
                self._pending_reactions[symbol].pop(price, None)

    def _check_wall_reaction_signal(
        self,
        symbol: str,
        snapshot: OrderBookSnapshot
    ) -> Optional[Signal]:
        """Перевірити сигнали на основі реакції на стіни."""
        reactions = self._wall_reactions.get(symbol, [])
        if not reactions:
            return None

        # Знайти непрооброблену реакцію
        for reaction in reactions:
            if reaction.signal_generated:
                continue

            # Перевірити свіжість
            age = (datetime.utcnow() - reaction.timestamp).total_seconds()
            if age > 30:  # Старіше 30 секунд - пропустити
                continue

            mid_price = snapshot.mid_price
            wall = reaction.wall

            if reaction.reaction == WallReaction.BOUNCE:
                # Bounce - торгуємо в напрямку відскоку
                if wall.wall_type == WallType.BID_WALL:
                    # Bounce від bid = LONG
                    signal_type = SignalType.LONG
                    target = mid_price * Decimal("1.005")  # +0.5%
                    stop = wall.price * Decimal("0.998")   # трохи нижче стіни
                else:
                    # Bounce від ask = SHORT
                    signal_type = SignalType.SHORT
                    target = mid_price * Decimal("0.995")
                    stop = wall.price * Decimal("1.002")

                reaction.signal_generated = True
                self._bounce_signals += 1

                return self.create_signal(
                    signal_type=signal_type,
                    symbol=symbol,
                    price=mid_price,
                    strength=0.7,
                    metadata={
                        "signal_source": "size_bounce",
                        "wall_price": float(wall.price),
                        "wall_volume_usd": float(wall.value_usd),
                        "wall_type": wall.wall_type.value,
                        "bounce_from": float(reaction.price_at_reaction),
                        "target_price": float(target),
                        "stop_price": float(stop),
                    }
                )

            elif reaction.reaction == WallReaction.BREAKOUT:
                # Breakout - торгуємо в напрямку пробою
                if wall.wall_type == WallType.BID_WALL:
                    # Пробій bid wall = SHORT (продовження вниз)
                    signal_type = SignalType.SHORT
                    target = mid_price * Decimal("0.99")   # -1%
                    stop = wall.price * Decimal("1.003")   # вище стіни
                else:
                    # Пробій ask wall = LONG (продовження вгору)
                    signal_type = SignalType.LONG
                    target = mid_price * Decimal("1.01")
                    stop = wall.price * Decimal("0.997")

                reaction.signal_generated = True
                self._breakout_signals += 1

                return self.create_signal(
                    signal_type=signal_type,
                    symbol=symbol,
                    price=mid_price,
                    strength=0.75,
                    metadata={
                        "signal_source": "size_breakout",
                        "wall_price": float(wall.price),
                        "wall_volume_usd": float(wall.value_usd),
                        "wall_type": wall.wall_type.value,
                        "breakout_from": float(reaction.price_at_reaction),
                        "target_price": float(target),
                        "stop_price": float(stop),
                    }
                )

        return None

    def _track_large_orders(
        self,
        symbol: str,
        snapshot: OrderBookSnapshot
    ) -> None:
        """
        Відстежувати великі ордери для виявлення Bid/Ask Flip.

        Концепція:
        Коли великий bid зникає і з'являється великий ask - це сигнал зміни настрою.
        """
        if symbol not in self._large_orders_history:
            self._large_orders_history[symbol] = {"bids": [], "asks": []}
        if symbol not in self._flip_events:
            self._flip_events[symbol] = []

        now = datetime.utcnow()
        threshold_usd = self._config.flip_min_volume_usd

        # Записати поточні великі bid'и
        for price, qty in snapshot.bids[:10]:
            value_usd = float(price * qty)
            if value_usd >= threshold_usd:
                self._large_orders_history[symbol]["bids"].append((now, price, qty))

        # Записати поточні великі ask'и
        for price, qty in snapshot.asks[:10]:
            value_usd = float(price * qty)
            if value_usd >= threshold_usd:
                self._large_orders_history[symbol]["asks"].append((now, price, qty))

        # Очистити старі записи
        cutoff = now - timedelta(seconds=self._config.flip_time_window_sec * 2)
        self._large_orders_history[symbol]["bids"] = [
            (t, p, q) for t, p, q in self._large_orders_history[symbol]["bids"]
            if t >= cutoff
        ]
        self._large_orders_history[symbol]["asks"] = [
            (t, p, q) for t, p, q in self._large_orders_history[symbol]["asks"]
            if t >= cutoff
        ]

        # Виявити flip'и
        self._detect_flips(symbol, snapshot)

    def _detect_flips(self, symbol: str, snapshot: OrderBookSnapshot) -> None:
        """Виявити flip події."""
        history = self._large_orders_history.get(symbol, {"bids": [], "asks": []})
        now = datetime.utcnow()
        window = timedelta(seconds=self._config.flip_time_window_sec)
        tolerance_pct = self._config.flip_price_tolerance_pct / 100

        # Перевірити bid -> ask flip
        recent_bids = [(t, p, q) for t, p, q in history["bids"] if now - t < window]
        current_ask_prices = {p for p, q in snapshot.asks[:10]}

        for bid_time, bid_price, bid_qty in recent_bids:
            # Перевірити чи цей bid зник
            bid_still_exists = any(
                abs(float((p - bid_price) / bid_price)) < tolerance_pct
                for p, q in snapshot.bids[:10]
            )

            if not bid_still_exists:
                # Перевірити чи з'явився ask на схожій ціні
                for ask_price, ask_qty in snapshot.asks[:10]:
                    price_diff_pct = float((ask_price - bid_price) / bid_price * 100)
                    if abs(price_diff_pct) < self._config.flip_price_tolerance_pct:
                        # Виявлено flip: bid став ask
                        flip = BidAskFlipEvent(
                            timestamp=now,
                            flip_type=BidAskFlipType.BID_TO_ASK,
                            price_level=bid_price,
                            original_volume=bid_qty,
                            new_volume=ask_qty,
                            price_change_pct=price_diff_pct,
                            is_same_price=abs(price_diff_pct) < 0.05
                        )
                        self._flip_events[symbol].append(flip)
                        logger.info(
                            f"[{self.name}] BID->ASK flip at {bid_price} "
                            f"(vol: {bid_qty} -> {ask_qty})"
                        )
                        break

        # Перевірити ask -> bid flip (аналогічно)
        recent_asks = [(t, p, q) for t, p, q in history["asks"] if now - t < window]

        for ask_time, ask_price, ask_qty in recent_asks:
            ask_still_exists = any(
                abs(float((p - ask_price) / ask_price)) < tolerance_pct
                for p, q in snapshot.asks[:10]
            )

            if not ask_still_exists:
                for bid_price, bid_qty in snapshot.bids[:10]:
                    price_diff_pct = float((bid_price - ask_price) / ask_price * 100)
                    if abs(price_diff_pct) < self._config.flip_price_tolerance_pct:
                        flip = BidAskFlipEvent(
                            timestamp=now,
                            flip_type=BidAskFlipType.ASK_TO_BID,
                            price_level=ask_price,
                            original_volume=ask_qty,
                            new_volume=bid_qty,
                            price_change_pct=price_diff_pct,
                            is_same_price=abs(price_diff_pct) < 0.05
                        )
                        self._flip_events[symbol].append(flip)
                        logger.info(
                            f"[{self.name}] ASK->BID flip at {ask_price} "
                            f"(vol: {ask_qty} -> {bid_qty})"
                        )
                        break

        # Обмежити історію flip'ів
        if len(self._flip_events[symbol]) > 50:
            self._flip_events[symbol] = self._flip_events[symbol][-50:]

    def _check_flip_signal(
        self,
        symbol: str,
        snapshot: OrderBookSnapshot
    ) -> Optional[Signal]:
        """Перевірити сигнал на основі Bid/Ask Flip."""
        flips = self._flip_events.get(symbol, [])
        if not flips:
            return None

        # Знайти свіжий flip
        now = datetime.utcnow()
        recent_flips = [f for f in flips if (now - f.timestamp).total_seconds() < 10]

        if not recent_flips:
            return None

        latest_flip = recent_flips[-1]
        mid_price = snapshot.mid_price

        if latest_flip.flip_type == BidAskFlipType.BID_TO_ASK:
            # Bid став ask = ведмежий сигнал (великий гравець перевернувся)
            signal_type = SignalType.SHORT
            target = mid_price * Decimal("0.995")
            stop = latest_flip.price_level * Decimal("1.005")
        else:
            # Ask став bid = бичачий сигнал
            signal_type = SignalType.LONG
            target = mid_price * Decimal("1.005")
            stop = latest_flip.price_level * Decimal("0.995")

        # Видалити оброблений flip
        self._flip_events[symbol] = [f for f in flips if f != latest_flip]
        self._flip_signals += 1

        return self.create_signal(
            signal_type=signal_type,
            symbol=symbol,
            price=mid_price,
            strength=0.7,
            metadata={
                "signal_source": "bid_ask_flip",
                "flip_type": latest_flip.flip_type.value,
                "flip_price": float(latest_flip.price_level),
                "original_volume": float(latest_flip.original_volume),
                "new_volume": float(latest_flip.new_volume),
                "is_same_price": latest_flip.is_same_price,
                "target_price": float(target),
                "stop_price": float(stop),
            }
        )

    def get_wall_reactions(self, symbol: str, limit: int = 10) -> List[WallReactionEvent]:
        """Отримати історію реакцій на стіни."""
        return self._wall_reactions.get(symbol, [])[-limit:]

    def get_flip_events(self, symbol: str, limit: int = 10) -> List[BidAskFlipEvent]:
        """Отримати історію flip подій."""
        return self._flip_events.get(symbol, [])[-limit:]

    def get_walls(self, symbol: str) -> List[OrderBookWall]:
        """Отримати поточні стіни."""
        return list(self._walls.get(symbol, {}).values())

    def get_icebergs(self, symbol: str) -> List[IcebergOrder]:
        """Отримати виявлені ікеберги."""
        return self._icebergs.get(symbol, [])

    def get_confirmed_support_resistance(
        self,
        symbol: str
    ) -> Tuple[List[Decimal], List[Decimal]]:
        """
        Отримати підтверджені рівні підтримки та опору.

        Returns:
            (supports, resistances)
        """
        walls = self._walls.get(symbol, {})

        supports = [
            w.price for w in walls.values()
            if w.wall_type == WallType.BID_WALL and
            w.times_seen >= self._config.wall_min_times_seen
        ]

        resistances = [
            w.price for w in walls.values()
            if w.wall_type == WallType.ASK_WALL and
            w.times_seen >= self._config.wall_min_times_seen
        ]

        return sorted(supports, reverse=True), sorted(resistances)

    @property
    def stats(self) -> Dict[str, Any]:
        """Отримати статистику."""
        base = super().stats
        base.update({
            "walls_detected": self._walls_detected,
            "spoofs_detected": self._spoofs_detected,
            "icebergs_detected": self._icebergs_detected,
            "frontrun_signals": self._frontrun_signals,
            "bounce_signals": self._bounce_signals,
            "breakout_signals": self._breakout_signals,
            "flip_signals": self._flip_signals,
        })
        return base

    def reset(self) -> None:
        """Скинути стан."""
        super().reset()
        self._book_history.clear()
        self._walls.clear()
        self._icebergs.clear()
        self._level_history.clear()
        self._level_fills.clear()
        self._frontrun_opportunities.clear()
        self._wall_reactions.clear()
        self._pending_reactions.clear()
        self._large_orders_history.clear()
        self._flip_events.clear()
        self._walls_detected = 0
        self._spoofs_detected = 0
        self._icebergs_detected = 0
        self._frontrun_signals = 0
        self._bounce_signals = 0
        self._breakout_signals = 0
        self._flip_signals = 0
