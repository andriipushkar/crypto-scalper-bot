"""
Impulse Scalping Strategy (Імпульсний скальпінг).

Стратегія, яка слідує за "поводирями" (індексами), використовуючи кореляцію
між глобальними ринками та криптовалютами для генерації короткострокових сигналів.

Концепція:
- Глобальні індекси (SPX, DXY, Gold) часто "ведуть" криптовалюти
- Різкий рух в індексі-поводирі може передбачити рух криптовалюти
- Стратегія відстежує імпульси в корельованих активах
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, Deque, Tuple
from enum import Enum

import aiohttp
from loguru import logger

from src.data.models import OrderBookSnapshot, Signal, SignalType, Trade
from src.strategy.base import BaseStrategy


class LeaderAsset(Enum):
    """
    Індекси-поводирі для імпульсної стратегії.

    Традиційні (з книги "Скальпинг"):
    - SPX, DXY, GOLD, VIX - класичні індекси

    Крипто-специфічні (адаптація для крипторинку):
    - BTC, ETH - основні криптовалюти як лідери для альткоїнів
    - BTC_DOM, ETH_DOM - домінація як індикатор потоків капіталу
    - TOTAL - загальна капіталізація ринку
    """
    # Традиційні індекси
    SPX = "SPX"           # S&P 500
    DXY = "DXY"           # US Dollar Index
    GOLD = "GOLD"         # Gold (XAU)
    VIX = "VIX"           # Volatility Index
    US10Y = "US10Y"       # US 10 Year Treasury Yield

    # Крипто-лідери (для альткоїнів)
    BTC = "BTC"           # Bitcoin - головний лідер для альткоїнів
    ETH = "ETH"           # Ethereum - лідер для DeFi/L2 токенів
    BTC_DOM = "BTC_DOM"   # Bitcoin Dominance (% ринку)
    ETH_DOM = "ETH_DOM"   # Ethereum Dominance
    TOTAL = "TOTAL"       # Total Crypto Market Cap
    TOTAL2 = "TOTAL2"     # Total Market Cap без BTC
    TOTAL3 = "TOTAL3"     # Total Market Cap без BTC та ETH


@dataclass
class LeaderConfig:
    """Конфігурація для індексу-поводиря."""
    asset: LeaderAsset
    correlation: float          # Кореляція з крипто (-1 до 1)
    impact_delay_ms: int        # Затримка впливу (мілісекунди)
    impulse_threshold: float    # Поріг імпульсу (%)
    weight: float = 1.0         # Вага в загальному сигналі


@dataclass
class PricePoint:
    """Точка ціни для історії."""
    timestamp: datetime
    price: Decimal
    volume: Decimal = Decimal("0")


@dataclass
class ImpulseEvent:
    """Подія імпульсу в індексі-поводирі."""
    asset: LeaderAsset
    timestamp: datetime
    direction: str              # "up" або "down"
    magnitude: float            # Величина руху (%)
    expected_crypto_impact: float  # Очікуваний вплив на крипто
    expiry: datetime            # Час закінчення сигналу


@dataclass
class ImpulseScalpingConfig:
    """Конфігурація імпульсної стратегії."""
    enabled: bool = True
    signal_cooldown: float = 5.0
    min_strength: float = 0.6

    # Поводирі та їх налаштування
    leaders: List[LeaderConfig] = field(default_factory=lambda: [
        # Традиційні індекси (для BTC та великих альткоїнів)
        LeaderConfig(
            asset=LeaderAsset.SPX,
            correlation=0.7,
            impact_delay_ms=500,
            impulse_threshold=0.15,
            weight=1.0
        ),
        LeaderConfig(
            asset=LeaderAsset.DXY,
            correlation=-0.6,
            impact_delay_ms=300,
            impulse_threshold=0.1,
            weight=0.8
        ),
        LeaderConfig(
            asset=LeaderAsset.GOLD,
            correlation=0.5,
            impact_delay_ms=800,
            impulse_threshold=0.2,
            weight=0.6
        ),
        LeaderConfig(
            asset=LeaderAsset.VIX,
            correlation=-0.65,
            impact_delay_ms=200,
            impulse_threshold=2.0,
            weight=0.9
        ),
        # Крипто-лідери (для альткоїнів)
        LeaderConfig(
            asset=LeaderAsset.BTC,
            correlation=0.85,          # Висока кореляція альткоїнів з BTC
            impact_delay_ms=100,       # Швидший вплив в крипто
            impulse_threshold=0.3,     # BTC рухається на 0.3%+ = імпульс
            weight=1.5                 # Найважливіший лідер
        ),
        LeaderConfig(
            asset=LeaderAsset.ETH,
            correlation=0.8,           # ETH також сильно корелює
            impact_delay_ms=150,
            impulse_threshold=0.4,
            weight=1.2
        ),
        LeaderConfig(
            asset=LeaderAsset.BTC_DOM,
            correlation=-0.5,          # Зростання домінації = падіння альтів
            impact_delay_ms=1000,      # Повільніший індикатор
            impulse_threshold=0.5,     # Зміна домінації на 0.5%+
            weight=0.7
        ),
        LeaderConfig(
            asset=LeaderAsset.TOTAL,
            correlation=0.9,           # Загальна капіталізація
            impact_delay_ms=200,
            impulse_threshold=0.5,
            weight=1.0
        ),
    ])

    # Таймінг
    lookback_seconds: int = 60
    impulse_validity_seconds: int = 30
    min_impulse_duration_ms: int = 100
    max_impulse_duration_ms: int = 5000

    # Фільтри
    min_volume_ratio: float = 1.5        # Мін. обсяг відносно середнього
    max_spread_bps: float = 10.0         # Макс. спред (базисні пункти)
    require_orderbook_confirmation: bool = True
    confirmation_imbalance: float = 1.2  # Поріг підтвердження ордербуком


class ImpulseScalpingStrategy(BaseStrategy):
    """
    Імпульсна стратегія скальпінгу.

    Відстежує глобальні індекси-"поводирі" та генерує сигнали
    на основі їх кореляції з криптовалютами.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Ініціалізація стратегії.

        Args:
            config: Конфігурація стратегії
        """
        super().__init__(config)

        # Парсинг конфігурації
        self._config = self._parse_config(config)

        # Історія цін для кожного поводиря
        self._leader_history: Dict[LeaderAsset, Deque[PricePoint]] = {
            leader.asset: deque(maxlen=1000)
            for leader in self._config.leaders
        }

        # Історія цін криптовалюти
        self._crypto_history: Dict[str, Deque[PricePoint]] = {}

        # Активні імпульсні події
        self._active_impulses: List[ImpulseEvent] = []

        # HTTP сесія для отримання даних індексів
        self._session: Optional[aiohttp.ClientSession] = None

        # Кеш останніх цін поводирів
        self._leader_prices: Dict[LeaderAsset, Decimal] = {}

        # Статистика
        self._impulses_detected = 0
        self._signals_from_impulses = 0

        logger.info(
            f"[{self.name}] Initialized with {len(self._config.leaders)} leaders, "
            f"lookback={self._config.lookback_seconds}s"
        )

    def _parse_config(self, config: Dict[str, Any]) -> ImpulseScalpingConfig:
        """Парсинг конфігурації."""
        leaders = []
        for leader_cfg in config.get("leaders", []):
            leaders.append(LeaderConfig(
                asset=LeaderAsset(leader_cfg.get("asset", "SPX")),
                correlation=leader_cfg.get("correlation", 0.7),
                impact_delay_ms=leader_cfg.get("impact_delay_ms", 500),
                impulse_threshold=leader_cfg.get("impulse_threshold", 0.15),
                weight=leader_cfg.get("weight", 1.0)
            ))

        return ImpulseScalpingConfig(
            enabled=config.get("enabled", True),
            signal_cooldown=config.get("signal_cooldown", 5.0),
            min_strength=config.get("min_strength", 0.6),
            leaders=leaders if leaders else ImpulseScalpingConfig().leaders,
            lookback_seconds=config.get("lookback_seconds", 60),
            impulse_validity_seconds=config.get("impulse_validity_seconds", 30),
            min_impulse_duration_ms=config.get("min_impulse_duration_ms", 100),
            max_impulse_duration_ms=config.get("max_impulse_duration_ms", 5000),
            min_volume_ratio=config.get("min_volume_ratio", 1.5),
            max_spread_bps=config.get("max_spread_bps", 10.0),
            require_orderbook_confirmation=config.get("require_orderbook_confirmation", True),
            confirmation_imbalance=config.get("confirmation_imbalance", 1.2)
        )

    async def start(self) -> None:
        """Запуск стратегії та підключення до джерел даних."""
        self._session = aiohttp.ClientSession()
        logger.info(f"[{self.name}] Started")

    async def stop(self) -> None:
        """Зупинка стратегії."""
        if self._session:
            await self._session.close()
            self._session = None
        logger.info(f"[{self.name}] Stopped")

    def on_orderbook(self, snapshot: OrderBookSnapshot) -> Optional[Signal]:
        """
        Обробка оновлення ордербуку.

        Args:
            snapshot: Знімок ордербуку

        Returns:
            Сигнал, якщо умови виконані
        """
        if not self.can_signal():
            return None

        symbol = snapshot.symbol

        # Оновити історію крипто
        if symbol not in self._crypto_history:
            self._crypto_history[symbol] = deque(maxlen=1000)

        self._crypto_history[symbol].append(PricePoint(
            timestamp=snapshot.timestamp,
            price=snapshot.mid_price,
            volume=snapshot.bid_volume(5) + snapshot.ask_volume(5)
        ))

        # Очистити застарілі імпульси
        self._cleanup_expired_impulses()

        # Перевірити активні імпульси
        if not self._active_impulses:
            return None

        # Перевірити умови входу
        signal = self._evaluate_entry(snapshot)

        return signal

    def on_trade(self, trade: Trade) -> Optional[Signal]:
        """
        Обробка угоди для аналізу обсягу.

        Args:
            trade: Торгова угода
        """
        # Оновити обсяг в історії
        symbol = trade.symbol
        if symbol in self._crypto_history and self._crypto_history[symbol]:
            last = self._crypto_history[symbol][-1]
            # Додаємо обсяг до останньої точки
            self._crypto_history[symbol][-1] = PricePoint(
                timestamp=last.timestamp,
                price=last.price,
                volume=last.volume + trade.quantity
            )

        return None

    def update_leader_price(
        self,
        asset: LeaderAsset,
        price: Decimal,
        volume: Decimal = Decimal("0"),
        timestamp: Optional[datetime] = None
    ) -> Optional[ImpulseEvent]:
        """
        Оновити ціну поводиря та перевірити на імпульс.

        Args:
            asset: Актив-поводир
            price: Поточна ціна
            volume: Обсяг
            timestamp: Час (за замовчуванням - зараз)

        Returns:
            ImpulseEvent якщо виявлено імпульс
        """
        ts = timestamp or datetime.utcnow()

        # Додати до історії
        if asset in self._leader_history:
            self._leader_history[asset].append(PricePoint(
                timestamp=ts,
                price=price,
                volume=volume
            ))

        # Оновити кеш
        self._leader_prices[asset] = price

        # Перевірити на імпульс
        impulse = self._detect_impulse(asset)

        if impulse:
            self._active_impulses.append(impulse)
            self._impulses_detected += 1
            logger.info(
                f"[{self.name}] Impulse detected in {asset.value}: "
                f"{impulse.direction} {impulse.magnitude:.2f}%"
            )

        return impulse

    def _detect_impulse(self, asset: LeaderAsset) -> Optional[ImpulseEvent]:
        """
        Виявити імпульс в активі-поводирі.

        Args:
            asset: Актив для аналізу

        Returns:
            ImpulseEvent якщо виявлено
        """
        history = self._leader_history.get(asset)
        if not history or len(history) < 10:
            return None

        # Знайти конфігурацію для цього поводиря
        leader_config = None
        for cfg in self._config.leaders:
            if cfg.asset == asset:
                leader_config = cfg
                break

        if not leader_config:
            return None

        # Отримати нещодавню історію
        recent = list(history)[-50:]
        if len(recent) < 2:
            return None

        current_price = recent[-1].price

        # Шукати швидкий рух
        lookback_cutoff = datetime.utcnow() - timedelta(
            seconds=self._config.lookback_seconds
        )

        # Знайти ціну на початку періоду
        start_price = None
        start_time = None
        for point in recent:
            if point.timestamp >= lookback_cutoff:
                start_price = point.price
                start_time = point.timestamp
                break

        if start_price is None or start_price == 0:
            return None

        # Розрахувати зміну
        change_pct = float((current_price - start_price) / start_price * 100)

        # Перевірити поріг
        if abs(change_pct) < leader_config.impulse_threshold:
            return None

        # Перевірити тривалість імпульсу
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        if duration_ms < self._config.min_impulse_duration_ms:
            return None
        if duration_ms > self._config.max_impulse_duration_ms:
            return None

        # Розрахувати очікуваний вплив на крипто
        expected_impact = change_pct * leader_config.correlation

        return ImpulseEvent(
            asset=asset,
            timestamp=datetime.utcnow(),
            direction="up" if change_pct > 0 else "down",
            magnitude=abs(change_pct),
            expected_crypto_impact=expected_impact,
            expiry=datetime.utcnow() + timedelta(
                seconds=self._config.impulse_validity_seconds
            )
        )

    def _evaluate_entry(self, snapshot: OrderBookSnapshot) -> Optional[Signal]:
        """
        Оцінити умови входу на основі активних імпульсів.

        Args:
            snapshot: Знімок ордербуку

        Returns:
            Сигнал якщо умови виконані
        """
        # Перевірити спред
        spread_bps = snapshot.spread_bps or Decimal("0")
        if float(spread_bps) > self._config.max_spread_bps:
            return None

        # Агрегувати сигнали від усіх імпульсів
        total_impact = 0.0
        total_weight = 0.0
        contributing_impulses = []

        for impulse in self._active_impulses:
            # Знайти конфігурацію
            leader_config = None
            for cfg in self._config.leaders:
                if cfg.asset == impulse.asset:
                    leader_config = cfg
                    break

            if not leader_config:
                continue

            # Перевірити затримку
            delay_elapsed = (datetime.utcnow() - impulse.timestamp).total_seconds() * 1000
            if delay_elapsed < leader_config.impact_delay_ms:
                continue

            total_impact += impulse.expected_crypto_impact * leader_config.weight
            total_weight += leader_config.weight
            contributing_impulses.append(impulse)

        if total_weight == 0:
            return None

        # Середньозважений вплив
        avg_impact = total_impact / total_weight

        # Визначити напрямок
        if abs(avg_impact) < 0.1:  # Мінімальний поріг
            return None

        # Перевірити підтвердження ордербуком
        if self._config.require_orderbook_confirmation:
            imbalance = float(snapshot.imbalance(5))

            if avg_impact > 0 and imbalance < self._config.confirmation_imbalance:
                return None
            if avg_impact < 0 and imbalance > 1 / self._config.confirmation_imbalance:
                return None

        # Розрахувати силу сигналу
        strength = min(abs(avg_impact) / 1.0, 1.0)  # Нормалізація

        # Бонус за кількість підтверджень
        if len(contributing_impulses) > 1:
            strength = min(strength + 0.1 * (len(contributing_impulses) - 1), 1.0)

        signal_type = SignalType.LONG if avg_impact > 0 else SignalType.SHORT

        self._signals_from_impulses += 1

        return self.create_signal(
            signal_type=signal_type,
            symbol=snapshot.symbol,
            price=snapshot.mid_price,
            strength=strength,
            metadata={
                "impulse_impact": avg_impact,
                "contributing_leaders": [i.asset.value for i in contributing_impulses],
                "impulses_count": len(contributing_impulses),
                "orderbook_imbalance": float(snapshot.imbalance(5)),
                "spread_bps": float(spread_bps),
            }
        )

    def _cleanup_expired_impulses(self) -> None:
        """Очистити застарілі імпульси."""
        now = datetime.utcnow()
        self._active_impulses = [
            imp for imp in self._active_impulses
            if imp.expiry > now
        ]

    def get_active_impulses(self) -> List[ImpulseEvent]:
        """Отримати список активних імпульсів."""
        self._cleanup_expired_impulses()
        return self._active_impulses.copy()

    def get_leader_prices(self) -> Dict[str, float]:
        """Отримати поточні ціни поводирів."""
        return {
            asset.value: float(price)
            for asset, price in self._leader_prices.items()
        }

    @property
    def stats(self) -> Dict[str, Any]:
        """Отримати статистику стратегії."""
        base_stats = super().stats
        base_stats.update({
            "impulses_detected": self._impulses_detected,
            "signals_from_impulses": self._signals_from_impulses,
            "active_impulses": len(self._active_impulses),
            "tracked_leaders": len(self._config.leaders),
        })
        return base_stats

    def reset(self) -> None:
        """Скинути стан стратегії."""
        super().reset()
        for history in self._leader_history.values():
            history.clear()
        self._crypto_history.clear()
        self._active_impulses.clear()
        self._leader_prices.clear()
        self._impulses_detected = 0
        self._signals_from_impulses = 0


class MultiTimeframeImpulseStrategy(ImpulseScalpingStrategy):
    """
    Мультитаймфреймова імпульсна стратегія.

    Аналізує імпульси на різних таймфреймах:
    - 1m, 5m, 15m для входів
    - 1h, 4h для підтвердження тренду
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Таймфрейми для аналізу
        self._timeframes = config.get("timeframes", [60, 300, 900])  # секунди
        self._trend_timeframes = config.get("trend_timeframes", [3600, 14400])

        # Історія по таймфреймах
        self._tf_history: Dict[int, Dict[LeaderAsset, Deque[PricePoint]]] = {
            tf: {asset: deque(maxlen=100) for asset in LeaderAsset}
            for tf in self._timeframes + self._trend_timeframes
        }

    def _detect_impulse(self, asset: LeaderAsset) -> Optional[ImpulseEvent]:
        """Виявити імпульс з мультитаймфреймовим підтвердженням."""
        # Базове виявлення
        base_impulse = super()._detect_impulse(asset)

        if not base_impulse:
            return None

        # Перевірити підтвердження на старших таймфреймах
        trend_confirmed = self._check_trend_alignment(asset, base_impulse.direction)

        if trend_confirmed:
            # Збільшити magnitude якщо тренд підтверджений
            base_impulse.expected_crypto_impact *= 1.2

        return base_impulse

    def _check_trend_alignment(self, asset: LeaderAsset, direction: str) -> bool:
        """
        Перевірити вирівнювання з трендом на старших таймфреймах.

        Args:
            asset: Актив
            direction: Напрямок імпульсу ("up" або "down")

        Returns:
            True якщо тренд підтверджує напрямок
        """
        history = self._leader_history.get(asset)
        if not history or len(history) < 50:
            return False

        recent = list(history)

        # Перевірити тренд за останню годину
        hour_ago = datetime.utcnow() - timedelta(hours=1)
        hour_prices = [p.price for p in recent if p.timestamp >= hour_ago]

        if len(hour_prices) < 10:
            return False

        # Простий тренд: порівняти початок і кінець
        trend_up = hour_prices[-1] > hour_prices[0]

        return (direction == "up" and trend_up) or (direction == "down" and not trend_up)
