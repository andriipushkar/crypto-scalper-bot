"""
Maker/Taker Fee Optimizer (Оптимізатор комісій).

Реалізує концепцію ECN rebates з книги "Скальпинг: практическое руководство трейдера",
адаптовану для крипторинку.

Концепція:
"В класичному скальпінгу важливо розуміти структуру комісій:
- Add liquidity (maker) = отримуємо rebate
- Remove liquidity (taker) = платимо комісію

В крипто:
- Maker ордери (limit) = нижча комісія
- Taker ордери (market) = вища комісія

Для скальпінгу з малим профітом оптимізація комісій критична!"

Функціонал:
1. Аналіз структури комісій біржі
2. Вибір оптимального типу ордера
3. Розрахунок break-even з урахуванням комісій
4. Smart order routing
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

from loguru import logger


class OrderExecutionType(Enum):
    """Тип виконання ордера."""
    MAKER = "maker"      # Limit order (add liquidity)
    TAKER = "taker"      # Market order (remove liquidity)
    POST_ONLY = "post_only"  # Гарантовано maker


class FeeLevel(Enum):
    """Рівень комісії на біржі."""
    VIP_0 = "vip_0"  # Базовий рівень
    VIP_1 = "vip_1"
    VIP_2 = "vip_2"
    VIP_3 = "vip_3"
    VIP_4 = "vip_4"
    VIP_5 = "vip_5"


@dataclass
class ExchangeFeeStructure:
    """Структура комісій біржі."""
    exchange: str
    fee_level: FeeLevel
    maker_fee_bps: float  # Базисні пункти (0.01%)
    taker_fee_bps: float
    has_rebate: bool = False  # Чи є негативна комісія для maker
    maker_rebate_bps: float = 0.0
    funding_rate_interval_hours: int = 8
    estimated_funding_rate_bps: float = 1.0  # Приблизна ставка фандингу

    @property
    def maker_fee(self) -> Decimal:
        """Комісія maker у відсотках."""
        if self.has_rebate:
            return Decimal(str(-self.maker_rebate_bps / 10000))
        return Decimal(str(self.maker_fee_bps / 10000))

    @property
    def taker_fee(self) -> Decimal:
        """Комісія taker у відсотках."""
        return Decimal(str(self.taker_fee_bps / 10000))

    @property
    def fee_difference_bps(self) -> float:
        """Різниця між taker та maker в bps."""
        return self.taker_fee_bps - self.maker_fee_bps


# Типові структури комісій для бірж
DEFAULT_FEE_STRUCTURES = {
    "binance": ExchangeFeeStructure(
        exchange="binance",
        fee_level=FeeLevel.VIP_0,
        maker_fee_bps=1.0,    # 0.01%
        taker_fee_bps=5.0,    # 0.05%
    ),
    "bybit": ExchangeFeeStructure(
        exchange="bybit",
        fee_level=FeeLevel.VIP_0,
        maker_fee_bps=1.0,
        taker_fee_bps=6.0,
    ),
    "okx": ExchangeFeeStructure(
        exchange="okx",
        fee_level=FeeLevel.VIP_0,
        maker_fee_bps=2.0,
        taker_fee_bps=5.0,
    ),
    "kraken": ExchangeFeeStructure(
        exchange="kraken",
        fee_level=FeeLevel.VIP_0,
        maker_fee_bps=2.0,
        taker_fee_bps=5.0,
    ),
    "kucoin": ExchangeFeeStructure(
        exchange="kucoin",
        fee_level=FeeLevel.VIP_0,
        maker_fee_bps=2.0,
        taker_fee_bps=6.0,
    ),
    "gateio": ExchangeFeeStructure(
        exchange="gateio",
        fee_level=FeeLevel.VIP_0,
        maker_fee_bps=2.0,
        taker_fee_bps=5.0,
    ),
}


@dataclass
class OrderExecutionPlan:
    """План виконання ордера."""
    execution_type: OrderExecutionType
    order_type: str  # "limit", "market", "post_only"
    price: Optional[Decimal]  # Ціна для limit ордерів
    estimated_fee_bps: float
    estimated_slippage_bps: float = 0.0
    total_cost_bps: float = 0.0
    time_priority: str = "normal"  # "urgent", "normal", "patient"
    notes: str = ""

    def __post_init__(self):
        self.total_cost_bps = self.estimated_fee_bps + self.estimated_slippage_bps


@dataclass
class TradeAnalysis:
    """Аналіз трейду з урахуванням комісій."""
    entry_price: Decimal
    target_price: Decimal
    stop_price: Decimal
    position_size: Decimal
    side: str  # "long" або "short"

    # Комісії
    entry_fee_bps: float
    exit_fee_bps: float
    total_fees_bps: float

    # Break-even
    break_even_price: Decimal
    break_even_move_bps: float

    # P&L
    gross_profit_if_target: Decimal
    net_profit_if_target: Decimal
    gross_loss_if_stop: Decimal
    net_loss_if_stop: Decimal

    # Risk/Reward
    risk_reward_gross: float
    risk_reward_net: float

    # Рекомендації
    is_profitable_after_fees: bool
    recommendation: str


@dataclass
class FeeOptimizerConfig:
    """Конфігурація оптимізатора комісій."""
    # Пріоритети
    prefer_maker: bool = True           # Надавати перевагу maker ордерам
    max_wait_for_fill_ms: int = 5000    # Макс. час очікування fill для maker

    # Пороги
    urgency_spread_bps: float = 5.0     # Спред > 5bps = urgent (market)
    min_profit_after_fees_bps: float = 5.0  # Мін. профіт після комісій

    # Post-only
    use_post_only: bool = True          # Використовувати post-only коли можливо
    post_only_price_offset_bps: float = 1.0  # Зсув ціни для post-only

    # Slippage estimation
    expected_slippage_bps: float = 2.0  # Очікуваний slippage для market


class FeeOptimizer:
    """
    Оптимізатор комісій для скальпінгу.

    Допомагає мінімізувати витрати на комісії через:
    1. Вибір оптимального типу ордера (maker vs taker)
    2. Розрахунок реального break-even
    3. Аналіз профітабельності з урахуванням комісій
    """

    def __init__(
        self,
        exchange: str,
        fee_structure: Optional[ExchangeFeeStructure] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Ініціалізація оптимізатора.

        Args:
            exchange: Назва біржі
            fee_structure: Кастомна структура комісій
            config: Конфігурація
        """
        self._exchange = exchange.lower()

        # Використати кастомну або дефолтну структуру комісій
        if fee_structure:
            self._fees = fee_structure
        else:
            self._fees = DEFAULT_FEE_STRUCTURES.get(
                self._exchange,
                DEFAULT_FEE_STRUCTURES["binance"]
            )

        self._config = self._parse_config(config or {})

        # Статистика
        self._total_orders = 0
        self._maker_orders = 0
        self._taker_orders = 0
        self._total_fees_saved_bps = 0.0

        logger.info(
            f"[FeeOptimizer] Initialized for {exchange}: "
            f"maker={self._fees.maker_fee_bps}bps, taker={self._fees.taker_fee_bps}bps"
        )

    def _parse_config(self, config: Dict[str, Any]) -> FeeOptimizerConfig:
        """Парсинг конфігурації."""
        return FeeOptimizerConfig(
            prefer_maker=config.get("prefer_maker", True),
            max_wait_for_fill_ms=config.get("max_wait_for_fill_ms", 5000),
            urgency_spread_bps=config.get("urgency_spread_bps", 5.0),
            min_profit_after_fees_bps=config.get("min_profit_after_fees_bps", 5.0),
            use_post_only=config.get("use_post_only", True),
            post_only_price_offset_bps=config.get("post_only_price_offset_bps", 1.0),
            expected_slippage_bps=config.get("expected_slippage_bps", 2.0)
        )

    def get_execution_plan(
        self,
        side: str,
        best_bid: Decimal,
        best_ask: Decimal,
        urgency: str = "normal"
    ) -> OrderExecutionPlan:
        """
        Отримати план виконання ордера.

        Args:
            side: "buy" або "sell"
            best_bid: Найкраща ціна покупки
            best_ask: Найкраща ціна продажу
            urgency: "urgent", "normal", "patient"

        Returns:
            План виконання з оптимальним типом ордера
        """
        spread_bps = float((best_ask - best_bid) / best_bid * 10000)

        # Якщо терміново або великий спред - market order
        if urgency == "urgent" or spread_bps > self._config.urgency_spread_bps:
            self._total_orders += 1
            self._taker_orders += 1

            return OrderExecutionPlan(
                execution_type=OrderExecutionType.TAKER,
                order_type="market",
                price=None,
                estimated_fee_bps=self._fees.taker_fee_bps,
                estimated_slippage_bps=self._config.expected_slippage_bps,
                time_priority="urgent",
                notes="Market order due to urgency or wide spread"
            )

        # Визначити ціну для limit ордера
        if side == "buy":
            if self._config.use_post_only:
                # Post-only: ставимо трохи нижче best bid
                offset = best_bid * Decimal(str(self._config.post_only_price_offset_bps / 10000))
                limit_price = best_bid - offset
            else:
                limit_price = best_bid
        else:  # sell
            if self._config.use_post_only:
                offset = best_ask * Decimal(str(self._config.post_only_price_offset_bps / 10000))
                limit_price = best_ask + offset
            else:
                limit_price = best_ask

        # Вибір між post_only та звичайним limit
        order_type = "post_only" if self._config.use_post_only else "limit"
        execution_type = OrderExecutionType.POST_ONLY if self._config.use_post_only else OrderExecutionType.MAKER

        self._total_orders += 1
        self._maker_orders += 1

        # Розрахувати скільки зекономлено
        saved = self._fees.taker_fee_bps - self._fees.maker_fee_bps
        self._total_fees_saved_bps += saved

        return OrderExecutionPlan(
            execution_type=execution_type,
            order_type=order_type,
            price=limit_price,
            estimated_fee_bps=self._fees.maker_fee_bps,
            estimated_slippage_bps=0.0,  # Limit = no slippage
            time_priority=urgency,
            notes=f"Maker order to save {saved:.1f}bps in fees"
        )

    def analyze_trade(
        self,
        entry_price: Decimal,
        target_price: Decimal,
        stop_price: Decimal,
        position_size: Decimal,
        side: str,
        entry_execution: OrderExecutionType = OrderExecutionType.MAKER,
        exit_execution: OrderExecutionType = OrderExecutionType.MAKER
    ) -> TradeAnalysis:
        """
        Аналізувати трейд з урахуванням комісій.

        Args:
            entry_price: Ціна входу
            target_price: Ціна тейк-профіту
            stop_price: Ціна стоп-лоссу
            position_size: Розмір позиції (в базовій валюті)
            side: "long" або "short"
            entry_execution: Тип виконання входу
            exit_execution: Тип виконання виходу

        Returns:
            Повний аналіз трейду
        """
        # Визначити комісії
        entry_fee_bps = (
            self._fees.maker_fee_bps if entry_execution != OrderExecutionType.TAKER
            else self._fees.taker_fee_bps
        )
        exit_fee_bps = (
            self._fees.maker_fee_bps if exit_execution != OrderExecutionType.TAKER
            else self._fees.taker_fee_bps
        )
        total_fees_bps = entry_fee_bps + exit_fee_bps

        # Розрахувати break-even
        total_fee_pct = Decimal(str(total_fees_bps / 10000))

        if side == "long":
            # Для long: break_even = entry * (1 + fees)
            break_even_price = entry_price * (1 + total_fee_pct)
            gross_profit_if_target = (target_price - entry_price) * position_size
            gross_loss_if_stop = (entry_price - stop_price) * position_size
        else:
            # Для short: break_even = entry * (1 - fees)
            break_even_price = entry_price * (1 - total_fee_pct)
            gross_profit_if_target = (entry_price - target_price) * position_size
            gross_loss_if_stop = (stop_price - entry_price) * position_size

        break_even_move_bps = float(abs(break_even_price - entry_price) / entry_price * 10000)

        # Розрахувати P&L
        position_value = entry_price * position_size
        total_fees = position_value * total_fee_pct

        net_profit_if_target = gross_profit_if_target - total_fees
        net_loss_if_stop = gross_loss_if_stop + total_fees

        # Risk/Reward
        risk_reward_gross = float(gross_profit_if_target / gross_loss_if_stop) if gross_loss_if_stop > 0 else 0
        risk_reward_net = float(net_profit_if_target / net_loss_if_stop) if net_loss_if_stop > 0 else 0

        # Рекомендації
        is_profitable = net_profit_if_target > 0
        target_move_bps = float(abs(target_price - entry_price) / entry_price * 10000)

        if not is_profitable:
            recommendation = "NOT RECOMMENDED: Target doesn't cover fees"
        elif target_move_bps < self._config.min_profit_after_fees_bps:
            recommendation = f"CAUTION: Target move ({target_move_bps:.1f}bps) is very small"
        elif risk_reward_net < 1.0:
            recommendation = f"CAUTION: R:R after fees ({risk_reward_net:.2f}) is below 1:1"
        else:
            recommendation = f"OK: Net R:R = {risk_reward_net:.2f}, fees = {total_fees_bps:.1f}bps"

        return TradeAnalysis(
            entry_price=entry_price,
            target_price=target_price,
            stop_price=stop_price,
            position_size=position_size,
            side=side,
            entry_fee_bps=entry_fee_bps,
            exit_fee_bps=exit_fee_bps,
            total_fees_bps=total_fees_bps,
            break_even_price=break_even_price,
            break_even_move_bps=break_even_move_bps,
            gross_profit_if_target=gross_profit_if_target,
            net_profit_if_target=net_profit_if_target,
            gross_loss_if_stop=gross_loss_if_stop,
            net_loss_if_stop=net_loss_if_stop,
            risk_reward_gross=risk_reward_gross,
            risk_reward_net=risk_reward_net,
            is_profitable_after_fees=is_profitable,
            recommendation=recommendation
        )

    def calculate_min_target_for_profit(
        self,
        entry_price: Decimal,
        side: str,
        min_profit_bps: Optional[float] = None
    ) -> Decimal:
        """
        Розрахувати мінімальну ціль для профітабельного трейду.

        Args:
            entry_price: Ціна входу
            side: "long" або "short"
            min_profit_bps: Мін. бажаний профіт в bps

        Returns:
            Мінімальна ціна цілі
        """
        if min_profit_bps is None:
            min_profit_bps = self._config.min_profit_after_fees_bps

        # Загальні комісії (entry + exit)
        # Припускаємо maker для обох
        total_fees_bps = self._fees.maker_fee_bps * 2
        required_move_bps = total_fees_bps + min_profit_bps

        move_pct = Decimal(str(required_move_bps / 10000))

        if side == "long":
            return entry_price * (1 + move_pct)
        else:
            return entry_price * (1 - move_pct)

    def get_fee_comparison(self) -> Dict[str, Any]:
        """Порівняння maker vs taker комісій."""
        maker_cost_per_10k = self._fees.maker_fee_bps
        taker_cost_per_10k = self._fees.taker_fee_bps
        saving_per_trade = taker_cost_per_10k - maker_cost_per_10k

        return {
            "exchange": self._exchange,
            "fee_level": self._fees.fee_level.value,
            "maker_fee_bps": maker_cost_per_10k,
            "taker_fee_bps": taker_cost_per_10k,
            "saving_per_trade_bps": saving_per_trade,
            "saving_per_round_trip_bps": saving_per_trade * 2,
            "annual_savings_example": {
                "trades_per_day": 100,
                "volume_per_trade_usd": 10000,
                "daily_savings_usd": saving_per_trade * 2 / 10000 * 10000 * 100,
                "annual_savings_usd": saving_per_trade * 2 / 10000 * 10000 * 100 * 365,
            }
        }

    def update_fee_structure(self, fee_structure: ExchangeFeeStructure) -> None:
        """Оновити структуру комісій."""
        self._fees = fee_structure
        logger.info(
            f"[FeeOptimizer] Updated fees: maker={fee_structure.maker_fee_bps}bps, "
            f"taker={fee_structure.taker_fee_bps}bps"
        )

    @property
    def stats(self) -> Dict[str, Any]:
        """Статистика оптимізатора."""
        maker_rate = (
            self._maker_orders / self._total_orders * 100
            if self._total_orders > 0 else 0
        )
        return {
            "exchange": self._exchange,
            "total_orders": self._total_orders,
            "maker_orders": self._maker_orders,
            "taker_orders": self._taker_orders,
            "maker_rate_percent": maker_rate,
            "total_fees_saved_bps": self._total_fees_saved_bps,
            "current_fees": {
                "maker_bps": self._fees.maker_fee_bps,
                "taker_bps": self._fees.taker_fee_bps,
            }
        }

    def reset_stats(self) -> None:
        """Скинути статистику."""
        self._total_orders = 0
        self._maker_orders = 0
        self._taker_orders = 0
        self._total_fees_saved_bps = 0.0
