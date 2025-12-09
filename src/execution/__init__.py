"""Order execution and management."""

from src.execution.binance_api import (
    BinanceFuturesAPI,
    BinanceAPIError,
    OrderExecutor,
)
from src.execution.paper_trading import (
    PaperTradingEngine,
    PaperTradingAPI,
    PaperOrder,
    PaperPosition,
    PaperBalance,
)
from src.execution.exchange_base import (
    BaseExchangeAPI,
    Exchange,
    ExchangeCredentials,
    ExchangeSymbolInfo,
    ExchangeOrder,
    ExchangePosition,
    ExchangeBalance,
    ExchangeTicker,
)
from src.execution.bybit_api import BybitFuturesAPI, BybitAPIError
from src.execution.okx_api import OKXFuturesAPI, OKXAPIError
from src.execution.kraken_api import KrakenFuturesAPI, KrakenAPIError
from src.execution.kucoin_api import KuCoinFuturesAPI, KuCoinAPIError
from src.execution.gateio_api import GateIOFuturesAPI, GateIOAPIError
from src.execution.exchange_factory import (
    ExchangeFactory,
    get_exchange_factory,
    create_exchange,
)
from src.execution.fee_optimizer import (
    FeeOptimizer,
    ExchangeFeeStructure,
    OrderExecutionType,
    OrderExecutionPlan,
    TradeAnalysis,
    FeeLevel,
)

__all__ = [
    # Exchange Base
    "BaseExchangeAPI",
    "Exchange",
    "ExchangeCredentials",
    "ExchangeSymbolInfo",
    "ExchangeOrder",
    "ExchangePosition",
    "ExchangeBalance",
    "ExchangeTicker",
    # Binance
    "BinanceFuturesAPI",
    "BinanceAPIError",
    "OrderExecutor",
    # Bybit
    "BybitFuturesAPI",
    "BybitAPIError",
    # OKX
    "OKXFuturesAPI",
    "OKXAPIError",
    # Kraken
    "KrakenFuturesAPI",
    "KrakenAPIError",
    # KuCoin
    "KuCoinFuturesAPI",
    "KuCoinAPIError",
    # Gate.io
    "GateIOFuturesAPI",
    "GateIOAPIError",
    # Paper Trading
    "PaperTradingEngine",
    "PaperTradingAPI",
    "PaperOrder",
    "PaperPosition",
    "PaperBalance",
    # Factory
    "ExchangeFactory",
    "get_exchange_factory",
    "create_exchange",
    # Fee Optimizer
    "FeeOptimizer",
    "ExchangeFeeStructure",
    "OrderExecutionType",
    "OrderExecutionPlan",
    "TradeAnalysis",
    "FeeLevel",
]
