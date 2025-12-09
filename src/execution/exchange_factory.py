"""
Exchange factory for creating exchange API instances.

Provides a unified interface for creating and managing multiple exchanges.
"""

import os
from typing import Dict, Optional, Type

from loguru import logger

from src.execution.exchange_base import (
    BaseExchangeAPI,
    Exchange,
    ExchangeCredentials,
)
from src.execution.binance_api import BinanceFuturesAPI
from src.execution.bybit_api import BybitFuturesAPI
from src.execution.okx_api import OKXFuturesAPI
from src.execution.paper_trading import PaperTradingEngine, PaperTradingAPI


# Exchange class registry
EXCHANGE_CLASSES: Dict[Exchange, Type[BaseExchangeAPI]] = {
    Exchange.BINANCE: BinanceFuturesAPI,
    Exchange.BYBIT: BybitFuturesAPI,
    Exchange.OKX: OKXFuturesAPI,
}


class ExchangeFactory:
    """
    Factory for creating exchange API instances.

    Usage:
        # Create from environment variables
        factory = ExchangeFactory()
        binance = factory.create(Exchange.BINANCE)
        bybit = factory.create(Exchange.BYBIT)

        # Create with explicit credentials
        creds = ExchangeCredentials(api_key="...", api_secret="...", testnet=True)
        api = factory.create(Exchange.BINANCE, credentials=creds)

        # Create paper trading instance
        paper = factory.create_paper_trading(initial_balance=100.0)
    """

    def __init__(self):
        self._instances: Dict[str, BaseExchangeAPI] = {}

    def create(
        self,
        exchange: Exchange,
        credentials: Optional[ExchangeCredentials] = None,
        testnet: bool = True,
    ) -> BaseExchangeAPI:
        """
        Create an exchange API instance.

        Args:
            exchange: Exchange type (BINANCE, BYBIT, OKX)
            credentials: Optional credentials (loads from env if not provided)
            testnet: Use testnet (default True)

        Returns:
            Exchange API instance
        """
        if credentials is None:
            credentials = self._load_credentials(exchange, testnet)

        if exchange not in EXCHANGE_CLASSES:
            raise ValueError(f"Unsupported exchange: {exchange}")

        api_class = EXCHANGE_CLASSES[exchange]
        instance = api_class(credentials)

        # Cache instance
        cache_key = f"{exchange.value}_{testnet}"
        self._instances[cache_key] = instance

        logger.info(f"Created {exchange.value} API ({'testnet' if testnet else 'mainnet'})")
        return instance

    def _load_credentials(
        self,
        exchange: Exchange,
        testnet: bool,
    ) -> ExchangeCredentials:
        """Load credentials from environment variables."""
        prefix = exchange.value.upper()

        if testnet:
            api_key = os.getenv(f"{prefix}_TESTNET_API_KEY", "")
            api_secret = os.getenv(f"{prefix}_TESTNET_API_SECRET", "")
        else:
            api_key = os.getenv(f"{prefix}_API_KEY", "")
            api_secret = os.getenv(f"{prefix}_API_SECRET", "")

        # OKX requires passphrase
        passphrase = None
        if exchange == Exchange.OKX:
            passphrase = os.getenv(f"{prefix}_PASSPHRASE", "")

        return ExchangeCredentials(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            testnet=testnet,
        )

    def create_paper_trading(
        self,
        initial_balance: float = 100.0,
        leverage: int = 10,
        commission_rate: float = 0.0004,
    ) -> PaperTradingAPI:
        """
        Create a paper trading instance.

        Args:
            initial_balance: Starting balance in USDT
            leverage: Default leverage
            commission_rate: Commission rate per trade

        Returns:
            Paper trading API (compatible with exchange APIs)
        """
        engine = PaperTradingEngine(
            initial_balance=initial_balance,
            leverage=leverage,
            commission_rate=commission_rate,
        )
        return PaperTradingAPI(engine)

    def get(self, exchange: Exchange, testnet: bool = True) -> Optional[BaseExchangeAPI]:
        """Get cached exchange instance."""
        cache_key = f"{exchange.value}_{testnet}"
        return self._instances.get(cache_key)

    async def close_all(self) -> None:
        """Close all exchange connections."""
        for name, instance in self._instances.items():
            try:
                await instance.close()
                logger.info(f"Closed {name} connection")
            except Exception as e:
                logger.error(f"Error closing {name}: {e}")
        self._instances.clear()


# Global factory instance
_factory: Optional[ExchangeFactory] = None


def get_exchange_factory() -> ExchangeFactory:
    """Get global exchange factory instance."""
    global _factory
    if _factory is None:
        _factory = ExchangeFactory()
    return _factory


def create_exchange(
    exchange: str,
    testnet: bool = True,
    credentials: Optional[ExchangeCredentials] = None,
) -> BaseExchangeAPI:
    """
    Convenience function to create an exchange API.

    Args:
        exchange: Exchange name ("binance", "bybit", "okx", "paper")
        testnet: Use testnet
        credentials: Optional credentials

    Returns:
        Exchange API instance
    """
    factory = get_exchange_factory()

    if exchange.lower() == "paper":
        return factory.create_paper_trading()

    try:
        exchange_type = Exchange(exchange.lower())
    except ValueError:
        raise ValueError(
            f"Unknown exchange: {exchange}. "
            f"Supported: binance, bybit, okx, paper"
        )

    return factory.create(exchange_type, credentials, testnet)
