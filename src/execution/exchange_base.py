"""
Abstract base class for exchange APIs.

Provides a unified interface for multiple exchanges (Binance, Bybit, OKX).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any

from src.data.models import Side, OrderType, OrderStatus


# =============================================================================
# Exchange Types
# =============================================================================

class Exchange(Enum):
    """Supported exchanges."""
    BINANCE = "binance"
    BYBIT = "bybit"
    OKX = "okx"


@dataclass
class ExchangeCredentials:
    """Exchange API credentials."""
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None  # Required for OKX
    testnet: bool = True


@dataclass
class ExchangeSymbolInfo:
    """Symbol trading rules."""
    symbol: str
    base_asset: str
    quote_asset: str
    price_precision: int
    quantity_precision: int
    min_quantity: Decimal
    max_quantity: Decimal
    min_notional: Decimal
    tick_size: Decimal
    step_size: Decimal


@dataclass
class ExchangeOrder:
    """Unified order representation."""
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: Side
    order_type: OrderType
    status: OrderStatus
    quantity: Decimal
    filled_quantity: Decimal
    price: Optional[Decimal]
    avg_fill_price: Optional[Decimal]
    commission: Decimal
    created_at: int  # timestamp ms
    updated_at: int


@dataclass
class ExchangePosition:
    """Unified position representation."""
    symbol: str
    side: Side
    quantity: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    leverage: int
    margin_type: str  # cross or isolated
    liquidation_price: Optional[Decimal]


@dataclass
class ExchangeBalance:
    """Unified balance representation."""
    asset: str
    wallet_balance: Decimal
    available_balance: Decimal
    unrealized_pnl: Decimal
    margin_balance: Decimal


@dataclass
class ExchangeTicker:
    """Unified ticker representation."""
    symbol: str
    last_price: Decimal
    bid_price: Decimal
    ask_price: Decimal
    volume_24h: Decimal
    price_change_24h: Decimal
    timestamp: int


# =============================================================================
# Abstract Exchange API
# =============================================================================

class BaseExchangeAPI(ABC):
    """
    Abstract base class for exchange APIs.

    All exchange implementations must inherit from this class
    and implement all abstract methods.
    """

    def __init__(self, credentials: ExchangeCredentials):
        self.credentials = credentials
        self.testnet = credentials.testnet
        self._session = None

    @property
    @abstractmethod
    def exchange(self) -> Exchange:
        """Return the exchange type."""
        pass

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Return the API base URL."""
        pass

    @property
    @abstractmethod
    def ws_url(self) -> str:
        """Return the WebSocket URL."""
        pass

    # -------------------------------------------------------------------------
    # Connection
    # -------------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Initialize connection to the exchange."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    async def ping(self) -> bool:
        """Check connection health."""
        pass

    # -------------------------------------------------------------------------
    # Market Data
    # -------------------------------------------------------------------------

    @abstractmethod
    async def get_exchange_info(self) -> Dict[str, ExchangeSymbolInfo]:
        """Get trading rules for all symbols."""
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> ExchangeTicker:
        """Get current ticker for a symbol."""
        pass

    @abstractmethod
    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Get orderbook snapshot."""
        pass

    @abstractmethod
    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get recent trades."""
        pass

    @abstractmethod
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get candlestick data."""
        pass

    # -------------------------------------------------------------------------
    # Account
    # -------------------------------------------------------------------------

    @abstractmethod
    async def get_balance(self) -> List[ExchangeBalance]:
        """Get account balance."""
        pass

    @abstractmethod
    async def get_positions(
        self,
        symbol: Optional[str] = None,
    ) -> List[ExchangePosition]:
        """Get open positions."""
        pass

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        pass

    @abstractmethod
    async def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        """Set margin type (cross/isolated)."""
        pass

    # -------------------------------------------------------------------------
    # Orders
    # -------------------------------------------------------------------------

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: Side,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
    ) -> ExchangeOrder:
        """Place an order."""
        pass

    @abstractmethod
    async def cancel_order(
        self,
        symbol: str,
        order_id: str,
    ) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    async def cancel_all_orders(self, symbol: str) -> int:
        """Cancel all open orders for a symbol. Returns count of cancelled orders."""
        pass

    @abstractmethod
    async def get_order(
        self,
        symbol: str,
        order_id: str,
    ) -> ExchangeOrder:
        """Get order by ID."""
        pass

    @abstractmethod
    async def get_open_orders(
        self,
        symbol: Optional[str] = None,
    ) -> List[ExchangeOrder]:
        """Get all open orders."""
        pass

    @abstractmethod
    async def get_order_history(
        self,
        symbol: str,
        limit: int = 50,
    ) -> List[ExchangeOrder]:
        """Get order history."""
        pass

    # -------------------------------------------------------------------------
    # Funding
    # -------------------------------------------------------------------------

    @abstractmethod
    async def get_funding_rate(self, symbol: str) -> Decimal:
        """Get current funding rate."""
        pass

    @abstractmethod
    async def get_funding_history(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get funding rate history."""
        pass

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    async def place_market_order(
        self,
        symbol: str,
        side: Side,
        quantity: Decimal,
        reduce_only: bool = False,
    ) -> ExchangeOrder:
        """Convenience method for market orders."""
        return await self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            reduce_only=reduce_only,
        )

    async def place_limit_order(
        self,
        symbol: str,
        side: Side,
        quantity: Decimal,
        price: Decimal,
        reduce_only: bool = False,
    ) -> ExchangeOrder:
        """Convenience method for limit orders."""
        return await self.place_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            reduce_only=reduce_only,
        )

    async def close_position(
        self,
        symbol: str,
        quantity: Optional[Decimal] = None,
    ) -> Optional[ExchangeOrder]:
        """Close an open position."""
        positions = await self.get_positions(symbol)
        if not positions:
            return None

        position = positions[0]
        close_side = Side.SELL if position.side == Side.BUY else Side.BUY
        close_qty = quantity or position.quantity

        return await self.place_market_order(
            symbol=symbol,
            side=close_side,
            quantity=close_qty,
            reduce_only=True,
        )

    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to exchange format.
        Override in subclasses if needed.
        """
        return symbol.upper().replace("-", "").replace("/", "")
