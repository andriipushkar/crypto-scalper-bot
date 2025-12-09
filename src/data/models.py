"""
Data models for the trading bot.

All market data structures and trading-related dataclasses.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict


class Side(Enum):
    """Order/position side."""
    BUY = "BUY"
    SELL = "SELL"

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self == Side.BUY else Side.BUY


class OrderType(Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


class OrderStatus(Enum):
    """Order status."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"


class TimeInForce(Enum):
    """Time in force for orders."""
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTX = "GTX"  # Post Only


class SignalType(Enum):
    """Trading signal type."""
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    NO_ACTION = "NO_ACTION"


# =============================================================================
# Market Data Models
# =============================================================================

@dataclass(frozen=True, slots=True)
class Trade:
    """
    Aggregated trade from exchange.

    Represents a single aggTrade event from Binance.
    """
    symbol: str
    trade_id: int
    price: Decimal
    quantity: Decimal
    timestamp: datetime
    is_buyer_maker: bool

    @property
    def side(self) -> Side:
        """Infer trade side: buyer maker = sell, seller maker = buy."""
        return Side.SELL if self.is_buyer_maker else Side.BUY

    @property
    def value(self) -> Decimal:
        """Trade value in quote currency (USDT)."""
        return self.price * self.quantity

    @classmethod
    def from_binance(cls, data: dict) -> "Trade":
        """Create Trade from Binance aggTrade message."""
        return cls(
            symbol=data["s"],
            trade_id=data["a"],
            price=Decimal(data["p"]),
            quantity=Decimal(data["q"]),
            timestamp=datetime.fromtimestamp(data["T"] / 1000),
            is_buyer_maker=data["m"],
        )


@dataclass(frozen=True, slots=True)
class OrderBookLevel:
    """Single price level in the order book."""
    price: Decimal
    quantity: Decimal

    @property
    def value(self) -> Decimal:
        """Value at this level in quote currency."""
        return self.price * self.quantity

    @classmethod
    def from_binance(cls, data: list) -> "OrderBookLevel":
        """Create from Binance format [price, quantity]."""
        return cls(
            price=Decimal(data[0]),
            quantity=Decimal(data[1]),
        )


@dataclass(slots=True)
class OrderBookSnapshot:
    """
    Order book snapshot at a point in time.

    Contains best bids and asks up to a certain depth.
    """
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]  # Sorted by price descending (best first)
    asks: List[OrderBookLevel]  # Sorted by price ascending (best first)
    last_update_id: int

    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        """Best (highest) bid."""
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        """Best (lowest) ask."""
        return self.asks[0] if self.asks else None

    @property
    def mid_price(self) -> Optional[Decimal]:
        """Mid price between best bid and ask."""
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return None

    @property
    def spread(self) -> Optional[Decimal]:
        """Absolute spread between best bid and ask."""
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None

    @property
    def spread_bps(self) -> Optional[Decimal]:
        """Spread in basis points."""
        if self.spread and self.mid_price:
            return (self.spread / self.mid_price) * 10000
        return None

    def bid_volume(self, levels: int = 5) -> Decimal:
        """Total bid volume for top N levels."""
        return sum(level.quantity for level in self.bids[:levels])

    def ask_volume(self, levels: int = 5) -> Decimal:
        """Total ask volume for top N levels."""
        return sum(level.quantity for level in self.asks[:levels])

    def imbalance(self, levels: int = 5) -> Decimal:
        """
        Order book imbalance ratio.

        > 1.0 = more bids (buying pressure)
        < 1.0 = more asks (selling pressure)
        """
        bid_vol = self.bid_volume(levels)
        ask_vol = self.ask_volume(levels)

        if ask_vol == 0:
            return Decimal("999.0")
        return bid_vol / ask_vol


@dataclass(slots=True)
class DepthUpdate:
    """
    Incremental order book update from WebSocket.

    Used to maintain local order book state.
    """
    symbol: str
    timestamp: datetime
    first_update_id: int
    final_update_id: int
    bids: List[OrderBookLevel]  # Updated bid levels
    asks: List[OrderBookLevel]  # Updated ask levels

    @classmethod
    def from_binance(cls, data: dict) -> "DepthUpdate":
        """Create from Binance depthUpdate message."""
        return cls(
            symbol=data["s"],
            timestamp=datetime.fromtimestamp(data["E"] / 1000),
            first_update_id=data["U"],
            final_update_id=data["u"],
            bids=[OrderBookLevel.from_binance(b) for b in data["b"]],
            asks=[OrderBookLevel.from_binance(a) for a in data["a"]],
        )


@dataclass(frozen=True, slots=True)
class MarkPrice:
    """
    Mark price and funding rate info.

    From Binance markPrice stream.
    """
    symbol: str
    mark_price: Decimal
    index_price: Decimal
    estimated_settle_price: Decimal
    funding_rate: Decimal
    next_funding_time: datetime
    timestamp: datetime

    @classmethod
    def from_binance(cls, data: dict) -> "MarkPrice":
        """Create from Binance markPrice message."""
        return cls(
            symbol=data["s"],
            mark_price=Decimal(data["p"]),
            index_price=Decimal(data["i"]),
            estimated_settle_price=Decimal(data.get("P", data["p"])),
            funding_rate=Decimal(data["r"]),
            next_funding_time=datetime.fromtimestamp(data["T"] / 1000),
            timestamp=datetime.fromtimestamp(data["E"] / 1000),
        )


# =============================================================================
# Trading Models
# =============================================================================

@dataclass(slots=True)
class Signal:
    """
    Trading signal generated by a strategy.
    """
    strategy: str
    signal_type: SignalType
    symbol: str
    timestamp: datetime
    strength: float  # 0.0 to 1.0
    price: Decimal
    metadata: Dict = field(default_factory=dict)

    @property
    def is_entry(self) -> bool:
        """Is this an entry signal?"""
        return self.signal_type in (SignalType.LONG, SignalType.SHORT)

    @property
    def is_exit(self) -> bool:
        """Is this an exit signal?"""
        return self.signal_type in (SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT)

    @property
    def side(self) -> Optional[Side]:
        """Get order side for this signal."""
        if self.signal_type == SignalType.LONG:
            return Side.BUY
        elif self.signal_type == SignalType.SHORT:
            return Side.SELL
        elif self.signal_type == SignalType.CLOSE_LONG:
            return Side.SELL
        elif self.signal_type == SignalType.CLOSE_SHORT:
            return Side.BUY
        return None


@dataclass(slots=True)
class Order:
    """
    Order representation.
    """
    order_id: str
    client_order_id: str
    symbol: str
    side: Side
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal]
    stop_price: Optional[Decimal]
    status: OrderStatus
    filled_quantity: Decimal
    average_price: Optional[Decimal]
    commission: Decimal
    created_at: datetime
    updated_at: datetime
    time_in_force: TimeInForce = TimeInForce.GTC

    @property
    def is_open(self) -> bool:
        """Is order still open?"""
        return self.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)

    @property
    def is_filled(self) -> bool:
        """Is order fully filled?"""
        return self.status == OrderStatus.FILLED

    @property
    def remaining_quantity(self) -> Decimal:
        """Remaining quantity to fill."""
        return self.quantity - self.filled_quantity


@dataclass(slots=True)
class Position:
    """
    Current position in a symbol.
    """
    symbol: str
    side: Side
    size: Decimal  # Positive for both long and short
    entry_price: Decimal
    mark_price: Decimal
    liquidation_price: Optional[Decimal]
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    leverage: int
    margin_type: str  # CROSSED or ISOLATED
    updated_at: datetime

    @property
    def notional_value(self) -> Decimal:
        """Position notional value."""
        return self.size * self.mark_price

    @property
    def is_long(self) -> bool:
        """Is this a long position?"""
        return self.side == Side.BUY

    @property
    def is_short(self) -> bool:
        """Is this a short position?"""
        return self.side == Side.SELL

    @property
    def pnl_percent(self) -> Decimal:
        """Unrealized P&L as percentage of entry."""
        if self.entry_price == 0:
            return Decimal("0")
        return (self.unrealized_pnl / (self.size * self.entry_price)) * 100


# =============================================================================
# Event Types for Event Bus
# =============================================================================

@dataclass(frozen=True, slots=True)
class TradeEvent:
    """Event wrapper for trade data."""
    trade: Trade


@dataclass(frozen=True, slots=True)
class DepthUpdateEvent:
    """Event wrapper for depth update."""
    update: DepthUpdate


@dataclass(frozen=True, slots=True)
class OrderBookEvent:
    """Event wrapper for order book snapshot."""
    snapshot: OrderBookSnapshot


@dataclass(frozen=True, slots=True)
class MarkPriceEvent:
    """Event wrapper for mark price update."""
    mark_price: MarkPrice


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """Event wrapper for trading signal."""
    signal: Signal


@dataclass(frozen=True, slots=True)
class OrderEvent:
    """Event wrapper for order updates."""
    order: Order
    event_type: str  # "new", "filled", "canceled", etc.


@dataclass(frozen=True, slots=True)
class PositionEvent:
    """Event wrapper for position updates."""
    position: Position
    event_type: str  # "open", "close", "update"
