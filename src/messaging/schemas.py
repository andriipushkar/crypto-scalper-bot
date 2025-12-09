"""
Event Schemas for Message Queue

Defines structured event types for Kafka messages.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime
from decimal import Decimal
from enum import Enum


@dataclass
class BaseEvent:
    """Base event schema."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "trading-bot"
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


# =============================================================================
# Market Data Events
# =============================================================================

@dataclass
class TickerData:
    """Ticker data structure."""
    symbol: str
    last_price: Decimal
    bid_price: Decimal
    ask_price: Decimal
    volume_24h: Decimal
    change_24h: Decimal
    high_24h: Decimal
    low_24h: Decimal


@dataclass
class MarketDataEvent(BaseEvent):
    """Market data event schema."""
    symbol: str = ""
    exchange: str = ""
    data_type: str = ""  # ticker, orderbook, trade, kline
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderBookData:
    """Order book snapshot."""
    symbol: str
    bids: List[List[str]]  # [[price, quantity], ...]
    asks: List[List[str]]
    last_update_id: int
    timestamp: datetime


@dataclass
class TradeData:
    """Trade data structure."""
    trade_id: str
    symbol: str
    price: Decimal
    quantity: Decimal
    side: str  # BUY or SELL
    timestamp: datetime


# =============================================================================
# Trading Events
# =============================================================================

@dataclass
class SignalEvent(BaseEvent):
    """Trading signal event schema."""
    symbol: str = ""
    strategy: str = ""
    signal_type: str = ""  # LONG, SHORT, NEUTRAL
    strength: float = 0.0
    entry_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderEvent(BaseEvent):
    """Order event schema."""
    order_id: str = ""
    client_order_id: str = ""
    symbol: str = ""
    side: str = ""  # BUY, SELL
    order_type: str = ""  # MARKET, LIMIT, STOP
    status: str = ""  # NEW, FILLED, CANCELLED, REJECTED
    quantity: Decimal = Decimal("0")
    price: Optional[Decimal] = None
    filled_quantity: Decimal = Decimal("0")
    average_price: Optional[Decimal] = None
    commission: Decimal = Decimal("0")
    exchange: str = ""
    strategy: str = ""
    error_message: Optional[str] = None


@dataclass
class PositionEvent(BaseEvent):
    """Position event schema."""
    symbol: str = ""
    side: str = ""  # LONG, SHORT
    action: str = ""  # OPENED, UPDATED, CLOSED
    quantity: Decimal = Decimal("0")
    entry_price: Decimal = Decimal("0")
    current_price: Optional[Decimal] = None
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    leverage: int = 1
    liquidation_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    exchange: str = ""


@dataclass
class TradeEvent(BaseEvent):
    """Completed trade event schema."""
    trade_id: str = ""
    symbol: str = ""
    side: str = ""
    entry_price: Decimal = Decimal("0")
    exit_price: Decimal = Decimal("0")
    quantity: Decimal = Decimal("0")
    pnl: Decimal = Decimal("0")
    pnl_percent: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")
    entry_time: datetime = field(default_factory=datetime.utcnow)
    exit_time: datetime = field(default_factory=datetime.utcnow)
    duration_seconds: int = 0
    strategy: str = ""
    exchange: str = ""
    tags: List[str] = field(default_factory=list)


# =============================================================================
# Alert Events
# =============================================================================

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AlertEvent(BaseEvent):
    """Alert event schema."""
    alert_type: str = ""
    severity: str = "info"
    title: str = ""
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False


# =============================================================================
# Risk Events
# =============================================================================

@dataclass
class RiskEvent(BaseEvent):
    """Risk management event schema."""
    event_type: str = ""  # limit_reached, drawdown, exposure
    risk_type: str = ""
    current_value: Decimal = Decimal("0")
    limit_value: Decimal = Decimal("0")
    percentage: Decimal = Decimal("0")
    action_taken: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Performance Events
# =============================================================================

@dataclass
class PerformanceSnapshot(BaseEvent):
    """Performance metrics snapshot."""
    period: str = ""  # daily, weekly, monthly
    total_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    average_trade_duration: int = 0
    best_trade: Decimal = Decimal("0")
    worst_trade: Decimal = Decimal("0")


# =============================================================================
# System Events
# =============================================================================

@dataclass
class SystemEvent(BaseEvent):
    """System status event schema."""
    event_type: str = ""  # started, stopped, connected, disconnected, error
    component: str = ""
    status: str = ""
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckEvent(BaseEvent):
    """Health check event schema."""
    component: str = ""
    status: str = ""  # healthy, degraded, unhealthy
    latency_ms: int = 0
    checks: Dict[str, bool] = field(default_factory=dict)
    message: str = ""
