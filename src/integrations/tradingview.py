"""
TradingView Integration

Webhook receiver for TradingView alerts and Pine Script signals.
"""
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import json
import asyncio

try:
    from fastapi import FastAPI, Request, HTTPException, Header, Depends
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from loguru import logger


class AlertAction(Enum):
    """TradingView alert actions."""
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"
    CUSTOM = "custom"


@dataclass
class TradingViewAlert:
    """Represents a TradingView alert."""
    alert_id: str
    timestamp: datetime
    ticker: str
    exchange: str
    action: AlertAction
    price: Decimal
    quantity: Optional[Decimal] = None
    position_size: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    comment: str = ""
    strategy_name: str = ""
    timeframe: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_webhook(cls, data: Dict[str, Any]) -> "TradingViewAlert":
        """Parse alert from webhook payload."""
        # Handle different payload formats
        ticker = (
            data.get('ticker') or
            data.get('symbol') or
            data.get('pair') or
            'UNKNOWN'
        )

        exchange = (
            data.get('exchange') or
            data.get('broker') or
            'BINANCE'
        )

        # Parse action
        action_str = str(data.get('action', data.get('order', 'custom'))).lower()
        action_map = {
            'buy': AlertAction.BUY,
            'long': AlertAction.BUY,
            'sell': AlertAction.SELL,
            'short': AlertAction.SELL,
            'close': AlertAction.CLOSE,
            'exit': AlertAction.CLOSE,
            'close_long': AlertAction.CLOSE_LONG,
            'close_short': AlertAction.CLOSE_SHORT,
            'tp': AlertAction.TAKE_PROFIT,
            'take_profit': AlertAction.TAKE_PROFIT,
            'sl': AlertAction.STOP_LOSS,
            'stop_loss': AlertAction.STOP_LOSS,
            'trailing': AlertAction.TRAILING_STOP,
            'scale_in': AlertAction.SCALE_IN,
            'scale_out': AlertAction.SCALE_OUT,
        }
        action = action_map.get(action_str, AlertAction.CUSTOM)

        # Parse price
        price_str = data.get('price') or data.get('close') or data.get('last') or '0'
        price = Decimal(str(price_str).replace(',', ''))

        # Parse optional fields
        quantity = None
        if data.get('quantity') or data.get('qty') or data.get('size'):
            qty_str = data.get('quantity') or data.get('qty') or data.get('size')
            quantity = Decimal(str(qty_str).replace(',', ''))

        position_size = None
        if data.get('position_size') or data.get('contracts'):
            pos_str = data.get('position_size') or data.get('contracts')
            position_size = Decimal(str(pos_str).replace(',', ''))

        take_profit = None
        if data.get('take_profit') or data.get('tp'):
            tp_str = data.get('take_profit') or data.get('tp')
            take_profit = Decimal(str(tp_str).replace(',', ''))

        stop_loss = None
        if data.get('stop_loss') or data.get('sl'):
            sl_str = data.get('stop_loss') or data.get('sl')
            stop_loss = Decimal(str(sl_str).replace(',', ''))

        return cls(
            alert_id=data.get('alert_id', f"tv-{datetime.now().timestamp()}"),
            timestamp=datetime.now(),
            ticker=ticker,
            exchange=exchange.upper(),
            action=action,
            price=price,
            quantity=quantity,
            position_size=position_size,
            take_profit=take_profit,
            stop_loss=stop_loss,
            comment=data.get('comment', data.get('message', '')),
            strategy_name=data.get('strategy', data.get('strategy_name', '')),
            timeframe=data.get('timeframe', data.get('interval', '')),
            raw_data=data,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "ticker": self.ticker,
            "exchange": self.exchange,
            "action": self.action.value,
            "price": str(self.price),
            "quantity": str(self.quantity) if self.quantity else None,
            "position_size": str(self.position_size) if self.position_size else None,
            "take_profit": str(self.take_profit) if self.take_profit else None,
            "stop_loss": str(self.stop_loss) if self.stop_loss else None,
            "comment": self.comment,
            "strategy_name": self.strategy_name,
            "timeframe": self.timeframe,
        }


# Alert handler callback type
AlertHandler = Callable[[TradingViewAlert], None]
AsyncAlertHandler = Callable[[TradingViewAlert], "asyncio.Future[None]"]


class TradingViewWebhook:
    """TradingView webhook receiver and processor."""

    def __init__(
        self,
        secret_key: Optional[str] = None,
        allowed_ips: Optional[List[str]] = None,
        validate_signature: bool = True,
    ):
        self.secret_key = secret_key
        self.allowed_ips = allowed_ips or []
        self.validate_signature = validate_signature and secret_key is not None

        self.handlers: List[AlertHandler] = []
        self.async_handlers: List[AsyncAlertHandler] = []
        self.alert_history: List[TradingViewAlert] = []
        self.max_history = 1000

        # Stats
        self.stats = {
            "total_received": 0,
            "total_processed": 0,
            "total_errors": 0,
            "by_action": {},
            "by_ticker": {},
        }

    def add_handler(self, handler: AlertHandler) -> None:
        """Add synchronous alert handler."""
        self.handlers.append(handler)
        logger.info(f"Added TradingView alert handler: {handler.__name__}")

    def add_async_handler(self, handler: AsyncAlertHandler) -> None:
        """Add async alert handler."""
        self.async_handlers.append(handler)
        logger.info(f"Added async TradingView alert handler: {handler.__name__}")

    def on_alert(self, func: AlertHandler) -> AlertHandler:
        """Decorator to register alert handler."""
        self.add_handler(func)
        return func

    def on_alert_async(self, func: AsyncAlertHandler) -> AsyncAlertHandler:
        """Decorator to register async alert handler."""
        self.add_async_handler(func)
        return func

    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature."""
        if not self.secret_key:
            return True

        expected = hmac.new(
            self.secret_key.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, signature)

    def verify_ip(self, ip: str) -> bool:
        """Verify request IP."""
        if not self.allowed_ips:
            return True
        return ip in self.allowed_ips

    def process_alert(self, data: Dict[str, Any]) -> TradingViewAlert:
        """Process incoming alert data."""
        self.stats["total_received"] += 1

        try:
            alert = TradingViewAlert.from_webhook(data)

            # Update stats
            action = alert.action.value
            self.stats["by_action"][action] = self.stats["by_action"].get(action, 0) + 1
            self.stats["by_ticker"][alert.ticker] = self.stats["by_ticker"].get(alert.ticker, 0) + 1

            # Store in history
            self.alert_history.append(alert)
            if len(self.alert_history) > self.max_history:
                self.alert_history = self.alert_history[-self.max_history:]

            # Call sync handlers
            for handler in self.handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Handler error: {e}")
                    self.stats["total_errors"] += 1

            self.stats["total_processed"] += 1

            logger.info(
                f"Processed TradingView alert: {alert.action.value} {alert.ticker} @ {alert.price}"
            )

            return alert

        except Exception as e:
            self.stats["total_errors"] += 1
            logger.error(f"Error processing alert: {e}")
            raise

    async def process_alert_async(self, data: Dict[str, Any]) -> TradingViewAlert:
        """Process incoming alert asynchronously."""
        alert = self.process_alert(data)

        # Call async handlers
        for handler in self.async_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Async handler error: {e}")
                self.stats["total_errors"] += 1

        return alert

    def get_stats(self) -> Dict[str, Any]:
        """Get webhook statistics."""
        return {
            **self.stats,
            "handlers_count": len(self.handlers) + len(self.async_handlers),
            "history_count": len(self.alert_history),
        }

    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return [a.to_dict() for a in self.alert_history[-limit:]]

    def create_fastapi_app(self, path: str = "/webhook/tradingview") -> "FastAPI":
        """Create FastAPI app with webhook endpoint."""
        if not HAS_FASTAPI:
            raise ImportError("FastAPI not installed")

        app = FastAPI(title="TradingView Webhook Receiver")

        @app.post(path)
        async def receive_webhook(
            request: Request,
            x_tv_signature: Optional[str] = Header(None, alias="X-TradingView-Signature"),
        ):
            """Receive TradingView webhook."""
            # Verify IP
            client_ip = request.client.host if request.client else ""
            if not self.verify_ip(client_ip):
                logger.warning(f"Unauthorized IP: {client_ip}")
                raise HTTPException(status_code=403, detail="Unauthorized IP")

            # Get payload
            body = await request.body()

            # Verify signature
            if self.validate_signature:
                if not x_tv_signature:
                    raise HTTPException(status_code=401, detail="Missing signature")
                if not self.verify_signature(body, x_tv_signature):
                    raise HTTPException(status_code=401, detail="Invalid signature")

            # Parse JSON
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                # Try form-encoded
                data = dict(await request.form())

            # Process alert
            try:
                alert = await self.process_alert_async(data)
                return JSONResponse({
                    "status": "success",
                    "alert_id": alert.alert_id,
                    "action": alert.action.value,
                    "ticker": alert.ticker,
                })
            except Exception as e:
                logger.error(f"Webhook processing error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/health")
        async def health():
            """Health check."""
            return {"status": "healthy"}

        @app.get("/stats")
        async def get_stats():
            """Get webhook stats."""
            return self.get_stats()

        @app.get("/alerts/recent")
        async def recent_alerts(limit: int = 10):
            """Get recent alerts."""
            return self.get_recent_alerts(limit)

        return app


# Pine Script template for sending alerts
PINE_SCRIPT_ALERT_TEMPLATE = '''
// TradingView Alert Message Template
// Copy this to your alert message in TradingView

{
    "ticker": "{{ticker}}",
    "exchange": "{{exchange}}",
    "action": "{{strategy.order.action}}",
    "price": {{close}},
    "quantity": {{strategy.order.contracts}},
    "position_size": {{strategy.position_size}},
    "strategy": "{{strategy.order.comment}}",
    "timeframe": "{{interval}}",
    "time": "{{time}}"
}
'''

PINE_SCRIPT_STRATEGY_TEMPLATE = '''
//@version=5
// Trading Bot Integration Strategy Template

strategy("Bot Integration", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=100)

// Parameters
fastLength = input.int(12, "Fast MA Length")
slowLength = input.int(26, "Slow MA Length")
signalLength = input.int(9, "Signal Length")

// Indicators
fastMA = ta.ema(close, fastLength)
slowMA = ta.ema(close, slowLength)
macd = fastMA - slowMA
signal = ta.ema(macd, signalLength)
histogram = macd - signal

// Entry conditions
longCondition = ta.crossover(macd, signal)
shortCondition = ta.crossunder(macd, signal)

// Execute trades
if longCondition
    strategy.entry("Long", strategy.long, comment="buy")
    alert('{"action":"buy","ticker":"' + syminfo.ticker + '","price":' + str.tostring(close) + '}', alert.freq_once_per_bar)

if shortCondition
    strategy.entry("Short", strategy.short, comment="sell")
    alert('{"action":"sell","ticker":"' + syminfo.ticker + '","price":' + str.tostring(close) + '}', alert.freq_once_per_bar)

// Exit conditions
if strategy.position_size > 0 and shortCondition
    strategy.close("Long", comment="close_long")

if strategy.position_size < 0 and longCondition
    strategy.close("Short", comment="close_short")

// Plot
plot(fastMA, color=color.blue, title="Fast MA")
plot(slowMA, color=color.red, title="Slow MA")
'''


def get_pine_script_templates() -> Dict[str, str]:
    """Get Pine Script templates for TradingView integration."""
    return {
        "alert_message": PINE_SCRIPT_ALERT_TEMPLATE,
        "strategy": PINE_SCRIPT_STRATEGY_TEMPLATE,
    }
