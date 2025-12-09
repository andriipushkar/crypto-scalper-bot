"""
Webhook API for receiving external trading signals.

Supports signals from:
- TradingView alerts
- Custom integrations
- Third-party signal providers
"""

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Awaitable

from fastapi import APIRouter, HTTPException, Request, Depends, Header
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
from loguru import logger

from src.data.models import Side, SignalType


# =============================================================================
# Models
# =============================================================================

class WebhookSignalType(str, Enum):
    """Webhook signal types."""
    BUY = "buy"
    SELL = "sell"
    LONG = "long"
    SHORT = "short"
    CLOSE = "close"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


class WebhookSignal(BaseModel):
    """
    Incoming webhook signal.

    Supports multiple formats including TradingView.
    """
    # Required fields
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    action: WebhookSignalType = Field(..., description="Signal action")

    # Optional fields
    price: Optional[float] = Field(None, description="Signal price")
    quantity: Optional[float] = Field(None, description="Position size")
    leverage: Optional[int] = Field(None, description="Leverage to use")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")

    # Metadata
    strategy: Optional[str] = Field(None, description="Strategy name")
    timeframe: Optional[str] = Field(None, description="Chart timeframe")
    comment: Optional[str] = Field(None, description="Signal comment")
    timestamp: Optional[int] = Field(None, description="Signal timestamp (ms)")

    # Authentication
    passphrase: Optional[str] = Field(None, description="Webhook passphrase")

    @validator("symbol")
    def normalize_symbol(cls, v):
        return v.upper().replace("-", "").replace("/", "").replace("PERP", "")

    @validator("action", pre=True)
    def normalize_action(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v

    class Config:
        use_enum_values = True


class TradingViewSignal(BaseModel):
    """
    TradingView alert webhook format.

    Example TradingView alert message:
    {
        "symbol": "{{ticker}}",
        "action": "{{strategy.order.action}}",
        "price": {{close}},
        "passphrase": "your_secret"
    }
    """
    ticker: Optional[str] = None
    symbol: Optional[str] = None
    action: str
    price: Optional[float] = None
    contracts: Optional[float] = None
    position_size: Optional[float] = None
    passphrase: Optional[str] = None
    strategy: Optional[str] = None
    interval: Optional[str] = None
    time: Optional[str] = None
    comment: Optional[str] = None

    def to_webhook_signal(self) -> WebhookSignal:
        """Convert to standard WebhookSignal."""
        symbol = self.symbol or self.ticker or ""
        symbol = symbol.upper().replace("-", "").replace("/", "")

        # Map TradingView actions
        action_map = {
            "buy": WebhookSignalType.BUY,
            "sell": WebhookSignalType.SELL,
            "long": WebhookSignalType.LONG,
            "short": WebhookSignalType.SHORT,
            "close": WebhookSignalType.CLOSE,
            "exit": WebhookSignalType.CLOSE,
        }

        action = action_map.get(self.action.lower(), WebhookSignalType.CLOSE)

        return WebhookSignal(
            symbol=symbol,
            action=action,
            price=self.price,
            quantity=self.contracts or self.position_size,
            strategy=self.strategy,
            timeframe=self.interval,
            comment=self.comment,
            passphrase=self.passphrase,
        )


class WebhookResponse(BaseModel):
    """Webhook response."""
    success: bool
    message: str
    signal_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


@dataclass
class WebhookConfig:
    """Webhook configuration."""
    enabled: bool = True
    passphrase: str = ""  # Required passphrase for authentication
    secret_key: str = ""  # For HMAC signature verification
    allowed_ips: List[str] = field(default_factory=list)  # IP whitelist
    rate_limit: int = 60  # Max signals per minute
    require_signature: bool = False


# =============================================================================
# Webhook Handler
# =============================================================================

SignalHandler = Callable[[WebhookSignal], Awaitable[Dict[str, Any]]]


class WebhookHandler:
    """
    Handles incoming webhook signals.

    Usage:
        handler = WebhookHandler(config)

        @handler.on_signal
        async def process_signal(signal: WebhookSignal) -> Dict:
            # Process the signal
            return {"executed": True}

        # In FastAPI app
        app.include_router(handler.router)
    """

    def __init__(self, config: WebhookConfig = None):
        if config:
            self.config = config
        else:
            self.config = WebhookConfig(
                passphrase=os.getenv("WEBHOOK_PASSPHRASE", ""),
                secret_key=os.getenv("WEBHOOK_SECRET_KEY", ""),
                allowed_ips=os.getenv("WEBHOOK_ALLOWED_IPS", "").split(","),
                require_signature=os.getenv("WEBHOOK_REQUIRE_SIGNATURE", "").lower() == "true",
            )

        self._signal_handlers: List[SignalHandler] = []
        self._signal_history: List[Dict[str, Any]] = []
        self._rate_limit_tracker: Dict[str, List[float]] = {}

        # Create router
        self.router = APIRouter(prefix="/webhook", tags=["Webhook"])
        self._setup_routes()

    def on_signal(self, handler: SignalHandler) -> SignalHandler:
        """Decorator to register signal handler."""
        self._signal_handlers.append(handler)
        return handler

    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""

        @self.router.post("/signal", response_model=WebhookResponse)
        async def receive_signal(
            request: Request,
            signal: WebhookSignal,
        ) -> WebhookResponse:
            """
            Receive a trading signal.

            Accepts standard webhook signal format.
            """
            return await self._process_signal(request, signal)

        @self.router.post("/tradingview", response_model=WebhookResponse)
        async def receive_tradingview(
            request: Request,
            signal: TradingViewSignal,
        ) -> WebhookResponse:
            """
            Receive TradingView alert.

            Accepts TradingView webhook format.
            """
            webhook_signal = signal.to_webhook_signal()
            return await self._process_signal(request, webhook_signal)

        @self.router.post("/raw", response_model=WebhookResponse)
        async def receive_raw(request: Request) -> WebhookResponse:
            """
            Receive raw webhook (flexible format).

            Attempts to parse various signal formats.
            """
            try:
                body = await request.json()
            except Exception:
                body = {}
                text = await request.body()
                if text:
                    # Try to parse as key=value pairs
                    for line in text.decode().split("\n"):
                        if "=" in line:
                            k, v = line.split("=", 1)
                            body[k.strip()] = v.strip()

            # Try to build signal from body
            signal = self._parse_raw_signal(body)
            if not signal:
                raise HTTPException(400, "Could not parse signal from request")

            return await self._process_signal(request, signal)

        @self.router.get("/history")
        async def get_history(limit: int = 50) -> List[Dict[str, Any]]:
            """Get recent signal history."""
            return self._signal_history[-limit:]

        @self.router.get("/health")
        async def health() -> Dict[str, Any]:
            """Webhook health check."""
            return {
                "status": "ok",
                "enabled": self.config.enabled,
                "handlers": len(self._signal_handlers),
                "signals_received": len(self._signal_history),
            }

    def _parse_raw_signal(self, body: Dict[str, Any]) -> Optional[WebhookSignal]:
        """Parse raw request body into signal."""
        # Find symbol
        symbol = body.get("symbol") or body.get("ticker") or body.get("pair")
        if not symbol:
            return None

        # Find action
        action = body.get("action") or body.get("side") or body.get("type")
        if not action:
            return None

        try:
            return WebhookSignal(
                symbol=str(symbol),
                action=action,
                price=float(body["price"]) if "price" in body else None,
                quantity=float(body.get("quantity") or body.get("size") or body.get("amount") or 0) or None,
                passphrase=body.get("passphrase") or body.get("key"),
                strategy=body.get("strategy"),
                comment=body.get("comment") or body.get("message"),
            )
        except Exception as e:
            logger.warning(f"Failed to parse raw signal: {e}")
            return None

    async def _process_signal(
        self,
        request: Request,
        signal: WebhookSignal,
    ) -> WebhookResponse:
        """Process incoming signal."""
        client_ip = request.client.host if request.client else "unknown"

        # Check if enabled
        if not self.config.enabled:
            raise HTTPException(503, "Webhook is disabled")

        # Check IP whitelist
        if self.config.allowed_ips and self.config.allowed_ips != [""]:
            if client_ip not in self.config.allowed_ips:
                logger.warning(f"Webhook request from unauthorized IP: {client_ip}")
                raise HTTPException(403, "IP not allowed")

        # Check rate limit
        if not self._check_rate_limit(client_ip):
            raise HTTPException(429, "Rate limit exceeded")

        # Check passphrase
        if self.config.passphrase:
            if signal.passphrase != self.config.passphrase:
                logger.warning(f"Webhook request with invalid passphrase from {client_ip}")
                raise HTTPException(401, "Invalid passphrase")

        # Check signature if required
        if self.config.require_signature:
            signature = request.headers.get("X-Webhook-Signature")
            if not self._verify_signature(await request.body(), signature):
                logger.warning(f"Webhook request with invalid signature from {client_ip}")
                raise HTTPException(401, "Invalid signature")

        # Generate signal ID
        signal_id = f"sig_{int(time.time() * 1000)}_{hash(signal.symbol + signal.action)}"

        # Log signal
        logger.info(
            f"Webhook signal received: {signal.action.upper()} {signal.symbol} "
            f"from {client_ip}"
        )

        # Record in history
        history_entry = {
            "id": signal_id,
            "symbol": signal.symbol,
            "action": signal.action,
            "price": signal.price,
            "quantity": signal.quantity,
            "strategy": signal.strategy,
            "ip": client_ip,
            "timestamp": datetime.now().isoformat(),
            "processed": False,
            "result": None,
        }
        self._signal_history.append(history_entry)

        # Trim history
        if len(self._signal_history) > 1000:
            self._signal_history = self._signal_history[-500:]

        # Process with handlers
        results = []
        for handler in self._signal_handlers:
            try:
                result = await handler(signal)
                results.append(result)
                history_entry["processed"] = True
                history_entry["result"] = result
            except Exception as e:
                logger.error(f"Signal handler error: {e}")
                history_entry["result"] = {"error": str(e)}

        return WebhookResponse(
            success=True,
            message=f"Signal {signal.action.upper()} {signal.symbol} received",
            signal_id=signal_id,
            data={"results": results} if results else None,
        )

    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check rate limit for client."""
        now = time.time()
        window = 60  # 1 minute

        if client_ip not in self._rate_limit_tracker:
            self._rate_limit_tracker[client_ip] = []

        # Remove old entries
        self._rate_limit_tracker[client_ip] = [
            t for t in self._rate_limit_tracker[client_ip]
            if now - t < window
        ]

        # Check limit
        if len(self._rate_limit_tracker[client_ip]) >= self.config.rate_limit:
            return False

        self._rate_limit_tracker[client_ip].append(now)
        return True

    def _verify_signature(self, body: bytes, signature: str) -> bool:
        """Verify HMAC signature."""
        if not signature or not self.config.secret_key:
            return False

        expected = hmac.new(
            self.config.secret_key.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(signature, expected)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_webhook_handler(
    passphrase: str = None,
    secret_key: str = None,
    allowed_ips: List[str] = None,
) -> WebhookHandler:
    """Create webhook handler with configuration."""
    config = WebhookConfig(
        passphrase=passphrase or os.getenv("WEBHOOK_PASSPHRASE", ""),
        secret_key=secret_key or os.getenv("WEBHOOK_SECRET_KEY", ""),
        allowed_ips=allowed_ips or [],
    )
    return WebhookHandler(config)


def signal_to_internal(signal: WebhookSignal) -> Dict[str, Any]:
    """Convert webhook signal to internal signal format."""
    action_map = {
        WebhookSignalType.BUY: SignalType.LONG,
        WebhookSignalType.LONG: SignalType.LONG,
        WebhookSignalType.SELL: SignalType.SHORT,
        WebhookSignalType.SHORT: SignalType.SHORT,
        WebhookSignalType.CLOSE: SignalType.CLOSE_LONG,  # Default
        WebhookSignalType.CLOSE_LONG: SignalType.CLOSE_LONG,
        WebhookSignalType.CLOSE_SHORT: SignalType.CLOSE_SHORT,
    }

    return {
        "symbol": signal.symbol,
        "signal_type": action_map.get(signal.action, SignalType.NO_ACTION),
        "price": Decimal(str(signal.price)) if signal.price else None,
        "quantity": Decimal(str(signal.quantity)) if signal.quantity else None,
        "leverage": signal.leverage,
        "take_profit": Decimal(str(signal.take_profit)) if signal.take_profit else None,
        "stop_loss": Decimal(str(signal.stop_loss)) if signal.stop_loss else None,
        "strategy": signal.strategy or "webhook",
        "strength": 1.0,  # Webhook signals are treated as strong
        "timestamp": datetime.now(),
    }
