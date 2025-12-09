"""
FastAPI application for the trading dashboard.

Provides:
- Real-time WebSocket updates
- REST API for bot control
- Performance metrics
- Trade history
- Secure authentication (JWT + API Key)
- Rate limiting
"""

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from src.data.models import Signal, SignalType, Order, Position, Side
from src.web.middleware import (
    init_security,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    RequestLoggingMiddleware,
    AuthMiddleware,
    require_auth,
    require_admin,
    get_current_user,
    jwt_auth,
    api_key_auth,
)


# =============================================================================
# Pydantic Models
# =============================================================================

class StatusResponse(BaseModel):
    """Bot status response."""
    status: str  # running, stopped, error
    uptime_seconds: float
    mode: str  # paper, live
    connected_exchanges: List[str]
    active_symbols: List[str]


class PositionResponse(BaseModel):
    """Position response."""
    symbol: str
    side: str
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    leverage: int


class TradeResponse(BaseModel):
    """Trade history response."""
    id: str
    symbol: str
    side: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    strategy: str


class MetricsResponse(BaseModel):
    """Performance metrics response."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float


class SignalResponse(BaseModel):
    """Signal response."""
    timestamp: str
    symbol: str
    type: str
    strength: float
    price: float
    strategy: str


class ConfigUpdateRequest(BaseModel):
    """Config update request."""
    key: str
    value: Any


class CommandRequest(BaseModel):
    """Bot command request."""
    command: str  # start, stop, pause, resume
    params: Optional[Dict[str, Any]] = None


class LoginRequest(BaseModel):
    """Login request."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


# =============================================================================
# New Request/Response Models for Trading Operations
# =============================================================================

class PlaceOrderRequest(BaseModel):
    """Place order request."""
    symbol: str
    side: str  # BUY, SELL
    quantity: float
    order_type: str = "MARKET"  # MARKET, LIMIT
    price: Optional[float] = None  # Required for LIMIT orders
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: Optional[int] = None


class ClosePositionRequest(BaseModel):
    """Close position request."""
    symbol: str
    quantity: Optional[float] = None  # None = close all


class SetLeverageRequest(BaseModel):
    """Set leverage request."""
    symbol: str
    leverage: int  # 1-125


class SetSLTPRequest(BaseModel):
    """Set stop-loss/take-profit request."""
    symbol: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class StrategyConfigRequest(BaseModel):
    """Strategy configuration request."""
    strategy_name: str
    enabled: bool
    params: Dict[str, Any]


class RiskConfigRequest(BaseModel):
    """Risk configuration request."""
    max_position_usd: Optional[float] = None
    max_daily_loss_usd: Optional[float] = None
    max_trades_per_day: Optional[int] = None
    max_open_positions: Optional[int] = None
    default_stop_loss_pct: Optional[float] = None
    default_take_profit_pct: Optional[float] = None
    risk_per_trade_pct: Optional[float] = None


class ExchangeConfigRequest(BaseModel):
    """Exchange configuration request."""
    exchange: str  # binance, bybit, okx, kraken, kucoin, gateio
    testnet: bool = True


class OrderResponse(BaseModel):
    """Order response."""
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    order_type: str
    status: str
    created_at: str


class LogEntry(BaseModel):
    """Log entry."""
    timestamp: str
    level: str
    message: str
    module: Optional[str] = None


# =============================================================================
# WebSocket Connection Manager
# =============================================================================

class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"WebSocket send error: {e}")

    async def send_personal(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send message to specific client."""
        await websocket.send_json(message)


# =============================================================================
# Shared State
# =============================================================================

class DashboardState:
    """Shared state for the dashboard."""

    def __init__(self):
        self.start_time = datetime.utcnow()
        self.status = "stopped"
        self.mode = "paper"
        self.symbols = ["BTCUSDT"]

        # Real-time data
        self.positions: Dict[str, Position] = {}
        self.signals: List[Signal] = []
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.orders: Dict[str, Dict] = {}  # Open orders

        # Metrics
        self.metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
        }

        # Market data
        self.prices: Dict[str, float] = {}
        self.orderbook_stats: Dict[str, Dict] = {}

        # Exchange configuration
        self.exchange_config = {
            "exchange": "binance",
            "testnet": True,
        }

        # Risk configuration
        self.risk_config = {
            "max_position_usd": 30.0,
            "max_daily_loss_usd": 5.0,
            "max_trades_per_day": 20,
            "max_open_positions": 1,
            "default_stop_loss_pct": 0.5,
            "default_take_profit_pct": 0.3,
            "risk_per_trade_pct": 1.0,
        }

        # Strategy configurations
        self.strategies_config = {
            "orderbook_imbalance": {
                "enabled": True,
                "imbalance_threshold": 1.5,
                "max_spread": 0.0005,
                "signal_cooldown": 10,
                "levels": 5,
                "min_volume_usd": 5000,
            },
            "volume_spike": {
                "enabled": True,
                "volume_multiplier": 3.0,
                "lookback_seconds": 60,
                "min_volume_usd": 10000,
                "signal_cooldown": 15,
            },
            "mean_reversion": {
                "enabled": False,
                "lookback_period": 20,
                "std_dev_multiplier": 2.0,
                "entry_z_score": 2.0,
            },
            "grid_trading": {
                "enabled": False,
                "grid_levels": 10,
                "range_percent": 0.05,
            },
            "dca": {
                "enabled": False,
                "mode": "hybrid",
                "interval_minutes": 60,
            },
            # ============================================
            # Advanced Scalping Strategies (Просунутий скальпінг)
            # ============================================
            "hybrid_scalping": {
                "enabled": True,
                "min_confirmations": 2,
                "min_combined_weight": 0.5,
                "signal_cooldown": 30,
                "description": "Гібридна стратегія (комбінує всі скальпінг-методи)",
            },
            "advanced_orderbook": {
                "enabled": True,
                "wall_multiplier": 5.0,
                "min_wall_value_usd": 50000,
                "frontrunning_enabled": True,
                "spoofing_detection": True,
                "iceberg_detection": True,
                "weight": 0.35,
                "description": "Аналіз стакану (стіни, фронтраннінг, спуфінг)",
            },
            "print_tape": {
                "enabled": True,
                "whale_threshold_usd": 50000,
                "cvd_window": 60,
                "delta_threshold": 60,
                "min_whale_trades": 3,
                "weight": 0.25,
                "description": "Лента принтів (CVD, whale trades)",
            },
            "cluster_analysis": {
                "enabled": True,
                "cluster_size_ticks": 10,
                "value_area_percent": 70,
                "imbalance_threshold": 65,
                "weight": 0.15,
                "description": "Кластерний аналіз (POC, Value Area, Delta)",
            },
            "impulse_scalping": {
                "enabled": False,
                "min_leader_move": 0.3,
                "impulse_ttl": 300,
                "correlation_lag": 60,
                "weight": 0.25,
                "description": "Імпульсний скальпінг (SPX/DXY/GOLD/VIX)",
            },
        }

        # Leverage per symbol
        self.leverage: Dict[str, int] = {"BTCUSDT": 10, "ETHUSDT": 10}

        # SL/TP per position
        self.sl_tp: Dict[str, Dict[str, float]] = {}

        # Logs buffer
        self.logs: List[Dict] = []
        self.max_logs = 1000

        # Alerts configuration
        self.alerts_enabled = True

    @property
    def uptime_seconds(self) -> float:
        return (datetime.utcnow() - self.start_time).total_seconds()

    def add_log(self, level: str, message: str, module: str = None):
        """Add log entry."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            "module": module,
        }
        self.logs.append(entry)
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]


# Global instances
manager = ConnectionManager()
state = DashboardState()


# =============================================================================
# FastAPI Application
# =============================================================================

def create_app(
    enable_auth: bool = True,
    rate_limit_rpm: int = 120,
) -> FastAPI:
    """
    Create FastAPI application with security features.

    Args:
        enable_auth: Enable authentication (default: True)
        rate_limit_rpm: Rate limit requests per minute (default: 120)
    """
    app = FastAPI(
        title="Crypto Scalper Bot",
        description="Real-time trading dashboard with secure API",
        version="1.0.0",
        docs_url="/docs" if os.getenv("ENABLE_DOCS", "true").lower() == "true" else None,
        redoc_url="/redoc" if os.getenv("ENABLE_DOCS", "true").lower() == "true" else None,
    )

    # Initialize security
    init_security(
        enable_jwt=enable_auth,
        enable_api_key=enable_auth,
        rate_limit_rpm=rate_limit_rpm,
    )

    # Security headers middleware (first, so headers are always added)
    app.add_middleware(SecurityHeadersMiddleware)

    # Request logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Rate limiting middleware
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=rate_limit_rpm,
        exempt_paths=["/health", "/docs", "/openapi.json", "/redoc", "/"],
    )

    # Authentication middleware (optional)
    if enable_auth:
        app.add_middleware(
            AuthMiddleware,
            public_paths=["/", "/health", "/docs", "/openapi.json", "/redoc", "/ws", "/api/auth/login"],
            auth_required=os.getenv("REQUIRE_AUTH", "false").lower() == "true",
        )

    # CORS middleware
    allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # =========================================================================
    # Authentication Endpoints
    # =========================================================================

    @app.post("/api/auth/login", response_model=TokenResponse)
    async def login(request: LoginRequest):
        """
        Login and get access token.

        Default credentials (change in production!):
        - username: admin
        - password: Set ADMIN_PASSWORD env var
        """
        # Get credentials from env
        admin_user = os.getenv("ADMIN_USERNAME", "admin")
        admin_pass = os.getenv("ADMIN_PASSWORD", "changeme123")

        if request.username != admin_user or request.password != admin_pass:
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials",
            )

        if not jwt_auth:
            raise HTTPException(
                status_code=503,
                detail="Authentication service unavailable",
            )

        # Create tokens
        access_token = jwt_auth.create_access_token({
            "sub": request.username,
            "is_admin": True,
            "permissions": ["read", "write", "admin"],
        })

        refresh_token = jwt_auth.create_refresh_token({
            "sub": request.username,
        })

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=jwt_auth.config.access_token_expire_minutes * 60,
        )

    @app.post("/api/auth/refresh", response_model=TokenResponse)
    async def refresh_token(
        authorization: str = Header(None),
    ):
        """Refresh access token using refresh token."""
        if not jwt_auth:
            raise HTTPException(status_code=503, detail="Auth service unavailable")

        token = jwt_auth.get_token_from_header(authorization)
        if not token:
            raise HTTPException(status_code=401, detail="Refresh token required")

        payload = jwt_auth.verify_token(token)
        if not payload or payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid refresh token")

        # Create new access token
        access_token = jwt_auth.create_access_token({
            "sub": payload["sub"],
            "is_admin": True,
            "permissions": ["read", "write", "admin"],
        })

        refresh_token = jwt_auth.create_refresh_token({"sub": payload["sub"]})

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=jwt_auth.config.access_token_expire_minutes * 60,
        )

    @app.get("/api/auth/me")
    async def get_current_user_info(user: dict = Depends(require_auth)):
        """Get current user information."""
        return {
            "type": user["type"],
            "username": user["data"].get("sub") or user["data"].get("name"),
            "permissions": user["data"].get("permissions", []),
        }

    # =========================================================================
    # WebSocket Endpoints
    # =========================================================================

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Main WebSocket endpoint for real-time updates."""
        await manager.connect(websocket)

        try:
            # Send initial state
            await manager.send_personal(websocket, {
                "type": "initial_state",
                "data": {
                    "status": state.status,
                    "mode": state.mode,
                    "symbols": state.symbols,
                    "positions": _serialize_positions(),
                    "metrics": state.metrics,
                }
            })

            while True:
                # Receive messages from client
                data = await websocket.receive_json()

                # Handle different message types
                msg_type = data.get("type")

                if msg_type == "subscribe":
                    # Subscribe to specific channels
                    channels = data.get("channels", [])
                    await manager.send_personal(websocket, {
                        "type": "subscribed",
                        "channels": channels,
                    })

                elif msg_type == "ping":
                    await manager.send_personal(websocket, {"type": "pong"})

        except WebSocketDisconnect:
            manager.disconnect(websocket)

    # =========================================================================
    # REST API Endpoints
    # =========================================================================

    @app.get("/api/status", response_model=StatusResponse)
    async def get_status():
        """Get bot status."""
        return StatusResponse(
            status=state.status,
            uptime_seconds=state.uptime_seconds,
            mode=state.mode,
            connected_exchanges=["binance_futures"],
            active_symbols=state.symbols,
        )

    @app.post("/api/command")
    async def execute_command(request: CommandRequest):
        """Execute bot command."""
        command = request.command

        if command == "start":
            state.status = "running"
            await manager.broadcast({"type": "status_change", "status": "running"})
            return {"success": True, "message": "Bot started"}

        elif command == "stop":
            state.status = "stopped"
            await manager.broadcast({"type": "status_change", "status": "stopped"})
            return {"success": True, "message": "Bot stopped"}

        elif command == "pause":
            state.status = "paused"
            await manager.broadcast({"type": "status_change", "status": "paused"})
            return {"success": True, "message": "Bot paused"}

        elif command == "resume":
            state.status = "running"
            await manager.broadcast({"type": "status_change", "status": "running"})
            return {"success": True, "message": "Bot resumed"}

        else:
            raise HTTPException(status_code=400, detail=f"Unknown command: {command}")

    @app.get("/api/positions", response_model=List[PositionResponse])
    async def get_positions():
        """Get current positions."""
        return _serialize_positions()

    @app.get("/api/trades", response_model=List[TradeResponse])
    async def get_trades(
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        symbol: Optional[str] = None,
    ):
        """Get trade history."""
        trades = state.trades

        if symbol:
            trades = [t for t in trades if t.get("symbol") == symbol]

        trades = trades[offset:offset + limit]

        return [
            TradeResponse(
                id=t.get("id", ""),
                symbol=t.get("symbol", ""),
                side=t.get("side", ""),
                entry_time=t.get("entry_time", ""),
                exit_time=t.get("exit_time", ""),
                entry_price=t.get("entry_price", 0),
                exit_price=t.get("exit_price", 0),
                quantity=t.get("quantity", 0),
                pnl=t.get("pnl", 0),
                pnl_pct=t.get("pnl_pct", 0),
                strategy=t.get("strategy", ""),
            )
            for t in trades
        ]

    @app.get("/api/metrics", response_model=MetricsResponse)
    async def get_metrics():
        """Get performance metrics."""
        return MetricsResponse(**state.metrics)

    @app.get("/api/signals", response_model=List[SignalResponse])
    async def get_signals(limit: int = Query(50, ge=1, le=500)):
        """Get recent signals."""
        signals = state.signals[-limit:]

        return [
            SignalResponse(
                timestamp=s.timestamp.isoformat(),
                symbol=s.symbol,
                type=s.signal_type.name,
                strength=s.strength,
                price=float(s.price),
                strategy=s.strategy,
            )
            for s in signals
        ]

    @app.get("/api/equity")
    async def get_equity_curve(
        start: Optional[str] = None,
        end: Optional[str] = None,
    ):
        """Get equity curve data."""
        curve = state.equity_curve

        if start:
            start_dt = datetime.fromisoformat(start)
            curve = [p for p in curve if datetime.fromisoformat(p["timestamp"]) >= start_dt]

        if end:
            end_dt = datetime.fromisoformat(end)
            curve = [p for p in curve if datetime.fromisoformat(p["timestamp"]) <= end_dt]

        return {"data": curve}

    @app.get("/api/market/{symbol}")
    async def get_market_data(symbol: str):
        """Get market data for symbol."""
        price = state.prices.get(symbol, 0)
        orderbook = state.orderbook_stats.get(symbol, {})

        return {
            "symbol": symbol,
            "price": price,
            "orderbook": orderbook,
        }

    @app.get("/api/config")
    async def get_config():
        """Get current configuration."""
        return {
            "mode": state.mode,
            "symbols": state.symbols,
            "leverage": state.leverage,
            "exchange": state.exchange_config,
            "risk": state.risk_config,
            "strategies": state.strategies_config,
        }

    @app.post("/api/config")
    async def update_config(request: ConfigUpdateRequest):
        """Update configuration."""
        key = request.key
        value = request.value

        if key == "mode":
            if value in ["paper", "live"]:
                state.mode = value
                return {"success": True, "message": f"Mode set to {value}"}
            raise HTTPException(status_code=400, detail="Invalid mode")

        if key == "symbols":
            if isinstance(value, list):
                state.symbols = value
                return {"success": True, "message": f"Symbols set to {value}"}
            raise HTTPException(status_code=400, detail="Symbols must be a list")

        raise HTTPException(status_code=400, detail=f"Unknown config key: {key}")

    # =========================================================================
    # Order Management Endpoints
    # =========================================================================

    @app.post("/api/order", response_model=OrderResponse)
    async def place_order(request: PlaceOrderRequest):
        """
        Place a new order.

        - **symbol**: Trading pair (e.g., BTCUSDT)
        - **side**: BUY or SELL
        - **quantity**: Order size
        - **order_type**: MARKET or LIMIT
        - **price**: Required for LIMIT orders
        - **stop_loss**: Optional stop-loss price
        - **take_profit**: Optional take-profit price
        """
        import uuid

        # Validate
        if request.side.upper() not in ["BUY", "SELL"]:
            raise HTTPException(status_code=400, detail="Side must be BUY or SELL")

        if request.order_type.upper() not in ["MARKET", "LIMIT"]:
            raise HTTPException(status_code=400, detail="Order type must be MARKET or LIMIT")

        if request.order_type.upper() == "LIMIT" and not request.price:
            raise HTTPException(status_code=400, detail="Price required for LIMIT orders")

        if request.quantity <= 0:
            raise HTTPException(status_code=400, detail="Quantity must be positive")

        # Set leverage if specified
        if request.leverage:
            state.leverage[request.symbol] = request.leverage

        # Get current price for market orders
        price = request.price or state.prices.get(request.symbol, 50000.0)

        # Create order
        order_id = str(uuid.uuid4())[:8]
        order = {
            "order_id": order_id,
            "symbol": request.symbol,
            "side": request.side.upper(),
            "quantity": request.quantity,
            "price": price,
            "order_type": request.order_type.upper(),
            "status": "FILLED" if request.order_type.upper() == "MARKET" else "NEW",
            "created_at": datetime.utcnow().isoformat(),
        }

        # For market orders, create position immediately
        if request.order_type.upper() == "MARKET":
            position = Position(
                symbol=request.symbol,
                side=Side.BUY if request.side.upper() == "BUY" else Side.SELL,
                size=Decimal(str(request.quantity)),
                entry_price=Decimal(str(price)),
                mark_price=Decimal(str(price)),
                liquidation_price=None,
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                leverage=state.leverage.get(request.symbol, 10),
                margin_type="CROSSED",
                updated_at=datetime.utcnow(),
            )
            state.positions[request.symbol] = position

            # Set SL/TP if provided
            if request.stop_loss or request.take_profit:
                state.sl_tp[request.symbol] = {
                    "stop_loss": request.stop_loss,
                    "take_profit": request.take_profit,
                }

            state.add_log("INFO", f"Order placed: {request.side} {request.quantity} {request.symbol} @ {price}", "orders")

            # Broadcast to WebSocket clients
            await manager.broadcast({
                "type": "order_placed",
                "data": order,
            })
            await manager.broadcast({
                "type": "position_update",
                "data": _serialize_positions(),
            })
        else:
            # Store limit order
            state.orders[order_id] = order

        return OrderResponse(**order)

    @app.post("/api/position/close")
    async def close_position(request: ClosePositionRequest):
        """
        Close an open position.

        - **symbol**: Position to close
        - **quantity**: Amount to close (None = close all)
        """
        if request.symbol not in state.positions:
            raise HTTPException(status_code=404, detail=f"No position found for {request.symbol}")

        position = state.positions[request.symbol]
        close_qty = request.quantity or float(position.size)

        # Calculate P&L
        current_price = state.prices.get(request.symbol, float(position.mark_price))
        entry_price = float(position.entry_price)

        if position.side == Side.BUY:
            pnl = (current_price - entry_price) * close_qty
        else:
            pnl = (entry_price - current_price) * close_qty

        # Record trade
        trade = {
            "id": str(uuid.uuid4())[:8] if 'uuid' in dir() else "manual",
            "symbol": request.symbol,
            "side": position.side.value,
            "entry_time": datetime.utcnow().isoformat(),
            "exit_time": datetime.utcnow().isoformat(),
            "entry_price": entry_price,
            "exit_price": current_price,
            "quantity": close_qty,
            "pnl": pnl,
            "pnl_pct": (pnl / (entry_price * close_qty)) * 100 if entry_price else 0,
            "strategy": "manual",
        }
        state.trades.append(trade)

        # Update metrics
        state.metrics["total_trades"] += 1
        state.metrics["total_pnl"] += pnl
        if pnl > 0:
            state.metrics["winning_trades"] += 1
            state.metrics["gross_profit"] += pnl
        else:
            state.metrics["losing_trades"] += 1
            state.metrics["gross_loss"] += abs(pnl)

        if state.metrics["total_trades"] > 0:
            state.metrics["win_rate"] = (state.metrics["winning_trades"] / state.metrics["total_trades"]) * 100

        if state.metrics["gross_loss"] > 0:
            state.metrics["profit_factor"] = state.metrics["gross_profit"] / state.metrics["gross_loss"]

        # Remove position
        del state.positions[request.symbol]
        if request.symbol in state.sl_tp:
            del state.sl_tp[request.symbol]

        state.add_log("INFO", f"Position closed: {request.symbol}, P&L: ${pnl:.2f}", "orders")

        # Broadcast
        await manager.broadcast({
            "type": "position_closed",
            "data": trade,
        })
        await manager.broadcast({
            "type": "position_update",
            "data": _serialize_positions(),
        })
        await manager.broadcast({
            "type": "metrics_update",
            "data": state.metrics,
        })

        return {
            "success": True,
            "message": f"Position closed: {request.symbol}",
            "pnl": pnl,
            "trade": trade,
        }

    @app.put("/api/leverage")
    async def set_leverage(request: SetLeverageRequest):
        """
        Set leverage for a symbol.

        - **symbol**: Trading pair
        - **leverage**: 1-125
        """
        if request.leverage < 1 or request.leverage > 125:
            raise HTTPException(status_code=400, detail="Leverage must be between 1 and 125")

        state.leverage[request.symbol] = request.leverage
        state.add_log("INFO", f"Leverage set: {request.symbol} = {request.leverage}x", "config")

        return {
            "success": True,
            "message": f"Leverage for {request.symbol} set to {request.leverage}x",
        }

    @app.put("/api/sl-tp")
    async def set_sl_tp(request: SetSLTPRequest):
        """
        Set stop-loss and/or take-profit for a position.

        - **symbol**: Position symbol
        - **stop_loss**: Stop-loss price
        - **take_profit**: Take-profit price
        """
        if request.symbol not in state.positions:
            raise HTTPException(status_code=404, detail=f"No position found for {request.symbol}")

        if request.symbol not in state.sl_tp:
            state.sl_tp[request.symbol] = {}

        if request.stop_loss is not None:
            state.sl_tp[request.symbol]["stop_loss"] = request.stop_loss

        if request.take_profit is not None:
            state.sl_tp[request.symbol]["take_profit"] = request.take_profit

        state.add_log("INFO", f"SL/TP set: {request.symbol} SL={request.stop_loss} TP={request.take_profit}", "config")

        return {
            "success": True,
            "message": f"SL/TP updated for {request.symbol}",
            "sl_tp": state.sl_tp[request.symbol],
        }

    @app.delete("/api/order/{order_id}")
    async def cancel_order(order_id: str):
        """Cancel a pending order."""
        if order_id not in state.orders:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")

        order = state.orders.pop(order_id)
        order["status"] = "CANCELLED"

        state.add_log("INFO", f"Order cancelled: {order_id}", "orders")

        return {
            "success": True,
            "message": f"Order {order_id} cancelled",
            "order": order,
        }

    # =========================================================================
    # Strategy Configuration Endpoints
    # =========================================================================

    @app.get("/api/strategies")
    async def get_strategies():
        """Get all strategy configurations."""
        return state.strategies_config

    @app.put("/api/strategies")
    async def update_strategy(request: StrategyConfigRequest):
        """
        Update strategy configuration.

        - **strategy_name**: Strategy to configure
        - **enabled**: Enable/disable strategy
        - **params**: Strategy parameters
        """
        if request.strategy_name not in state.strategies_config:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown strategy: {request.strategy_name}. Available: {list(state.strategies_config.keys())}"
            )

        state.strategies_config[request.strategy_name]["enabled"] = request.enabled
        state.strategies_config[request.strategy_name].update(request.params)

        state.add_log("INFO", f"Strategy updated: {request.strategy_name} enabled={request.enabled}", "config")

        return {
            "success": True,
            "message": f"Strategy {request.strategy_name} updated",
            "config": state.strategies_config[request.strategy_name],
        }

    @app.post("/api/strategies/{strategy_name}/toggle")
    async def toggle_strategy(strategy_name: str):
        """Toggle strategy enabled/disabled."""
        if strategy_name not in state.strategies_config:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy_name}")

        current = state.strategies_config[strategy_name]["enabled"]
        state.strategies_config[strategy_name]["enabled"] = not current

        new_state = "enabled" if not current else "disabled"
        state.add_log("INFO", f"Strategy {strategy_name} {new_state}", "config")

        return {
            "success": True,
            "message": f"Strategy {strategy_name} {new_state}",
            "enabled": not current,
        }

    # =========================================================================
    # Risk Configuration Endpoints
    # =========================================================================

    @app.get("/api/risk")
    async def get_risk_config():
        """Get risk management configuration."""
        return state.risk_config

    @app.put("/api/risk")
    async def update_risk_config(request: RiskConfigRequest):
        """
        Update risk management configuration.

        All fields are optional - only provided fields will be updated.
        """
        updates = request.dict(exclude_unset=True, exclude_none=True)

        for key, value in updates.items():
            if key in state.risk_config:
                state.risk_config[key] = value

        state.add_log("INFO", f"Risk config updated: {updates}", "config")

        return {
            "success": True,
            "message": "Risk configuration updated",
            "config": state.risk_config,
        }

    # =========================================================================
    # Exchange Configuration Endpoints
    # =========================================================================

    @app.get("/api/exchange")
    async def get_exchange_config():
        """Get exchange configuration."""
        return {
            "current": state.exchange_config,
            "available": ["binance", "bybit", "okx", "kraken", "kucoin", "gateio"],
        }

    @app.put("/api/exchange")
    async def update_exchange_config(request: ExchangeConfigRequest):
        """
        Update exchange configuration.

        - **exchange**: Exchange name
        - **testnet**: Use testnet (recommended for testing)
        """
        available = ["binance", "bybit", "okx", "kraken", "kucoin", "gateio"]
        if request.exchange.lower() not in available:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown exchange: {request.exchange}. Available: {available}"
            )

        state.exchange_config["exchange"] = request.exchange.lower()
        state.exchange_config["testnet"] = request.testnet

        state.add_log("INFO", f"Exchange changed to {request.exchange} (testnet={request.testnet})", "config")

        return {
            "success": True,
            "message": f"Exchange set to {request.exchange}",
            "config": state.exchange_config,
        }

    # =========================================================================
    # Logs Endpoint
    # =========================================================================

    @app.get("/api/logs")
    async def get_logs(
        limit: int = Query(100, ge=1, le=1000),
        level: Optional[str] = None,
        module: Optional[str] = None,
    ):
        """
        Get recent logs.

        - **limit**: Number of logs to return
        - **level**: Filter by level (INFO, WARNING, ERROR)
        - **module**: Filter by module
        """
        logs = state.logs

        if level:
            logs = [l for l in logs if l["level"] == level.upper()]

        if module:
            logs = [l for l in logs if l.get("module") == module]

        return {
            "total": len(state.logs),
            "filtered": len(logs),
            "logs": logs[-limit:],
        }

    @app.delete("/api/logs")
    async def clear_logs():
        """Clear all logs."""
        state.logs.clear()
        return {"success": True, "message": "Logs cleared"}

    # =========================================================================
    # Alerts Configuration
    # =========================================================================

    @app.get("/api/alerts")
    async def get_alerts_config():
        """Get alerts configuration."""
        return {"enabled": state.alerts_enabled}

    @app.put("/api/alerts")
    async def update_alerts(enabled: bool = True):
        """Enable/disable alerts."""
        state.alerts_enabled = enabled
        return {
            "success": True,
            "message": f"Alerts {'enabled' if enabled else 'disabled'}",
        }

    # =========================================================================
    # Health Check
    # =========================================================================

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

    # =========================================================================
    # Dashboard HTML
    # =========================================================================

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Serve dashboard HTML."""
        return get_dashboard_html()

    return app


# =============================================================================
# Helper Functions
# =============================================================================

def _serialize_positions() -> List[Dict]:
    """Serialize positions for API response."""
    return [
        {
            "symbol": pos.symbol,
            "side": pos.side.value,
            "size": float(pos.size),
            "entry_price": float(pos.entry_price),
            "mark_price": float(pos.mark_price),
            "unrealized_pnl": float(pos.unrealized_pnl),
            "unrealized_pnl_pct": float(pos.unrealized_pnl / pos.entry_price * 100) if pos.entry_price else 0,
            "leverage": pos.leverage,
        }
        for pos in state.positions.values()
    ]


def get_dashboard_html() -> str:
    """Generate dashboard HTML."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crypto Scalper Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .status-running { color: #16a34a; }
        .status-stopped { color: #dc2626; }
        .status-paused { color: #d97706; }
        .pnl-positive { color: #16a34a; }
        .pnl-negative { color: #dc2626; }
        .tab-active { border-bottom: 2px solid #2563eb; color: #2563eb; }
        .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 50; }
        .modal.active { display: flex; align-items: center; justify-content: center; }
        .toast { position: fixed; bottom: 20px; right: 20px; padding: 12px 24px; border-radius: 8px; z-index: 100; animation: slideIn 0.3s; color: white; }
        .toast-success { background: #16a34a; }
        .toast-error { background: #dc2626; }
        @keyframes slideIn { from { transform: translateX(100%); } to { transform: translateX(0); } }
        .card { background: white; border: 1px solid #e5e7eb; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .btn { color: white !important; }
        button[class*="bg-green"], button[class*="bg-red"], button[class*="bg-yellow"], button[class*="bg-blue"], button[class*="bg-purple"], button[class*="bg-orange"] { color: white !important; }
    </style>
</head>
<body class="bg-gray-100 text-gray-900 min-h-screen">
    <!-- Navigation -->
    <nav class="bg-white border-b border-gray-200 shadow-sm">
        <div class="container mx-auto px-4">
            <div class="flex items-center justify-between h-16">
                <h1 class="text-xl font-bold">🤖 Crypto Scalper Bot</h1>
                <div class="flex items-center gap-4">
                    <span id="status" class="px-3 py-1 rounded-full text-sm font-medium bg-gray-200">--</span>
                    <button onclick="sendCommand('start')" class="bg-green-600 hover:bg-green-700 px-4 py-2 rounded text-sm">▶ Start</button>
                    <button onclick="sendCommand('pause')" class="bg-yellow-600 hover:bg-yellow-700 px-4 py-2 rounded text-sm">⏸ Pause</button>
                    <button onclick="sendCommand('stop')" class="bg-red-600 hover:bg-red-700 px-4 py-2 rounded text-sm">⏹ Stop</button>
                </div>
            </div>
        </div>
    </nav>

    <!-- Tabs -->
    <div class="bg-white border-b border-gray-200">
        <div class="container mx-auto px-4">
            <div class="flex gap-6">
                <button onclick="showTab('dashboard')" class="tab-btn tab-active py-4 px-2" data-tab="dashboard">📊 Dashboard</button>
                <button onclick="showTab('trading')" class="tab-btn py-4 px-2 text-gray-500 hover:text-gray-900" data-tab="trading">💹 Trading</button>
                <button onclick="showTab('strategies')" class="tab-btn py-4 px-2 text-gray-500 hover:text-gray-900" data-tab="strategies">🎯 Strategies</button>
                <button onclick="showTab('settings')" class="tab-btn py-4 px-2 text-gray-500 hover:text-gray-900" data-tab="settings">⚙️ Settings</button>
                <button onclick="showTab('logs')" class="tab-btn py-4 px-2 text-gray-500 hover:text-gray-900" data-tab="logs">📋 Logs</button>
            </div>
        </div>
    </div>

    <div class="container mx-auto px-4 py-6">
        <!-- Dashboard Tab -->
        <div id="tab-dashboard" class="tab-content">
            <!-- Metrics Cards -->
            <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
                <div class="card rounded-lg p-4">
                    <div class="text-gray-500 text-xs">Total P&L</div>
                    <div id="totalPnl" class="text-xl font-bold">$0.00</div>
                </div>
                <div class="card rounded-lg p-4">
                    <div class="text-gray-500 text-xs">Daily P&L</div>
                    <div id="dailyPnl" class="text-xl font-bold">$0.00</div>
                </div>
                <div class="card rounded-lg p-4">
                    <div class="text-gray-500 text-xs">Win Rate</div>
                    <div id="winRate" class="text-xl font-bold">0%</div>
                </div>
                <div class="card rounded-lg p-4">
                    <div class="text-gray-500 text-xs">Trades</div>
                    <div id="totalTrades" class="text-xl font-bold">0</div>
                </div>
                <div class="card rounded-lg p-4">
                    <div class="text-gray-500 text-xs">Profit Factor</div>
                    <div id="profitFactor" class="text-xl font-bold">0.00</div>
                </div>
                <div class="card rounded-lg p-4">
                    <div class="text-gray-500 text-xs">Max Drawdown</div>
                    <div id="maxDrawdown" class="text-xl font-bold text-red-600">0%</div>
                </div>
            </div>

            <!-- Main Content -->
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <!-- Equity Chart -->
                <div class="lg:col-span-2 card rounded-lg p-4">
                    <h2 class="text-lg font-semibold mb-4">📈 Equity Curve</h2>
                    <canvas id="equityChart" height="200"></canvas>
                </div>

                <!-- Positions -->
                <div class="card rounded-lg p-4">
                    <div class="flex justify-between items-center mb-4">
                        <h2 class="text-lg font-semibold">📊 Open Positions</h2>
                        <button onclick="closeAllPositions()" class="text-xs bg-red-600 hover:bg-red-700 px-2 py-1 rounded text-gray-900">Close All</button>
                    </div>
                    <div id="positions" class="space-y-2 max-h-64 overflow-y-auto">
                        <p class="text-gray-500 text-sm">No open positions</p>
                    </div>
                </div>
            </div>

            <!-- Signals & Trades -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                <div class="card rounded-lg p-4">
                    <h2 class="text-lg font-semibold mb-4">⚡ Recent Signals</h2>
                    <div id="signals" class="space-y-2 max-h-64 overflow-y-auto">
                        <p class="text-gray-500 text-sm">Waiting for signals...</p>
                    </div>
                </div>
                <div class="card rounded-lg p-4">
                    <h2 class="text-lg font-semibold mb-4">💰 Recent Trades</h2>
                    <div id="trades" class="space-y-2 max-h-64 overflow-y-auto">
                        <p class="text-gray-500 text-sm">No trades yet</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Trading Tab -->
        <div id="tab-trading" class="tab-content hidden">
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <!-- Place Order Form -->
                <div class="card rounded-lg p-6">
                    <h2 class="text-lg font-semibold mb-4">📝 Place Order</h2>
                    <form id="orderForm" class="space-y-4">
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <label class="block text-sm text-gray-500 mb-1">Symbol</label>
                                <select id="orderSymbol" class="w-full bg-gray-100 rounded px-3 py-2">
                                    <option>BTCUSDT</option>
                                    <option>ETHUSDT</option>
                                    <option>SOLUSDT</option>
                                </select>
                            </div>
                            <div>
                                <label class="block text-sm text-gray-500 mb-1">Side</label>
                                <div class="flex gap-2">
                                    <button type="button" onclick="setSide('BUY')" id="buyBtn" class="flex-1 bg-green-600 hover:bg-green-700 py-2 rounded font-medium">BUY</button>
                                    <button type="button" onclick="setSide('SELL')" id="sellBtn" class="flex-1 bg-gray-100 hover:bg-red-600 py-2 rounded font-medium">SELL</button>
                                </div>
                            </div>
                        </div>
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <label class="block text-sm text-gray-500 mb-1">Order Type</label>
                                <select id="orderType" class="w-full bg-gray-100 rounded px-3 py-2" onchange="togglePriceField()">
                                    <option value="MARKET">Market</option>
                                    <option value="LIMIT">Limit</option>
                                </select>
                            </div>
                            <div>
                                <label class="block text-sm text-gray-500 mb-1">Quantity</label>
                                <input type="number" id="orderQty" step="0.0001" class="w-full bg-gray-100 rounded px-3 py-2" placeholder="0.001">
                            </div>
                        </div>
                        <div id="priceField" class="hidden">
                            <label class="block text-sm text-gray-500 mb-1">Price</label>
                            <input type="number" id="orderPrice" step="0.01" class="w-full bg-gray-100 rounded px-3 py-2" placeholder="50000">
                        </div>
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <label class="block text-sm text-gray-500 mb-1">Stop Loss (optional)</label>
                                <input type="number" id="orderSL" step="0.01" class="w-full bg-gray-100 rounded px-3 py-2" placeholder="49000">
                            </div>
                            <div>
                                <label class="block text-sm text-gray-500 mb-1">Take Profit (optional)</label>
                                <input type="number" id="orderTP" step="0.01" class="w-full bg-gray-100 rounded px-3 py-2" placeholder="51000">
                            </div>
                        </div>
                        <div>
                            <label class="block text-sm text-gray-500 mb-1">Leverage</label>
                            <input type="range" id="orderLeverage" min="1" max="50" value="10" class="w-full" oninput="updateLeverageLabel()">
                            <div class="text-center text-sm"><span id="leverageLabel">10</span>x</div>
                        </div>
                        <button type="submit" class="w-full bg-blue-600 hover:bg-blue-700 py-3 rounded font-medium">Place Order</button>
                    </form>
                </div>

                <!-- Quick Actions & Position Management -->
                <div class="space-y-6">
                    <!-- Set SL/TP -->
                    <div class="card rounded-lg p-6">
                        <h2 class="text-lg font-semibold mb-4">🎯 Set SL/TP</h2>
                        <div class="space-y-4">
                            <div>
                                <label class="block text-sm text-gray-500 mb-1">Symbol</label>
                                <select id="sltpSymbol" class="w-full bg-gray-100 rounded px-3 py-2">
                                    <option>BTCUSDT</option>
                                    <option>ETHUSDT</option>
                                </select>
                            </div>
                            <div class="grid grid-cols-2 gap-4">
                                <div>
                                    <label class="block text-sm text-gray-500 mb-1">Stop Loss</label>
                                    <input type="number" id="slPrice" step="0.01" class="w-full bg-gray-100 rounded px-3 py-2">
                                </div>
                                <div>
                                    <label class="block text-sm text-gray-500 mb-1">Take Profit</label>
                                    <input type="number" id="tpPrice" step="0.01" class="w-full bg-gray-100 rounded px-3 py-2">
                                </div>
                            </div>
                            <button onclick="setSLTP()" class="w-full bg-purple-600 hover:bg-purple-700 py-2 rounded">Update SL/TP</button>
                        </div>
                    </div>

                    <!-- Leverage Settings -->
                    <div class="card rounded-lg p-6">
                        <h2 class="text-lg font-semibold mb-4">⚡ Leverage</h2>
                        <div class="space-y-4">
                            <div>
                                <label class="block text-sm text-gray-500 mb-1">Symbol</label>
                                <select id="levSymbol" class="w-full bg-gray-100 rounded px-3 py-2">
                                    <option>BTCUSDT</option>
                                    <option>ETHUSDT</option>
                                </select>
                            </div>
                            <div>
                                <label class="block text-sm text-gray-500 mb-1">Leverage: <span id="levValue">10</span>x</label>
                                <input type="range" id="levSlider" min="1" max="125" value="10" class="w-full" oninput="document.getElementById('levValue').textContent=this.value">
                            </div>
                            <button onclick="setLeverage()" class="w-full bg-orange-600 hover:bg-orange-700 py-2 rounded">Set Leverage</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Strategies Tab -->
        <div id="tab-strategies" class="tab-content hidden">
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6" id="strategiesGrid">
                <!-- Strategies will be loaded here -->
            </div>
        </div>

        <!-- Settings Tab -->
        <div id="tab-settings" class="tab-content hidden">
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <!-- Risk Settings -->
                <div class="card rounded-lg p-6">
                    <h2 class="text-lg font-semibold mb-4">🛡️ Risk Management</h2>
                    <form id="riskForm" class="space-y-4">
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <label class="block text-sm text-gray-500 mb-1">Max Position ($)</label>
                                <input type="number" id="riskMaxPos" class="w-full bg-gray-100 rounded px-3 py-2" value="30">
                            </div>
                            <div>
                                <label class="block text-sm text-gray-500 mb-1">Max Daily Loss ($)</label>
                                <input type="number" id="riskMaxLoss" class="w-full bg-gray-100 rounded px-3 py-2" value="5">
                            </div>
                        </div>
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <label class="block text-sm text-gray-500 mb-1">Max Trades/Day</label>
                                <input type="number" id="riskMaxTrades" class="w-full bg-gray-100 rounded px-3 py-2" value="20">
                            </div>
                            <div>
                                <label class="block text-sm text-gray-500 mb-1">Max Open Positions</label>
                                <input type="number" id="riskMaxPositions" class="w-full bg-gray-100 rounded px-3 py-2" value="1">
                            </div>
                        </div>
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <label class="block text-sm text-gray-500 mb-1">Default SL (%)</label>
                                <input type="number" id="riskSL" step="0.1" class="w-full bg-gray-100 rounded px-3 py-2" value="0.5">
                            </div>
                            <div>
                                <label class="block text-sm text-gray-500 mb-1">Default TP (%)</label>
                                <input type="number" id="riskTP" step="0.1" class="w-full bg-gray-100 rounded px-3 py-2" value="0.3">
                            </div>
                        </div>
                        <div>
                            <label class="block text-sm text-gray-500 mb-1">Risk Per Trade (%)</label>
                            <input type="number" id="riskPerTrade" step="0.1" class="w-full bg-gray-100 rounded px-3 py-2" value="1">
                        </div>
                        <button type="submit" class="w-full bg-blue-600 hover:bg-blue-700 py-2 rounded">Save Risk Settings</button>
                    </form>
                </div>

                <!-- Exchange Settings -->
                <div class="card rounded-lg p-6">
                    <h2 class="text-lg font-semibold mb-4">🏦 Exchange</h2>
                    <div class="space-y-4">
                        <div>
                            <label class="block text-sm text-gray-500 mb-1">Exchange</label>
                            <select id="exchangeSelect" class="w-full bg-gray-100 rounded px-3 py-2">
                                <option value="binance">Binance</option>
                                <option value="bybit">Bybit</option>
                                <option value="okx">OKX</option>
                                <option value="kraken">Kraken</option>
                                <option value="kucoin">KuCoin</option>
                                <option value="gateio">Gate.io</option>
                            </select>
                        </div>
                        <div class="flex items-center gap-2">
                            <input type="checkbox" id="testnetMode" checked class="rounded">
                            <label for="testnetMode" class="text-sm">Use Testnet</label>
                        </div>
                        <button onclick="saveExchange()" class="w-full bg-blue-600 hover:bg-blue-700 py-2 rounded">Save Exchange</button>
                    </div>

                    <h3 class="text-md font-semibold mt-6 mb-4">📋 Trading Symbols</h3>
                    <div class="flex flex-wrap gap-2" id="symbolsList">
                        <span class="bg-blue-600 px-3 py-1 rounded-full text-sm">BTCUSDT</span>
                        <span class="bg-gray-100 px-3 py-1 rounded-full text-sm cursor-pointer hover:bg-blue-600">ETHUSDT</span>
                        <span class="bg-gray-100 px-3 py-1 rounded-full text-sm cursor-pointer hover:bg-blue-600">SOLUSDT</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Logs Tab -->
        <div id="tab-logs" class="tab-content hidden">
            <div class="card rounded-lg p-4">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-lg font-semibold">📋 System Logs</h2>
                    <div class="flex gap-2">
                        <select id="logFilter" class="bg-gray-100 rounded px-3 py-1 text-sm" onchange="loadLogs()">
                            <option value="">All Levels</option>
                            <option value="INFO">INFO</option>
                            <option value="WARNING">WARNING</option>
                            <option value="ERROR">ERROR</option>
                        </select>
                        <button onclick="clearLogs()" class="bg-red-600 hover:bg-red-700 px-3 py-1 rounded text-sm">Clear</button>
                        <button onclick="loadLogs()" class="bg-blue-600 hover:bg-blue-700 px-3 py-1 rounded text-sm">Refresh</button>
                    </div>
                </div>
                <div id="logsContainer" class="bg-gray-50 rounded p-4 h-96 overflow-y-auto font-mono text-sm">
                    <p class="text-gray-500">Loading logs...</p>
                </div>
            </div>
        </div>
    </div>

    <!-- Toast Container -->
    <div id="toastContainer"></div>

    <script>
        let ws;
        let equityChart;

        // Initialize WebSocket
        function connectWebSocket() {
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                console.log('WebSocket connected');
                ws.send(JSON.stringify({ type: 'subscribe', channels: ['all'] }));
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };

            ws.onclose = () => {
                console.log('WebSocket disconnected, reconnecting...');
                setTimeout(connectWebSocket, 3000);
            };
        }

        // Handle WebSocket messages
        function handleMessage(data) {
            switch (data.type) {
                case 'initial_state':
                    updateStatus(data.data.status);
                    updateMetrics(data.data.metrics);
                    updatePositions(data.data.positions);
                    break;
                case 'status_change':
                    updateStatus(data.status);
                    break;
                case 'metrics_update':
                    updateMetrics(data.data);
                    break;
                case 'position_update':
                    updatePositions(data.data);
                    break;
                case 'signal':
                    addSignal(data.data);
                    break;
                case 'trade':
                    addTrade(data.data);
                    break;
                case 'equity_update':
                    updateEquityChart(data.data);
                    break;
            }
        }

        // Update UI functions
        function updateStatus(status) {
            const statusEl = document.getElementById('status');
            statusEl.textContent = status.charAt(0).toUpperCase() + status.slice(1);
            statusEl.className = `text-lg font-medium status-${status}`;
        }

        function updateMetrics(metrics) {
            const pnl = metrics.total_pnl || 0;
            document.getElementById('totalPnl').textContent = `$${pnl.toFixed(2)}`;
            document.getElementById('totalPnl').className = `text-2xl font-bold ${pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}`;
            document.getElementById('winRate').textContent = `${(metrics.win_rate || 0).toFixed(1)}%`;
            document.getElementById('totalTrades').textContent = metrics.total_trades || 0;
            document.getElementById('profitFactor').textContent = (metrics.profit_factor || 0).toFixed(2);
        }

        function updatePositions(positions) {
            const container = document.getElementById('positions');
            if (!positions || positions.length === 0) {
                container.innerHTML = '<p class="text-gray-500">No open positions</p>';
                return;
            }

            container.innerHTML = positions.map(pos => `
                <div class="bg-gray-100 rounded p-2">
                    <div class="flex justify-between items-center">
                        <span class="font-medium">${pos.symbol}</span>
                        <span class="${pos.side === 'BUY' ? 'text-green-400' : 'text-red-400'}">${pos.side}</span>
                    </div>
                    <div class="flex justify-between text-sm text-gray-500">
                        <span>Size: ${pos.size}</span>
                        <span class="${pos.unrealized_pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">
                            $${pos.unrealized_pnl.toFixed(2)}
                        </span>
                    </div>
                </div>
            `).join('');
        }

        function addSignal(signal) {
            const container = document.getElementById('signals');
            if (container.querySelector('.text-gray-500')) {
                container.innerHTML = '';
            }

            const signalEl = document.createElement('div');
            signalEl.className = 'bg-gray-100 rounded p-2 text-sm';
            signalEl.innerHTML = `
                <div class="flex justify-between">
                    <span class="${signal.type === 'LONG' ? 'text-green-400' : 'text-red-400'}">${signal.type}</span>
                    <span>${signal.symbol}</span>
                </div>
                <div class="text-gray-500">${signal.strategy} (${(signal.strength * 100).toFixed(0)}%)</div>
            `;
            container.prepend(signalEl);

            // Keep only last 20 signals
            while (container.children.length > 20) {
                container.removeChild(container.lastChild);
            }
        }

        function addTrade(trade) {
            const container = document.getElementById('trades');
            if (container.querySelector('.text-gray-500')) {
                container.innerHTML = '';
            }

            const tradeEl = document.createElement('div');
            tradeEl.className = 'bg-gray-100 rounded p-2 text-sm';
            tradeEl.innerHTML = `
                <div class="flex justify-between">
                    <span>${trade.symbol}</span>
                    <span class="${trade.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">$${trade.pnl.toFixed(2)}</span>
                </div>
                <div class="text-gray-500">${trade.side} | ${trade.strategy}</div>
            `;
            container.prepend(tradeEl);

            while (container.children.length > 20) {
                container.removeChild(container.lastChild);
            }
        }

        // Initialize chart
        function initChart() {
            const ctx = document.getElementById('equityChart').getContext('2d');
            equityChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Equity',
                        data: [],
                        borderColor: '#22c55e',
                        backgroundColor: 'rgba(34, 197, 94, 0.1)',
                        fill: true,
                        tension: 0.4,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { display: false },
                        y: { grid: { color: '#374151' } }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
        }

        function updateEquityChart(data) {
            if (equityChart && data) {
                equityChart.data.labels.push(data.timestamp);
                equityChart.data.datasets[0].data.push(data.equity);

                // Keep last 100 points
                if (equityChart.data.labels.length > 100) {
                    equityChart.data.labels.shift();
                    equityChart.data.datasets[0].data.shift();
                }

                equityChart.update('none');
            }
        }

        // Send command to bot
        async function sendCommand(command) {
            try {
                const response = await fetch('/api/command', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command })
                });
                const data = await response.json();
                showToast(data.message, 'success');
            } catch (error) {
                showToast('Command failed: ' + error, 'error');
            }
        }

        // Fetch initial data
        async function fetchInitialData() {
            try {
                const metricsRes = await fetch('/api/metrics');
                const metrics = await metricsRes.json();
                updateMetrics(metrics);

                const positionsRes = await fetch('/api/positions');
                const positions = await positionsRes.json();
                updatePositions(positions);

                const statusRes = await fetch('/api/status');
                const status = await statusRes.json();
                updateStatus(status.status);

                loadStrategies();
                loadRiskSettings();
            } catch (error) {
                console.error('Failed to fetch initial data:', error);
            }
        }

        // Tab switching
        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
            document.querySelectorAll('.tab-btn').forEach(el => {
                el.classList.remove('tab-active');
                el.classList.add('text-gray-500');
            });
            document.getElementById('tab-' + tabName).classList.remove('hidden');
            document.querySelector(`[data-tab="${tabName}"]`).classList.add('tab-active');
            document.querySelector(`[data-tab="${tabName}"]`).classList.remove('text-gray-500');

            if (tabName === 'logs') loadLogs();
            if (tabName === 'strategies') loadStrategies();
        }

        // Toast notifications
        function showToast(message, type = 'success') {
            const container = document.getElementById('toastContainer');
            const toast = document.createElement('div');
            toast.className = `toast toast-${type}`;
            toast.textContent = message;
            container.appendChild(toast);
            setTimeout(() => toast.remove(), 3000);
        }

        // Order form state
        let orderSide = 'BUY';

        function setSide(side) {
            orderSide = side;
            document.getElementById('buyBtn').className = side === 'BUY'
                ? 'flex-1 bg-green-600 hover:bg-green-700 py-2 rounded font-medium'
                : 'flex-1 bg-gray-100 hover:bg-green-600 py-2 rounded font-medium';
            document.getElementById('sellBtn').className = side === 'SELL'
                ? 'flex-1 bg-red-600 hover:bg-red-700 py-2 rounded font-medium'
                : 'flex-1 bg-gray-100 hover:bg-red-600 py-2 rounded font-medium';
        }

        function togglePriceField() {
            const priceField = document.getElementById('priceField');
            const orderType = document.getElementById('orderType').value;
            priceField.classList.toggle('hidden', orderType === 'MARKET');
        }

        function updateLeverageLabel() {
            document.getElementById('leverageLabel').textContent = document.getElementById('orderLeverage').value;
        }

        // Place order
        document.getElementById('orderForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const order = {
                symbol: document.getElementById('orderSymbol').value,
                side: orderSide,
                quantity: parseFloat(document.getElementById('orderQty').value),
                order_type: document.getElementById('orderType').value,
                leverage: parseInt(document.getElementById('orderLeverage').value),
            };
            if (order.order_type === 'LIMIT') {
                order.price = parseFloat(document.getElementById('orderPrice').value);
            }
            const sl = document.getElementById('orderSL').value;
            const tp = document.getElementById('orderTP').value;
            if (sl) order.stop_loss = parseFloat(sl);
            if (tp) order.take_profit = parseFloat(tp);

            try {
                const res = await fetch('/api/order', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(order)
                });
                const data = await res.json();
                if (res.ok) {
                    showToast(`Order placed: ${order.side} ${order.quantity} ${order.symbol}`, 'success');
                } else {
                    showToast(data.detail || 'Order failed', 'error');
                }
            } catch (err) {
                showToast('Order failed: ' + err, 'error');
            }
        });

        // Set SL/TP
        async function setSLTP() {
            const symbol = document.getElementById('sltpSymbol').value;
            const sl = document.getElementById('slPrice').value;
            const tp = document.getElementById('tpPrice').value;
            try {
                const res = await fetch('/api/sl-tp', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol, stop_loss: sl ? parseFloat(sl) : null, take_profit: tp ? parseFloat(tp) : null })
                });
                const data = await res.json();
                showToast(data.message || 'SL/TP updated', res.ok ? 'success' : 'error');
            } catch (err) {
                showToast('Failed to set SL/TP', 'error');
            }
        }

        // Set Leverage
        async function setLeverage() {
            const symbol = document.getElementById('levSymbol').value;
            const leverage = parseInt(document.getElementById('levSlider').value);
            try {
                const res = await fetch('/api/leverage', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol, leverage })
                });
                const data = await res.json();
                showToast(data.message || 'Leverage updated', res.ok ? 'success' : 'error');
            } catch (err) {
                showToast('Failed to set leverage', 'error');
            }
        }

        // Close all positions
        async function closeAllPositions() {
            if (!confirm('Close ALL positions?')) return;
            const positions = await fetch('/api/positions').then(r => r.json());
            for (const pos of positions) {
                await fetch('/api/position/close', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol: pos.symbol })
                });
            }
            showToast('All positions closed', 'success');
        }

        // Strategies
        const scalping_strategies = ['hybrid_scalping', 'advanced_orderbook', 'print_tape', 'cluster_analysis', 'impulse_scalping'];
        const basic_strategies = ['orderbook_imbalance', 'volume_spike', 'mean_reversion', 'grid_trading', 'dca'];

        function getStrategyIcon(name) {
            const icons = {
                'hybrid_scalping': '🎯',
                'advanced_orderbook': '📊',
                'print_tape': '📜',
                'cluster_analysis': '🔬',
                'impulse_scalping': '⚡',
                'orderbook_imbalance': '📈',
                'volume_spike': '📢',
                'mean_reversion': '🔄',
                'grid_trading': '📐',
                'dca': '💰',
            };
            return icons[name] || '📌';
        }

        function formatStrategyName(name) {
            return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
        }

        async function loadStrategies() {
            try {
                const res = await fetch('/api/strategies');
                const strategies = await res.json();
                const grid = document.getElementById('strategiesGrid');

                // Group strategies
                const scalping = Object.entries(strategies).filter(([n]) => scalping_strategies.includes(n));
                const basic = Object.entries(strategies).filter(([n]) => basic_strategies.includes(n));

                let html = '';

                // Scalping section
                html += '<div class="col-span-full"><h2 class="text-xl font-bold mb-4 text-blue-400">🎯 Просунутий Скальпінг</h2></div>';
                html += scalping.map(([name, config]) => renderStrategyCard(name, config)).join('');

                // Basic section
                html += '<div class="col-span-full mt-6"><h2 class="text-xl font-bold mb-4 text-gray-500">📊 Базові Стратегії</h2></div>';
                html += basic.map(([name, config]) => renderStrategyCard(name, config)).join('');

                grid.innerHTML = html;
            } catch (err) {
                console.error('Failed to load strategies:', err);
            }
        }

        function renderStrategyCard(name, config) {
            const description = config.description || '';
            const hasWeight = config.weight !== undefined;

            return `
                <div class="card rounded-lg p-6 ${config.enabled ? 'border border-green-600/30' : ''}">
                    <div class="flex justify-between items-start mb-3">
                        <div>
                            <h3 class="text-lg font-semibold">${getStrategyIcon(name)} ${formatStrategyName(name)}</h3>
                            ${description ? `<p class="text-xs text-gray-500 mt-1">${description}</p>` : ''}
                        </div>
                        <button onclick="toggleStrategy('${name}')" class="px-3 py-1 rounded text-sm transition-all ${config.enabled ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-100 hover:bg-gray-600'}">
                            ${config.enabled ? '✓ Увімкнено' : 'Вимкнено'}
                        </button>
                    </div>
                    ${hasWeight ? `<div class="mb-3"><span class="text-xs bg-blue-600/30 text-blue-400 px-2 py-1 rounded">Вага: ${(config.weight * 100).toFixed(0)}%</span></div>` : ''}
                    <div class="space-y-1 text-sm text-gray-500">
                        ${Object.entries(config).filter(([k]) => !['enabled', 'description', 'weight'].includes(k)).slice(0, 5).map(([k, v]) => `
                            <div class="flex justify-between">
                                <span>${k.replace(/_/g, ' ')}:</span>
                                <span class="text-gray-900">${typeof v === 'boolean' ? (v ? '✓' : '✗') : v}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        async function toggleStrategy(name) {
            try {
                const res = await fetch(`/api/strategies/${name}/toggle`, { method: 'POST' });
                const data = await res.json();
                showToast(data.message, 'success');
                loadStrategies();
            } catch (err) {
                showToast('Failed to toggle strategy', 'error');
            }
        }

        // Risk settings
        async function loadRiskSettings() {
            try {
                const res = await fetch('/api/risk');
                const config = await res.json();
                document.getElementById('riskMaxPos').value = config.max_position_usd || 30;
                document.getElementById('riskMaxLoss').value = config.max_daily_loss_usd || 5;
                document.getElementById('riskMaxTrades').value = config.max_trades_per_day || 20;
                document.getElementById('riskMaxPositions').value = config.max_open_positions || 1;
                document.getElementById('riskSL').value = config.default_stop_loss_pct || 0.5;
                document.getElementById('riskTP').value = config.default_take_profit_pct || 0.3;
                document.getElementById('riskPerTrade').value = config.risk_per_trade_pct || 1;
            } catch (err) {
                console.error('Failed to load risk settings:', err);
            }
        }

        document.getElementById('riskForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const config = {
                max_position_usd: parseFloat(document.getElementById('riskMaxPos').value),
                max_daily_loss_usd: parseFloat(document.getElementById('riskMaxLoss').value),
                max_trades_per_day: parseInt(document.getElementById('riskMaxTrades').value),
                max_open_positions: parseInt(document.getElementById('riskMaxPositions').value),
                default_stop_loss_pct: parseFloat(document.getElementById('riskSL').value),
                default_take_profit_pct: parseFloat(document.getElementById('riskTP').value),
                risk_per_trade_pct: parseFloat(document.getElementById('riskPerTrade').value),
            };
            try {
                const res = await fetch('/api/risk', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });
                const data = await res.json();
                showToast(data.message || 'Risk settings saved', res.ok ? 'success' : 'error');
            } catch (err) {
                showToast('Failed to save risk settings', 'error');
            }
        });

        // Exchange
        async function saveExchange() {
            const exchange = document.getElementById('exchangeSelect').value;
            const testnet = document.getElementById('testnetMode').checked;
            try {
                const res = await fetch('/api/exchange', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ exchange, testnet })
                });
                const data = await res.json();
                showToast(data.message || 'Exchange saved', res.ok ? 'success' : 'error');
            } catch (err) {
                showToast('Failed to save exchange', 'error');
            }
        }

        // Logs
        async function loadLogs() {
            try {
                const level = document.getElementById('logFilter').value;
                const res = await fetch(`/api/logs?limit=200${level ? '&level=' + level : ''}`);
                const data = await res.json();
                const container = document.getElementById('logsContainer');
                if (data.logs.length === 0) {
                    container.innerHTML = '<p class="text-gray-500">No logs</p>';
                    return;
                }
                container.innerHTML = data.logs.map(log => {
                    const color = log.level === 'ERROR' ? 'text-red-400' : log.level === 'WARNING' ? 'text-yellow-400' : 'text-gray-300';
                    return `<div class="${color}">[${log.timestamp.slice(11, 19)}] [${log.level}] ${log.message}</div>`;
                }).join('');
                container.scrollTop = container.scrollHeight;
            } catch (err) {
                console.error('Failed to load logs:', err);
            }
        }

        async function clearLogs() {
            await fetch('/api/logs', { method: 'DELETE' });
            loadLogs();
            showToast('Logs cleared', 'success');
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            initChart();
            connectWebSocket();
            fetchInitialData();
        });
    </script>
</body>
</html>
"""


# =============================================================================
# Broadcast Functions (for integration with trading engine)
# =============================================================================

async def broadcast_signal(signal: Signal):
    """Broadcast signal to all connected clients."""
    await manager.broadcast({
        "type": "signal",
        "data": {
            "timestamp": signal.timestamp.isoformat(),
            "symbol": signal.symbol,
            "type": signal.signal_type.name,
            "strength": signal.strength,
            "price": float(signal.price),
            "strategy": signal.strategy,
        }
    })


async def broadcast_trade(trade: Dict):
    """Broadcast trade to all connected clients."""
    await manager.broadcast({
        "type": "trade",
        "data": trade,
    })


async def broadcast_position_update(positions: List[Position]):
    """Broadcast position update."""
    await manager.broadcast({
        "type": "position_update",
        "data": _serialize_positions(),
    })


async def broadcast_metrics_update(metrics: Dict):
    """Broadcast metrics update."""
    state.metrics.update(metrics)
    await manager.broadcast({
        "type": "metrics_update",
        "data": state.metrics,
    })


async def broadcast_equity_point(timestamp: datetime, equity: float):
    """Broadcast equity curve point."""
    point = {"timestamp": timestamp.isoformat(), "equity": equity}
    state.equity_curve.append(point)

    # Keep last 1000 points
    if len(state.equity_curve) > 1000:
        state.equity_curve = state.equity_curve[-1000:]

    await manager.broadcast({
        "type": "equity_update",
        "data": point,
    })


# =============================================================================
# Application Instance
# =============================================================================

app = create_app()
