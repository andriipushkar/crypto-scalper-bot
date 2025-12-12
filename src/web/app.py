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
from src.data.trade_history_db import trade_history_db
from src.web.dashboard_html import get_dashboard_html
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
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


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
        self.mode = "paper"  # paper, live, backtest
        self.selected_symbols = ["BTCUSDT"]  # Вибрані для торгівлі

        # Доступні символи з цінами
        self.available_symbols = {
            "BTCUSDT": {"name": "Bitcoin", "price": 97245.0, "change_24h": 2.34, "volume_24h": 45000000000},
            "ETHUSDT": {"name": "Ethereum", "price": 3642.0, "change_24h": 1.87, "volume_24h": 18000000000},
            "SOLUSDT": {"name": "Solana", "price": 227.0, "change_24h": -0.54, "volume_24h": 4500000000},
            "XRPUSDT": {"name": "Ripple", "price": 2.31, "change_24h": 5.21, "volume_24h": 8000000000},
            "DOGEUSDT": {"name": "Dogecoin", "price": 0.408, "change_24h": 3.12, "volume_24h": 2500000000},
            "BNBUSDT": {"name": "BNB", "price": 715.0, "change_24h": 1.45, "volume_24h": 1800000000},
            "ADAUSDT": {"name": "Cardano", "price": 1.05, "change_24h": 2.8, "volume_24h": 1200000000},
            "AVAXUSDT": {"name": "Avalanche", "price": 45.2, "change_24h": 4.1, "volume_24h": 900000000},
            "DOTUSDT": {"name": "Polkadot", "price": 8.45, "change_24h": 1.9, "volume_24h": 650000000},
            "LINKUSDT": {"name": "Chainlink", "price": 24.8, "change_24h": 3.5, "volume_24h": 780000000},
            "MATICUSDT": {"name": "Polygon", "price": 0.52, "change_24h": 2.1, "volume_24h": 420000000},
            "UNIUSDT": {"name": "Uniswap", "price": 14.2, "change_24h": 1.3, "volume_24h": 350000000},
        }

        # Real-time data
        self.positions: Dict[str, Position] = {}
        self.signals: List[Signal] = []
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.orders: Dict[str, Dict] = {}  # Open orders

        # Баланс
        self.balance = {
            "total": 1000.0,
            "available": 1000.0,
            "in_positions": 0.0,
            "unrealized_pnl": 0.0,
            "margin_used": 0.0,
        }

        # Metrics
        self.metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "daily_pnl": 0.0,
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

        # Доступні біржі
        self.available_exchanges = {
            "binance": {"name": "Binance", "testnet": True, "futures": True, "logo": "🟡"},
            "bybit": {"name": "Bybit", "testnet": True, "futures": True, "logo": "🟠"},
            "okx": {"name": "OKX", "testnet": True, "futures": True, "logo": "⚫"},
            "kraken": {"name": "Kraken", "testnet": False, "futures": True, "logo": "🟣"},
            "kucoin": {"name": "KuCoin", "testnet": True, "futures": True, "logo": "🟢"},
            "gateio": {"name": "Gate.io", "testnet": True, "futures": True, "logo": "🔵"},
        }

        # Вибрані біржі (може бути декілька)
        self.selected_exchanges = ["binance"]

        # Exchange configuration (legacy - для сумісності)
        self.exchange_config = {
            "exchange": "binance",
            "testnet": True,
        }

        # Backtest configuration
        self.backtest_config = {
            "start_date": "2024-01-01",
            "end_date": "2024-12-01",
            "initial_balance": 1000.0,
            "running": False,
            "progress": 0,
            "results": None,
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
        # Based on backtest results (2025-01-01 to 2025-12-12)
        self.strategies_config = {
            # ============================================
            # TOP PERFORMERS - ENABLED
            # ============================================
            "cluster_analysis": {
                "enabled": True,  # +162.6% return, Sharpe 0.96, MaxDD 14.6%
                "cluster_size_ticks": 10,
                "value_area_percent": 70,
                "imbalance_threshold": 65,
                "weight": 0.35,
                "description": "Кластерний аналіз (POC, Value Area, Delta)",
                "backtest_return": "+162.6%",
            },
            "mean_reversion": {
                "enabled": True,  # +107.1% return, Sharpe 0.68, MaxDD 15.7%
                "lookback_period": 20,
                "std_dev_multiplier": 2.0,
                "entry_z_score": 2.0,
                "description": "Mean Reversion (Z-score based)",
                "backtest_return": "+107.1%",
            },
            "volume_spike": {
                "enabled": True,  # +47.6% return, Sharpe 0.47, MaxDD 11.5%
                "volume_multiplier": 3.0,
                "lookback_seconds": 60,
                "min_volume_usd": 10000,
                "signal_cooldown": 15,
                "description": "Volume Spike Detection",
                "backtest_return": "+47.6%",
            },
            "orderbook_imbalance": {
                "enabled": True,  # +34.9% return, Sharpe 0.42, MaxDD 15.3%
                "imbalance_threshold": 1.5,
                "max_spread": 0.0005,
                "signal_cooldown": 10,
                "levels": 5,
                "min_volume_usd": 5000,
                "description": "Orderbook Imbalance",
                "backtest_return": "+34.9%",
            },
            "hybrid_scalping": {
                "enabled": True,  # +20.3% return, Sharpe 0.36, MaxDD 14.8%
                "min_confirmations": 2,
                "min_combined_weight": 0.5,
                "signal_cooldown": 30,
                "description": "Гібридна стратегія (комбінує всі скальпінг-методи)",
                "backtest_return": "+20.3%",
            },
            "print_tape": {
                "enabled": True,  # +18.5% return, Sharpe 0.33, MaxDD 16.1%
                "whale_threshold_usd": 50000,
                "cvd_window": 60,
                "delta_threshold": 60,
                "min_whale_trades": 3,
                "weight": 0.25,
                "description": "Лента принтів (CVD, whale trades)",
                "backtest_return": "+18.5%",
            },
            "order_flow": {
                "enabled": True,  # +9.1% return, Sharpe 0.32, MaxDD 25.1%
                "aggressive_threshold": 0.7,
                "volume_ratio_min": 1.5,
                "weight": 0.20,
                "description": "Order Flow Analysis",
                "backtest_return": "+9.1%",
            },
            # ============================================
            # OPTIMIZED V9 - HYBRID TA FILTERS + DYNAMIC BLACKLIST
            # ============================================
            "impulse_scalping": {
                "enabled": True,  # V9: +52.3% return, WR 44.7%, PF 1.30
                "macd_fast": 12,
                "macd_slow": 26,
                "trend_sma": 50,
                # Volatility filters
                "vol_filter_enabled": True,
                "vol_low_threshold": 0.8,  # Skip if ATR% < 0.8
                "vol_high_threshold": 2.5,
                # Technical Analysis filters
                "adx_min": 20,  # ADX > 20 for trending market
                "efficiency_ratio_min": 0.3,  # ER > 0.3 for clean trends
                # Entry filters
                "trend_strength_min": 1.0,  # Price > 1% from SMA
                "candle_body_ratio": 0.6,  # Body > 60% of range
                # Dynamic blacklist (poor historical performance)
                "blacklist": ["ATOMUSDT", "MKRUSDT", "THETAUSDT", "APEUSDT",
                              "XTZUSDT", "FILUSDT", "EOSUSDT"],
                "weight": 0.25,
                "description": "Volatility-Adaptive MACD with ADX/ER Filters (V9)",
                "backtest_return": "+52.3%",
                "backtest_winrate": "44.7%",
                "backtest_pf": "1.30",
                "backtest_trades": 1388,
            },
            # ============================================
            # UNPROFITABLE - DISABLED
            # ============================================
            "advanced_orderbook": {
                "enabled": False,  # -15.0% return, MaxDD 29.1%
                "wall_multiplier": 5.0,
                "min_wall_value_usd": 50000,
                "frontrunning_enabled": True,
                "spoofing_detection": True,
                "iceberg_detection": True,
                "weight": 0.35,
                "description": "Аналіз стакану (стіни, фронтраннінг, спуфінг)",
                "backtest_return": "-15.0%",
                "reason_disabled": "Negative return, needs V2 optimization",
            },
            "dca": {
                "enabled": False,  # -15.6% return, MaxDD 25.1%
                "mode": "hybrid",
                "interval_minutes": 60,
                "description": "Dollar Cost Averaging",
                "backtest_return": "-15.6%",
                "reason_disabled": "Loses money without trend filter",
            },
            "grid_trading": {
                "enabled": False,  # -61.1% return, MaxDD 105.5% - DANGEROUS
                "grid_levels": 10,
                "range_percent": 0.05,
                "description": "Grid Trading",
                "backtest_return": "-61.1%",
                "reason_disabled": "DANGEROUS - extreme drawdown in trending markets",
            },
            "dca_grid": {
                "enabled": False,  # -77.1% return, MaxDD 83.1% - DANGEROUS
                "dca_levels": 3,
                "grid_levels": 5,
                "description": "DCA + Grid Combined",
                "backtest_return": "-77.1%",
                "reason_disabled": "DANGEROUS - averaging down destroys capital",
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

# Initialize trade history database
trade_history_db.connect()

# Load trades from database into state
try:
    db_trades = trade_history_db.get_trades(limit=100)
    state.trades = [
        {
            'id': t.get('trade_id', ''),
            'symbol': t.get('symbol', ''),
            'side': t.get('side', ''),
            'entry_time': t.get('entry_time', ''),
            'exit_time': t.get('exit_time', ''),
            'entry_price': t.get('entry_price', 0),
            'exit_price': t.get('exit_price', 0),
            'quantity': t.get('quantity', 0),
            'pnl': t.get('pnl', 0),
            'pnl_pct': t.get('pnl_pct', 0),
            'strategy': t.get('strategy', 'manual'),
        }
        for t in db_trades
    ]
    # Update metrics from DB stats
    db_stats = trade_history_db.get_stats()
    state.metrics.update({
        'total_trades': db_stats.get('total_trades', 0),
        'winning_trades': db_stats.get('winning_trades', 0),
        'losing_trades': db_stats.get('losing_trades', 0),
        'win_rate': db_stats.get('win_rate', 0),
        'total_pnl': db_stats.get('total_pnl', 0),
        'gross_profit': db_stats.get('gross_profit', 0),
        'gross_loss': db_stats.get('gross_loss', 0),
        'profit_factor': db_stats.get('profit_factor', 0),
    })
    logger.info(f"Loaded {len(state.trades)} trades from database")
except Exception as e:
    logger.error(f"Failed to load trades from database: {e}")


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
                    "symbols": state.selected_symbols,
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
            active_symbols=state.selected_symbols,
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

    @app.get("/api/trades/history")
    async def get_trades_history(limit: int = Query(50, ge=1, le=500)):
        """Get trade history for dashboard (simplified format)."""
        trades = state.trades[-limit:] if state.trades else []
        # Reverse to get newest first
        trades = list(reversed(trades))
        return [
            {
                "symbol": t.get("symbol", ""),
                "side": t.get("side", ""),
                "entry_price": t.get("entry_price", 0),
                "exit_price": t.get("exit_price", 0),
                "pnl": t.get("pnl", 0),
                "pnl_pct": t.get("pnl_pct", 0),
                "strategy": t.get("strategy", "Manual"),
                "closed_at": t.get("exit_time", t.get("closed_at", "")),
            }
            for t in trades
        ]

    # =========================================================================
    # Database Trade History Endpoints
    # =========================================================================

    @app.get("/api/trades/db")
    async def get_trades_from_db(
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """Get trade history from database with filters."""
        trades = trade_history_db.get_trades(
            limit=limit,
            offset=offset,
            symbol=symbol,
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
        )
        total = trade_history_db.get_trade_count(symbol=symbol, strategy=strategy)
        return {
            "trades": trades,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    @app.get("/api/trades/stats")
    async def get_trading_stats(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """Get trading statistics from database."""
        return trade_history_db.get_stats(start_date=start_date, end_date=end_date)

    @app.get("/api/trades/daily")
    async def get_daily_pnl(days: int = Query(30, ge=1, le=365)):
        """Get daily P&L for last N days."""
        return trade_history_db.get_daily_pnl(days=days)

    @app.get("/api/trades/by-symbol")
    async def get_pnl_by_symbol():
        """Get P&L breakdown by symbol."""
        return trade_history_db.get_pnl_by_symbol()

    @app.get("/api/trades/by-strategy")
    async def get_pnl_by_strategy():
        """Get P&L breakdown by strategy."""
        return trade_history_db.get_pnl_by_strategy()

    @app.delete("/api/trades/db/{trade_id}")
    async def delete_trade_from_db(trade_id: str):
        """Delete a trade from database."""
        success = trade_history_db.delete_trade(trade_id)
        if success:
            return {"success": True, "message": f"Trade {trade_id} deleted"}
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")

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
            "symbols": state.selected_symbols,
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
                state.selected_symbols = value
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

        # Save to database
        try:
            trade_history_db.save_trade(trade)
            logger.info(f"Trade saved to DB: {trade.get('id')}")
        except Exception as e:
            logger.error(f"Failed to save trade to DB: {e}")

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

    @app.get("/api/exchanges")
    async def get_exchanges():
        """Get all exchanges with selection status."""
        return {
            "available": state.available_exchanges,
            "selected": state.selected_exchanges,
        }

    @app.post("/api/exchanges/select")
    async def select_exchanges(exchanges: List[str]):
        """Select exchanges for trading (can select multiple)."""
        available = list(state.available_exchanges.keys())
        invalid = [e for e in exchanges if e not in available]
        if invalid:
            raise HTTPException(status_code=400, detail=f"Unknown exchanges: {invalid}")

        if not exchanges:
            raise HTTPException(status_code=400, detail="At least one exchange must be selected")

        state.selected_exchanges = exchanges

        # Update legacy config with first selected exchange
        state.exchange_config["exchange"] = exchanges[0]

        state.add_log("INFO", f"Selected exchanges: {exchanges}", "config")

        return {"success": True, "selected": exchanges}

    @app.post("/api/exchanges/{exchange}/toggle")
    async def toggle_exchange(exchange: str):
        """Toggle exchange selection."""
        if exchange not in state.available_exchanges:
            raise HTTPException(status_code=404, detail=f"Exchange {exchange} not found")

        if exchange in state.selected_exchanges:
            if len(state.selected_exchanges) <= 1:
                raise HTTPException(status_code=400, detail="At least one exchange must be selected")
            state.selected_exchanges.remove(exchange)
        else:
            state.selected_exchanges.append(exchange)

        # Update legacy config
        state.exchange_config["exchange"] = state.selected_exchanges[0]

        return {
            "success": True,
            "exchange": exchange,
            "enabled": exchange in state.selected_exchanges,
            "selected": state.selected_exchanges,
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
    # Balance Endpoints
    # =========================================================================

    @app.get("/api/balance")
    async def get_balance():
        """Get account balance."""
        # Оновлюємо баланс на основі позицій
        total_in_positions = sum(
            float(pos.size) * float(pos.entry_price) / pos.leverage
            for pos in state.positions.values()
        )
        total_unrealized = sum(
            float(pos.unrealized_pnl)
            for pos in state.positions.values()
        )
        state.balance["in_positions"] = total_in_positions
        state.balance["unrealized_pnl"] = total_unrealized
        state.balance["available"] = state.balance["total"] - total_in_positions + state.metrics["total_pnl"]

        return state.balance

    @app.put("/api/balance")
    async def set_balance(amount: float = Query(..., gt=0)):
        """Set initial balance (for paper trading)."""
        if state.mode == "live":
            raise HTTPException(status_code=400, detail="Cannot set balance in live mode")

        state.balance["total"] = amount
        state.balance["available"] = amount
        state.add_log("INFO", f"Balance set to ${amount}", "balance")

        return {"success": True, "balance": state.balance}

    # =========================================================================
    # Symbols Management
    # =========================================================================

    @app.get("/api/symbols")
    async def get_symbols():
        """Get all available symbols with prices."""
        return {
            "available": state.available_symbols,
            "selected": state.selected_symbols,
        }

    @app.post("/api/symbols/select")
    async def select_symbols(symbols: List[str]):
        """Select symbols for trading."""
        # Валідуємо символи
        invalid = [s for s in symbols if s not in state.available_symbols]
        if invalid:
            raise HTTPException(status_code=400, detail=f"Invalid symbols: {invalid}")

        state.selected_symbols = symbols
        state.add_log("INFO", f"Selected symbols: {symbols}", "config")

        await manager.broadcast({
            "type": "symbols_update",
            "data": {"selected": symbols}
        })

        return {"success": True, "selected": symbols}

    @app.post("/api/symbols/add")
    async def add_symbol(symbol: str, name: str = "Unknown"):
        """Add a new symbol to available list."""
        symbol = symbol.upper()
        if symbol in state.available_symbols:
            raise HTTPException(status_code=400, detail=f"Symbol {symbol} already exists")

        state.available_symbols[symbol] = {
            "name": name,
            "price": 0.0,
            "change_24h": 0.0,
            "volume_24h": 0,
        }

        return {"success": True, "symbol": symbol}

    @app.delete("/api/symbols/{symbol}")
    async def remove_symbol(symbol: str):
        """Remove a symbol from available list."""
        symbol = symbol.upper()
        if symbol not in state.available_symbols:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

        if len(state.available_symbols) <= 1:
            raise HTTPException(status_code=400, detail="Cannot remove last symbol")

        # Remove from available
        del state.available_symbols[symbol]

        # Remove from selected if present
        if symbol in state.selected_symbols:
            state.selected_symbols.remove(symbol)
            # Ensure at least one symbol is selected
            if not state.selected_symbols and state.available_symbols:
                state.selected_symbols.append(list(state.available_symbols.keys())[0])

        return {"success": True, "symbol": symbol, "selected": state.selected_symbols}

    # =========================================================================
    # Trading Mode
    # =========================================================================

    @app.get("/api/mode")
    async def get_mode():
        """Get current trading mode."""
        return {
            "mode": state.mode,
            "available_modes": ["paper", "live", "backtest"],
            "backtest_config": state.backtest_config if state.mode == "backtest" else None,
        }

    @app.put("/api/mode")
    async def set_mode(mode: str):
        """Set trading mode: paper, live, or backtest."""
        if mode not in ["paper", "live", "backtest"]:
            raise HTTPException(status_code=400, detail="Mode must be: paper, live, or backtest")

        old_mode = state.mode
        state.mode = mode

        # Якщо перемикаємось на live - попередження
        if mode == "live" and old_mode != "live":
            state.add_log("WARNING", "Switched to LIVE trading mode!", "config")

        # Якщо перемикаємось на backtest - зупиняємо бота
        if mode == "backtest":
            state.status = "stopped"
            state.add_log("INFO", "Switched to backtest mode", "config")

        await manager.broadcast({
            "type": "mode_change",
            "data": {"mode": mode}
        })

        return {"success": True, "mode": mode}

    # =========================================================================
    # Backtest Endpoints
    # =========================================================================

    @app.get("/api/backtest/config")
    async def get_backtest_config():
        """Get backtest configuration."""
        return state.backtest_config

    @app.put("/api/backtest/config")
    async def update_backtest_config(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_balance: Optional[float] = None,
    ):
        """Update backtest configuration."""
        if start_date:
            state.backtest_config["start_date"] = start_date
        if end_date:
            state.backtest_config["end_date"] = end_date
        if initial_balance:
            state.backtest_config["initial_balance"] = initial_balance

        return {"success": True, "config": state.backtest_config}

    @app.post("/api/backtest/run")
    async def run_backtest():
        """Start backtest with current configuration using real historical data."""
        if state.mode != "backtest":
            raise HTTPException(status_code=400, detail="Switch to backtest mode first")

        if state.backtest_config["running"]:
            raise HTTPException(status_code=400, detail="Backtest already running")

        state.backtest_config["running"] = True
        state.backtest_config["progress"] = 0
        state.add_log("INFO", "Backtest started - downloading historical data...", "backtest")

        from src.backtesting.data import HistoricalDataLoader, OHLCV
        from src.backtesting.engine import BacktestEngine, BacktestConfig, OrderSide

        async def real_backtest():
            try:
                # Parse dates
                start_date = datetime.strptime(state.backtest_config["start_date"], "%Y-%m-%d")
                end_date = datetime.strptime(state.backtest_config["end_date"], "%Y-%m-%d")

                # Initialize data loader
                data_loader = HistoricalDataLoader()
                symbols = state.selected_symbols or ["BTCUSDT"]
                timeframe = "1h"  # 1-hour candles

                # Download data for all symbols
                all_data: Dict[str, List[OHLCV]] = {}
                total_symbols = len(symbols)

                for idx, symbol in enumerate(symbols):
                    state.backtest_config["progress"] = int((idx / total_symbols) * 30)
                    state.add_log("INFO", f"Downloading {symbol}...", "backtest")

                    await manager.broadcast({
                        "type": "backtest_progress",
                        "data": {"progress": state.backtest_config["progress"], "status": f"Downloading {symbol}..."}
                    })

                    try:
                        candles = await data_loader.get_data(
                            symbol=symbol,
                            timeframe=timeframe,
                            start_time=start_date,
                            end_time=end_date,
                            use_cache=True
                        )
                        if candles:
                            all_data[symbol] = candles
                            state.add_log("INFO", f"Downloaded {len(candles)} candles for {symbol}", "backtest")
                    except Exception as e:
                        state.add_log("WARNING", f"Failed to download {symbol}: {e}", "backtest")

                if not all_data:
                    raise ValueError("No data downloaded for any symbol")

                # Configure backtest engine
                config = BacktestConfig(
                    initial_balance=state.backtest_config["initial_balance"],
                    commission_rate=0.0004,  # 0.04% Binance futures fee
                    slippage=0.0001,  # 0.01%
                    leverage=10,
                )
                engine = BacktestEngine(config)

                # Simple scalping strategy
                def scalping_strategy(eng: BacktestEngine, symbol: str, candle: OHLCV, history: List[OHLCV]):
                    if len(history) < 20:
                        return

                    # Calculate simple indicators
                    closes = [c.close for c in history[-20:]]
                    sma_short = sum(closes[-5:]) / 5
                    sma_long = sum(closes[-20:]) / 20

                    current_price = candle.close
                    has_position = symbol in eng.positions

                    # Entry logic
                    if not has_position:
                        # Bullish crossover
                        if sma_short > sma_long * 1.001:  # 0.1% above
                            qty = (eng.get_available_balance() * 0.1) / current_price
                            if qty > 0:
                                order = eng.place_order(symbol, OrderSide.BUY, qty)
                                if order and order.filled:
                                    # Set SL/TP
                                    eng.set_stop_loss(symbol, current_price * 0.99)  # 1% SL
                                    eng.set_take_profit(symbol, current_price * 1.015)  # 1.5% TP
                        # Bearish crossover (short)
                        elif sma_short < sma_long * 0.999:  # 0.1% below
                            qty = (eng.get_available_balance() * 0.1) / current_price
                            if qty > 0 and config.allow_shorting:
                                order = eng.place_order(symbol, OrderSide.SELL, qty)
                                if order and order.filled:
                                    eng.set_stop_loss(symbol, current_price * 1.01)
                                    eng.set_take_profit(symbol, current_price * 0.985)

                state.add_log("INFO", "Running backtest...", "backtest")
                state.backtest_config["progress"] = 35

                # Run backtest
                result = engine.run(
                    strategy=scalping_strategy,
                    data=all_data,
                    warmup_period=50
                )

                state.backtest_config["progress"] = 95

                # Convert results
                trades_data = [
                    {
                        "id": t.id,
                        "symbol": t.symbol,
                        "side": t.side.upper(),
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "quantity": t.quantity,
                        "pnl": round(t.pnl, 2),
                        "pnl_pct": round(t.pnl_pct, 2),
                        "entry_time": t.entry_time.isoformat(),
                        "exit_time": t.exit_time.isoformat(),
                        "commission": round(t.commission, 4),
                    }
                    for t in result.trades
                ]

                equity_curve = [e["equity"] for e in result.equity_curve]

                state.backtest_config["results"] = {
                    "total_trades": result.total_trades,
                    "winning_trades": result.winning_trades,
                    "losing_trades": result.losing_trades,
                    "win_rate": round(result.win_rate, 2),
                    "total_pnl": round(result.total_return, 2),
                    "total_return_pct": round(result.total_return_pct, 2),
                    "profit_factor": round(result.profit_factor, 2) if result.profit_factor != float('inf') else 999,
                    "max_drawdown": round(result.max_drawdown, 2),
                    "max_drawdown_pct": round(result.max_drawdown_pct, 2),
                    "sharpe_ratio": round(result.sharpe_ratio, 2),
                    "avg_trade": round(result.avg_trade, 2),
                    "avg_win": round(result.avg_win, 2),
                    "avg_loss": round(result.avg_loss, 2),
                    "largest_win": round(result.largest_win, 2),
                    "largest_loss": round(result.largest_loss, 2),
                    "max_consecutive_wins": result.max_consecutive_wins,
                    "max_consecutive_losses": result.max_consecutive_losses,
                    "initial_balance": result.initial_balance,
                    "final_balance": round(result.final_balance, 2),
                    "start_date": result.start_time.isoformat(),
                    "end_date": result.end_time.isoformat(),
                    "symbols_tested": list(all_data.keys()),
                    "trades": trades_data[-100:],  # Last 100 trades
                    "equity_curve": equity_curve,
                }

                state.backtest_config["progress"] = 100
                state.backtest_config["running"] = False

                state.add_log(
                    "INFO",
                    f"Backtest completed: {result.total_trades} trades, "
                    f"P&L: ${result.total_return:.2f} ({result.total_return_pct:.1f}%), "
                    f"Win Rate: {result.win_rate:.1f}%",
                    "backtest"
                )

                await manager.broadcast({
                    "type": "backtest_complete",
                    "data": state.backtest_config["results"]
                })

            except Exception as e:
                state.backtest_config["running"] = False
                state.add_log("ERROR", f"Backtest failed: {str(e)}", "backtest")
                logger.exception("Backtest error")

                await manager.broadcast({
                    "type": "backtest_error",
                    "data": {"error": str(e)}
                })

        asyncio.create_task(real_backtest())

        return {"success": True, "message": "Backtest started - downloading historical data..."}

    @app.get("/api/backtest/results")
    async def get_backtest_results():
        """Get backtest results."""
        return {
            "running": state.backtest_config["running"],
            "progress": state.backtest_config["progress"],
            "results": state.backtest_config["results"],
        }

    @app.post("/api/backtest/stop")
    async def stop_backtest():
        """Stop running backtest."""
        state.backtest_config["running"] = False
        return {"success": True, "message": "Backtest stopped"}

    # =========================================================================
    # Full State Endpoint
    # =========================================================================

    @app.get("/api/state")
    async def get_full_state():
        """Get full dashboard state."""
        return {
            "status": state.status,
            "mode": state.mode,
            "exchange": state.exchange_config,
            "exchanges": {
                "available": state.available_exchanges,
                "selected": state.selected_exchanges,
            },
            "symbols": {
                "available": state.available_symbols,
                "selected": state.selected_symbols,
            },
            "balance": state.balance,
            "positions": _serialize_positions(),
            "metrics": state.metrics,
            "strategies": state.strategies_config,
            "risk": state.risk_config,
            "backtest": state.backtest_config,
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
            "size_usdt": float(pos.size) * float(pos.mark_price) if pos.mark_price else 0,
            "entry_price": float(pos.entry_price),
            "mark_price": float(pos.mark_price),
            "unrealized_pnl": float(pos.unrealized_pnl),
            "unrealized_pnl_pct": float(pos.unrealized_pnl / pos.entry_price * 100) if pos.entry_price else 0,
            "leverage": pos.leverage,
            "stop_loss": state.sl_tp.get(pos.symbol, {}).get("stop_loss"),
            "take_profit": state.sl_tp.get(pos.symbol, {}).get("take_profit"),
        }
        for pos in state.positions.values()
    ]


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
