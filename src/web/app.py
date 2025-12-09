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

    @property
    def uptime_seconds(self) -> float:
        return (datetime.utcnow() - self.start_time).total_seconds()


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
            "leverage": 10,
            "max_position_pct": 0.5,
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
        .status-running { color: #22c55e; }
        .status-stopped { color: #ef4444; }
        .status-paused { color: #f59e0b; }
        .pnl-positive { color: #22c55e; }
        .pnl-negative { color: #ef4444; }
    </style>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <div class="container mx-auto px-4 py-6">
        <!-- Header -->
        <header class="flex justify-between items-center mb-8">
            <h1 class="text-2xl font-bold">Crypto Scalper Bot</h1>
            <div class="flex items-center gap-4">
                <span id="status" class="text-lg font-medium">--</span>
                <button id="startBtn" onclick="sendCommand('start')" class="bg-green-600 hover:bg-green-700 px-4 py-2 rounded">Start</button>
                <button id="stopBtn" onclick="sendCommand('stop')" class="bg-red-600 hover:bg-red-700 px-4 py-2 rounded">Stop</button>
            </div>
        </header>

        <!-- Metrics Cards -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div class="bg-gray-800 rounded-lg p-4">
                <div class="text-gray-400 text-sm">Total P&L</div>
                <div id="totalPnl" class="text-2xl font-bold">$0.00</div>
            </div>
            <div class="bg-gray-800 rounded-lg p-4">
                <div class="text-gray-400 text-sm">Win Rate</div>
                <div id="winRate" class="text-2xl font-bold">0%</div>
            </div>
            <div class="bg-gray-800 rounded-lg p-4">
                <div class="text-gray-400 text-sm">Total Trades</div>
                <div id="totalTrades" class="text-2xl font-bold">0</div>
            </div>
            <div class="bg-gray-800 rounded-lg p-4">
                <div class="text-gray-400 text-sm">Profit Factor</div>
                <div id="profitFactor" class="text-2xl font-bold">0.00</div>
            </div>
        </div>

        <!-- Main Content -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Equity Chart -->
            <div class="lg:col-span-2 bg-gray-800 rounded-lg p-4">
                <h2 class="text-lg font-semibold mb-4">Equity Curve</h2>
                <canvas id="equityChart" height="200"></canvas>
            </div>

            <!-- Positions -->
            <div class="bg-gray-800 rounded-lg p-4">
                <h2 class="text-lg font-semibold mb-4">Open Positions</h2>
                <div id="positions" class="space-y-2">
                    <p class="text-gray-400">No open positions</p>
                </div>
            </div>
        </div>

        <!-- Signals & Trades -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
            <!-- Recent Signals -->
            <div class="bg-gray-800 rounded-lg p-4">
                <h2 class="text-lg font-semibold mb-4">Recent Signals</h2>
                <div id="signals" class="space-y-2 max-h-64 overflow-y-auto">
                    <p class="text-gray-400">Waiting for signals...</p>
                </div>
            </div>

            <!-- Recent Trades -->
            <div class="bg-gray-800 rounded-lg p-4">
                <h2 class="text-lg font-semibold mb-4">Recent Trades</h2>
                <div id="trades" class="space-y-2 max-h-64 overflow-y-auto">
                    <p class="text-gray-400">No trades yet</p>
                </div>
            </div>
        </div>
    </div>

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
                container.innerHTML = '<p class="text-gray-400">No open positions</p>';
                return;
            }

            container.innerHTML = positions.map(pos => `
                <div class="bg-gray-700 rounded p-2">
                    <div class="flex justify-between items-center">
                        <span class="font-medium">${pos.symbol}</span>
                        <span class="${pos.side === 'BUY' ? 'text-green-400' : 'text-red-400'}">${pos.side}</span>
                    </div>
                    <div class="flex justify-between text-sm text-gray-400">
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
            if (container.querySelector('.text-gray-400')) {
                container.innerHTML = '';
            }

            const signalEl = document.createElement('div');
            signalEl.className = 'bg-gray-700 rounded p-2 text-sm';
            signalEl.innerHTML = `
                <div class="flex justify-between">
                    <span class="${signal.type === 'LONG' ? 'text-green-400' : 'text-red-400'}">${signal.type}</span>
                    <span>${signal.symbol}</span>
                </div>
                <div class="text-gray-400">${signal.strategy} (${(signal.strength * 100).toFixed(0)}%)</div>
            `;
            container.prepend(signalEl);

            // Keep only last 20 signals
            while (container.children.length > 20) {
                container.removeChild(container.lastChild);
            }
        }

        function addTrade(trade) {
            const container = document.getElementById('trades');
            if (container.querySelector('.text-gray-400')) {
                container.innerHTML = '';
            }

            const tradeEl = document.createElement('div');
            tradeEl.className = 'bg-gray-700 rounded p-2 text-sm';
            tradeEl.innerHTML = `
                <div class="flex justify-between">
                    <span>${trade.symbol}</span>
                    <span class="${trade.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">$${trade.pnl.toFixed(2)}</span>
                </div>
                <div class="text-gray-400">${trade.side} | ${trade.strategy}</div>
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
                console.log(data.message);
            } catch (error) {
                console.error('Command failed:', error);
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
            } catch (error) {
                console.error('Failed to fetch initial data:', error);
            }
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
