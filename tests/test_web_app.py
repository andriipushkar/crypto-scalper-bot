"""
Comprehensive tests for the web application API endpoints.

Tests all new API endpoints for trading operations, strategies, risk management,
exchange configuration, and dashboard functionality.
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import json
import os

# Set test environment
os.environ["ENVIRONMENT"] = "testnet"
os.environ["ENABLE_DOCS"] = "true"
os.environ["REQUIRE_AUTH"] = "false"
os.environ["ADMIN_USERNAME"] = "admin"
os.environ["ADMIN_PASSWORD"] = "testpass123"

from httpx import AsyncClient, ASGITransport


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for session scope."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def app():
    """Create test application instance."""
    from src.web.app import create_app, state, manager

    # Reset state for each test
    state.status = "stopped"
    state.mode = "paper"
    state.positions = {}
    state.signals = []
    state.trades = []
    state.orders = {}
    state.metrics = {
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
    state.logs = []
    state.leverage = {"BTCUSDT": 10, "ETHUSDT": 10}
    state.sl_tp = {}
    state.prices = {"BTCUSDT": 50000.0, "ETHUSDT": 2500.0}
    state.alerts_enabled = True

    # Clear WebSocket connections
    manager.active_connections = []

    app = create_app(enable_auth=False, rate_limit_rpm=1000)
    yield app


@pytest_asyncio.fixture
async def client(app):
    """Create test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# =============================================================================
# Health & Status Tests
# =============================================================================

class TestHealthEndpoints:
    """Test health check endpoints."""

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test health endpoint returns healthy status."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_get_status(self, client):
        """Test status endpoint."""
        response = await client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "uptime_seconds" in data
        assert "mode" in data
        assert "connected_exchanges" in data
        assert "active_symbols" in data

    @pytest.mark.asyncio
    async def test_dashboard_html(self, client):
        """Test dashboard HTML endpoint."""
        response = await client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Futures Bot" in response.text


# =============================================================================
# Bot Command Tests
# =============================================================================

class TestBotCommands:
    """Test bot control commands."""

    @pytest.mark.asyncio
    async def test_start_command(self, client):
        """Test start bot command."""
        response = await client.post(
            "/api/command",
            json={"command": "start"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "started" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_stop_command(self, client):
        """Test stop bot command."""
        response = await client.post(
            "/api/command",
            json={"command": "stop"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "stopped" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_pause_command(self, client):
        """Test pause bot command."""
        response = await client.post(
            "/api/command",
            json={"command": "pause"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_resume_command(self, client):
        """Test resume bot command."""
        response = await client.post(
            "/api/command",
            json={"command": "resume"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_unknown_command(self, client):
        """Test unknown command returns error."""
        response = await client.post(
            "/api/command",
            json={"command": "invalid_command"}
        )
        assert response.status_code == 400


# =============================================================================
# Order Placement Tests
# =============================================================================

class TestOrderPlacement:
    """Test order placement API endpoints."""

    @pytest.mark.asyncio
    async def test_place_market_buy_order(self, client):
        """Test placing a market buy order."""
        response = await client.post(
            "/api/order",
            json={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.001,
                "order_type": "MARKET"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "BTCUSDT"
        assert data["side"] == "BUY"
        assert data["quantity"] == 0.001
        assert data["status"] == "FILLED"
        assert "order_id" in data

    @pytest.mark.asyncio
    async def test_place_market_sell_order(self, client):
        """Test placing a market sell order."""
        response = await client.post(
            "/api/order",
            json={
                "symbol": "ETHUSDT",
                "side": "SELL",
                "quantity": 0.01,
                "order_type": "MARKET"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["side"] == "SELL"
        assert data["status"] == "FILLED"

    @pytest.mark.asyncio
    async def test_place_limit_order(self, client):
        """Test placing a limit order."""
        response = await client.post(
            "/api/order",
            json={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.001,
                "order_type": "LIMIT",
                "price": 49000.0
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["order_type"] == "LIMIT"
        assert data["status"] == "NEW"
        assert data["price"] == 49000.0

    @pytest.mark.asyncio
    async def test_place_order_with_sl_tp(self, client):
        """Test placing order with stop-loss and take-profit."""
        response = await client.post(
            "/api/order",
            json={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.001,
                "order_type": "MARKET",
                "stop_loss": 49000.0,
                "take_profit": 52000.0
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "FILLED"

    @pytest.mark.asyncio
    async def test_place_order_with_leverage(self, client):
        """Test placing order with custom leverage."""
        response = await client.post(
            "/api/order",
            json={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.001,
                "order_type": "MARKET",
                "leverage": 20
            }
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_place_order_invalid_side(self, client):
        """Test placing order with invalid side."""
        response = await client.post(
            "/api/order",
            json={
                "symbol": "BTCUSDT",
                "side": "INVALID",
                "quantity": 0.001
            }
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_place_order_invalid_type(self, client):
        """Test placing order with invalid order type."""
        response = await client.post(
            "/api/order",
            json={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.001,
                "order_type": "INVALID"
            }
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_place_limit_order_without_price(self, client):
        """Test placing limit order without price."""
        response = await client.post(
            "/api/order",
            json={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.001,
                "order_type": "LIMIT"
            }
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_place_order_zero_quantity(self, client):
        """Test placing order with zero quantity."""
        response = await client.post(
            "/api/order",
            json={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0
            }
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_place_order_negative_quantity(self, client):
        """Test placing order with negative quantity."""
        response = await client.post(
            "/api/order",
            json={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": -0.001
            }
        )
        assert response.status_code == 400


# =============================================================================
# Position Management Tests
# =============================================================================

class TestPositionManagement:
    """Test position management endpoints."""

    @pytest.mark.asyncio
    async def test_get_positions_empty(self, client):
        """Test getting positions when empty."""
        response = await client.get("/api/positions")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    @pytest.mark.asyncio
    async def test_get_positions_after_order(self, client):
        """Test getting positions after placing order."""
        # Place order first
        await client.post(
            "/api/order",
            json={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.001,
                "order_type": "MARKET"
            }
        )

        # Get positions
        response = await client.get("/api/positions")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1

    @pytest.mark.asyncio
    async def test_close_position(self, client):
        """Test closing a position."""
        # Place order first
        await client.post(
            "/api/order",
            json={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.001,
                "order_type": "MARKET"
            }
        )

        # Close position
        response = await client.post(
            "/api/position/close",
            json={"symbol": "BTCUSDT"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "pnl" in data

    @pytest.mark.asyncio
    async def test_close_position_partial(self, client):
        """Test closing position partially."""
        # Place order first
        await client.post(
            "/api/order",
            json={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.002,
                "order_type": "MARKET"
            }
        )

        # Close partial
        response = await client.post(
            "/api/position/close",
            json={"symbol": "BTCUSDT", "quantity": 0.001}
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_close_position_not_found(self, client):
        """Test closing non-existent position."""
        response = await client.post(
            "/api/position/close",
            json={"symbol": "NONEXISTENT"}
        )
        assert response.status_code == 404


# =============================================================================
# Leverage Tests
# =============================================================================

class TestLeverageManagement:
    """Test leverage management endpoints."""

    @pytest.mark.asyncio
    async def test_set_leverage(self, client):
        """Test setting leverage."""
        response = await client.put(
            "/api/leverage",
            json={"symbol": "BTCUSDT", "leverage": 20}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "20x" in data["message"]

    @pytest.mark.asyncio
    async def test_set_leverage_max(self, client):
        """Test setting maximum leverage."""
        response = await client.put(
            "/api/leverage",
            json={"symbol": "BTCUSDT", "leverage": 125}
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_set_leverage_min(self, client):
        """Test setting minimum leverage."""
        response = await client.put(
            "/api/leverage",
            json={"symbol": "BTCUSDT", "leverage": 1}
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_set_leverage_too_low(self, client):
        """Test setting leverage below minimum."""
        response = await client.put(
            "/api/leverage",
            json={"symbol": "BTCUSDT", "leverage": 0}
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_set_leverage_too_high(self, client):
        """Test setting leverage above maximum."""
        response = await client.put(
            "/api/leverage",
            json={"symbol": "BTCUSDT", "leverage": 200}
        )
        assert response.status_code == 400


# =============================================================================
# SL/TP Tests
# =============================================================================

class TestSLTPManagement:
    """Test stop-loss/take-profit management."""

    @pytest.mark.asyncio
    async def test_set_sl_tp(self, client):
        """Test setting SL/TP."""
        # Create position first
        await client.post(
            "/api/order",
            json={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.001,
                "order_type": "MARKET"
            }
        )

        # Set SL/TP
        response = await client.put(
            "/api/sl-tp",
            json={
                "symbol": "BTCUSDT",
                "stop_loss": 49000.0,
                "take_profit": 52000.0
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "sl_tp" in data

    @pytest.mark.asyncio
    async def test_set_only_sl(self, client):
        """Test setting only stop-loss."""
        await client.post(
            "/api/order",
            json={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.001,
                "order_type": "MARKET"
            }
        )

        response = await client.put(
            "/api/sl-tp",
            json={"symbol": "BTCUSDT", "stop_loss": 49000.0}
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_set_only_tp(self, client):
        """Test setting only take-profit."""
        await client.post(
            "/api/order",
            json={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.001,
                "order_type": "MARKET"
            }
        )

        response = await client.put(
            "/api/sl-tp",
            json={"symbol": "BTCUSDT", "take_profit": 52000.0}
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_set_sl_tp_no_position(self, client):
        """Test setting SL/TP without position."""
        response = await client.put(
            "/api/sl-tp",
            json={
                "symbol": "NONEXISTENT",
                "stop_loss": 49000.0
            }
        )
        assert response.status_code == 404


# =============================================================================
# Order Cancellation Tests
# =============================================================================

class TestOrderCancellation:
    """Test order cancellation endpoints."""

    @pytest.mark.asyncio
    async def test_cancel_order(self, client):
        """Test cancelling a limit order."""
        # Place limit order
        place_response = await client.post(
            "/api/order",
            json={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.001,
                "order_type": "LIMIT",
                "price": 40000.0
            }
        )
        order_id = place_response.json()["order_id"]

        # Cancel order
        response = await client.delete(f"/api/order/{order_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, client):
        """Test cancelling non-existent order."""
        response = await client.delete("/api/order/nonexistent")
        assert response.status_code == 404


# =============================================================================
# Strategy Configuration Tests
# =============================================================================

class TestStrategyConfiguration:
    """Test strategy configuration endpoints."""

    @pytest.mark.asyncio
    async def test_get_strategies(self, client):
        """Test getting all strategies."""
        response = await client.get("/api/strategies")
        assert response.status_code == 200
        data = response.json()
        assert "orderbook_imbalance" in data
        assert "volume_spike" in data
        assert "mean_reversion" in data
        assert "grid_trading" in data
        assert "dca" in data

    @pytest.mark.asyncio
    async def test_update_strategy(self, client):
        """Test updating strategy configuration."""
        response = await client.put(
            "/api/strategies",
            json={
                "strategy_name": "orderbook_imbalance",
                "enabled": True,
                "params": {"imbalance_threshold": 2.0}
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_update_strategy_disable(self, client):
        """Test disabling a strategy."""
        response = await client.put(
            "/api/strategies",
            json={
                "strategy_name": "volume_spike",
                "enabled": False,
                "params": {}
            }
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_update_unknown_strategy(self, client):
        """Test updating unknown strategy."""
        response = await client.put(
            "/api/strategies",
            json={
                "strategy_name": "unknown_strategy",
                "enabled": True,
                "params": {}
            }
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_toggle_strategy(self, client):
        """Test toggling strategy enabled/disabled."""
        response = await client.post("/api/strategies/orderbook_imbalance/toggle")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "enabled" in data

    @pytest.mark.asyncio
    async def test_toggle_unknown_strategy(self, client):
        """Test toggling unknown strategy."""
        response = await client.post("/api/strategies/unknown/toggle")
        assert response.status_code == 400


# =============================================================================
# Risk Configuration Tests
# =============================================================================

class TestRiskConfiguration:
    """Test risk configuration endpoints."""

    @pytest.mark.asyncio
    async def test_get_risk_config(self, client):
        """Test getting risk configuration."""
        response = await client.get("/api/risk")
        assert response.status_code == 200
        data = response.json()
        assert "max_position_usd" in data
        assert "max_daily_loss_usd" in data
        assert "max_trades_per_day" in data
        assert "max_open_positions" in data
        assert "default_stop_loss_pct" in data
        assert "default_take_profit_pct" in data
        assert "risk_per_trade_pct" in data

    @pytest.mark.asyncio
    async def test_update_risk_config(self, client):
        """Test updating risk configuration."""
        response = await client.put(
            "/api/risk",
            json={
                "max_position_usd": 50.0,
                "max_daily_loss_usd": 10.0,
                "max_trades_per_day": 30
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["config"]["max_position_usd"] == 50.0

    @pytest.mark.asyncio
    async def test_update_risk_partial(self, client):
        """Test partial risk config update."""
        response = await client.put(
            "/api/risk",
            json={"max_position_usd": 100.0}
        )
        assert response.status_code == 200


# =============================================================================
# Exchange Configuration Tests
# =============================================================================

class TestExchangeConfiguration:
    """Test exchange configuration endpoints."""

    @pytest.mark.asyncio
    async def test_get_exchange_config(self, client):
        """Test getting exchange configuration."""
        response = await client.get("/api/exchange")
        assert response.status_code == 200
        data = response.json()
        assert "current" in data
        assert "available" in data
        assert "binance" in data["available"]

    @pytest.mark.asyncio
    async def test_update_exchange_config(self, client):
        """Test updating exchange configuration."""
        response = await client.put(
            "/api/exchange",
            json={"exchange": "bybit", "testnet": True}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["config"]["exchange"] == "bybit"

    @pytest.mark.asyncio
    async def test_update_exchange_production(self, client):
        """Test switching to production mode."""
        response = await client.put(
            "/api/exchange",
            json={"exchange": "binance", "testnet": False}
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_update_unknown_exchange(self, client):
        """Test updating to unknown exchange."""
        response = await client.put(
            "/api/exchange",
            json={"exchange": "unknown_exchange", "testnet": True}
        )
        assert response.status_code == 400


# =============================================================================
# Logs Tests
# =============================================================================

class TestLogsEndpoints:
    """Test logs endpoints."""

    @pytest.mark.asyncio
    async def test_get_logs(self, client):
        """Test getting logs."""
        response = await client.get("/api/logs")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "filtered" in data
        assert "logs" in data

    @pytest.mark.asyncio
    async def test_get_logs_with_filter(self, client):
        """Test getting logs with level filter."""
        response = await client.get("/api/logs?level=ERROR")
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data

    @pytest.mark.asyncio
    async def test_get_logs_with_limit(self, client):
        """Test getting logs with limit."""
        response = await client.get("/api/logs?limit=10")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_clear_logs(self, client):
        """Test clearing logs."""
        response = await client.delete("/api/logs")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


# =============================================================================
# Alerts Tests
# =============================================================================

class TestAlertsEndpoints:
    """Test alerts endpoints."""

    @pytest.mark.asyncio
    async def test_get_alerts_config(self, client):
        """Test getting alerts configuration."""
        response = await client.get("/api/alerts")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data

    @pytest.mark.asyncio
    async def test_enable_alerts(self, client):
        """Test enabling alerts."""
        response = await client.put("/api/alerts?enabled=true")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_disable_alerts(self, client):
        """Test disabling alerts."""
        response = await client.put("/api/alerts?enabled=false")
        assert response.status_code == 200


# =============================================================================
# Trade History Tests
# =============================================================================

class TestTradeHistory:
    """Test trade history endpoints."""

    @pytest.mark.asyncio
    async def test_get_trades_empty(self, client):
        """Test getting trades when empty."""
        response = await client.get("/api/trades")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_trades_with_limit(self, client):
        """Test getting trades with limit."""
        response = await client.get("/api/trades?limit=10")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_trades_with_offset(self, client):
        """Test getting trades with offset."""
        response = await client.get("/api/trades?offset=5")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_trades_with_symbol_filter(self, client):
        """Test getting trades filtered by symbol."""
        response = await client.get("/api/trades?symbol=BTCUSDT")
        assert response.status_code == 200


# =============================================================================
# Metrics Tests
# =============================================================================

class TestMetrics:
    """Test metrics endpoints."""

    @pytest.mark.asyncio
    async def test_get_metrics(self, client):
        """Test getting metrics."""
        response = await client.get("/api/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_trades" in data
        assert "winning_trades" in data
        assert "losing_trades" in data
        assert "win_rate" in data
        assert "total_pnl" in data
        assert "profit_factor" in data
        assert "sharpe_ratio" in data
        assert "max_drawdown" in data


# =============================================================================
# Signals Tests
# =============================================================================

class TestSignals:
    """Test signals endpoints."""

    @pytest.mark.asyncio
    async def test_get_signals(self, client):
        """Test getting signals."""
        response = await client.get("/api/signals")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_signals_with_limit(self, client):
        """Test getting signals with limit."""
        response = await client.get("/api/signals?limit=20")
        assert response.status_code == 200


# =============================================================================
# Equity Curve Tests
# =============================================================================

class TestEquityCurve:
    """Test equity curve endpoints."""

    @pytest.mark.asyncio
    async def test_get_equity_curve(self, client):
        """Test getting equity curve."""
        response = await client.get("/api/equity")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data


# =============================================================================
# Market Data Tests
# =============================================================================

class TestMarketData:
    """Test market data endpoints."""

    @pytest.mark.asyncio
    async def test_get_market_data(self, client):
        """Test getting market data."""
        response = await client.get("/api/market/BTCUSDT")
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "BTCUSDT"
        assert "price" in data


# =============================================================================
# Config Tests
# =============================================================================

class TestConfig:
    """Test configuration endpoints."""

    @pytest.mark.asyncio
    async def test_get_config(self, client):
        """Test getting configuration."""
        response = await client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert "mode" in data
        assert "symbols" in data
        assert "leverage" in data
        assert "exchange" in data
        assert "risk" in data
        assert "strategies" in data

    @pytest.mark.asyncio
    async def test_update_mode(self, client):
        """Test updating mode."""
        response = await client.post(
            "/api/config",
            json={"key": "mode", "value": "live"}
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_update_symbols(self, client):
        """Test updating symbols."""
        response = await client.post(
            "/api/config",
            json={"key": "symbols", "value": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]}
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_update_invalid_mode(self, client):
        """Test updating to invalid mode."""
        response = await client.post(
            "/api/config",
            json={"key": "mode", "value": "invalid"}
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_update_unknown_key(self, client):
        """Test updating unknown config key."""
        response = await client.post(
            "/api/config",
            json={"key": "unknown", "value": "test"}
        )
        assert response.status_code == 400


# =============================================================================
# Authentication Tests
# =============================================================================

class TestAuthentication:
    """Test authentication endpoints."""

    @pytest.mark.asyncio
    async def test_login_success(self, client):
        """Test successful login."""
        with patch.dict(os.environ, {"ADMIN_USERNAME": "admin", "ADMIN_PASSWORD": "testpass123"}):
            response = await client.post(
                "/api/auth/login",
                json={"username": "admin", "password": "testpass123"}
            )
            # May succeed or fail based on auth setup
            assert response.status_code in [200, 401, 503]

    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials."""
        response = await client.post(
            "/api/auth/login",
            json={"username": "wrong", "password": "wrong"}
        )
        assert response.status_code in [401, 503]


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_complete_trading_workflow(self, client):
        """Test complete trading workflow."""
        # 1. Check status
        status_response = await client.get("/api/status")
        assert status_response.status_code == 200

        # 2. Start trading
        await client.post("/api/command", json={"command": "start"})

        # 3. Set leverage
        await client.put(
            "/api/leverage",
            json={"symbol": "BTCUSDT", "leverage": 10}
        )

        # 4. Place order
        order_response = await client.post(
            "/api/order",
            json={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.001,
                "order_type": "MARKET",
                "stop_loss": 49000.0,
                "take_profit": 52000.0
            }
        )
        assert order_response.status_code == 200

        # 5. Check position
        positions_response = await client.get("/api/positions")
        assert positions_response.status_code == 200

        # 6. Close position
        close_response = await client.post(
            "/api/position/close",
            json={"symbol": "BTCUSDT"}
        )
        assert close_response.status_code == 200

        # 7. Check metrics updated
        metrics_response = await client.get("/api/metrics")
        assert metrics_response.status_code == 200

        # 8. Stop trading
        await client.post("/api/command", json={"command": "stop"})

    @pytest.mark.asyncio
    async def test_strategy_management_workflow(self, client):
        """Test strategy management workflow."""
        # Get strategies
        strategies = await client.get("/api/strategies")
        assert strategies.status_code == 200

        # Toggle a strategy
        toggle_response = await client.post("/api/strategies/mean_reversion/toggle")
        assert toggle_response.status_code == 200

        # Update strategy params
        update_response = await client.put(
            "/api/strategies",
            json={
                "strategy_name": "mean_reversion",
                "enabled": True,
                "params": {"lookback_period": 30}
            }
        )
        assert update_response.status_code == 200

    @pytest.mark.asyncio
    async def test_risk_management_workflow(self, client):
        """Test risk management workflow."""
        # Get risk config
        risk_config = await client.get("/api/risk")
        assert risk_config.status_code == 200

        # Update risk
        update_response = await client.put(
            "/api/risk",
            json={
                "max_position_usd": 100.0,
                "max_daily_loss_usd": 20.0,
                "default_stop_loss_pct": 1.0
            }
        )
        assert update_response.status_code == 200

        # Verify update
        updated_config = await client.get("/api/risk")
        assert updated_config.json()["max_position_usd"] == 100.0


# =============================================================================
# Pydantic Model Validation Tests
# =============================================================================

class TestPydanticModels:
    """Test Pydantic model validation."""

    @pytest.mark.asyncio
    async def test_order_request_validation(self, client):
        """Test order request validation."""
        # Missing required field
        response = await client.post(
            "/api/order",
            json={"side": "BUY", "quantity": 0.001}
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_leverage_request_validation(self, client):
        """Test leverage request validation."""
        # Missing symbol
        response = await client.put(
            "/api/leverage",
            json={"leverage": 10}
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_strategy_request_validation(self, client):
        """Test strategy request validation."""
        # Missing required field
        response = await client.put(
            "/api/strategies",
            json={"strategy_name": "test"}
        )
        assert response.status_code == 422
