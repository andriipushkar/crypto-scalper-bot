"""
Tests for WebSocket functionality in the web application.

Tests WebSocket connections, message handling, and broadcast functionality.
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import json
import os

from httpx import AsyncClient, ASGITransport

os.environ["ENVIRONMENT"] = "testnet"
os.environ["REQUIRE_AUTH"] = "false"


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

    # Reset state
    state.status = "stopped"
    state.mode = "paper"
    state.positions = {}
    state.signals = []
    state.trades = []
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
    state.prices = {"BTCUSDT": 50000.0, "ETHUSDT": 2500.0}

    # Clear connections
    manager.active_connections = []

    app = create_app(enable_auth=False, rate_limit_rpm=1000)
    yield app


# =============================================================================
# Connection Manager Tests
# =============================================================================

class TestConnectionManager:
    """Test ConnectionManager class."""

    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """Test manager initialization."""
        from src.web.app import ConnectionManager

        manager = ConnectionManager()
        assert manager.active_connections == []

    @pytest.mark.asyncio
    async def test_connect(self):
        """Test WebSocket connection."""
        from src.web.app import ConnectionManager

        manager = ConnectionManager()

        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()

        await manager.connect(mock_ws)

        mock_ws.accept.assert_called_once()
        assert mock_ws in manager.active_connections

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test WebSocket disconnection."""
        from src.web.app import ConnectionManager

        manager = ConnectionManager()

        mock_ws = AsyncMock()
        manager.active_connections.append(mock_ws)

        manager.disconnect(mock_ws)

        assert mock_ws not in manager.active_connections

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self):
        """Test disconnect for non-connected WebSocket."""
        from src.web.app import ConnectionManager

        manager = ConnectionManager()

        mock_ws = AsyncMock()
        # Should not raise error
        manager.disconnect(mock_ws)

    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Test broadcast to all connections."""
        from src.web.app import ConnectionManager

        manager = ConnectionManager()

        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()

        manager.active_connections = [mock_ws1, mock_ws2]

        message = {"type": "test", "data": "hello"}
        await manager.broadcast(message)

        mock_ws1.send_json.assert_called_once_with(message)
        mock_ws2.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_broadcast_with_error(self):
        """Test broadcast handles errors gracefully."""
        from src.web.app import ConnectionManager

        manager = ConnectionManager()

        mock_ws1 = AsyncMock()
        mock_ws1.send_json.side_effect = Exception("Connection error")
        mock_ws2 = AsyncMock()

        manager.active_connections = [mock_ws1, mock_ws2]

        message = {"type": "test", "data": "hello"}
        # Should not raise error
        await manager.broadcast(message)

        # ws2 should still receive the message
        mock_ws2.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_send_personal(self):
        """Test send to specific connection."""
        from src.web.app import ConnectionManager

        manager = ConnectionManager()

        mock_ws = AsyncMock()

        message = {"type": "personal", "data": "hello"}
        await manager.send_personal(mock_ws, message)

        mock_ws.send_json.assert_called_once_with(message)


# =============================================================================
# DashboardState Tests
# =============================================================================

class TestDashboardState:
    """Test DashboardState class."""

    def test_state_initialization(self):
        """Test state initialization."""
        from src.web.app import DashboardState

        state = DashboardState()

        assert state.status == "stopped"
        assert state.mode == "paper"
        assert state.positions == {}
        assert state.signals == []
        assert state.trades == []

    def test_uptime_seconds(self):
        """Test uptime calculation."""
        from src.web.app import DashboardState

        state = DashboardState()
        uptime = state.uptime_seconds

        assert uptime >= 0

    def test_add_log(self):
        """Test adding log entry."""
        from src.web.app import DashboardState

        state = DashboardState()
        state.add_log("INFO", "Test message", "test_module")

        assert len(state.logs) == 1
        assert state.logs[0]["level"] == "INFO"
        assert state.logs[0]["message"] == "Test message"
        assert state.logs[0]["module"] == "test_module"

    def test_add_log_max_limit(self):
        """Test log limit enforcement."""
        from src.web.app import DashboardState

        state = DashboardState()
        state.max_logs = 10

        for i in range(15):
            state.add_log("INFO", f"Message {i}")

        assert len(state.logs) == 10
        assert state.logs[0]["message"] == "Message 5"


# =============================================================================
# Broadcast Functions Tests
# =============================================================================

class TestBroadcastFunctions:
    """Test broadcast helper functions."""

    @pytest.mark.asyncio
    async def test_broadcast_signal(self):
        """Test broadcasting signal."""
        from src.web.app import broadcast_signal, manager
        from src.data.models import Signal, SignalType
        from decimal import Decimal
        from datetime import datetime

        mock_ws = AsyncMock()
        manager.active_connections = [mock_ws]

        signal = Signal(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            signal_type=SignalType.LONG,
            strength=0.8,
            price=Decimal("50000"),
            strategy="test_strategy"
        )

        await broadcast_signal(signal)

        mock_ws.send_json.assert_called_once()
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "signal"
        assert call_args["data"]["symbol"] == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_broadcast_trade(self):
        """Test broadcasting trade."""
        from src.web.app import broadcast_trade, manager

        mock_ws = AsyncMock()
        manager.active_connections = [mock_ws]

        trade = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "pnl": 50.0
        }

        await broadcast_trade(trade)

        mock_ws.send_json.assert_called_once()
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "trade"
        assert call_args["data"] == trade

    @pytest.mark.asyncio
    async def test_broadcast_position_update(self):
        """Test broadcasting position update."""
        from src.web.app import broadcast_position_update, manager

        mock_ws = AsyncMock()
        manager.active_connections = [mock_ws]

        await broadcast_position_update([])

        mock_ws.send_json.assert_called_once()
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "position_update"

    @pytest.mark.asyncio
    async def test_broadcast_metrics_update(self):
        """Test broadcasting metrics update."""
        from src.web.app import broadcast_metrics_update, manager, state

        mock_ws = AsyncMock()
        manager.active_connections = [mock_ws]

        metrics = {"total_pnl": 100.0}
        await broadcast_metrics_update(metrics)

        mock_ws.send_json.assert_called_once()
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "metrics_update"
        assert state.metrics["total_pnl"] == 100.0

    @pytest.mark.asyncio
    async def test_broadcast_equity_point(self):
        """Test broadcasting equity curve point."""
        from src.web.app import broadcast_equity_point, manager, state
        from datetime import datetime

        mock_ws = AsyncMock()
        manager.active_connections = [mock_ws]

        timestamp = datetime.now()
        await broadcast_equity_point(timestamp, 1000.0)

        mock_ws.send_json.assert_called_once()
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "equity_update"
        assert call_args["data"]["equity"] == 1000.0
        assert len(state.equity_curve) >= 1


# =============================================================================
# WebSocket Integration Tests
# =============================================================================

class TestWebSocketIntegration:
    """Integration tests for WebSocket functionality."""

    @pytest.mark.asyncio
    async def test_order_broadcasts_position_update(self, app):
        """Test that placing order broadcasts position update."""
        from src.web.app import manager

        mock_ws = AsyncMock()
        manager.active_connections = [mock_ws]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
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

        # Check that broadcasts were sent
        assert mock_ws.send_json.called

    @pytest.mark.asyncio
    async def test_close_position_broadcasts_updates(self, app):
        """Test that closing position broadcasts updates."""
        from src.web.app import manager, state
        from src.data.models import Position, Side
        from decimal import Decimal

        # Setup position
        from datetime import datetime
        state.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            side=Side.BUY,
            size=Decimal("0.001"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50500"),
            liquidation_price=None,
            unrealized_pnl=Decimal("0.50"),
            realized_pnl=Decimal("0"),
            leverage=10,
            margin_type="CROSSED",
            updated_at=datetime.utcnow(),
        )

        mock_ws = AsyncMock()
        manager.active_connections = [mock_ws]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/position/close",
                json={"symbol": "BTCUSDT"}
            )
            assert response.status_code == 200

        # Check that broadcasts were sent
        assert mock_ws.send_json.called

    @pytest.mark.asyncio
    async def test_command_broadcasts_status_change(self, app):
        """Test that bot commands broadcast status changes."""
        from src.web.app import manager

        mock_ws = AsyncMock()
        manager.active_connections = [mock_ws]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/command",
                json={"command": "start"}
            )
            assert response.status_code == 200

        # Check that status change was broadcast
        mock_ws.send_json.assert_called()
        calls = mock_ws.send_json.call_args_list
        status_calls = [c for c in calls if c[0][0].get("type") == "status_change"]
        assert len(status_calls) > 0


# =============================================================================
# Pydantic Models Tests
# =============================================================================

class TestPydanticModels:
    """Test Pydantic model definitions."""

    def test_status_response_model(self):
        """Test StatusResponse model."""
        from src.web.app import StatusResponse

        response = StatusResponse(
            status="running",
            uptime_seconds=3600.0,
            mode="paper",
            connected_exchanges=["binance"],
            active_symbols=["BTCUSDT"]
        )

        assert response.status == "running"
        assert response.uptime_seconds == 3600.0

    def test_position_response_model(self):
        """Test PositionResponse model."""
        from src.web.app import PositionResponse

        response = PositionResponse(
            symbol="BTCUSDT",
            side="BUY",
            size=0.001,
            entry_price=50000.0,
            mark_price=50500.0,
            unrealized_pnl=0.50,
            unrealized_pnl_pct=1.0,
            leverage=10
        )

        assert response.symbol == "BTCUSDT"
        assert response.unrealized_pnl == 0.50

    def test_trade_response_model(self):
        """Test TradeResponse model."""
        from src.web.app import TradeResponse

        response = TradeResponse(
            id="trade-1",
            symbol="BTCUSDT",
            side="BUY",
            entry_time="2024-01-01T00:00:00",
            exit_time="2024-01-01T01:00:00",
            entry_price=50000.0,
            exit_price=50500.0,
            quantity=0.001,
            pnl=0.50,
            pnl_pct=1.0,
            strategy="test"
        )

        assert response.pnl == 0.50

    def test_metrics_response_model(self):
        """Test MetricsResponse model."""
        from src.web.app import MetricsResponse

        response = MetricsResponse(
            total_trades=100,
            winning_trades=70,
            losing_trades=30,
            win_rate=70.0,
            total_pnl=500.0,
            gross_profit=700.0,
            gross_loss=200.0,
            profit_factor=3.5,
            sharpe_ratio=2.0,
            sortino_ratio=2.5,
            max_drawdown=100.0,
            max_drawdown_pct=5.0
        )

        assert response.win_rate == 70.0
        assert response.profit_factor == 3.5

    def test_place_order_request_model(self):
        """Test PlaceOrderRequest model."""
        from src.web.app import PlaceOrderRequest

        request = PlaceOrderRequest(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.001,
            order_type="MARKET",
            stop_loss=49000.0,
            take_profit=52000.0
        )

        assert request.symbol == "BTCUSDT"
        assert request.stop_loss == 49000.0

    def test_close_position_request_model(self):
        """Test ClosePositionRequest model."""
        from src.web.app import ClosePositionRequest

        request = ClosePositionRequest(symbol="BTCUSDT", quantity=0.0005)

        assert request.symbol == "BTCUSDT"
        assert request.quantity == 0.0005

    def test_set_leverage_request_model(self):
        """Test SetLeverageRequest model."""
        from src.web.app import SetLeverageRequest

        request = SetLeverageRequest(symbol="BTCUSDT", leverage=20)

        assert request.leverage == 20

    def test_set_sltp_request_model(self):
        """Test SetSLTPRequest model."""
        from src.web.app import SetSLTPRequest

        request = SetSLTPRequest(
            symbol="BTCUSDT",
            stop_loss=49000.0,
            take_profit=52000.0
        )

        assert request.stop_loss == 49000.0
        assert request.take_profit == 52000.0

    def test_strategy_config_request_model(self):
        """Test StrategyConfigRequest model."""
        from src.web.app import StrategyConfigRequest

        request = StrategyConfigRequest(
            strategy_name="orderbook_imbalance",
            enabled=True,
            params={"threshold": 1.5}
        )

        assert request.strategy_name == "orderbook_imbalance"
        assert request.enabled is True

    def test_risk_config_request_model(self):
        """Test RiskConfigRequest model."""
        from src.web.app import RiskConfigRequest

        request = RiskConfigRequest(
            max_position_usd=100.0,
            max_daily_loss_usd=20.0,
            max_trades_per_day=50
        )

        assert request.max_position_usd == 100.0

    def test_exchange_config_request_model(self):
        """Test ExchangeConfigRequest model."""
        from src.web.app import ExchangeConfigRequest

        request = ExchangeConfigRequest(exchange="bybit", testnet=True)

        assert request.exchange == "bybit"
        assert request.testnet is True

    def test_order_response_model(self):
        """Test OrderResponse model."""
        from src.web.app import OrderResponse

        response = OrderResponse(
            order_id="order-1",
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.001,
            price=50000.0,
            order_type="MARKET",
            status="FILLED",
            created_at="2024-01-01T00:00:00"
        )

        assert response.status == "FILLED"
        assert response.order_type == "MARKET"

    def test_log_entry_model(self):
        """Test LogEntry model."""
        from src.web.app import LogEntry

        entry = LogEntry(
            timestamp="2024-01-01T00:00:00",
            level="INFO",
            message="Test message",
            module="test"
        )

        assert entry.level == "INFO"

    def test_login_request_model(self):
        """Test LoginRequest model."""
        from src.web.app import LoginRequest

        request = LoginRequest(username="admin", password="secret")

        assert request.username == "admin"

    def test_token_response_model(self):
        """Test TokenResponse model."""
        from src.web.app import TokenResponse

        response = TokenResponse(
            access_token="token123",
            refresh_token="refresh123",
            expires_in=3600
        )

        assert response.token_type == "bearer"


# =============================================================================
# Helper Functions Tests
# =============================================================================

class TestHelperFunctions:
    """Test helper functions."""

    def test_serialize_positions_empty(self):
        """Test serializing empty positions."""
        from src.web.app import _serialize_positions, state

        state.positions = {}
        result = _serialize_positions()

        assert result == []

    def test_serialize_positions_with_data(self):
        """Test serializing positions with data."""
        from src.web.app import _serialize_positions, state
        from src.data.models import Position, Side
        from decimal import Decimal
        from datetime import datetime

        state.positions = {
            "BTCUSDT": Position(
                symbol="BTCUSDT",
                side=Side.BUY,
                size=Decimal("0.001"),
                entry_price=Decimal("50000"),
                mark_price=Decimal("50500"),
                liquidation_price=None,
                unrealized_pnl=Decimal("0.50"),
                realized_pnl=Decimal("0"),
                leverage=10,
                margin_type="CROSSED",
                updated_at=datetime.utcnow(),
            )
        }

        result = _serialize_positions()

        assert len(result) == 1
        assert result[0]["symbol"] == "BTCUSDT"
        assert result[0]["side"] == "BUY"
        assert result[0]["size"] == 0.001

    def test_get_dashboard_html(self):
        """Test getting dashboard HTML."""
        from src.web.app import get_dashboard_html

        html = get_dashboard_html()

        assert "<!DOCTYPE html>" in html
        assert "Crypto Scalper Dashboard" in html
        assert "WebSocket" in html


# =============================================================================
# App Factory Tests
# =============================================================================

class TestAppFactory:
    """Test application factory."""

    def test_create_app_default(self):
        """Test creating app with defaults."""
        from src.web.app import create_app

        app = create_app()

        assert app is not None
        assert app.title == "Crypto Scalper Bot"

    def test_create_app_no_auth(self):
        """Test creating app without authentication."""
        from src.web.app import create_app

        app = create_app(enable_auth=False)

        assert app is not None

    def test_create_app_custom_rate_limit(self):
        """Test creating app with custom rate limit."""
        from src.web.app import create_app

        app = create_app(rate_limit_rpm=60)

        assert app is not None

    def test_app_has_required_routes(self):
        """Test app has all required routes."""
        from src.web.app import create_app

        app = create_app()

        routes = [route.path for route in app.routes]

        assert "/health" in routes
        assert "/api/status" in routes
        assert "/api/command" in routes
        assert "/api/order" in routes
        assert "/api/positions" in routes
        assert "/api/leverage" in routes
        assert "/api/sl-tp" in routes
        assert "/api/strategies" in routes
        assert "/api/risk" in routes
        assert "/api/exchange" in routes
        assert "/api/logs" in routes
        assert "/api/alerts" in routes


# =============================================================================
# Metrics Calculation Tests
# =============================================================================

class TestMetricsCalculation:
    """Test metrics calculation in position closing."""

    @pytest.mark.asyncio
    async def test_metrics_update_on_winning_trade(self, app):
        """Test metrics update on winning trade."""
        from src.web.app import state
        from src.data.models import Position, Side
        from decimal import Decimal
        from datetime import datetime

        # Setup position with profit potential
        state.positions["BTCUSDT"] = Position(
            symbol="BTCUSDT",
            side=Side.BUY,
            size=Decimal("0.001"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("51000"),  # Price went up
            liquidation_price=None,
            unrealized_pnl=Decimal("1.0"),
            realized_pnl=Decimal("0"),
            leverage=10,
            margin_type="CROSSED",
            updated_at=datetime.utcnow(),
        )
        state.prices["BTCUSDT"] = 51000.0  # Current price higher
        state.metrics["total_trades"] = 0
        state.metrics["winning_trades"] = 0

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/position/close",
                json={"symbol": "BTCUSDT"}
            )
            assert response.status_code == 200
            data = response.json()

            # PnL should be positive (bought at 50000, sold at 51000)
            assert data["pnl"] > 0

        # Metrics should be updated
        assert state.metrics["total_trades"] == 1
        assert state.metrics["winning_trades"] == 1

    @pytest.mark.asyncio
    async def test_metrics_update_on_losing_trade(self, app):
        """Test metrics update on losing trade."""
        from src.web.app import state
        from src.data.models import Position, Side
        from decimal import Decimal
        from datetime import datetime

        # Setup position with loss potential
        state.positions["ETHUSDT"] = Position(
            symbol="ETHUSDT",
            side=Side.BUY,
            size=Decimal("0.01"),
            entry_price=Decimal("2500"),
            mark_price=Decimal("2400"),  # Price went down
            liquidation_price=None,
            unrealized_pnl=Decimal("-1.0"),
            realized_pnl=Decimal("0"),
            leverage=10,
            margin_type="CROSSED",
            updated_at=datetime.utcnow(),
        )
        state.prices["ETHUSDT"] = 2400.0  # Current price lower
        state.metrics["total_trades"] = 0
        state.metrics["losing_trades"] = 0

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/position/close",
                json={"symbol": "ETHUSDT"}
            )
            assert response.status_code == 200
            data = response.json()

            # PnL should be negative
            assert data["pnl"] < 0

        # Metrics should be updated
        assert state.metrics["total_trades"] == 1
        assert state.metrics["losing_trades"] == 1


# =============================================================================
# Strategy Config Storage Tests
# =============================================================================

class TestStrategyConfigStorage:
    """Test strategy configuration storage."""

    @pytest.mark.asyncio
    async def test_strategy_toggle_persists(self, app):
        """Test that strategy toggle persists in state."""
        from src.web.app import state

        initial_state = state.strategies_config["mean_reversion"]["enabled"]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/strategies/mean_reversion/toggle")
            assert response.status_code == 200

        # State should be toggled
        assert state.strategies_config["mean_reversion"]["enabled"] != initial_state

    @pytest.mark.asyncio
    async def test_strategy_params_update(self, app):
        """Test strategy parameters update."""
        from src.web.app import state

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.put(
                "/api/strategies",
                json={
                    "strategy_name": "orderbook_imbalance",
                    "enabled": True,
                    "params": {"imbalance_threshold": 3.0, "levels": 10}
                }
            )
            assert response.status_code == 200

        # Params should be updated
        assert state.strategies_config["orderbook_imbalance"]["imbalance_threshold"] == 3.0
        assert state.strategies_config["orderbook_imbalance"]["levels"] == 10
