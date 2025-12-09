"""
Comprehensive tests for Telegram Bot Controller.

Tests all commands, authentication, notifications, and integration with trading system.
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import os

# Set test environment
os.environ["TELEGRAM_BOT_TOKEN"] = "test_token"
os.environ["TELEGRAM_ALLOWED_USERS"] = "123456,789012"
os.environ["TELEGRAM_ADMIN_USERS"] = "123456"

from src.notifications.telegram_bot import (
    TelegramBotController,
    TelegramBotConfig,
    BotState,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def config():
    """Create test configuration."""
    return TelegramBotConfig(
        bot_token="test_token_12345",
        allowed_users=[123456, 789012],
        admin_users=[123456],
        enable_commands=True,
    )


@pytest.fixture
def mock_callbacks():
    """Create mock callbacks for bot controller."""
    return {
        "get_status": lambda: {
            "state": "running",
            "paused": False,
            "uptime": "1h 30m",
            "strategies": 2,
            "exchange": "Binance",
        },
        "get_balance": AsyncMock(return_value={
            "wallet_balance": 1000.0,
            "available_balance": 950.0,
            "margin_balance": 50.0,
            "unrealized_pnl": 25.0,
        }),
        "get_positions": AsyncMock(return_value=[
            {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.001,
                "entry_price": 50000.0,
                "mark_price": 50500.0,
                "unrealized_pnl": 0.50,
            },
        ]),
        "get_orders": AsyncMock(return_value=[
            {
                "order_id": "test-order-1",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "quantity": 0.001,
                "price": 49000.0,
            },
        ]),
        "get_trades": AsyncMock(return_value=[
            {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.001,
                "price": 50000.0,
                "realized_pnl": 10.0,
            },
        ]),
        "get_pnl": lambda: {
            "total_pnl": 150.0,
            "daily_pnl": 25.0,
            "trades": 10,
            "win_rate": 70.0,
            "profit_factor": 2.5,
            "winning_trades": 7,
            "losing_trades": 3,
            "max_drawdown": 5.0,
            "sharpe_ratio": 1.8,
        },
        "get_risk": lambda: {
            "drawdown": 3.5,
            "exposure": 25.0,
            "leverage_used": 5.0,
            "available_margin": 900.0,
        },
        "start_trading": AsyncMock(return_value=True),
        "stop_trading": AsyncMock(return_value=True),
        "close_position": AsyncMock(return_value=True),
        "cancel_order": AsyncMock(return_value=True),
        "set_leverage": AsyncMock(return_value=True),
    }


@pytest_asyncio.fixture
async def bot_controller(config, mock_callbacks):
    """Create bot controller with mocks."""
    controller = TelegramBotController(
        config=config,
        **mock_callbacks
    )
    # Mock API call
    controller._api_call = AsyncMock(return_value={"ok": True})
    controller.send_message = AsyncMock(return_value=True)

    # Set up additional mock callbacks for new commands
    controller._place_order = AsyncMock(return_value={"success": True})
    controller._set_sl_tp = AsyncMock(return_value=True)
    controller._get_daily_stats = lambda: {
        "date": "2024-01-01",
        "trades": 5,
        "pnl": 50.0,
        "win_rate": 80.0,
        "best_trade": 30.0,
        "worst_trade": -10.0,
    }
    controller._get_strategies = lambda: {
        "orderbook_imbalance": {"enabled": True},
        "volume_spike": {"enabled": True},
        "mean_reversion": {"enabled": False},
    }
    controller._toggle_strategy = AsyncMock(return_value=True)
    controller._close_all_positions = AsyncMock(return_value=2)

    yield controller


# =============================================================================
# Configuration Tests
# =============================================================================

class TestBotConfiguration:
    """Test bot configuration."""

    def test_config_creation(self, config):
        """Test configuration creation."""
        assert config.bot_token == "test_token_12345"
        assert 123456 in config.allowed_users
        assert 123456 in config.admin_users

    def test_is_configured(self, config):
        """Test is_configured property."""
        controller = TelegramBotController(config=config)
        assert controller.is_configured is True

    def test_not_configured(self):
        """Test not configured when no token."""
        # Pass empty config to override environment variables
        empty_config = TelegramBotConfig(
            bot_token="",
            allowed_users=[],
            admin_users=[],
        )
        controller = TelegramBotController(config=empty_config)
        assert controller.is_configured is False

    def test_parse_user_ids(self, config):
        """Test parsing user IDs from string."""
        controller = TelegramBotController(config=config)
        ids = controller._parse_user_ids("123,456,789")
        assert ids == [123, 456, 789]

    def test_parse_user_ids_empty(self, config):
        """Test parsing empty user IDs."""
        controller = TelegramBotController(config=config)
        ids = controller._parse_user_ids("")
        assert ids == []

    def test_parse_user_ids_invalid(self, config):
        """Test parsing invalid user IDs."""
        controller = TelegramBotController(config=config)
        ids = controller._parse_user_ids("abc,def")
        assert ids == []


# =============================================================================
# Authorization Tests
# =============================================================================

class TestAuthorization:
    """Test user authorization."""

    @pytest.mark.asyncio
    async def test_authorized_user(self, bot_controller):
        """Test authorized user."""
        assert bot_controller._is_authorized(123456) is True
        assert bot_controller._is_authorized(789012) is True

    @pytest.mark.asyncio
    async def test_unauthorized_user(self, bot_controller):
        """Test unauthorized user."""
        assert bot_controller._is_authorized(999999) is False

    @pytest.mark.asyncio
    async def test_is_admin(self, bot_controller):
        """Test admin user."""
        assert bot_controller._is_admin(123456) is True
        assert bot_controller._is_admin(789012) is False


# =============================================================================
# Basic Command Tests
# =============================================================================

class TestBasicCommands:
    """Test basic commands."""

    @pytest.mark.asyncio
    async def test_start_command(self, bot_controller):
        """Test /start command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_start(ctx)
        assert "Welcome" in response

    @pytest.mark.asyncio
    async def test_help_command(self, bot_controller):
        """Test /help command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_help(ctx)
        assert "Available Commands" in response
        assert "/status" in response
        assert "/positions" in response

    @pytest.mark.asyncio
    async def test_status_command(self, bot_controller):
        """Test /status command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_status(ctx)
        assert "Bot Status" in response
        assert "RUNNING" in response

    @pytest.mark.asyncio
    async def test_balance_command(self, bot_controller):
        """Test /balance command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_balance(ctx)
        assert "Account Balance" in response
        assert "$1,000.00" in response

    @pytest.mark.asyncio
    async def test_positions_command(self, bot_controller):
        """Test /positions command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_positions(ctx)
        assert "Open Positions" in response
        assert "BTCUSDT" in response

    @pytest.mark.asyncio
    async def test_positions_empty(self, bot_controller):
        """Test /positions with no positions."""
        bot_controller._get_positions = AsyncMock(return_value=[])
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_positions(ctx)
        assert "No open positions" in response

    @pytest.mark.asyncio
    async def test_orders_command(self, bot_controller):
        """Test /orders command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_orders(ctx)
        assert "Open Orders" in response

    @pytest.mark.asyncio
    async def test_orders_empty(self, bot_controller):
        """Test /orders with no orders."""
        bot_controller._get_orders = AsyncMock(return_value=[])
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_orders(ctx)
        assert "No open orders" in response

    @pytest.mark.asyncio
    async def test_pnl_command(self, bot_controller):
        """Test /pnl command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_pnl(ctx)
        assert "Profit & Loss" in response
        assert "150.00" in response

    @pytest.mark.asyncio
    async def test_trades_command(self, bot_controller):
        """Test /trades command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_trades(ctx)
        assert "Recent Trades" in response

    @pytest.mark.asyncio
    async def test_trades_with_limit(self, bot_controller):
        """Test /trades with limit."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["10"], "is_admin": True}
        response = await bot_controller._cmd_trades(ctx)
        assert "Recent Trades" in response

    @pytest.mark.asyncio
    async def test_risk_command(self, bot_controller):
        """Test /risk command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_risk(ctx)
        assert "Risk Metrics" in response
        assert "Drawdown" in response


# =============================================================================
# Admin Control Commands Tests
# =============================================================================

class TestAdminCommands:
    """Test admin control commands."""

    @pytest.mark.asyncio
    async def test_start_trading_command(self, bot_controller):
        """Test /start_trading command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_start_trading(ctx)
        assert "started" in response.lower()
        assert bot_controller._state == BotState.RUNNING

    @pytest.mark.asyncio
    async def test_start_trading_no_permission(self, bot_controller):
        """Test /start_trading without admin permission."""
        ctx = {"chat_id": 789012, "user_id": 789012, "args": [], "is_admin": False}
        response = await bot_controller._cmd_start_trading(ctx)
        assert "Admin permission required" in response

    @pytest.mark.asyncio
    async def test_stop_trading_command(self, bot_controller):
        """Test /stop_trading command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_stop_trading(ctx)
        assert "stopped" in response.lower()
        assert bot_controller._state == BotState.STOPPED

    @pytest.mark.asyncio
    async def test_pause_command(self, bot_controller):
        """Test /pause command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_pause(ctx)
        assert "paused" in response.lower()
        assert bot_controller._state == BotState.PAUSED

    @pytest.mark.asyncio
    async def test_resume_command(self, bot_controller):
        """Test /resume command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_resume(ctx)
        assert "resumed" in response.lower()
        assert bot_controller._state == BotState.RUNNING

    @pytest.mark.asyncio
    async def test_close_position_command(self, bot_controller):
        """Test /close command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["BTCUSDT"], "is_admin": True}
        response = await bot_controller._cmd_close(ctx)
        assert "closed" in response.lower()

    @pytest.mark.asyncio
    async def test_close_position_no_args(self, bot_controller):
        """Test /close without symbol."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_close(ctx)
        assert "Usage" in response

    @pytest.mark.asyncio
    async def test_cancel_order_command(self, bot_controller):
        """Test /cancel command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["test-order-1"], "is_admin": True}
        response = await bot_controller._cmd_cancel(ctx)
        assert "cancelled" in response.lower()

    @pytest.mark.asyncio
    async def test_cancel_order_no_args(self, bot_controller):
        """Test /cancel without order ID."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_cancel(ctx)
        assert "Usage" in response

    @pytest.mark.asyncio
    async def test_leverage_command(self, bot_controller):
        """Test /leverage command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["BTCUSDT", "10"], "is_admin": True}
        response = await bot_controller._cmd_leverage(ctx)
        assert "set to 10x" in response.lower()

    @pytest.mark.asyncio
    async def test_leverage_invalid_value(self, bot_controller):
        """Test /leverage with invalid value."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["BTCUSDT", "invalid"], "is_admin": True}
        response = await bot_controller._cmd_leverage(ctx)
        assert "Invalid" in response

    @pytest.mark.asyncio
    async def test_leverage_out_of_range(self, bot_controller):
        """Test /leverage with out of range value."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["BTCUSDT", "200"], "is_admin": True}
        response = await bot_controller._cmd_leverage(ctx)
        assert "must be between" in response.lower()


# =============================================================================
# Trading Commands Tests
# =============================================================================

class TestTradingCommands:
    """Test trading commands."""

    @pytest.mark.asyncio
    async def test_buy_command(self, bot_controller):
        """Test /buy command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["BTCUSDT", "0.001"], "is_admin": True}
        response = await bot_controller._cmd_buy(ctx)
        assert "Order Placed" in response or "BUY" in response

    @pytest.mark.asyncio
    async def test_buy_with_price(self, bot_controller):
        """Test /buy with limit price."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["BTCUSDT", "0.001", "50000"], "is_admin": True}
        response = await bot_controller._cmd_buy(ctx)
        assert "Order Placed" in response or "Limit" in response

    @pytest.mark.asyncio
    async def test_buy_no_args(self, bot_controller):
        """Test /buy without arguments."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_buy(ctx)
        assert "Usage" in response

    @pytest.mark.asyncio
    async def test_buy_invalid_quantity(self, bot_controller):
        """Test /buy with invalid quantity."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["BTCUSDT", "invalid"], "is_admin": True}
        response = await bot_controller._cmd_buy(ctx)
        assert "Invalid" in response

    @pytest.mark.asyncio
    async def test_sell_command(self, bot_controller):
        """Test /sell command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["BTCUSDT", "0.001"], "is_admin": True}
        response = await bot_controller._cmd_sell(ctx)
        assert "Order Placed" in response or "SELL" in response

    @pytest.mark.asyncio
    async def test_sell_with_price(self, bot_controller):
        """Test /sell with limit price."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["BTCUSDT", "0.001", "52000"], "is_admin": True}
        response = await bot_controller._cmd_sell(ctx)
        assert "Order Placed" in response or "Limit" in response

    @pytest.mark.asyncio
    async def test_sell_no_permission(self, bot_controller):
        """Test /sell without admin permission."""
        ctx = {"chat_id": 789012, "user_id": 789012, "args": ["BTCUSDT", "0.001"], "is_admin": False}
        response = await bot_controller._cmd_sell(ctx)
        assert "Admin permission required" in response

    @pytest.mark.asyncio
    async def test_sl_command(self, bot_controller):
        """Test /sl command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["BTCUSDT", "49000"], "is_admin": True}
        response = await bot_controller._cmd_set_sl(ctx)
        assert "Stop-Loss" in response or "set" in response.lower()

    @pytest.mark.asyncio
    async def test_sl_no_args(self, bot_controller):
        """Test /sl without arguments."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_set_sl(ctx)
        assert "Usage" in response

    @pytest.mark.asyncio
    async def test_tp_command(self, bot_controller):
        """Test /tp command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["BTCUSDT", "52000"], "is_admin": True}
        response = await bot_controller._cmd_set_tp(ctx)
        assert "Take-Profit" in response or "set" in response.lower()

    @pytest.mark.asyncio
    async def test_sltp_command(self, bot_controller):
        """Test /sltp command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["BTCUSDT", "49000", "52000"], "is_admin": True}
        response = await bot_controller._cmd_set_sltp(ctx)
        assert "SL/TP" in response

    @pytest.mark.asyncio
    async def test_sltp_no_args(self, bot_controller):
        """Test /sltp without arguments."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_set_sltp(ctx)
        assert "Usage" in response


# =============================================================================
# Statistics Commands Tests
# =============================================================================

class TestStatisticsCommands:
    """Test statistics commands."""

    @pytest.mark.asyncio
    async def test_stats_command(self, bot_controller):
        """Test /stats command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_stats(ctx)
        assert "Trading Statistics" in response
        assert "Total P&L" in response

    @pytest.mark.asyncio
    async def test_daily_stats_command(self, bot_controller):
        """Test /daily command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_daily_stats(ctx)
        assert "Daily Statistics" in response or "Trading Statistics" in response

    @pytest.mark.asyncio
    async def test_weekly_stats_command(self, bot_controller):
        """Test /weekly command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_weekly_stats(ctx)
        assert "Weekly Statistics" in response

    @pytest.mark.asyncio
    async def test_monthly_stats_command(self, bot_controller):
        """Test /monthly command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_monthly_stats(ctx)
        assert "Monthly Statistics" in response


# =============================================================================
# Alert Commands Tests
# =============================================================================

class TestAlertCommands:
    """Test alert commands."""

    @pytest.mark.asyncio
    async def test_alerts_status(self, bot_controller):
        """Test /alerts command without args."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_alerts(ctx)
        assert "Alerts Configuration" in response

    @pytest.mark.asyncio
    async def test_alerts_enable(self, bot_controller):
        """Test /alerts on command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["on"], "is_admin": True}
        response = await bot_controller._cmd_alerts(ctx)
        assert "enabled" in response.lower()
        assert bot_controller._alerts_enabled is True

    @pytest.mark.asyncio
    async def test_alerts_disable(self, bot_controller):
        """Test /alerts off command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["off"], "is_admin": True}
        response = await bot_controller._cmd_alerts(ctx)
        assert "disabled" in response.lower()
        assert bot_controller._alerts_enabled is False


# =============================================================================
# Strategy Commands Tests
# =============================================================================

class TestStrategyCommands:
    """Test strategy commands."""

    @pytest.mark.asyncio
    async def test_strategies_command(self, bot_controller):
        """Test /strategies command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_strategies(ctx)
        assert "Trading Strategies" in response or "Available Strategies" in response

    @pytest.mark.asyncio
    async def test_enable_strategy_command(self, bot_controller):
        """Test /enable command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["mean_reversion"], "is_admin": True}
        response = await bot_controller._cmd_enable_strategy(ctx)
        assert "enabled" in response.lower()

    @pytest.mark.asyncio
    async def test_enable_strategy_no_args(self, bot_controller):
        """Test /enable without arguments."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_enable_strategy(ctx)
        assert "Usage" in response

    @pytest.mark.asyncio
    async def test_disable_strategy_command(self, bot_controller):
        """Test /disable command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["volume_spike"], "is_admin": True}
        response = await bot_controller._cmd_disable_strategy(ctx)
        assert "disabled" in response.lower()

    @pytest.mark.asyncio
    async def test_disable_strategy_no_permission(self, bot_controller):
        """Test /disable without admin permission."""
        ctx = {"chat_id": 789012, "user_id": 789012, "args": ["volume_spike"], "is_admin": False}
        response = await bot_controller._cmd_disable_strategy(ctx)
        assert "Admin permission required" in response


# =============================================================================
# Exchange Commands Tests
# =============================================================================

class TestExchangeCommands:
    """Test exchange commands."""

    @pytest.mark.asyncio
    async def test_exchange_status(self, bot_controller):
        """Test /exchange without args."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_exchange(ctx)
        assert "Exchange Configuration" in response

    @pytest.mark.asyncio
    async def test_exchange_change(self, bot_controller):
        """Test /exchange with exchange name."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["bybit"], "is_admin": True}
        response = await bot_controller._cmd_exchange(ctx)
        assert "bybit" in response.lower()

    @pytest.mark.asyncio
    async def test_exchange_unknown(self, bot_controller):
        """Test /exchange with unknown exchange."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["unknown_exchange"], "is_admin": True}
        response = await bot_controller._cmd_exchange(ctx)
        assert "Unknown exchange" in response


# =============================================================================
# Quick Action Commands Tests
# =============================================================================

class TestQuickActionCommands:
    """Test quick action commands."""

    @pytest.mark.asyncio
    async def test_closeall_command(self, bot_controller):
        """Test /closeall command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_close_all(ctx)
        assert "Closed" in response

    @pytest.mark.asyncio
    async def test_closeall_no_permission(self, bot_controller):
        """Test /closeall without admin permission."""
        ctx = {"chat_id": 789012, "user_id": 789012, "args": [], "is_admin": False}
        response = await bot_controller._cmd_close_all(ctx)
        assert "Admin permission required" in response

    @pytest.mark.asyncio
    async def test_panic_command(self, bot_controller):
        """Test /panic command."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await bot_controller._cmd_panic(ctx)
        assert "PANIC" in response or "Emergency" in response
        assert bot_controller._state == BotState.STOPPED

    @pytest.mark.asyncio
    async def test_panic_no_permission(self, bot_controller):
        """Test /panic without admin permission."""
        ctx = {"chat_id": 789012, "user_id": 789012, "args": [], "is_admin": False}
        response = await bot_controller._cmd_panic(ctx)
        assert "Admin permission required" in response


# =============================================================================
# Notification Tests
# =============================================================================

class TestNotifications:
    """Test automatic notifications."""

    @pytest.mark.asyncio
    async def test_send_alert(self, bot_controller):
        """Test sending alert."""
        bot_controller._alerts_enabled = True
        bot_controller._alert_chat_ids = [123456]

        await bot_controller.send_alert("Test Alert", "This is a test message", "info")

        bot_controller.send_message.assert_called()

    @pytest.mark.asyncio
    async def test_send_alert_disabled(self, bot_controller):
        """Test sending alert when disabled."""
        bot_controller._alerts_enabled = False
        bot_controller.send_message.reset_mock()

        await bot_controller.send_alert("Test Alert", "This is a test message", "info")

        bot_controller.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_notify_position_opened(self, bot_controller):
        """Test position opened notification."""
        bot_controller._alerts_enabled = True
        bot_controller._alert_chat_ids = [123456]

        await bot_controller.notify_position_opened(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.001,
            price=50000.0,
            leverage=10
        )

        bot_controller.send_message.assert_called()

    @pytest.mark.asyncio
    async def test_notify_position_closed(self, bot_controller):
        """Test position closed notification."""
        bot_controller._alerts_enabled = True
        bot_controller._alert_chat_ids = [123456]

        await bot_controller.notify_position_closed(
            symbol="BTCUSDT",
            side="BUY",
            pnl=50.0,
            pnl_pct=2.5,
            exit_price=51000.0,
            duration="1h 30m"
        )

        bot_controller.send_message.assert_called()

    @pytest.mark.asyncio
    async def test_notify_stop_loss_hit(self, bot_controller):
        """Test stop-loss notification."""
        bot_controller._alerts_enabled = True
        bot_controller._alert_chat_ids = [123456]

        await bot_controller.notify_stop_loss_hit(
            symbol="BTCUSDT",
            pnl=-25.0,
            sl_price=49000.0
        )

        bot_controller.send_message.assert_called()

    @pytest.mark.asyncio
    async def test_notify_take_profit_hit(self, bot_controller):
        """Test take-profit notification."""
        bot_controller._alerts_enabled = True
        bot_controller._alert_chat_ids = [123456]

        await bot_controller.notify_take_profit_hit(
            symbol="BTCUSDT",
            pnl=75.0,
            tp_price=52000.0
        )

        bot_controller.send_message.assert_called()

    @pytest.mark.asyncio
    async def test_notify_daily_limit_reached(self, bot_controller):
        """Test daily limit notification."""
        bot_controller._alerts_enabled = True
        bot_controller._alert_chat_ids = [123456]

        await bot_controller.notify_daily_limit_reached(
            current_loss=100.0,
            limit=100.0
        )

        bot_controller.send_message.assert_called()

    @pytest.mark.asyncio
    async def test_notify_error(self, bot_controller):
        """Test error notification."""
        bot_controller._alerts_enabled = True
        bot_controller._alert_chat_ids = [123456]

        await bot_controller.notify_error(
            error_type="Connection Error",
            message="Failed to connect to exchange"
        )

        bot_controller.send_message.assert_called()

    @pytest.mark.asyncio
    async def test_notify_signal(self, bot_controller):
        """Test signal notification."""
        bot_controller._alerts_enabled = True
        bot_controller._alert_chat_ids = [123456]

        await bot_controller.notify_signal(
            symbol="BTCUSDT",
            signal_type="LONG",
            strength=0.85,
            price=50000.0,
            strategy="orderbook_imbalance"
        )

        bot_controller.send_message.assert_called()

    @pytest.mark.asyncio
    async def test_notify_signal_disabled(self, bot_controller):
        """Test signal notification when disabled."""
        bot_controller._alerts_enabled = False
        bot_controller.send_message.reset_mock()

        await bot_controller.notify_signal(
            symbol="BTCUSDT",
            signal_type="LONG",
            strength=0.85,
            price=50000.0,
            strategy="orderbook_imbalance"
        )

        bot_controller.send_message.assert_not_called()


# =============================================================================
# Command Handling Tests
# =============================================================================

class TestCommandHandling:
    """Test command handling."""

    @pytest.mark.asyncio
    async def test_handle_command(self, bot_controller):
        """Test command handling."""
        await bot_controller._handle_command(123456, 123456, "/status")
        bot_controller.send_message.assert_called()

    @pytest.mark.asyncio
    async def test_handle_unknown_command(self, bot_controller):
        """Test unknown command handling."""
        await bot_controller._handle_command(123456, 123456, "/unknown_command")
        bot_controller.send_message.assert_called()
        call_args = bot_controller.send_message.call_args
        assert "Unknown command" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_handle_command_with_botname(self, bot_controller):
        """Test command handling with bot name suffix."""
        await bot_controller._handle_command(123456, 123456, "/status@testbot")
        bot_controller.send_message.assert_called()


# =============================================================================
# Message Handling Tests
# =============================================================================

class TestMessageHandling:
    """Test message handling."""

    @pytest.mark.asyncio
    async def test_handle_update_with_command(self, bot_controller):
        """Test handling update with command."""
        update = {
            "message": {
                "text": "/status",
                "chat": {"id": 123456},
                "from": {"id": 123456, "username": "testuser"}
            }
        }

        await bot_controller._handle_update(update)
        bot_controller.send_message.assert_called()

    @pytest.mark.asyncio
    async def test_handle_update_unauthorized(self, bot_controller):
        """Test handling update from unauthorized user."""
        update = {
            "message": {
                "text": "/status",
                "chat": {"id": 999999},
                "from": {"id": 999999, "username": "unauthorized"}
            }
        }

        await bot_controller._handle_update(update)
        bot_controller.send_message.assert_called()
        call_args = bot_controller.send_message.call_args
        assert "Unauthorized" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_handle_update_no_text(self, bot_controller):
        """Test handling update without text."""
        update = {
            "message": {
                "chat": {"id": 123456},
                "from": {"id": 123456, "username": "testuser"}
            }
        }

        await bot_controller._handle_update(update)
        # Should not call send_message for empty text

    @pytest.mark.asyncio
    async def test_handle_update_empty(self, bot_controller):
        """Test handling empty update."""
        update = {}
        await bot_controller._handle_update(update)
        # Should not raise error


# =============================================================================
# Lifecycle Tests
# =============================================================================

class TestLifecycle:
    """Test bot lifecycle."""

    @pytest.mark.asyncio
    async def test_start_not_configured(self):
        """Test starting bot when not configured."""
        controller = TelegramBotController()
        await controller.start()
        # Should not raise error

    @pytest.mark.asyncio
    async def test_stop(self, bot_controller):
        """Test stopping bot."""
        bot_controller._running = True
        bot_controller._session = AsyncMock()

        await bot_controller.stop()

        assert bot_controller._running is False
        assert bot_controller._state == BotState.STOPPED


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""

    @pytest.mark.asyncio
    async def test_balance_not_available(self, config):
        """Test balance when callback not set."""
        controller = TelegramBotController(config=config)
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await controller._cmd_balance(ctx)
        assert "not available" in response.lower()

    @pytest.mark.asyncio
    async def test_positions_not_available(self, config):
        """Test positions when callback not set."""
        controller = TelegramBotController(config=config)
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await controller._cmd_positions(ctx)
        assert "not available" in response.lower()

    @pytest.mark.asyncio
    async def test_orders_not_available(self, config):
        """Test orders when callback not set."""
        controller = TelegramBotController(config=config)
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await controller._cmd_orders(ctx)
        assert "not available" in response.lower()

    @pytest.mark.asyncio
    async def test_trades_not_available(self, config):
        """Test trades when callback not set."""
        controller = TelegramBotController(config=config)
        ctx = {"chat_id": 123456, "user_id": 123456, "args": [], "is_admin": True}
        response = await controller._cmd_trades(ctx)
        assert "not available" in response.lower()

    @pytest.mark.asyncio
    async def test_buy_not_available(self, config):
        """Test buy when place_order not set."""
        controller = TelegramBotController(config=config)
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["BTCUSDT", "0.001"], "is_admin": True}
        response = await controller._cmd_buy(ctx)
        assert "not available" in response.lower()

    @pytest.mark.asyncio
    async def test_sl_not_available(self, config):
        """Test SL when callback not set."""
        controller = TelegramBotController(config=config)
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["BTCUSDT", "49000"], "is_admin": True}
        response = await controller._cmd_set_sl(ctx)
        assert "not available" in response.lower()

    @pytest.mark.asyncio
    async def test_invalid_price_in_sl(self, bot_controller):
        """Test SL with invalid price."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["BTCUSDT", "invalid"], "is_admin": True}
        response = await bot_controller._cmd_set_sl(ctx)
        assert "Invalid" in response

    @pytest.mark.asyncio
    async def test_invalid_price_in_buy(self, bot_controller):
        """Test buy with invalid price."""
        ctx = {"chat_id": 123456, "user_id": 123456, "args": ["BTCUSDT", "0.001", "invalid"], "is_admin": True}
        response = await bot_controller._cmd_buy(ctx)
        assert "Invalid" in response


# =============================================================================
# Factory Tests
# =============================================================================

class TestFactory:
    """Test factory function."""

    def test_create_telegram_bot_controller(self):
        """Test factory function."""
        from src.notifications.telegram_bot import create_telegram_bot_controller

        controller = create_telegram_bot_controller()
        assert controller is not None
        assert isinstance(controller, TelegramBotController)
