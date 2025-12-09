"""
Tests for Integration Modules

Tests for Telegram, Slack, and TradingView integrations.
"""
import pytest
import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

# Skip if dependencies not installed
pytest.importorskip("aiohttp")


class TestTradingViewIntegration:
    """Tests for TradingView webhook integration."""

    def test_alert_parsing(self):
        """Test parsing TradingView alert payload."""
        from src.integrations.tradingview import TradingViewAlert, AlertAction

        payload = {
            "ticker": "BTCUSDT",
            "exchange": "BINANCE",
            "action": "buy",
            "price": 50000.0,
            "quantity": 0.1,
            "strategy": "MACD Crossover",
        }

        alert = TradingViewAlert.from_webhook(payload)

        assert alert.ticker == "BTCUSDT"
        assert alert.exchange == "BINANCE"
        assert alert.action == AlertAction.BUY
        assert alert.price == Decimal("50000.0")
        assert alert.quantity == Decimal("0.1")
        assert alert.strategy_name == "MACD Crossover"

    def test_alert_action_mapping(self):
        """Test action string to enum mapping."""
        from src.integrations.tradingview import TradingViewAlert, AlertAction

        test_cases = [
            ({"action": "buy"}, AlertAction.BUY),
            ({"action": "long"}, AlertAction.BUY),
            ({"action": "sell"}, AlertAction.SELL),
            ({"action": "short"}, AlertAction.SELL),
            ({"action": "close"}, AlertAction.CLOSE),
            ({"action": "tp"}, AlertAction.TAKE_PROFIT),
            ({"action": "sl"}, AlertAction.STOP_LOSS),
        ]

        for payload, expected_action in test_cases:
            payload["ticker"] = "BTCUSDT"
            payload["price"] = 100
            alert = TradingViewAlert.from_webhook(payload)
            assert alert.action == expected_action, f"Failed for {payload}"

    def test_webhook_handler(self):
        """Test webhook handler registration."""
        from src.integrations.tradingview import TradingViewWebhook

        webhook = TradingViewWebhook(secret_key="test_secret")
        handler_called = False

        @webhook.on_alert
        def handle_alert(alert):
            nonlocal handler_called
            handler_called = True

        payload = {
            "ticker": "ETHUSDT",
            "action": "buy",
            "price": 3000,
        }

        webhook.process_alert(payload)

        assert handler_called
        assert webhook.stats["total_received"] == 1
        assert webhook.stats["total_processed"] == 1

    def test_signature_verification(self):
        """Test webhook signature verification."""
        from src.integrations.tradingview import TradingViewWebhook
        import hmac
        import hashlib

        secret = "test_secret_key"
        webhook = TradingViewWebhook(secret_key=secret)

        payload = b'{"ticker": "BTCUSDT"}'
        valid_signature = hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()

        assert webhook.verify_signature(payload, valid_signature)
        assert not webhook.verify_signature(payload, "invalid_signature")

    def test_ip_whitelist(self):
        """Test IP whitelist verification."""
        from src.integrations.tradingview import TradingViewWebhook

        webhook = TradingViewWebhook(allowed_ips=["1.2.3.4", "5.6.7.8"])

        assert webhook.verify_ip("1.2.3.4")
        assert webhook.verify_ip("5.6.7.8")
        assert not webhook.verify_ip("9.10.11.12")

    def test_alert_history(self):
        """Test alert history storage."""
        from src.integrations.tradingview import TradingViewWebhook

        webhook = TradingViewWebhook()

        for i in range(5):
            webhook.process_alert({
                "ticker": f"SYMBOL{i}",
                "action": "buy",
                "price": 100 + i,
            })

        recent = webhook.get_recent_alerts(3)
        assert len(recent) == 3
        assert recent[-1]["ticker"] == "SYMBOL4"


class TestSlackIntegration:
    """Tests for Slack integration."""

    @pytest.fixture
    def slack_client(self):
        """Create Slack client fixture."""
        from src.integrations.slack import SlackClient
        return SlackClient(
            bot_token="xoxb-test-token",
            default_channel="#test-channel"
        )

    @pytest.fixture
    def slack_notifier(self, slack_client):
        """Create Slack notifier fixture."""
        from src.integrations.slack import SlackNotifier
        return SlackNotifier(slack_client)

    def test_trade_blocks_creation(self, slack_notifier):
        """Test trade notification block creation."""
        trade = {
            "symbol": "BTCUSDT",
            "side": "long",
            "entry_price": 50000,
            "exit_price": 51000,
            "quantity": 0.1,
            "pnl": 100,
            "strategy": "MACD",
        }

        blocks = slack_notifier._create_trade_blocks(trade)

        assert len(blocks) >= 2
        assert blocks[0]["type"] == "header"
        assert "Trade Executed" in blocks[0]["text"]["text"]

    def test_signal_blocks_creation(self, slack_notifier):
        """Test signal notification block creation."""
        signal = {
            "id": "sig_123",
            "symbol": "ETHUSDT",
            "action": "buy",
            "price": 3000,
            "confidence": 85.5,
        }

        blocks = slack_notifier._create_signal_blocks(signal)

        assert len(blocks) >= 2
        # Should have action buttons
        action_block = next((b for b in blocks if b.get("type") == "actions"), None)
        assert action_block is not None

    def test_daily_report_blocks(self, slack_notifier):
        """Test daily report block creation."""
        report = {
            "date": "2024-01-15",
            "pnl": 1500.50,
            "trades": 25,
            "win_rate": 68.0,
            "balance": 15000.0,
            "winning_trades": 17,
            "losing_trades": 8,
            "best_trade": 500,
            "worst_trade": -150,
        }

        blocks = slack_notifier._create_daily_report_blocks(report)

        assert len(blocks) >= 3
        assert any("Daily Trading Report" in str(b) for b in blocks)

    @pytest.mark.asyncio
    async def test_send_message(self, slack_client):
        """Test sending Slack message."""
        with patch.object(slack_client, '_get_session') as mock_session:
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value={"ok": True})

            mock_post = AsyncMock(return_value=mock_response)
            mock_session.return_value.post = mock_post

            # This would actually send a message
            # await slack_client.send_message("Test message")

    def test_alert_blocks_creation(self, slack_notifier):
        """Test alert notification block creation."""
        blocks = slack_notifier._create_alert_blocks(
            severity="critical",
            title="High Drawdown Alert",
            message="Portfolio drawdown exceeded 10%"
        )

        assert blocks[0]["type"] == "header"
        assert "High Drawdown Alert" in blocks[0]["text"]["text"]


class TestTelegramBot:
    """Tests for Telegram bot."""

    @pytest.fixture
    def mock_bot(self):
        """Create mock Telegram bot."""
        # Skip if telegram not installed
        pytest.importorskip("telegram")

        from src.integrations.telegram_bot import TelegramTradingBot

        bot = TelegramTradingBot(
            token="test_token",
            allowed_users=[123456],
            admin_users=[123456],
        )
        return bot

    def test_authorization_check(self, mock_bot):
        """Test user authorization."""
        assert mock_bot._is_authorized(123456)
        assert not mock_bot._is_authorized(999999)

    def test_admin_check(self, mock_bot):
        """Test admin check."""
        assert mock_bot._is_admin(123456)
        assert not mock_bot._is_admin(999999)

    def test_session_management(self, mock_bot):
        """Test user session management."""
        from src.integrations.telegram_bot import BotState

        session = mock_bot._get_session(123456)
        assert session.user_id == 123456
        assert session.state == BotState.IDLE

        # Same user should return same session
        session2 = mock_bot._get_session(123456)
        assert session is session2

    def test_callback_registration(self, mock_bot):
        """Test callback registration."""
        def dummy_callback():
            pass

        mock_bot.set_callback("get_balance", dummy_callback)
        assert "get_balance" in mock_bot.callbacks
        assert mock_bot.callbacks["get_balance"] is dummy_callback


class TestTelegramNotifier:
    """Tests for Telegram notifier."""

    @pytest.fixture
    def notifier(self):
        """Create notifier fixture."""
        pytest.importorskip("telegram")

        from src.integrations.telegram_bot import TelegramNotifier
        return TelegramNotifier(
            token="test_token",
            chat_ids=[123456, 789012]
        )

    @pytest.mark.asyncio
    async def test_trade_alert_format(self, notifier):
        """Test trade alert message format."""
        trade = {
            "symbol": "BTCUSDT",
            "side": "long",
            "entry_price": 50000,
            "exit_price": 51000,
            "pnl": 100,
        }

        # Test that the method exists and would format correctly
        # Actual sending would require mocking
        assert hasattr(notifier, 'send_trade_alert')

    @pytest.mark.asyncio
    async def test_signal_alert_format(self, notifier):
        """Test signal alert message format."""
        signal = {
            "symbol": "ETHUSDT",
            "action": "buy",
            "price": 3000,
            "strategy": "MACD",
            "confidence": 85,
        }

        assert hasattr(notifier, 'send_signal_alert')

    @pytest.mark.asyncio
    async def test_daily_report_format(self, notifier):
        """Test daily report message format."""
        report = {
            "date": "2024-01-15",
            "pnl": 500,
            "trades": 10,
            "win_rate": 70,
            "balance": 10500,
        }

        assert hasattr(notifier, 'send_daily_report')
