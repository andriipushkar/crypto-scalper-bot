"""
Tests for Advanced Alert System module.

100% coverage for:
- Alert and AlertCondition dataclasses
- All AlertConditionCheckers
- NotificationSender
- AdvancedAlertSystem
- Convenience alert creation methods
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from collections import defaultdict

from src.alerts.alert_system import (
    AdvancedAlertSystem,
    AlertSystemConfig,
    Alert,
    AlertCondition,
    TriggeredAlert,
    AlertType,
    AlertPriority,
    AlertStatus,
    NotificationChannel,
    NotificationSender,
    PriceAlertChecker,
    VolumeAlertChecker,
    WhaleAlertChecker,
    OnChainMetricsChecker,
    TechnicalIndicatorChecker,
    create_alert_system,
)


# =============================================================================
# AlertCondition Tests
# =============================================================================

class TestAlertCondition:
    """Tests for AlertCondition dataclass."""

    def test_condition_creation(self):
        """Test condition creation."""
        condition = AlertCondition(
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
            comparison="above",
            timeframe="5m",
        )

        assert condition.alert_type == AlertType.PRICE_ABOVE
        assert condition.symbol == "BTCUSDT"
        assert condition.threshold == Decimal("50000")

    def test_condition_to_dict(self):
        """Test condition serialization."""
        condition = AlertCondition(
            alert_type=AlertType.RSI_OVERBOUGHT,
            symbol="ETHUSDT",
            threshold=Decimal("70"),
            params={"period": 14},
        )

        data = condition.to_dict()

        assert data["alert_type"] == "rsi_overbought"
        assert data["symbol"] == "ETHUSDT"
        assert data["threshold"] == "70"
        assert data["params"]["period"] == 14

    def test_condition_from_dict(self):
        """Test condition deserialization."""
        data = {
            "alert_type": "volume_spike",
            "symbol": "BTCUSDT",
            "threshold": "3.0",
            "comparison": "above",
            "timeframe": "1h",
            "params": {"lookback": 20},
        }

        condition = AlertCondition.from_dict(data)

        assert condition.alert_type == AlertType.VOLUME_SPIKE
        assert condition.symbol == "BTCUSDT"
        assert condition.threshold == Decimal("3.0")


# =============================================================================
# Alert Tests
# =============================================================================

class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_creation(self):
        """Test alert creation."""
        condition = AlertCondition(
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
        )

        alert = Alert(
            alert_id="test_001",
            name="BTC Above 50K",
            condition=condition,
            priority=AlertPriority.HIGH,
        )

        assert alert.alert_id == "test_001"
        assert alert.name == "BTC Above 50K"
        assert alert.status == AlertStatus.ACTIVE

    def test_can_trigger_active(self):
        """Test can_trigger for active alert."""
        condition = AlertCondition(
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
        )

        alert = Alert(
            alert_id="test",
            name="Test",
            condition=condition,
            status=AlertStatus.ACTIVE,
        )

        assert alert.can_trigger() is True

    def test_can_trigger_disabled(self):
        """Test can_trigger for disabled alert."""
        condition = AlertCondition(
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
        )

        alert = Alert(
            alert_id="test",
            name="Test",
            condition=condition,
            status=AlertStatus.DISABLED,
        )

        assert alert.can_trigger() is False

    def test_can_trigger_expired(self):
        """Test can_trigger for expired alert."""
        condition = AlertCondition(
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
        )

        alert = Alert(
            alert_id="test",
            name="Test",
            condition=condition,
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )

        assert alert.can_trigger() is False

    def test_can_trigger_max_triggers_reached(self):
        """Test can_trigger when max triggers reached."""
        condition = AlertCondition(
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
        )

        alert = Alert(
            alert_id="test",
            name="Test",
            condition=condition,
            max_triggers=5,
            trigger_count=5,
        )

        assert alert.can_trigger() is False

    def test_can_trigger_cooldown(self):
        """Test can_trigger during cooldown."""
        condition = AlertCondition(
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
        )

        alert = Alert(
            alert_id="test",
            name="Test",
            condition=condition,
            cooldown_seconds=300,
            last_triggered=datetime.utcnow() - timedelta(seconds=60),  # 60s ago
        )

        assert alert.can_trigger() is False

    def test_can_trigger_active_hours(self):
        """Test can_trigger with active hours."""
        condition = AlertCondition(
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
        )

        current_hour = datetime.utcnow().hour

        alert_in_hours = Alert(
            alert_id="test1",
            name="Test",
            condition=condition,
            active_hours=[current_hour],
        )

        alert_out_hours = Alert(
            alert_id="test2",
            name="Test",
            condition=condition,
            active_hours=[(current_hour + 12) % 24],  # Different hour
        )

        assert alert_in_hours.can_trigger() is True
        assert alert_out_hours.can_trigger() is False

    def test_format_message_custom_template(self):
        """Test message formatting with custom template."""
        condition = AlertCondition(
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
        )

        alert = Alert(
            alert_id="test",
            name="Test",
            condition=condition,
            message_template="Price is now {current_value}! Target was {threshold}",
        )

        message = alert.format_message({
            "current_value": Decimal("51000"),
            "threshold": Decimal("50000"),
        })

        assert "51000" in message
        assert "50000" in message

    def test_format_message_default(self):
        """Test default message formatting."""
        condition = AlertCondition(
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
        )

        alert = Alert(
            alert_id="test",
            name="BTC Alert",
            condition=condition,
            priority=AlertPriority.HIGH,
        )

        message = alert.format_message({"current_value": Decimal("51000")})

        assert "HIGH" in message
        assert "BTC Alert" in message
        assert "BTCUSDT" in message
        assert "51000" in message


# =============================================================================
# PriceAlertChecker Tests
# =============================================================================

class TestPriceAlertChecker:
    """Tests for PriceAlertChecker."""

    @pytest.fixture
    def checker(self):
        """Create checker instance."""
        return PriceAlertChecker()

    @pytest.mark.asyncio
    async def test_price_above_triggered(self, checker):
        """Test PRICE_ABOVE alert triggered."""
        condition = AlertCondition(
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
        )

        market_data = {"price": 51000}

        result = await checker.check(condition, market_data)

        assert result is not None
        assert result["current_value"] == Decimal("51000")
        assert ">" in result["condition_met"]

    @pytest.mark.asyncio
    async def test_price_above_not_triggered(self, checker):
        """Test PRICE_ABOVE alert not triggered."""
        condition = AlertCondition(
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
        )

        market_data = {"price": 49000}

        result = await checker.check(condition, market_data)

        assert result is None

    @pytest.mark.asyncio
    async def test_price_below_triggered(self, checker):
        """Test PRICE_BELOW alert triggered."""
        condition = AlertCondition(
            alert_type=AlertType.PRICE_BELOW,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
        )

        market_data = {"price": 49000}

        result = await checker.check(condition, market_data)

        assert result is not None
        assert "<" in result["condition_met"]

    @pytest.mark.asyncio
    async def test_price_cross_above(self, checker):
        """Test PRICE_CROSS alert when crossing above."""
        condition = AlertCondition(
            alert_type=AlertType.PRICE_CROSS,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
        )

        # First call - set last price below threshold
        checker._last_prices["BTCUSDT"] = Decimal("49000")

        # Price crosses above
        market_data = {"price": 51000}

        result = await checker.check(condition, market_data)

        assert result is not None
        assert "above" in result["condition_met"]

    @pytest.mark.asyncio
    async def test_price_cross_below(self, checker):
        """Test PRICE_CROSS alert when crossing below."""
        condition = AlertCondition(
            alert_type=AlertType.PRICE_CROSS,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
        )

        # First call - set last price above threshold
        checker._last_prices["BTCUSDT"] = Decimal("51000")

        # Price crosses below
        market_data = {"price": 49000}

        result = await checker.check(condition, market_data)

        assert result is not None
        assert "below" in result["condition_met"]

    @pytest.mark.asyncio
    async def test_price_change_pct(self, checker):
        """Test PRICE_CHANGE_PCT alert."""
        condition = AlertCondition(
            alert_type=AlertType.PRICE_CHANGE_PCT,
            symbol="BTCUSDT",
            threshold=Decimal("5"),  # 5%
        )

        # Set last price
        checker._last_prices["BTCUSDT"] = Decimal("50000")

        # Price moves 6%
        market_data = {"price": 53000}

        result = await checker.check(condition, market_data)

        assert result is not None
        assert "up" in result["condition_met"]

    @pytest.mark.asyncio
    async def test_price_zero(self, checker):
        """Test with zero price."""
        condition = AlertCondition(
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
        )

        market_data = {"price": 0}

        result = await checker.check(condition, market_data)

        assert result is None


# =============================================================================
# VolumeAlertChecker Tests
# =============================================================================

class TestVolumeAlertChecker:
    """Tests for VolumeAlertChecker."""

    @pytest.fixture
    def checker(self):
        """Create checker instance."""
        return VolumeAlertChecker(lookback_periods=20)

    @pytest.mark.asyncio
    async def test_volume_spike(self, checker):
        """Test VOLUME_SPIKE alert."""
        condition = AlertCondition(
            alert_type=AlertType.VOLUME_SPIKE,
            symbol="BTCUSDT",
            threshold=Decimal("3"),  # 3x average
        )

        # Build history with normal volume
        for i in range(10):
            await checker.check(condition, {"volume": 1000})

        # Spike volume
        market_data = {"volume": 5000}

        result = await checker.check(condition, market_data)

        assert result is not None
        assert "spike" in result["condition_met"]

    @pytest.mark.asyncio
    async def test_volume_above(self, checker):
        """Test VOLUME_ABOVE alert."""
        condition = AlertCondition(
            alert_type=AlertType.VOLUME_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("10000"),
        )

        market_data = {"volume": 15000}

        result = await checker.check(condition, market_data)

        assert result is not None
        assert "15,000" in result["condition_met"]

    @pytest.mark.asyncio
    async def test_oi_change(self, checker):
        """Test OI_CHANGE alert."""
        condition = AlertCondition(
            alert_type=AlertType.OI_CHANGE,
            symbol="BTCUSDT",
            threshold=Decimal("10"),  # 10%
        )

        market_data = {
            "volume": 1000,
            "open_interest": 5000000,
            "oi_change_pct": 15,
        }

        result = await checker.check(condition, market_data)

        assert result is not None
        assert "increased" in result["condition_met"]

    @pytest.mark.asyncio
    async def test_volume_zero(self, checker):
        """Test with zero volume."""
        condition = AlertCondition(
            alert_type=AlertType.VOLUME_SPIKE,
            symbol="BTCUSDT",
            threshold=Decimal("3"),
        )

        market_data = {"volume": 0}

        result = await checker.check(condition, market_data)

        assert result is None


# =============================================================================
# WhaleAlertChecker Tests
# =============================================================================

class TestWhaleAlertChecker:
    """Tests for WhaleAlertChecker."""

    @pytest.fixture
    def checker(self):
        """Create checker instance."""
        return WhaleAlertChecker()

    @pytest.mark.asyncio
    async def test_whale_transfer(self, checker):
        """Test WHALE_TRANSFER alert."""
        condition = AlertCondition(
            alert_type=AlertType.WHALE_TRANSFER,
            symbol="BTCUSDT",
            threshold=Decimal("1000000"),  # $1M
        )

        market_data = {
            "whale_transfers": [
                {
                    "tx_hash": "abc123",
                    "amount_usd": 2000000,
                    "type": "transfer",
                    "from": "0x123...",
                    "to": "0x456...",
                }
            ]
        }

        result = await checker.check(condition, market_data)

        assert result is not None
        assert result["current_value"] == Decimal("2000000")
        assert "abc123" in result["tx_hash"]

    @pytest.mark.asyncio
    async def test_whale_transfer_duplicate(self, checker):
        """Test that duplicate transactions are ignored."""
        condition = AlertCondition(
            alert_type=AlertType.WHALE_TRANSFER,
            symbol="BTCUSDT",
            threshold=Decimal("1000000"),
        )

        market_data = {
            "whale_transfers": [
                {
                    "tx_hash": "abc123",
                    "amount_usd": 2000000,
                    "type": "transfer",
                    "from": "0x123",
                    "to": "0x456",
                }
            ]
        }

        # First call
        result1 = await checker.check(condition, market_data)
        assert result1 is not None

        # Second call with same tx
        result2 = await checker.check(condition, market_data)
        assert result2 is None

    @pytest.mark.asyncio
    async def test_whale_exchange_deposit(self, checker):
        """Test WHALE_EXCHANGE_DEPOSIT alert."""
        condition = AlertCondition(
            alert_type=AlertType.WHALE_EXCHANGE_DEPOSIT,
            symbol="BTCUSDT",
            threshold=Decimal("1000000"),
        )

        market_data = {
            "whale_transfers": [
                {
                    "tx_hash": "deposit123",
                    "amount_usd": 5000000,
                    "type": "deposit",
                    "from": "0x123",
                    "to": "exchange",
                    "to_exchange": True,
                }
            ]
        }

        result = await checker.check(condition, market_data)

        assert result is not None

    @pytest.mark.asyncio
    async def test_whale_exchange_withdrawal(self, checker):
        """Test WHALE_EXCHANGE_WITHDRAWAL alert."""
        condition = AlertCondition(
            alert_type=AlertType.WHALE_EXCHANGE_WITHDRAWAL,
            symbol="BTCUSDT",
            threshold=Decimal("1000000"),
        )

        market_data = {
            "whale_transfers": [
                {
                    "tx_hash": "withdraw123",
                    "amount_usd": 3000000,
                    "type": "withdrawal",
                    "from": "exchange",
                    "to": "0x789",
                    "from_exchange": True,
                }
            ]
        }

        result = await checker.check(condition, market_data)

        assert result is not None

    @pytest.mark.asyncio
    async def test_whale_below_threshold(self, checker):
        """Test whale transfer below threshold."""
        condition = AlertCondition(
            alert_type=AlertType.WHALE_TRANSFER,
            symbol="BTCUSDT",
            threshold=Decimal("5000000"),  # $5M
        )

        market_data = {
            "whale_transfers": [
                {
                    "tx_hash": "small123",
                    "amount_usd": 1000000,  # Only $1M
                    "type": "transfer",
                    "from": "0x123",
                    "to": "0x456",
                }
            ]
        }

        result = await checker.check(condition, market_data)

        assert result is None


# =============================================================================
# OnChainMetricsChecker Tests
# =============================================================================

class TestOnChainMetricsChecker:
    """Tests for OnChainMetricsChecker."""

    @pytest.fixture
    def checker(self):
        """Create checker instance."""
        return OnChainMetricsChecker()

    @pytest.mark.asyncio
    async def test_funding_rate_above(self, checker):
        """Test FUNDING_RATE alert above threshold."""
        condition = AlertCondition(
            alert_type=AlertType.FUNDING_RATE,
            symbol="BTCUSDT",
            threshold=Decimal("0.1"),
            comparison="above",
        )

        market_data = {
            "onchain_metrics": {
                "funding_rate": 0.15,
            }
        }

        result = await checker.check(condition, market_data)

        assert result is not None
        assert result["metric"] == "funding_rate"

    @pytest.mark.asyncio
    async def test_funding_rate_extreme(self, checker):
        """Test FUNDING_RATE extreme alert."""
        condition = AlertCondition(
            alert_type=AlertType.FUNDING_RATE,
            symbol="BTCUSDT",
            threshold=Decimal("0.1"),
            comparison="extreme",
        )

        market_data = {
            "onchain_metrics": {
                "funding_rate": -0.15,  # Extreme negative
            }
        }

        result = await checker.check(condition, market_data)

        assert result is not None
        assert "extreme" in result["condition_met"]

    @pytest.mark.asyncio
    async def test_nvt_ratio(self, checker):
        """Test NVT_RATIO alert."""
        condition = AlertCondition(
            alert_type=AlertType.NVT_RATIO,
            symbol="BTCUSDT",
            threshold=Decimal("100"),
            comparison="above",
        )

        market_data = {
            "onchain_metrics": {
                "nvt_ratio": 150,
            }
        }

        result = await checker.check(condition, market_data)

        assert result is not None

    @pytest.mark.asyncio
    async def test_missing_metric(self, checker):
        """Test with missing metric."""
        condition = AlertCondition(
            alert_type=AlertType.MVRV_RATIO,
            symbol="BTCUSDT",
            threshold=Decimal("3"),
        )

        market_data = {
            "onchain_metrics": {}  # No MVRV
        }

        result = await checker.check(condition, market_data)

        assert result is None


# =============================================================================
# TechnicalIndicatorChecker Tests
# =============================================================================

class TestTechnicalIndicatorChecker:
    """Tests for TechnicalIndicatorChecker."""

    @pytest.fixture
    def checker(self):
        """Create checker instance."""
        return TechnicalIndicatorChecker()

    @pytest.mark.asyncio
    async def test_rsi_overbought(self, checker):
        """Test RSI_OVERBOUGHT alert."""
        condition = AlertCondition(
            alert_type=AlertType.RSI_OVERBOUGHT,
            symbol="BTCUSDT",
            threshold=Decimal("70"),
        )

        market_data = {
            "indicators": {
                "rsi": 75,
            }
        }

        result = await checker.check(condition, market_data)

        assert result is not None
        assert "overbought" in result["condition_met"]

    @pytest.mark.asyncio
    async def test_rsi_oversold(self, checker):
        """Test RSI_OVERSOLD alert."""
        condition = AlertCondition(
            alert_type=AlertType.RSI_OVERSOLD,
            symbol="BTCUSDT",
            threshold=Decimal("30"),
        )

        market_data = {
            "indicators": {
                "rsi": 25,
            }
        }

        result = await checker.check(condition, market_data)

        assert result is not None
        assert "oversold" in result["condition_met"]

    @pytest.mark.asyncio
    async def test_macd_bullish_cross(self, checker):
        """Test MACD_CROSS bullish crossover."""
        condition = AlertCondition(
            alert_type=AlertType.MACD_CROSS,
            symbol="BTCUSDT",
            threshold=Decimal("0"),
        )

        # Set previous values (MACD below signal)
        checker._indicator_values["BTCUSDT"]["macd"] = -10
        checker._indicator_values["BTCUSDT"]["macd_signal"] = -5

        # MACD crosses above signal
        market_data = {
            "indicators": {
                "macd": 5,
                "macd_signal": 0,
            }
        }

        result = await checker.check(condition, market_data)

        assert result is not None
        assert "bullish" in result["condition_met"]

    @pytest.mark.asyncio
    async def test_macd_bearish_cross(self, checker):
        """Test MACD_CROSS bearish crossover."""
        condition = AlertCondition(
            alert_type=AlertType.MACD_CROSS,
            symbol="BTCUSDT",
            threshold=Decimal("0"),
        )

        # Set previous values (MACD above signal)
        checker._indicator_values["BTCUSDT"]["macd"] = 10
        checker._indicator_values["BTCUSDT"]["macd_signal"] = 5

        # MACD crosses below signal
        market_data = {
            "indicators": {
                "macd": -5,
                "macd_signal": 0,
            }
        }

        result = await checker.check(condition, market_data)

        assert result is not None
        assert "bearish" in result["condition_met"]

    @pytest.mark.asyncio
    async def test_bb_breakout_upper(self, checker):
        """Test BB_BREAKOUT upper band."""
        condition = AlertCondition(
            alert_type=AlertType.BB_BREAKOUT,
            symbol="BTCUSDT",
            threshold=Decimal("0"),
        )

        market_data = {
            "price": 52000,
            "indicators": {
                "bb_upper": 51000,
                "bb_lower": 49000,
            }
        }

        result = await checker.check(condition, market_data)

        assert result is not None
        assert "above upper" in result["condition_met"]

    @pytest.mark.asyncio
    async def test_bb_breakout_lower(self, checker):
        """Test BB_BREAKOUT lower band."""
        condition = AlertCondition(
            alert_type=AlertType.BB_BREAKOUT,
            symbol="BTCUSDT",
            threshold=Decimal("0"),
        )

        market_data = {
            "price": 48000,
            "indicators": {
                "bb_upper": 51000,
                "bb_lower": 49000,
            }
        }

        result = await checker.check(condition, market_data)

        assert result is not None
        assert "below lower" in result["condition_met"]


# =============================================================================
# NotificationSender Tests
# =============================================================================

class TestNotificationSender:
    """Tests for NotificationSender."""

    @pytest.fixture
    def sender(self):
        """Create sender instance."""
        return NotificationSender(
            telegram_bot_token="test_token",
            telegram_chat_id="test_chat",
            discord_webhook_url="https://discord.webhook/test",
            webhook_urls={"default": "https://webhook.test"},
        )

    @pytest.mark.asyncio
    async def test_send_telegram(self, sender):
        """Test Telegram notification."""
        with patch("aiohttp.ClientSession") as mock_session_cls:
            # Create proper async context manager mocks
            response = MagicMock()
            response.status = 200

            post_context = MagicMock()
            post_context.__aenter__ = AsyncMock(return_value=response)
            post_context.__aexit__ = AsyncMock(return_value=None)

            session = MagicMock()
            session.post = MagicMock(return_value=post_context)

            session_context = MagicMock()
            session_context.__aenter__ = AsyncMock(return_value=session)
            session_context.__aexit__ = AsyncMock(return_value=None)

            mock_session_cls.return_value = session_context

            result = await sender._send_telegram("Test message", AlertPriority.HIGH)

            assert result is True

    @pytest.mark.asyncio
    async def test_send_telegram_not_configured(self):
        """Test Telegram when not configured."""
        sender = NotificationSender()

        result = await sender._send_telegram("Test", AlertPriority.MEDIUM)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_discord(self, sender):
        """Test Discord notification."""
        with patch("aiohttp.ClientSession") as mock_session_cls:
            # Create proper async context manager mocks
            response = MagicMock()
            response.status = 204

            post_context = MagicMock()
            post_context.__aenter__ = AsyncMock(return_value=response)
            post_context.__aexit__ = AsyncMock(return_value=None)

            session = MagicMock()
            session.post = MagicMock(return_value=post_context)

            session_context = MagicMock()
            session_context.__aenter__ = AsyncMock(return_value=session)
            session_context.__aexit__ = AsyncMock(return_value=None)

            mock_session_cls.return_value = session_context

            result = await sender._send_discord("Test message", AlertPriority.CRITICAL)

            assert result is True

    @pytest.mark.asyncio
    async def test_send_discord_not_configured(self):
        """Test Discord when not configured."""
        sender = NotificationSender()

        result = await sender._send_discord("Test", AlertPriority.MEDIUM)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_webhook(self, sender):
        """Test webhook notification."""
        with patch("aiohttp.ClientSession") as mock_session_cls:
            # Create proper async context manager mocks
            response = MagicMock()
            response.status = 200

            post_context = MagicMock()
            post_context.__aenter__ = AsyncMock(return_value=response)
            post_context.__aexit__ = AsyncMock(return_value=None)

            session = MagicMock()
            session.post = MagicMock(return_value=post_context)

            session_context = MagicMock()
            session_context.__aenter__ = AsyncMock(return_value=session)
            session_context.__aexit__ = AsyncMock(return_value=None)

            mock_session_cls.return_value = session_context

            result = await sender._send_webhook("Test", AlertPriority.MEDIUM, {"extra": "data"})

            assert result is True

    def test_send_console(self, sender):
        """Test console notification."""
        result = sender._send_console("Test message", AlertPriority.HIGH)

        assert result is True

    @pytest.mark.asyncio
    async def test_send_routing(self, sender):
        """Test send method routing."""
        with patch.object(sender, '_send_telegram', new_callable=AsyncMock) as mock:
            mock.return_value = True
            await sender.send(NotificationChannel.TELEGRAM, "Test", AlertPriority.MEDIUM)
            mock.assert_called_once()

        with patch.object(sender, '_send_discord', new_callable=AsyncMock) as mock:
            mock.return_value = True
            await sender.send(NotificationChannel.DISCORD, "Test", AlertPriority.MEDIUM)
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_unsupported_channel(self, sender):
        """Test unsupported channel."""
        result = await sender.send(NotificationChannel.SMS, "Test", AlertPriority.MEDIUM)

        assert result is False


# =============================================================================
# AdvancedAlertSystem Tests
# =============================================================================

class TestAdvancedAlertSystem:
    """Tests for AdvancedAlertSystem."""

    @pytest.fixture
    def config(self):
        """Create config."""
        return AlertSystemConfig(
            check_interval_seconds=1.0,
            max_alerts_per_user=10,
            max_triggers_per_minute=5,
        )

    @pytest.fixture
    def system(self, config):
        """Create alert system."""
        return AdvancedAlertSystem(config=config)

    @pytest.fixture
    async def started_system(self, system):
        """Start system."""
        await system.start()
        yield system
        await system.stop()

    # -------------------------------------------------------------------------
    # Alert CRUD Tests
    # -------------------------------------------------------------------------

    def test_create_alert(self, system):
        """Test alert creation."""
        alert = system.create_alert(
            name="BTC Above 50K",
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
        )

        assert alert.name == "BTC Above 50K"
        assert alert.alert_id in system._alerts

    def test_create_alert_max_per_user(self, system):
        """Test max alerts per user limit."""
        for i in range(10):
            system.create_alert(
                name=f"Alert {i}",
                alert_type=AlertType.PRICE_ABOVE,
                symbol="BTCUSDT",
                threshold=Decimal("50000"),
                user_id="user123",
            )

        with pytest.raises(ValueError):
            system.create_alert(
                name="One too many",
                alert_type=AlertType.PRICE_ABOVE,
                symbol="BTCUSDT",
                threshold=Decimal("50000"),
                user_id="user123",
            )

    def test_get_alert(self, system):
        """Test getting alert by ID."""
        alert = system.create_alert(
            name="Test",
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
        )

        found = system.get_alert(alert.alert_id)
        assert found == alert

        not_found = system.get_alert("nonexistent")
        assert not_found is None

    def test_get_alerts_filtered(self, system):
        """Test getting alerts with filters."""
        system.create_alert(
            name="BTC Alert",
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
            user_id="user1",
        )

        system.create_alert(
            name="ETH Alert",
            alert_type=AlertType.RSI_OVERBOUGHT,
            symbol="ETHUSDT",
            threshold=Decimal("70"),
            user_id="user2",
        )

        # Filter by user
        user1_alerts = system.get_alerts(user_id="user1")
        assert len(user1_alerts) == 1

        # Filter by symbol
        btc_alerts = system.get_alerts(symbol="BTCUSDT")
        assert len(btc_alerts) == 1

        # Filter by type
        rsi_alerts = system.get_alerts(alert_type=AlertType.RSI_OVERBOUGHT)
        assert len(rsi_alerts) == 1

    def test_update_alert(self, system):
        """Test updating alert."""
        alert = system.create_alert(
            name="Original",
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
        )

        updated = system.update_alert(
            alert.alert_id,
            name="Updated",
            priority=AlertPriority.CRITICAL,
        )

        assert updated.name == "Updated"
        assert updated.priority == AlertPriority.CRITICAL

    def test_delete_alert(self, system):
        """Test deleting alert."""
        alert = system.create_alert(
            name="To Delete",
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
        )

        result = system.delete_alert(alert.alert_id)
        assert result is True
        assert system.get_alert(alert.alert_id) is None

        result = system.delete_alert("nonexistent")
        assert result is False

    def test_disable_enable_alert(self, system):
        """Test disabling and enabling alert."""
        alert = system.create_alert(
            name="Test",
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
        )

        system.disable_alert(alert.alert_id)
        assert alert.status == AlertStatus.DISABLED

        system.enable_alert(alert.alert_id)
        assert alert.status == AlertStatus.ACTIVE

    # -------------------------------------------------------------------------
    # Alert Trigger Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_check_alert_triggered(self, system):
        """Test alert check and trigger."""
        alert = system.create_alert(
            name="Test",
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
            channels=[NotificationChannel.CONSOLE],
        )

        market_data = {"BTCUSDT": {"price": 51000}}

        with patch.object(system._notifier, 'send', new_callable=AsyncMock) as mock:
            mock.return_value = True
            await system._check_alert(alert, market_data)

            assert alert.trigger_count == 1
            assert alert.last_triggered is not None

    @pytest.mark.asyncio
    async def test_trigger_once(self, system):
        """Test trigger_once behavior."""
        alert = system.create_alert(
            name="Test",
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
            trigger_once=True,
            channels=[NotificationChannel.CONSOLE],
        )

        with patch.object(system._notifier, 'send', new_callable=AsyncMock) as mock:
            mock.return_value = True
            await system._trigger_alert(alert, {"current_value": Decimal("51000")})

            assert alert.status == AlertStatus.TRIGGERED

    @pytest.mark.asyncio
    async def test_max_triggers_reached(self, system):
        """Test max_triggers behavior."""
        alert = system.create_alert(
            name="Test",
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
            max_triggers=1,
            channels=[NotificationChannel.CONSOLE],
        )

        with patch.object(system._notifier, 'send', new_callable=AsyncMock) as mock:
            mock.return_value = True
            await system._trigger_alert(alert, {"current_value": Decimal("51000")})

            assert alert.status == AlertStatus.TRIGGERED
            assert alert.trigger_count == 1

    # -------------------------------------------------------------------------
    # History and Statistics Tests
    # -------------------------------------------------------------------------

    def test_get_trigger_history(self, system):
        """Test getting trigger history."""
        system._triggered_alerts = [
            TriggeredAlert(
                trigger_id="t1",
                alert_id="a1",
                triggered_at=datetime.utcnow(),
                condition_met="Test",
                current_value=Decimal("50000"),
                context={},
            ),
            TriggeredAlert(
                trigger_id="t2",
                alert_id="a2",
                triggered_at=datetime.utcnow(),
                condition_met="Test",
                current_value=Decimal("50000"),
                context={},
            ),
        ]

        # All history
        history = system.get_trigger_history()
        assert len(history) == 2

        # Filter by alert
        history = system.get_trigger_history(alert_id="a1")
        assert len(history) == 1

        # Limit
        history = system.get_trigger_history(limit=1)
        assert len(history) == 1

    def test_get_statistics(self, system):
        """Test getting statistics."""
        system.create_alert(
            name="Active",
            alert_type=AlertType.PRICE_ABOVE,
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
        )

        alert = system.create_alert(
            name="Disabled",
            alert_type=AlertType.RSI_OVERBOUGHT,
            symbol="ETHUSDT",
            threshold=Decimal("70"),
        )
        system.disable_alert(alert.alert_id)

        stats = system.get_statistics()

        assert stats["total_alerts"] == 2
        assert stats["active_alerts"] == 1
        assert stats["disabled_alerts"] == 1
        assert "price_above" in stats["alerts_by_type"]

    # -------------------------------------------------------------------------
    # Convenience Method Tests
    # -------------------------------------------------------------------------

    def test_create_price_alert(self, system):
        """Test convenience price alert creation."""
        alert = system.create_price_alert(
            symbol="BTCUSDT",
            threshold=Decimal("50000"),
            direction="above",
        )

        assert alert.condition.alert_type == AlertType.PRICE_ABOVE
        assert "above" in alert.name

        alert = system.create_price_alert(
            symbol="BTCUSDT",
            threshold=Decimal("45000"),
            direction="below",
        )

        assert alert.condition.alert_type == AlertType.PRICE_BELOW

    def test_create_volume_spike_alert(self, system):
        """Test convenience volume spike alert creation."""
        alert = system.create_volume_spike_alert(
            symbol="BTCUSDT",
            multiplier=Decimal("5"),
        )

        assert alert.condition.alert_type == AlertType.VOLUME_SPIKE
        assert alert.condition.threshold == Decimal("5")

    def test_create_whale_alert(self, system):
        """Test convenience whale alert creation."""
        alert = system.create_whale_alert(
            symbol="BTCUSDT",
            min_amount_usd=Decimal("2000000"),
            transfer_type="deposit",
        )

        assert alert.condition.alert_type == AlertType.WHALE_EXCHANGE_DEPOSIT
        assert alert.condition.threshold == Decimal("2000000")

    def test_create_rsi_alert(self, system):
        """Test convenience RSI alert creation."""
        alert = system.create_rsi_alert(
            symbol="BTCUSDT",
            threshold=Decimal("75"),
            condition="overbought",
        )

        assert alert.condition.alert_type == AlertType.RSI_OVERBOUGHT

        alert = system.create_rsi_alert(
            symbol="BTCUSDT",
            threshold=Decimal("25"),
            condition="oversold",
        )

        assert alert.condition.alert_type == AlertType.RSI_OVERSOLD

    def test_create_funding_rate_alert(self, system):
        """Test convenience funding rate alert creation."""
        alert = system.create_funding_rate_alert(
            symbol="BTCUSDT",
            threshold=Decimal("0.05"),
        )

        assert alert.condition.alert_type == AlertType.FUNDING_RATE
        assert alert.condition.comparison == "extreme"


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateAlertSystem:
    """Tests for create_alert_system factory."""

    def test_create_with_defaults(self):
        """Test creating system with defaults."""
        system = create_alert_system()

        assert system is not None
        assert system.config is not None

    def test_create_with_credentials(self):
        """Test creating system with credentials."""
        system = create_alert_system(
            telegram_bot_token="token",
            telegram_chat_id="chat",
            discord_webhook_url="https://discord.webhook",
        )

        assert system.config.telegram_bot_token == "token"
        assert system.config.telegram_chat_id == "chat"

    def test_create_with_callback(self):
        """Test creating system with market data callback."""
        def callback():
            return {"BTCUSDT": {"price": 50000}}

        system = create_alert_system(market_data_callback=callback)

        assert system.market_data_callback == callback
