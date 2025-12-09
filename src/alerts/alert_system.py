"""
Advanced Alert System.

Provides flexible alerting capabilities for:
- Price alerts (above, below, cross)
- Volume alerts (spike detection)
- Whale movement tracking
- On-chain metrics monitoring
- Technical indicator alerts
- Custom condition alerts
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set, Union
import json
import aiohttp
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of alerts."""

    # Price alerts
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    PRICE_CROSS = "price_cross"
    PRICE_CHANGE_PCT = "price_change_pct"

    # Volume alerts
    VOLUME_SPIKE = "volume_spike"
    VOLUME_ABOVE = "volume_above"
    OI_CHANGE = "oi_change"  # Open Interest

    # Whale alerts
    WHALE_TRANSFER = "whale_transfer"
    WHALE_EXCHANGE_DEPOSIT = "whale_exchange_deposit"
    WHALE_EXCHANGE_WITHDRAWAL = "whale_exchange_withdrawal"

    # On-chain metrics
    ACTIVE_ADDRESSES = "active_addresses"
    NVT_RATIO = "nvt_ratio"
    MVRV_RATIO = "mvrv_ratio"
    FUNDING_RATE = "funding_rate"
    LIQUIDATION_VOLUME = "liquidation_volume"

    # Technical indicators
    RSI_OVERBOUGHT = "rsi_overbought"
    RSI_OVERSOLD = "rsi_oversold"
    MACD_CROSS = "macd_cross"
    BB_BREAKOUT = "bb_breakout"  # Bollinger Bands
    SUPPORT_BREAK = "support_break"
    RESISTANCE_BREAK = "resistance_break"

    # Market structure
    NEW_HIGH = "new_high"
    NEW_LOW = "new_low"
    TREND_CHANGE = "trend_change"

    # Custom
    CUSTOM = "custom"


class AlertPriority(Enum):
    """Priority levels for alerts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Status of an alert."""

    ACTIVE = "active"
    TRIGGERED = "triggered"
    EXPIRED = "expired"
    DISABLED = "disabled"
    COOLDOWN = "cooldown"


class NotificationChannel(Enum):
    """Notification delivery channels."""

    TELEGRAM = "telegram"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    CONSOLE = "console"


@dataclass
class AlertCondition:
    """Base alert condition configuration."""

    alert_type: AlertType
    symbol: str
    threshold: Decimal
    comparison: str = "above"  # above, below, cross, change
    timeframe: str = "1m"  # For indicator-based alerts

    # Additional parameters for specific alert types
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_type": self.alert_type.value,
            "symbol": self.symbol,
            "threshold": str(self.threshold),
            "comparison": self.comparison,
            "timeframe": self.timeframe,
            "params": self.params
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlertCondition":
        """Create from dictionary."""
        return cls(
            alert_type=AlertType(data["alert_type"]),
            symbol=data["symbol"],
            threshold=Decimal(data["threshold"]),
            comparison=data.get("comparison", "above"),
            timeframe=data.get("timeframe", "1m"),
            params=data.get("params", {})
        )


@dataclass
class Alert:
    """Alert definition."""

    alert_id: str
    name: str
    condition: AlertCondition
    priority: AlertPriority = AlertPriority.MEDIUM
    status: AlertStatus = AlertStatus.ACTIVE

    # Notification settings
    channels: List[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.TELEGRAM]
    )
    message_template: Optional[str] = None

    # Trigger settings
    trigger_once: bool = False  # If True, disable after first trigger
    cooldown_seconds: int = 300  # Minimum time between triggers
    max_triggers: Optional[int] = None  # Max number of triggers

    # Time constraints
    active_hours: Optional[List[int]] = None  # Hours when alert is active (0-23)
    expires_at: Optional[datetime] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    user_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def can_trigger(self) -> bool:
        """Check if alert can be triggered."""
        if self.status != AlertStatus.ACTIVE:
            return False

        # Check expiration
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False

        # Check max triggers
        if self.max_triggers and self.trigger_count >= self.max_triggers:
            return False

        # Check cooldown
        if self.last_triggered:
            elapsed = (datetime.utcnow() - self.last_triggered).total_seconds()
            if elapsed < self.cooldown_seconds:
                return False

        # Check active hours
        if self.active_hours:
            current_hour = datetime.utcnow().hour
            if current_hour not in self.active_hours:
                return False

        return True

    def format_message(self, context: Dict[str, Any]) -> str:
        """Format alert message with context."""
        if self.message_template:
            try:
                return self.message_template.format(**context)
            except KeyError:
                pass

        # Default message format
        return (
            f"[{self.priority.value.upper()}] {self.name}\n"
            f"Symbol: {self.condition.symbol}\n"
            f"Type: {self.condition.alert_type.value}\n"
            f"Threshold: {self.condition.threshold}\n"
            f"Current: {context.get('current_value', 'N/A')}"
        )


@dataclass
class TriggeredAlert:
    """Record of a triggered alert."""

    trigger_id: str
    alert_id: str
    triggered_at: datetime
    condition_met: str
    current_value: Any
    context: Dict[str, Any]
    notifications_sent: List[str] = field(default_factory=list)


class AlertConditionChecker(ABC):
    """Base class for alert condition checkers."""

    @abstractmethod
    async def check(
        self,
        condition: AlertCondition,
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if condition is met.

        Returns context dict if triggered, None otherwise.
        """
        pass


class PriceAlertChecker(AlertConditionChecker):
    """Checker for price-based alerts."""

    def __init__(self):
        self._last_prices: Dict[str, Decimal] = {}

    async def check(
        self,
        condition: AlertCondition,
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check price conditions."""
        current_price = Decimal(str(market_data.get("price", 0)))
        if current_price == 0:
            return None

        last_price = self._last_prices.get(condition.symbol)
        self._last_prices[condition.symbol] = current_price

        triggered = False
        condition_met = ""

        if condition.alert_type == AlertType.PRICE_ABOVE:
            if current_price > condition.threshold:
                triggered = True
                condition_met = f"Price {current_price} > {condition.threshold}"

        elif condition.alert_type == AlertType.PRICE_BELOW:
            if current_price < condition.threshold:
                triggered = True
                condition_met = f"Price {current_price} < {condition.threshold}"

        elif condition.alert_type == AlertType.PRICE_CROSS:
            if last_price:
                # Crossed from below
                if last_price < condition.threshold <= current_price:
                    triggered = True
                    condition_met = f"Price crossed above {condition.threshold}"
                # Crossed from above
                elif last_price > condition.threshold >= current_price:
                    triggered = True
                    condition_met = f"Price crossed below {condition.threshold}"

        elif condition.alert_type == AlertType.PRICE_CHANGE_PCT:
            if last_price and last_price > 0:
                change_pct = ((current_price - last_price) / last_price) * 100
                if abs(change_pct) >= condition.threshold:
                    triggered = True
                    direction = "up" if change_pct > 0 else "down"
                    condition_met = f"Price moved {direction} {abs(change_pct):.2f}%"

        if triggered:
            return {
                "current_value": current_price,
                "previous_value": last_price,
                "threshold": condition.threshold,
                "condition_met": condition_met
            }

        return None


class VolumeAlertChecker(AlertConditionChecker):
    """Checker for volume-based alerts."""

    def __init__(self, lookback_periods: int = 20):
        self.lookback_periods = lookback_periods
        self._volume_history: Dict[str, List[Decimal]] = defaultdict(list)

    async def check(
        self,
        condition: AlertCondition,
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check volume conditions."""
        current_volume = Decimal(str(market_data.get("volume", 0)))
        if current_volume == 0:
            return None

        # Update history
        history = self._volume_history[condition.symbol]
        history.append(current_volume)
        if len(history) > self.lookback_periods:
            history.pop(0)

        triggered = False
        condition_met = ""

        if condition.alert_type == AlertType.VOLUME_SPIKE:
            if len(history) >= 5:
                avg_volume = sum(history[:-1]) / len(history[:-1])
                if avg_volume > 0:
                    spike_ratio = current_volume / avg_volume
                    if spike_ratio >= condition.threshold:
                        triggered = True
                        condition_met = f"Volume spike: {spike_ratio:.1f}x average"

        elif condition.alert_type == AlertType.VOLUME_ABOVE:
            if current_volume > condition.threshold:
                triggered = True
                condition_met = f"Volume {current_volume:,.0f} > {condition.threshold:,.0f}"

        elif condition.alert_type == AlertType.OI_CHANGE:
            oi = Decimal(str(market_data.get("open_interest", 0)))
            oi_change_pct = Decimal(str(market_data.get("oi_change_pct", 0)))
            if abs(oi_change_pct) >= condition.threshold:
                triggered = True
                direction = "increased" if oi_change_pct > 0 else "decreased"
                condition_met = f"OI {direction} by {abs(oi_change_pct):.1f}%"

        if triggered:
            return {
                "current_value": current_volume,
                "avg_volume": sum(history) / len(history) if history else Decimal("0"),
                "condition_met": condition_met
            }

        return None


class WhaleAlertChecker(AlertConditionChecker):
    """Checker for whale movement alerts."""

    def __init__(self, whale_api_url: Optional[str] = None):
        self.whale_api_url = whale_api_url
        self._processed_txs: Set[str] = set()

    async def check(
        self,
        condition: AlertCondition,
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check whale movement conditions."""
        whale_data = market_data.get("whale_transfers", [])

        for transfer in whale_data:
            tx_hash = transfer.get("tx_hash", "")
            if tx_hash in self._processed_txs:
                continue

            amount_usd = Decimal(str(transfer.get("amount_usd", 0)))

            if amount_usd >= condition.threshold:
                self._processed_txs.add(tx_hash)

                # Keep set size manageable
                if len(self._processed_txs) > 10000:
                    self._processed_txs = set(list(self._processed_txs)[-5000:])

                transfer_type = transfer.get("type", "transfer")
                from_addr = transfer.get("from", "")[:10]
                to_addr = transfer.get("to", "")[:10]

                # Check if matches specific alert type
                if condition.alert_type == AlertType.WHALE_EXCHANGE_DEPOSIT:
                    if not transfer.get("to_exchange"):
                        continue
                elif condition.alert_type == AlertType.WHALE_EXCHANGE_WITHDRAWAL:
                    if not transfer.get("from_exchange"):
                        continue

                return {
                    "current_value": amount_usd,
                    "tx_hash": tx_hash,
                    "from": from_addr,
                    "to": to_addr,
                    "transfer_type": transfer_type,
                    "condition_met": f"Whale {transfer_type}: ${amount_usd:,.0f}"
                }

        return None


class OnChainMetricsChecker(AlertConditionChecker):
    """Checker for on-chain metrics alerts."""

    async def check(
        self,
        condition: AlertCondition,
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check on-chain metric conditions."""
        metrics = market_data.get("onchain_metrics", {})

        metric_map = {
            AlertType.ACTIVE_ADDRESSES: "active_addresses",
            AlertType.NVT_RATIO: "nvt_ratio",
            AlertType.MVRV_RATIO: "mvrv_ratio",
            AlertType.FUNDING_RATE: "funding_rate",
            AlertType.LIQUIDATION_VOLUME: "liquidation_volume",
        }

        metric_key = metric_map.get(condition.alert_type)
        if not metric_key or metric_key not in metrics:
            return None

        current_value = Decimal(str(metrics[metric_key]))

        triggered = False
        condition_met = ""

        if condition.comparison == "above":
            if current_value > condition.threshold:
                triggered = True
                condition_met = f"{metric_key} {current_value} > {condition.threshold}"
        elif condition.comparison == "below":
            if current_value < condition.threshold:
                triggered = True
                condition_met = f"{metric_key} {current_value} < {condition.threshold}"
        elif condition.comparison == "extreme":
            # Check for extreme values (used for funding rate, etc.)
            if abs(current_value) > condition.threshold:
                triggered = True
                condition_met = f"{metric_key} at extreme: {current_value}"

        if triggered:
            return {
                "current_value": current_value,
                "metric": metric_key,
                "condition_met": condition_met
            }

        return None


class TechnicalIndicatorChecker(AlertConditionChecker):
    """Checker for technical indicator alerts."""

    def __init__(self):
        self._indicator_values: Dict[str, Dict[str, Any]] = defaultdict(dict)

    async def check(
        self,
        condition: AlertCondition,
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check technical indicator conditions."""
        indicators = market_data.get("indicators", {})
        symbol = condition.symbol

        triggered = False
        condition_met = ""
        current_value = None

        if condition.alert_type == AlertType.RSI_OVERBOUGHT:
            rsi = indicators.get("rsi")
            if rsi and rsi >= condition.threshold:
                triggered = True
                current_value = rsi
                condition_met = f"RSI overbought: {rsi:.1f}"

        elif condition.alert_type == AlertType.RSI_OVERSOLD:
            rsi = indicators.get("rsi")
            threshold = condition.threshold if condition.threshold > 0 else Decimal("30")
            if rsi and rsi <= threshold:
                triggered = True
                current_value = rsi
                condition_met = f"RSI oversold: {rsi:.1f}"

        elif condition.alert_type == AlertType.MACD_CROSS:
            macd = indicators.get("macd")
            signal = indicators.get("macd_signal")
            prev_macd = self._indicator_values[symbol].get("macd")
            prev_signal = self._indicator_values[symbol].get("macd_signal")

            if all(v is not None for v in [macd, signal, prev_macd, prev_signal]):
                # Bullish cross
                if prev_macd < prev_signal and macd >= signal:
                    triggered = True
                    current_value = macd
                    condition_met = "MACD bullish crossover"
                # Bearish cross
                elif prev_macd > prev_signal and macd <= signal:
                    triggered = True
                    current_value = macd
                    condition_met = "MACD bearish crossover"

            # Update stored values
            if macd is not None:
                self._indicator_values[symbol]["macd"] = macd
            if signal is not None:
                self._indicator_values[symbol]["macd_signal"] = signal

        elif condition.alert_type == AlertType.BB_BREAKOUT:
            price = Decimal(str(market_data.get("price", 0)))
            bb_upper = indicators.get("bb_upper")
            bb_lower = indicators.get("bb_lower")

            if bb_upper and price > bb_upper:
                triggered = True
                current_value = price
                condition_met = f"Price broke above upper BB ({bb_upper:.2f})"
            elif bb_lower and price < bb_lower:
                triggered = True
                current_value = price
                condition_met = f"Price broke below lower BB ({bb_lower:.2f})"

        if triggered:
            return {
                "current_value": current_value,
                "indicators": indicators,
                "condition_met": condition_met
            }

        return None


class NotificationSender:
    """Handles sending notifications through various channels."""

    def __init__(
        self,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        discord_webhook_url: Optional[str] = None,
        webhook_urls: Optional[Dict[str, str]] = None,
        email_config: Optional[Dict[str, str]] = None
    ):
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.discord_webhook_url = discord_webhook_url
        self.webhook_urls = webhook_urls or {}
        self.email_config = email_config or {}

    async def send(
        self,
        channel: NotificationChannel,
        message: str,
        priority: AlertPriority,
        extra: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send notification through specified channel."""
        try:
            if channel == NotificationChannel.TELEGRAM:
                return await self._send_telegram(message, priority)
            elif channel == NotificationChannel.DISCORD:
                return await self._send_discord(message, priority)
            elif channel == NotificationChannel.WEBHOOK:
                return await self._send_webhook(message, priority, extra)
            elif channel == NotificationChannel.CONSOLE:
                return self._send_console(message, priority)
            else:
                logger.warning(f"Unsupported notification channel: {channel}")
                return False
        except Exception as e:
            logger.error(f"Error sending notification via {channel}: {e}")
            return False

    async def _send_telegram(self, message: str, priority: AlertPriority) -> bool:
        """Send Telegram notification."""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            logger.warning("Telegram not configured")
            return False

        # Add priority emoji
        priority_emoji = {
            AlertPriority.LOW: "",
            AlertPriority.MEDIUM: "",
            AlertPriority.HIGH: "",
            AlertPriority.CRITICAL: ""
        }
        formatted_message = f"{priority_emoji[priority]} {message}"

        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": formatted_message,
            "parse_mode": "HTML"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                return response.status == 200

    async def _send_discord(self, message: str, priority: AlertPriority) -> bool:
        """Send Discord notification."""
        if not self.discord_webhook_url:
            logger.warning("Discord webhook not configured")
            return False

        # Color based on priority
        color_map = {
            AlertPriority.LOW: 0x808080,  # Gray
            AlertPriority.MEDIUM: 0x3498db,  # Blue
            AlertPriority.HIGH: 0xf39c12,  # Orange
            AlertPriority.CRITICAL: 0xe74c3c  # Red
        }

        payload = {
            "embeds": [{
                "title": f"Alert [{priority.value.upper()}]",
                "description": message,
                "color": color_map[priority],
                "timestamp": datetime.utcnow().isoformat()
            }]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.discord_webhook_url, json=payload) as response:
                return response.status == 204

    async def _send_webhook(
        self,
        message: str,
        priority: AlertPriority,
        extra: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send webhook notification."""
        webhook_url = self.webhook_urls.get("default")
        if not webhook_url:
            logger.warning("Webhook URL not configured")
            return False

        payload = {
            "message": message,
            "priority": priority.value,
            "timestamp": datetime.utcnow().isoformat(),
            **(extra or {})
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                return response.status in [200, 201, 204]

    def _send_console(self, message: str, priority: AlertPriority) -> bool:
        """Send console notification."""
        prefix = f"[{priority.value.upper()}]"
        print(f"\n{'='*50}\nALERT {prefix}\n{message}\n{'='*50}\n")
        return True


@dataclass
class AlertSystemConfig:
    """Configuration for the alert system."""

    # Check intervals
    check_interval_seconds: float = 5.0
    cleanup_interval_seconds: float = 3600.0  # 1 hour

    # Limits
    max_alerts_per_user: int = 100
    max_triggers_per_minute: int = 30

    # Data sources
    price_source: str = "websocket"  # websocket, rest
    whale_tracking_enabled: bool = True
    onchain_metrics_enabled: bool = False

    # Notification settings
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    discord_webhook_url: Optional[str] = None
    webhook_urls: Dict[str, str] = field(default_factory=dict)

    # Persistence
    storage_path: Optional[str] = None


class AdvancedAlertSystem:
    """
    Advanced alert system for crypto trading.

    Supports multiple alert types, notification channels,
    and complex condition checking.
    """

    def __init__(
        self,
        config: AlertSystemConfig,
        market_data_callback: Optional[Callable[[], Dict[str, Any]]] = None
    ):
        self.config = config
        self.market_data_callback = market_data_callback

        # Alert storage
        self._alerts: Dict[str, Alert] = {}
        self._triggered_alerts: List[TriggeredAlert] = []
        self._triggers_this_minute: int = 0
        self._last_trigger_reset: datetime = datetime.utcnow()

        # Condition checkers
        self._checkers: Dict[AlertType, AlertConditionChecker] = {
            AlertType.PRICE_ABOVE: PriceAlertChecker(),
            AlertType.PRICE_BELOW: PriceAlertChecker(),
            AlertType.PRICE_CROSS: PriceAlertChecker(),
            AlertType.PRICE_CHANGE_PCT: PriceAlertChecker(),
            AlertType.VOLUME_SPIKE: VolumeAlertChecker(),
            AlertType.VOLUME_ABOVE: VolumeAlertChecker(),
            AlertType.OI_CHANGE: VolumeAlertChecker(),
            AlertType.WHALE_TRANSFER: WhaleAlertChecker(),
            AlertType.WHALE_EXCHANGE_DEPOSIT: WhaleAlertChecker(),
            AlertType.WHALE_EXCHANGE_WITHDRAWAL: WhaleAlertChecker(),
            AlertType.ACTIVE_ADDRESSES: OnChainMetricsChecker(),
            AlertType.NVT_RATIO: OnChainMetricsChecker(),
            AlertType.MVRV_RATIO: OnChainMetricsChecker(),
            AlertType.FUNDING_RATE: OnChainMetricsChecker(),
            AlertType.LIQUIDATION_VOLUME: OnChainMetricsChecker(),
            AlertType.RSI_OVERBOUGHT: TechnicalIndicatorChecker(),
            AlertType.RSI_OVERSOLD: TechnicalIndicatorChecker(),
            AlertType.MACD_CROSS: TechnicalIndicatorChecker(),
            AlertType.BB_BREAKOUT: TechnicalIndicatorChecker(),
        }

        # Notification sender
        self._notifier = NotificationSender(
            telegram_bot_token=config.telegram_bot_token,
            telegram_chat_id=config.telegram_chat_id,
            discord_webhook_url=config.discord_webhook_url,
            webhook_urls=config.webhook_urls
        )

        self._running = False
        self._check_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the alert system."""
        self._running = True
        self._check_task = asyncio.create_task(self._check_loop())
        logger.info("Advanced Alert System started")

    async def stop(self) -> None:
        """Stop the alert system."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("Advanced Alert System stopped")

    def create_alert(
        self,
        name: str,
        alert_type: AlertType,
        symbol: str,
        threshold: Decimal,
        comparison: str = "above",
        priority: AlertPriority = AlertPriority.MEDIUM,
        channels: Optional[List[NotificationChannel]] = None,
        message_template: Optional[str] = None,
        trigger_once: bool = False,
        cooldown_seconds: int = 300,
        max_triggers: Optional[int] = None,
        expires_at: Optional[datetime] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Create a new alert."""
        # Check user limits
        if user_id:
            user_alerts = [a for a in self._alerts.values() if a.user_id == user_id]
            if len(user_alerts) >= self.config.max_alerts_per_user:
                raise ValueError(f"User has reached max alerts limit ({self.config.max_alerts_per_user})")

        # Generate alert ID
        alert_id = hashlib.md5(
            f"{name}{symbol}{alert_type.value}{datetime.utcnow().timestamp()}".encode()
        ).hexdigest()[:16]

        condition = AlertCondition(
            alert_type=alert_type,
            symbol=symbol,
            threshold=threshold,
            comparison=comparison,
            params=params or {}
        )

        alert = Alert(
            alert_id=alert_id,
            name=name,
            condition=condition,
            priority=priority,
            channels=channels or [NotificationChannel.TELEGRAM],
            message_template=message_template,
            trigger_once=trigger_once,
            cooldown_seconds=cooldown_seconds,
            max_triggers=max_triggers,
            expires_at=expires_at,
            user_id=user_id,
            tags=tags or []
        )

        self._alerts[alert_id] = alert
        logger.info(f"Created alert: {alert_id} - {name}")

        return alert

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID."""
        return self._alerts.get(alert_id)

    def get_alerts(
        self,
        user_id: Optional[str] = None,
        symbol: Optional[str] = None,
        alert_type: Optional[AlertType] = None,
        status: Optional[AlertStatus] = None
    ) -> List[Alert]:
        """Get alerts with optional filters."""
        alerts = list(self._alerts.values())

        if user_id:
            alerts = [a for a in alerts if a.user_id == user_id]
        if symbol:
            alerts = [a for a in alerts if a.condition.symbol == symbol]
        if alert_type:
            alerts = [a for a in alerts if a.condition.alert_type == alert_type]
        if status:
            alerts = [a for a in alerts if a.status == status]

        return alerts

    def update_alert(
        self,
        alert_id: str,
        **updates
    ) -> Optional[Alert]:
        """Update an existing alert."""
        alert = self._alerts.get(alert_id)
        if not alert:
            return None

        for key, value in updates.items():
            if hasattr(alert, key):
                setattr(alert, key, value)
            elif hasattr(alert.condition, key):
                setattr(alert.condition, key, value)

        return alert

    def delete_alert(self, alert_id: str) -> bool:
        """Delete an alert."""
        if alert_id in self._alerts:
            del self._alerts[alert_id]
            logger.info(f"Deleted alert: {alert_id}")
            return True
        return False

    def disable_alert(self, alert_id: str) -> bool:
        """Disable an alert."""
        alert = self._alerts.get(alert_id)
        if alert:
            alert.status = AlertStatus.DISABLED
            return True
        return False

    def enable_alert(self, alert_id: str) -> bool:
        """Enable a disabled alert."""
        alert = self._alerts.get(alert_id)
        if alert:
            alert.status = AlertStatus.ACTIVE
            return True
        return False

    async def _check_loop(self) -> None:
        """Main alert checking loop."""
        while self._running:
            try:
                # Reset trigger counter every minute
                now = datetime.utcnow()
                if (now - self._last_trigger_reset).total_seconds() >= 60:
                    self._triggers_this_minute = 0
                    self._last_trigger_reset = now

                # Get market data
                market_data = {}
                if self.market_data_callback:
                    try:
                        market_data = self.market_data_callback()
                    except Exception as e:
                        logger.error(f"Error getting market data: {e}")

                # Check each active alert
                for alert in list(self._alerts.values()):
                    if not alert.can_trigger():
                        continue

                    if self._triggers_this_minute >= self.config.max_triggers_per_minute:
                        logger.warning("Max triggers per minute reached")
                        break

                    await self._check_alert(alert, market_data)

                await asyncio.sleep(self.config.check_interval_seconds)

            except Exception as e:
                logger.error(f"Error in alert check loop: {e}")
                await asyncio.sleep(1)

    async def _check_alert(
        self,
        alert: Alert,
        market_data: Dict[str, Any]
    ) -> None:
        """Check a single alert."""
        checker = self._checkers.get(alert.condition.alert_type)
        if not checker:
            return

        # Get symbol-specific data
        symbol = alert.condition.symbol
        symbol_data = market_data.get(symbol, market_data)

        # Check condition
        context = await checker.check(alert.condition, symbol_data)
        if not context:
            return

        # Trigger the alert
        await self._trigger_alert(alert, context)

    async def _trigger_alert(
        self,
        alert: Alert,
        context: Dict[str, Any]
    ) -> None:
        """Trigger an alert and send notifications."""
        # Update alert state
        alert.last_triggered = datetime.utcnow()
        alert.trigger_count += 1
        self._triggers_this_minute += 1

        # Create trigger record
        trigger = TriggeredAlert(
            trigger_id=f"{alert.alert_id}_{datetime.utcnow().timestamp()}",
            alert_id=alert.alert_id,
            triggered_at=datetime.utcnow(),
            condition_met=context.get("condition_met", ""),
            current_value=context.get("current_value"),
            context=context
        )

        # Format message
        message = alert.format_message(context)

        # Send notifications
        for channel in alert.channels:
            success = await self._notifier.send(
                channel=channel,
                message=message,
                priority=alert.priority,
                extra={"alert_id": alert.alert_id, "symbol": alert.condition.symbol}
            )
            if success:
                trigger.notifications_sent.append(channel.value)

        self._triggered_alerts.append(trigger)
        logger.info(
            f"Alert triggered: {alert.name} - {context.get('condition_met', '')}"
        )

        # Handle trigger_once
        if alert.trigger_once:
            alert.status = AlertStatus.TRIGGERED

        # Handle max_triggers
        if alert.max_triggers and alert.trigger_count >= alert.max_triggers:
            alert.status = AlertStatus.TRIGGERED

    def get_trigger_history(
        self,
        alert_id: Optional[str] = None,
        limit: int = 100
    ) -> List[TriggeredAlert]:
        """Get alert trigger history."""
        history = self._triggered_alerts

        if alert_id:
            history = [t for t in history if t.alert_id == alert_id]

        return sorted(history, key=lambda t: t.triggered_at, reverse=True)[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get alert system statistics."""
        active_alerts = len([a for a in self._alerts.values() if a.status == AlertStatus.ACTIVE])
        disabled_alerts = len([a for a in self._alerts.values() if a.status == AlertStatus.DISABLED])
        triggered_today = len([
            t for t in self._triggered_alerts
            if t.triggered_at.date() == datetime.utcnow().date()
        ])

        alerts_by_type = defaultdict(int)
        for alert in self._alerts.values():
            alerts_by_type[alert.condition.alert_type.value] += 1

        return {
            "total_alerts": len(self._alerts),
            "active_alerts": active_alerts,
            "disabled_alerts": disabled_alerts,
            "triggered_today": triggered_today,
            "total_triggers": len(self._triggered_alerts),
            "triggers_this_minute": self._triggers_this_minute,
            "alerts_by_type": dict(alerts_by_type)
        }

    # Convenience methods for common alert types

    def create_price_alert(
        self,
        symbol: str,
        threshold: Decimal,
        direction: str = "above",
        name: Optional[str] = None,
        **kwargs
    ) -> Alert:
        """Create a price alert."""
        alert_type = AlertType.PRICE_ABOVE if direction == "above" else AlertType.PRICE_BELOW
        name = name or f"{symbol} price {direction} {threshold}"
        return self.create_alert(
            name=name,
            alert_type=alert_type,
            symbol=symbol,
            threshold=threshold,
            comparison=direction,
            **kwargs
        )

    def create_volume_spike_alert(
        self,
        symbol: str,
        multiplier: Decimal = Decimal("3"),
        name: Optional[str] = None,
        **kwargs
    ) -> Alert:
        """Create a volume spike alert."""
        name = name or f"{symbol} volume spike {multiplier}x"
        return self.create_alert(
            name=name,
            alert_type=AlertType.VOLUME_SPIKE,
            symbol=symbol,
            threshold=multiplier,
            **kwargs
        )

    def create_whale_alert(
        self,
        symbol: str,
        min_amount_usd: Decimal = Decimal("1000000"),
        transfer_type: str = "any",
        name: Optional[str] = None,
        **kwargs
    ) -> Alert:
        """Create a whale movement alert."""
        type_map = {
            "any": AlertType.WHALE_TRANSFER,
            "deposit": AlertType.WHALE_EXCHANGE_DEPOSIT,
            "withdrawal": AlertType.WHALE_EXCHANGE_WITHDRAWAL
        }
        alert_type = type_map.get(transfer_type, AlertType.WHALE_TRANSFER)
        name = name or f"{symbol} whale {transfer_type} >${min_amount_usd:,.0f}"
        return self.create_alert(
            name=name,
            alert_type=alert_type,
            symbol=symbol,
            threshold=min_amount_usd,
            **kwargs
        )

    def create_rsi_alert(
        self,
        symbol: str,
        threshold: Decimal = Decimal("70"),
        condition: str = "overbought",
        name: Optional[str] = None,
        **kwargs
    ) -> Alert:
        """Create an RSI alert."""
        alert_type = AlertType.RSI_OVERBOUGHT if condition == "overbought" else AlertType.RSI_OVERSOLD
        name = name or f"{symbol} RSI {condition}"
        return self.create_alert(
            name=name,
            alert_type=alert_type,
            symbol=symbol,
            threshold=threshold,
            **kwargs
        )

    def create_funding_rate_alert(
        self,
        symbol: str,
        threshold: Decimal = Decimal("0.1"),
        name: Optional[str] = None,
        **kwargs
    ) -> Alert:
        """Create a funding rate alert."""
        name = name or f"{symbol} extreme funding rate"
        return self.create_alert(
            name=name,
            alert_type=AlertType.FUNDING_RATE,
            symbol=symbol,
            threshold=threshold,
            comparison="extreme",
            **kwargs
        )


# Convenience function to create the alert system
def create_alert_system(
    telegram_bot_token: Optional[str] = None,
    telegram_chat_id: Optional[str] = None,
    discord_webhook_url: Optional[str] = None,
    market_data_callback: Optional[Callable[[], Dict[str, Any]]] = None
) -> AdvancedAlertSystem:
    """Create and configure the advanced alert system."""
    config = AlertSystemConfig(
        telegram_bot_token=telegram_bot_token,
        telegram_chat_id=telegram_chat_id,
        discord_webhook_url=discord_webhook_url
    )

    return AdvancedAlertSystem(
        config=config,
        market_data_callback=market_data_callback
    )
