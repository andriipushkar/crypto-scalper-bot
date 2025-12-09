"""
Mobile push notifications for trading alerts.

Supports multiple push notification services:
- Firebase Cloud Messaging (FCM) for Android/iOS
- Pushover for simple mobile notifications
- NTFY.sh for self-hosted notifications
"""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional

import aiohttp
from loguru import logger


# =============================================================================
# Notification Types
# =============================================================================

class NotificationPriority(Enum):
    """Push notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationType(Enum):
    """Types of trading notifications."""
    TRADE_EXECUTED = "trade_executed"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    STOP_LOSS_HIT = "stop_loss_hit"
    TAKE_PROFIT_HIT = "take_profit_hit"
    LIQUIDATION_RISK = "liquidation_risk"
    DRAWDOWN_WARNING = "drawdown_warning"
    CONNECTION_LOST = "connection_lost"
    CONNECTION_RESTORED = "connection_restored"
    DAILY_SUMMARY = "daily_summary"
    ERROR = "error"


@dataclass
class PushNotification:
    """Push notification data."""
    title: str
    body: str
    notification_type: NotificationType
    priority: NotificationPriority = NotificationPriority.NORMAL
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    # Optional fields
    image_url: Optional[str] = None
    action_url: Optional[str] = None
    sound: str = "default"
    badge: int = 1


# =============================================================================
# Base Provider
# =============================================================================

class PushProvider(ABC):
    """Abstract base class for push notification providers."""

    @abstractmethod
    async def send(self, notification: PushNotification) -> bool:
        """Send a push notification."""
        pass

    @abstractmethod
    async def send_to_topic(self, notification: PushNotification, topic: str) -> bool:
        """Send notification to a topic/channel."""
        pass

    async def health_check(self) -> bool:
        """Check if the provider is accessible."""
        return True


# =============================================================================
# Firebase Cloud Messaging (FCM)
# =============================================================================

class FCMProvider(PushProvider):
    """
    Firebase Cloud Messaging provider.

    Requires:
        - FIREBASE_SERVER_KEY: FCM server key
        - Device tokens registered from mobile app
    """

    FCM_URL = "https://fcm.googleapis.com/fcm/send"

    def __init__(self, server_key: str):
        self.server_key = server_key
        self._device_tokens: List[str] = []
        self._session: Optional[aiohttp.ClientSession] = None

    def register_device(self, token: str) -> None:
        """Register a device token for notifications."""
        if token not in self._device_tokens:
            self._device_tokens.append(token)
            logger.info(f"Registered FCM device: {token[:20]}...")

    def unregister_device(self, token: str) -> None:
        """Unregister a device token."""
        if token in self._device_tokens:
            self._device_tokens.remove(token)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"key={self.server_key}",
                    "Content-Type": "application/json",
                }
            )
        return self._session

    def _build_payload(
        self,
        notification: PushNotification,
        target: str,
        is_topic: bool = False,
    ) -> Dict[str, Any]:
        """Build FCM payload."""
        payload = {
            "notification": {
                "title": notification.title,
                "body": notification.body,
                "sound": notification.sound,
                "badge": notification.badge,
            },
            "data": {
                "type": notification.notification_type.value,
                "timestamp": notification.timestamp.isoformat(),
                **notification.data,
            },
            "priority": "high" if notification.priority in [
                NotificationPriority.HIGH,
                NotificationPriority.URGENT
            ] else "normal",
        }

        if is_topic:
            payload["to"] = f"/topics/{target}"
        else:
            payload["to"] = target

        if notification.image_url:
            payload["notification"]["image"] = notification.image_url

        if notification.action_url:
            payload["data"]["action_url"] = notification.action_url

        return payload

    async def send(self, notification: PushNotification) -> bool:
        """Send notification to all registered devices."""
        if not self._device_tokens:
            logger.warning("No FCM devices registered")
            return False

        session = await self._get_session()
        success_count = 0

        for token in self._device_tokens:
            payload = self._build_payload(notification, token)

            try:
                async with session.post(self.FCM_URL, json=payload) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if result.get("success", 0) > 0:
                            success_count += 1
                        else:
                            logger.warning(f"FCM delivery failed: {result}")
                    else:
                        logger.error(f"FCM request failed: {resp.status}")
            except Exception as e:
                logger.error(f"FCM send error: {e}")

        return success_count > 0

    async def send_to_topic(self, notification: PushNotification, topic: str) -> bool:
        """Send notification to a topic."""
        session = await self._get_session()
        payload = self._build_payload(notification, topic, is_topic=True)

        try:
            async with session.post(self.FCM_URL, json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("message_id") is not None
                else:
                    logger.error(f"FCM topic send failed: {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"FCM topic send error: {e}")
            return False

    async def close(self) -> None:
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()


# =============================================================================
# Pushover
# =============================================================================

class PushoverProvider(PushProvider):
    """
    Pushover notification provider.

    Requires:
        - PUSHOVER_APP_TOKEN: Application API token
        - PUSHOVER_USER_KEY: User/Group key
    """

    PUSHOVER_URL = "https://api.pushover.net/1/messages.json"

    PRIORITY_MAP = {
        NotificationPriority.LOW: -1,
        NotificationPriority.NORMAL: 0,
        NotificationPriority.HIGH: 1,
        NotificationPriority.URGENT: 2,
    }

    SOUND_MAP = {
        NotificationType.TRADE_EXECUTED: "cashregister",
        NotificationType.POSITION_OPENED: "bugle",
        NotificationType.POSITION_CLOSED: "magic",
        NotificationType.STOP_LOSS_HIT: "falling",
        NotificationType.TAKE_PROFIT_HIT: "cosmic",
        NotificationType.LIQUIDATION_RISK: "siren",
        NotificationType.DRAWDOWN_WARNING: "vibrate",
        NotificationType.CONNECTION_LOST: "alien",
        NotificationType.CONNECTION_RESTORED: "echo",
        NotificationType.ERROR: "mechanical",
    }

    def __init__(self, app_token: str, user_key: str):
        self.app_token = app_token
        self.user_key = user_key
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def send(self, notification: PushNotification) -> bool:
        """Send Pushover notification."""
        session = await self._get_session()

        priority = self.PRIORITY_MAP.get(notification.priority, 0)
        sound = self.SOUND_MAP.get(notification.notification_type, "pushover")

        data = {
            "token": self.app_token,
            "user": self.user_key,
            "title": notification.title,
            "message": notification.body,
            "priority": priority,
            "sound": sound,
            "timestamp": int(notification.timestamp.timestamp()),
        }

        # Emergency priority requires retry/expire
        if priority == 2:
            data["retry"] = 60  # Retry every 60 seconds
            data["expire"] = 3600  # Expire after 1 hour

        if notification.action_url:
            data["url"] = notification.action_url
            data["url_title"] = "Open Dashboard"

        if notification.image_url:
            data["attachment_base64"] = ""  # Would need to fetch and encode

        try:
            async with session.post(self.PUSHOVER_URL, data=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("status") == 1
                else:
                    text = await resp.text()
                    logger.error(f"Pushover failed: {resp.status} - {text}")
                    return False
        except Exception as e:
            logger.error(f"Pushover send error: {e}")
            return False

    async def send_to_topic(self, notification: PushNotification, topic: str) -> bool:
        """Pushover doesn't support topics, send to default user."""
        return await self.send(notification)

    async def health_check(self) -> bool:
        """Validate Pushover credentials."""
        session = await self._get_session()

        url = "https://api.pushover.net/1/users/validate.json"
        data = {
            "token": self.app_token,
            "user": self.user_key,
        }

        try:
            async with session.post(url, data=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("status") == 1
                return False
        except Exception:
            return False

    async def close(self) -> None:
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()


# =============================================================================
# NTFY.sh (Self-hosted option)
# =============================================================================

class NTFYProvider(PushProvider):
    """
    NTFY.sh notification provider.

    NTFY is a simple pub-sub notification service.
    Can be self-hosted or use ntfy.sh cloud.

    Requires:
        - NTFY_SERVER: Server URL (default: https://ntfy.sh)
        - NTFY_TOPIC: Topic name (your unique channel)
        - NTFY_TOKEN: Optional auth token
    """

    PRIORITY_MAP = {
        NotificationPriority.LOW: "low",
        NotificationPriority.NORMAL: "default",
        NotificationPriority.HIGH: "high",
        NotificationPriority.URGENT: "urgent",
    }

    def __init__(
        self,
        topic: str,
        server: str = "https://ntfy.sh",
        token: Optional[str] = None,
    ):
        self.server = server.rstrip("/")
        self.topic = topic
        self.token = token
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    def _get_tags(self, notification: PushNotification) -> List[str]:
        """Get emoji tags based on notification type."""
        tag_map = {
            NotificationType.TRADE_EXECUTED: ["moneybag", "chart_with_upwards_trend"],
            NotificationType.POSITION_OPENED: ["rocket"],
            NotificationType.POSITION_CLOSED: ["checkered_flag"],
            NotificationType.STOP_LOSS_HIT: ["warning", "chart_with_downwards_trend"],
            NotificationType.TAKE_PROFIT_HIT: ["tada", "star"],
            NotificationType.LIQUIDATION_RISK: ["rotating_light", "skull"],
            NotificationType.DRAWDOWN_WARNING: ["warning"],
            NotificationType.CONNECTION_LOST: ["x"],
            NotificationType.CONNECTION_RESTORED: ["white_check_mark"],
            NotificationType.DAILY_SUMMARY: ["clipboard", "bar_chart"],
            NotificationType.ERROR: ["x", "bug"],
        }
        return tag_map.get(notification.notification_type, [])

    async def send(self, notification: PushNotification) -> bool:
        """Send NTFY notification."""
        return await self.send_to_topic(notification, self.topic)

    async def send_to_topic(self, notification: PushNotification, topic: str) -> bool:
        """Send notification to a specific topic."""
        session = await self._get_session()
        url = f"{self.server}/{topic}"

        headers = {
            "Title": notification.title,
            "Priority": self.PRIORITY_MAP.get(notification.priority, "default"),
            "Tags": ",".join(self._get_tags(notification)),
        }

        if notification.action_url:
            headers["Click"] = notification.action_url

        if notification.image_url:
            headers["Attach"] = notification.image_url

        # Add custom data as JSON action
        if notification.data:
            headers["X-Data"] = json.dumps(notification.data)

        try:
            async with session.post(url, data=notification.body, headers=headers) as resp:
                if resp.status == 200:
                    return True
                else:
                    text = await resp.text()
                    logger.error(f"NTFY failed: {resp.status} - {text}")
                    return False
        except Exception as e:
            logger.error(f"NTFY send error: {e}")
            return False

    async def health_check(self) -> bool:
        """Check if NTFY server is accessible."""
        session = await self._get_session()

        try:
            async with session.get(f"{self.server}/v1/health") as resp:
                return resp.status == 200
        except Exception:
            # Fallback: try to access topic info
            try:
                async with session.get(f"{self.server}/{self.topic}/json") as resp:
                    return resp.status in [200, 404]  # 404 is OK, means server works
            except Exception:
                return False

    async def close(self) -> None:
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()


# =============================================================================
# Push Notification Manager
# =============================================================================

class PushNotificationManager:
    """
    Manages multiple push notification providers.

    Usage:
        manager = PushNotificationManager()

        # Add providers
        manager.add_provider("pushover", PushoverProvider(app_token, user_key))
        manager.add_provider("ntfy", NTFYProvider(topic="trading-alerts"))

        # Send notification
        await manager.notify(
            title="Trade Executed",
            body="Bought 0.01 BTC at $50,000",
            notification_type=NotificationType.TRADE_EXECUTED,
        )
    """

    def __init__(self):
        self._providers: Dict[str, PushProvider] = {}
        self._enabled = True

    def add_provider(self, name: str, provider: PushProvider) -> None:
        """Add a notification provider."""
        self._providers[name] = provider
        logger.info(f"Added push provider: {name}")

    def remove_provider(self, name: str) -> None:
        """Remove a notification provider."""
        if name in self._providers:
            del self._providers[name]

    def enable(self) -> None:
        """Enable notifications."""
        self._enabled = True

    def disable(self) -> None:
        """Disable notifications."""
        self._enabled = False

    async def notify(
        self,
        title: str,
        body: str,
        notification_type: NotificationType,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        data: Dict[str, Any] = None,
        **kwargs,
    ) -> Dict[str, bool]:
        """
        Send notification through all providers.

        Returns:
            Dict mapping provider name to success status
        """
        if not self._enabled:
            return {}

        notification = PushNotification(
            title=title,
            body=body,
            notification_type=notification_type,
            priority=priority,
            data=data or {},
            **kwargs,
        )

        results = {}
        tasks = []

        for name, provider in self._providers.items():
            tasks.append(self._send_with_provider(name, provider, notification))

        if tasks:
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            for (name, _), result in zip(self._providers.items(), completed):
                if isinstance(result, Exception):
                    logger.error(f"Provider {name} error: {result}")
                    results[name] = False
                else:
                    results[name] = result

        return results

    async def _send_with_provider(
        self,
        name: str,
        provider: PushProvider,
        notification: PushNotification,
    ) -> bool:
        """Send notification with a single provider."""
        try:
            return await provider.send(notification)
        except Exception as e:
            logger.error(f"Provider {name} failed: {e}")
            return False

    async def notify_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        pnl: Optional[float] = None,
    ) -> Dict[str, bool]:
        """Send trade notification."""
        if pnl is not None:
            # Position closed
            pnl_emoji = "+" if pnl >= 0 else ""
            title = f"Position Closed: {symbol}"
            body = f"{side} {quantity} @ ${price:,.2f}\nP&L: {pnl_emoji}${pnl:.2f}"
            ntype = NotificationType.POSITION_CLOSED
            priority = NotificationPriority.HIGH if abs(pnl) > 10 else NotificationPriority.NORMAL
        else:
            # Position opened
            title = f"Position Opened: {symbol}"
            body = f"{side} {quantity} @ ${price:,.2f}"
            ntype = NotificationType.POSITION_OPENED
            priority = NotificationPriority.NORMAL

        return await self.notify(
            title=title,
            body=body,
            notification_type=ntype,
            priority=priority,
            data={"symbol": symbol, "side": side, "quantity": quantity, "price": price},
        )

    async def notify_stop_loss(
        self,
        symbol: str,
        loss: float,
        price: float,
    ) -> Dict[str, bool]:
        """Send stop loss notification."""
        return await self.notify(
            title=f"Stop Loss Hit: {symbol}",
            body=f"Loss: -${abs(loss):.2f} at ${price:,.2f}",
            notification_type=NotificationType.STOP_LOSS_HIT,
            priority=NotificationPriority.HIGH,
            data={"symbol": symbol, "loss": loss, "price": price},
        )

    async def notify_take_profit(
        self,
        symbol: str,
        profit: float,
        price: float,
    ) -> Dict[str, bool]:
        """Send take profit notification."""
        return await self.notify(
            title=f"Take Profit Hit: {symbol}",
            body=f"Profit: +${profit:.2f} at ${price:,.2f}",
            notification_type=NotificationType.TAKE_PROFIT_HIT,
            priority=NotificationPriority.HIGH,
            data={"symbol": symbol, "profit": profit, "price": price},
        )

    async def notify_liquidation_risk(
        self,
        symbol: str,
        margin_ratio: float,
    ) -> Dict[str, bool]:
        """Send liquidation risk warning."""
        return await self.notify(
            title=f"LIQUIDATION RISK: {symbol}",
            body=f"Margin ratio: {margin_ratio:.1f}%\nReduce position immediately!",
            notification_type=NotificationType.LIQUIDATION_RISK,
            priority=NotificationPriority.URGENT,
            data={"symbol": symbol, "margin_ratio": margin_ratio},
        )

    async def notify_drawdown(
        self,
        current_drawdown: float,
        max_drawdown: float,
    ) -> Dict[str, bool]:
        """Send drawdown warning."""
        return await self.notify(
            title="Drawdown Warning",
            body=f"Current: {current_drawdown:.1f}%\nMax allowed: {max_drawdown:.1f}%",
            notification_type=NotificationType.DRAWDOWN_WARNING,
            priority=NotificationPriority.HIGH,
            data={"current": current_drawdown, "max": max_drawdown},
        )

    async def notify_daily_summary(
        self,
        total_pnl: float,
        trades: int,
        win_rate: float,
        best_trade: float,
        worst_trade: float,
    ) -> Dict[str, bool]:
        """Send daily summary notification."""
        pnl_emoji = "+" if total_pnl >= 0 else ""

        body = (
            f"P&L: {pnl_emoji}${total_pnl:.2f}\n"
            f"Trades: {trades} ({win_rate:.0f}% win rate)\n"
            f"Best: +${best_trade:.2f}\n"
            f"Worst: -${abs(worst_trade):.2f}"
        )

        return await self.notify(
            title="Daily Trading Summary",
            body=body,
            notification_type=NotificationType.DAILY_SUMMARY,
            priority=NotificationPriority.NORMAL,
            data={
                "pnl": total_pnl,
                "trades": trades,
                "win_rate": win_rate,
            },
        )

    async def notify_error(self, error: str, context: str = "") -> Dict[str, bool]:
        """Send error notification."""
        body = error
        if context:
            body = f"{context}: {error}"

        return await self.notify(
            title="Trading Bot Error",
            body=body,
            notification_type=NotificationType.ERROR,
            priority=NotificationPriority.HIGH,
        )

    async def notify_connection(self, connected: bool, service: str = "WebSocket") -> Dict[str, bool]:
        """Send connection status notification."""
        if connected:
            return await self.notify(
                title="Connection Restored",
                body=f"{service} connection restored",
                notification_type=NotificationType.CONNECTION_RESTORED,
                priority=NotificationPriority.NORMAL,
            )
        else:
            return await self.notify(
                title="Connection Lost",
                body=f"{service} connection lost",
                notification_type=NotificationType.CONNECTION_LOST,
                priority=NotificationPriority.HIGH,
            )

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all providers."""
        results = {}

        for name, provider in self._providers.items():
            try:
                results[name] = await provider.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                results[name] = False

        return results

    async def close(self) -> None:
        """Close all providers."""
        for name, provider in self._providers.items():
            try:
                if hasattr(provider, "close"):
                    await provider.close()
            except Exception as e:
                logger.error(f"Error closing provider {name}: {e}")


# =============================================================================
# Factory Function
# =============================================================================

def create_push_manager_from_env() -> PushNotificationManager:
    """
    Create PushNotificationManager from environment variables.

    Environment variables:
        - PUSHOVER_APP_TOKEN, PUSHOVER_USER_KEY: Pushover credentials
        - NTFY_TOPIC, NTFY_SERVER, NTFY_TOKEN: NTFY configuration
        - FIREBASE_SERVER_KEY: FCM server key
    """
    manager = PushNotificationManager()

    # Pushover
    pushover_token = os.getenv("PUSHOVER_APP_TOKEN")
    pushover_user = os.getenv("PUSHOVER_USER_KEY")
    if pushover_token and pushover_user:
        manager.add_provider("pushover", PushoverProvider(pushover_token, pushover_user))
        logger.info("Pushover push notifications enabled")

    # NTFY
    ntfy_topic = os.getenv("NTFY_TOPIC")
    if ntfy_topic:
        ntfy_server = os.getenv("NTFY_SERVER", "https://ntfy.sh")
        ntfy_token = os.getenv("NTFY_TOKEN")
        manager.add_provider("ntfy", NTFYProvider(ntfy_topic, ntfy_server, ntfy_token))
        logger.info(f"NTFY push notifications enabled (topic: {ntfy_topic})")

    # Firebase
    firebase_key = os.getenv("FIREBASE_SERVER_KEY")
    if firebase_key:
        manager.add_provider("fcm", FCMProvider(firebase_key))
        logger.info("Firebase Cloud Messaging enabled")

    if not manager._providers:
        logger.warning("No push notification providers configured")

    return manager
