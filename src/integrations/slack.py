"""
Slack Integration

Slack notifications and interactive bot for trading operations.
"""
import asyncio
import json
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
import aiohttp

from loguru import logger


@dataclass
class SlackMessage:
    """Slack message structure."""
    channel: str
    text: str
    blocks: Optional[List[Dict]] = None
    attachments: Optional[List[Dict]] = None
    thread_ts: Optional[str] = None


class SlackClient:
    """Slack API client for notifications and interactions."""

    def __init__(
        self,
        bot_token: str,
        webhook_url: Optional[str] = None,
        default_channel: str = "#trading-alerts",
    ):
        self.bot_token = bot_token
        self.webhook_url = webhook_url
        self.default_channel = default_channel
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.bot_token}"}
            )
        return self._session

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def send_message(
        self,
        text: str,
        channel: Optional[str] = None,
        blocks: Optional[List[Dict]] = None,
        attachments: Optional[List[Dict]] = None,
        thread_ts: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a message to Slack."""
        session = await self._get_session()

        payload = {
            "channel": channel or self.default_channel,
            "text": text,
        }

        if blocks:
            payload["blocks"] = blocks
        if attachments:
            payload["attachments"] = attachments
        if thread_ts:
            payload["thread_ts"] = thread_ts

        async with session.post(
            "https://slack.com/api/chat.postMessage",
            json=payload,
        ) as response:
            result = await response.json()
            if not result.get("ok"):
                logger.error(f"Slack API error: {result.get('error')}")
            return result

    async def send_webhook(self, text: str, blocks: Optional[List[Dict]] = None):
        """Send message via webhook (simpler, no token needed)."""
        if not self.webhook_url:
            raise ValueError("Webhook URL not configured")

        session = await self._get_session()

        payload = {"text": text}
        if blocks:
            payload["blocks"] = blocks

        async with session.post(self.webhook_url, json=payload) as response:
            return response.status == 200

    async def update_message(
        self,
        channel: str,
        ts: str,
        text: str,
        blocks: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Update an existing message."""
        session = await self._get_session()

        payload = {
            "channel": channel,
            "ts": ts,
            "text": text,
        }
        if blocks:
            payload["blocks"] = blocks

        async with session.post(
            "https://slack.com/api/chat.update",
            json=payload,
        ) as response:
            return await response.json()

    async def add_reaction(self, channel: str, ts: str, emoji: str) -> Dict[str, Any]:
        """Add reaction to a message."""
        session = await self._get_session()

        async with session.post(
            "https://slack.com/api/reactions.add",
            json={
                "channel": channel,
                "timestamp": ts,
                "name": emoji,
            },
        ) as response:
            return await response.json()


class SlackNotifier:
    """Trading notifications for Slack."""

    def __init__(self, client: SlackClient):
        self.client = client

    def _create_trade_blocks(self, trade: Dict[str, Any]) -> List[Dict]:
        """Create blocks for trade notification."""
        pnl = trade.get('pnl', 0)
        color = "#36a64f" if pnl >= 0 else "#dc3545"
        emoji = ":chart_with_upwards_trend:" if pnl >= 0 else ":chart_with_downwards_trend:"

        return [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Trade Executed",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Symbol:*\n{trade.get('symbol')}"},
                    {"type": "mrkdwn", "text": f"*Side:*\n{trade.get('side', '').upper()}"},
                    {"type": "mrkdwn", "text": f"*Entry:*\n${trade.get('entry_price', 0):,.2f}"},
                    {"type": "mrkdwn", "text": f"*Exit:*\n${trade.get('exit_price', 0):,.2f}"},
                    {"type": "mrkdwn", "text": f"*Quantity:*\n{trade.get('quantity', 0)}"},
                    {"type": "mrkdwn", "text": f"*P&L:*\n${pnl:+,.2f}"},
                ]
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"Strategy: {trade.get('strategy', 'Unknown')} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}
                ]
            }
        ]

    def _create_signal_blocks(self, signal: Dict[str, Any]) -> List[Dict]:
        """Create blocks for signal notification."""
        action = signal.get('action', 'unknown').upper()
        emoji = ":large_green_circle:" if action == "BUY" else ":red_circle:" if action == "SELL" else ":white_circle:"

        return [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Trading Signal",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Symbol:*\n{signal.get('symbol')}"},
                    {"type": "mrkdwn", "text": f"*Action:*\n{action}"},
                    {"type": "mrkdwn", "text": f"*Price:*\n${signal.get('price', 0):,.2f}"},
                    {"type": "mrkdwn", "text": f"*Confidence:*\n{signal.get('confidence', 0):.1f}%"},
                ]
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Execute", "emoji": True},
                        "style": "primary",
                        "action_id": f"execute_signal_{signal.get('id', 'unknown')}"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Ignore", "emoji": True},
                        "action_id": f"ignore_signal_{signal.get('id', 'unknown')}"
                    }
                ]
            }
        ]

    def _create_alert_blocks(self, severity: str, title: str, message: str) -> List[Dict]:
        """Create blocks for alert notification."""
        emoji_map = {
            "critical": ":rotating_light:",
            "warning": ":warning:",
            "info": ":information_source:",
        }
        emoji = emoji_map.get(severity, ":bell:")

        return [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {title}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": message}
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"Severity: {severity.upper()} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}
                ]
            }
        ]

    def _create_daily_report_blocks(self, report: Dict[str, Any]) -> List[Dict]:
        """Create blocks for daily report."""
        pnl = report.get('pnl', 0)
        emoji = ":chart_with_upwards_trend:" if pnl >= 0 else ":chart_with_downwards_trend:"

        return [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Daily Trading Report",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*{report.get('date', 'Today')}*"}
            },
            {"type": "divider"},
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*P&L:*\n${pnl:+,.2f}"},
                    {"type": "mrkdwn", "text": f"*Total Trades:*\n{report.get('trades', 0)}"},
                    {"type": "mrkdwn", "text": f"*Win Rate:*\n{report.get('win_rate', 0):.1f}%"},
                    {"type": "mrkdwn", "text": f"*Balance:*\n${report.get('balance', 0):,.2f}"},
                    {"type": "mrkdwn", "text": f"*Winning:*\n{report.get('winning_trades', 0)}"},
                    {"type": "mrkdwn", "text": f"*Losing:*\n{report.get('losing_trades', 0)}"},
                ]
            },
            {"type": "divider"},
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Best Trade:*\n${report.get('best_trade', 0):+,.2f}"},
                    {"type": "mrkdwn", "text": f"*Worst Trade:*\n${report.get('worst_trade', 0):+,.2f}"},
                ]
            }
        ]

    async def send_trade_notification(self, trade: Dict[str, Any], channel: Optional[str] = None):
        """Send trade execution notification."""
        blocks = self._create_trade_blocks(trade)
        pnl = trade.get('pnl', 0)
        text = f"Trade: {trade.get('side', '').upper()} {trade.get('symbol')} P&L: ${pnl:+,.2f}"

        await self.client.send_message(text=text, blocks=blocks, channel=channel)

    async def send_signal_notification(self, signal: Dict[str, Any], channel: Optional[str] = None):
        """Send signal notification with action buttons."""
        blocks = self._create_signal_blocks(signal)
        text = f"Signal: {signal.get('action', '').upper()} {signal.get('symbol')}"

        await self.client.send_message(text=text, blocks=blocks, channel=channel)

    async def send_alert(
        self,
        severity: str,
        title: str,
        message: str,
        channel: Optional[str] = None
    ):
        """Send alert notification."""
        blocks = self._create_alert_blocks(severity, title, message)
        text = f"[{severity.upper()}] {title}"

        await self.client.send_message(text=text, blocks=blocks, channel=channel)

    async def send_daily_report(self, report: Dict[str, Any], channel: Optional[str] = None):
        """Send daily performance report."""
        blocks = self._create_daily_report_blocks(report)
        pnl = report.get('pnl', 0)
        text = f"Daily Report: P&L ${pnl:+,.2f}"

        await self.client.send_message(text=text, blocks=blocks, channel=channel)

    async def send_position_update(self, position: Dict[str, Any], channel: Optional[str] = None):
        """Send position update notification."""
        pnl = position.get('unrealized_pnl', 0)
        emoji = ":arrow_up:" if pnl >= 0 else ":arrow_down:"

        text = (
            f"{emoji} *Position Update*\n"
            f"Symbol: {position.get('symbol')}\n"
            f"Side: {position.get('side', '').upper()}\n"
            f"Entry: ${position.get('entry_price', 0):,.2f}\n"
            f"Current: ${position.get('current_price', 0):,.2f}\n"
            f"Unrealized P&L: ${pnl:+,.2f}"
        )

        await self.client.send_message(text=text, channel=channel)

    async def send_error(self, error: str, details: str = "", channel: Optional[str] = None):
        """Send error notification."""
        await self.send_alert(
            severity="critical",
            title="Error",
            message=f"```{error}```\n{details}",
            channel=channel
        )


class SlackInteractiveHandler:
    """Handle Slack interactive components (buttons, modals)."""

    def __init__(self, signing_secret: str):
        self.signing_secret = signing_secret
        self.handlers: Dict[str, Callable] = {}

    def register_handler(self, action_id_prefix: str, handler: Callable):
        """Register handler for action ID prefix."""
        self.handlers[action_id_prefix] = handler

    async def handle_interaction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming interaction payload."""
        actions = payload.get("actions", [])

        for action in actions:
            action_id = action.get("action_id", "")

            for prefix, handler in self.handlers.items():
                if action_id.startswith(prefix):
                    return await handler(payload, action)

        return {"text": "Action not recognized"}

    def verify_signature(self, signature: str, timestamp: str, body: bytes) -> bool:
        """Verify Slack request signature."""
        import hmac
        import hashlib

        sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
        my_signature = (
            "v0=" + hmac.new(
                self.signing_secret.encode(),
                sig_basestring.encode(),
                hashlib.sha256
            ).hexdigest()
        )

        return hmac.compare_digest(my_signature, signature)


# Quick setup helper
def create_slack_notifier(
    bot_token: str,
    webhook_url: Optional[str] = None,
    default_channel: str = "#trading-alerts",
) -> SlackNotifier:
    """Create a configured Slack notifier."""
    client = SlackClient(
        bot_token=bot_token,
        webhook_url=webhook_url,
        default_channel=default_channel,
    )
    return SlackNotifier(client)
