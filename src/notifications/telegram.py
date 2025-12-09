"""
Telegram Bot for trading notifications.

Sends alerts for:
- Trade signals
- Order executions
- Position changes
- Daily summaries
- Error alerts
"""

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List

import aiohttp
from loguru import logger

from src.data.models import Signal, SignalType, Order, Position, Side


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""
    bot_token: str
    chat_id: str
    enabled: bool = True
    send_signals: bool = True
    send_orders: bool = True
    send_positions: bool = True
    send_errors: bool = True
    send_daily_summary: bool = True
    min_signal_strength: float = 0.6  # Only notify for strong signals


class TelegramNotifier:
    """
    Telegram notification service.

    Sends formatted messages to a Telegram chat/channel.
    """

    API_URL = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, config: TelegramConfig = None):
        """
        Initialize Telegram notifier.

        Args:
            config: Telegram configuration (or from env vars)
        """
        if config:
            self.config = config
        else:
            self.config = TelegramConfig(
                bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
                chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
                enabled=os.getenv("TELEGRAM_ENABLED", "true").lower() == "true",
            )

        self._session: Optional[aiohttp.ClientSession] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._rate_limit_delay = 0.05  # 20 messages per second max

        # Stats
        self._messages_sent = 0
        self._errors = 0

    @property
    def is_configured(self) -> bool:
        """Check if Telegram is properly configured."""
        return bool(self.config.bot_token and self.config.chat_id)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get notification statistics."""
        return {
            "enabled": self.config.enabled,
            "configured": self.is_configured,
            "messages_sent": self._messages_sent,
            "errors": self._errors,
            "queue_size": self._message_queue.qsize(),
        }

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start the notification service."""
        if not self.is_configured:
            logger.warning("Telegram not configured, notifications disabled")
            return

        self._session = aiohttp.ClientSession()
        self._running = True

        # Start message processor
        asyncio.create_task(self._process_queue())

        logger.info("Telegram notifier started")

        # Send startup message
        await self.send_system_message("🤖 Bot started")

    async def stop(self) -> None:
        """Stop the notification service."""
        self._running = False

        # Send shutdown message
        if self.is_configured:
            await self.send_system_message("🛑 Bot stopped")

        if self._session:
            await self._session.close()
            self._session = None

        logger.info(f"Telegram notifier stopped. Sent {self._messages_sent} messages")

    # =========================================================================
    # Message Sending
    # =========================================================================

    async def _send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message to Telegram.

        Args:
            text: Message text (HTML formatted)
            parse_mode: Parse mode (HTML or Markdown)

        Returns:
            True if sent successfully
        """
        if not self._session or not self.is_configured:
            return False

        url = self.API_URL.format(token=self.config.bot_token, method="sendMessage")

        payload = {
            "chat_id": self.config.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            async with self._session.post(url, json=payload) as resp:
                if resp.status == 200:
                    self._messages_sent += 1
                    return True
                else:
                    error = await resp.text()
                    logger.error(f"Telegram API error: {error}")
                    self._errors += 1
                    return False

        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            self._errors += 1
            return False

    async def _process_queue(self) -> None:
        """Process message queue with rate limiting."""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )

                await self._send_message(message)
                await asyncio.sleep(self._rate_limit_delay)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Queue processing error: {e}")

    def queue_message(self, text: str) -> None:
        """Add message to queue for sending."""
        if self.config.enabled and self.is_configured:
            self._message_queue.put_nowait(text)

    # =========================================================================
    # Notification Methods
    # =========================================================================

    async def send_system_message(self, message: str) -> None:
        """Send a system message."""
        text = f"<b>System</b>\n{message}\n\n<i>{datetime.utcnow().strftime('%H:%M:%S UTC')}</i>"
        self.queue_message(text)

    async def notify_signal(self, signal: Signal) -> None:
        """
        Send notification for a trading signal.

        Args:
            signal: Trading signal
        """
        if not self.config.send_signals:
            return

        if signal.strength < self.config.min_signal_strength:
            return

        if signal.signal_type == SignalType.NO_ACTION:
            return

        # Emoji based on signal type
        emoji = {
            SignalType.LONG: "🟢",
            SignalType.SHORT: "🔴",
            SignalType.CLOSE_LONG: "📤",
            SignalType.CLOSE_SHORT: "📤",
        }.get(signal.signal_type, "⚪")

        strength_bar = "█" * int(signal.strength * 10) + "░" * (10 - int(signal.strength * 10))

        text = f"""
{emoji} <b>SIGNAL: {signal.signal_type.name}</b>

📊 <b>{signal.symbol}</b>
💰 Price: <code>${float(signal.price):,.2f}</code>
📈 Strength: [{strength_bar}] {signal.strength:.0%}
🎯 Strategy: {signal.strategy}

<i>{signal.timestamp.strftime('%H:%M:%S UTC')}</i>
"""
        self.queue_message(text.strip())

    async def notify_order(self, order: Order, event_type: str = "new") -> None:
        """
        Send notification for order events.

        Args:
            order: Order object
            event_type: Event type (new, filled, canceled)
        """
        if not self.config.send_orders:
            return

        emoji = {
            "new": "📝",
            "filled": "✅",
            "canceled": "❌",
            "rejected": "⛔",
        }.get(event_type, "📋")

        side_emoji = "🟢" if order.side == Side.BUY else "🔴"

        text = f"""
{emoji} <b>ORDER {event_type.upper()}</b>

{side_emoji} {order.side.value} {order.symbol}
📦 Quantity: <code>{float(order.quantity):.6f}</code>
💵 Price: <code>${float(order.price) if order.price else 'MARKET':,.2f}</code>
📋 Type: {order.order_type.value}
🔖 ID: <code>{order.order_id}</code>

<i>{datetime.utcnow().strftime('%H:%M:%S UTC')}</i>
"""
        self.queue_message(text.strip())

    async def notify_position_open(self, position: Position, entry_order: Order = None) -> None:
        """Send notification for new position."""
        if not self.config.send_positions:
            return

        side_emoji = "🟢" if position.side == Side.BUY else "🔴"
        direction = "LONG" if position.side == Side.BUY else "SHORT"

        text = f"""
📈 <b>POSITION OPENED</b>

{side_emoji} <b>{direction} {position.symbol}</b>
📦 Size: <code>{float(position.size):.6f}</code>
💰 Entry: <code>${float(position.entry_price):,.2f}</code>
⚡ Leverage: {position.leverage}x
💵 Value: <code>${float(position.notional_value):,.2f}</code>

<i>{datetime.utcnow().strftime('%H:%M:%S UTC')}</i>
"""
        self.queue_message(text.strip())

    async def notify_position_close(
        self,
        position: Position,
        pnl: Decimal,
        pnl_pct: float,
    ) -> None:
        """Send notification for closed position."""
        if not self.config.send_positions:
            return

        pnl_emoji = "💚" if pnl > 0 else "❤️" if pnl < 0 else "🤍"
        pnl_sign = "+" if pnl > 0 else ""

        text = f"""
📉 <b>POSITION CLOSED</b>

{pnl_emoji} <b>{position.symbol}</b>
💰 P&L: <code>{pnl_sign}${float(pnl):,.2f}</code> ({pnl_sign}{pnl_pct:.2f}%)
📦 Size: <code>{float(position.size):.6f}</code>
💵 Entry → Exit: <code>${float(position.entry_price):,.2f}</code> → <code>${float(position.mark_price):,.2f}</code>

<i>{datetime.utcnow().strftime('%H:%M:%S UTC')}</i>
"""
        self.queue_message(text.strip())

    async def notify_daily_summary(self, stats: Dict[str, Any]) -> None:
        """
        Send daily performance summary.

        Args:
            stats: Daily statistics dictionary
        """
        if not self.config.send_daily_summary:
            return

        pnl = stats.get("total_pnl", 0)
        pnl_emoji = "📈" if pnl > 0 else "📉" if pnl < 0 else "➡️"
        pnl_sign = "+" if pnl > 0 else ""

        trades = stats.get("trades", 0)
        wins = stats.get("wins", 0)
        win_rate = (wins / trades * 100) if trades > 0 else 0

        text = f"""
📊 <b>DAILY SUMMARY</b>

{pnl_emoji} <b>P&L: {pnl_sign}${pnl:,.2f}</b>

📈 Trades: {trades}
✅ Wins: {wins}
❌ Losses: {stats.get('losses', 0)}
🎯 Win Rate: {win_rate:.1f}%

📉 Max Drawdown: ${stats.get('max_drawdown', 0):,.2f}
⚡ Profit Factor: {stats.get('profit_factor', 0):.2f}

<i>{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</i>
"""
        self.queue_message(text.strip())

    async def notify_error(self, error: str, context: str = "") -> None:
        """
        Send error notification.

        Args:
            error: Error message
            context: Additional context
        """
        if not self.config.send_errors:
            return

        text = f"""
🚨 <b>ERROR</b>

<code>{error}</code>

{f'Context: {context}' if context else ''}

<i>{datetime.utcnow().strftime('%H:%M:%S UTC')}</i>
"""
        self.queue_message(text.strip())

    async def notify_custom(self, title: str, message: str, emoji: str = "📢") -> None:
        """
        Send custom notification.

        Args:
            title: Message title
            message: Message body
            emoji: Emoji prefix
        """
        text = f"""
{emoji} <b>{title}</b>

{message}

<i>{datetime.utcnow().strftime('%H:%M:%S UTC')}</i>
"""
        self.queue_message(text.strip())


# =============================================================================
# Factory function
# =============================================================================

def create_telegram_notifier(config: dict = None) -> TelegramNotifier:
    """
    Create Telegram notifier from config dictionary.

    Args:
        config: Optional config dictionary

    Returns:
        Configured TelegramNotifier
    """
    if config:
        telegram_config = TelegramConfig(
            bot_token=config.get("bot_token", os.getenv("TELEGRAM_BOT_TOKEN", "")),
            chat_id=config.get("chat_id", os.getenv("TELEGRAM_CHAT_ID", "")),
            enabled=config.get("enabled", True),
            send_signals=config.get("send_signals", True),
            send_orders=config.get("send_orders", True),
            send_positions=config.get("send_positions", True),
            send_errors=config.get("send_errors", True),
            send_daily_summary=config.get("send_daily_summary", True),
            min_signal_strength=config.get("min_signal_strength", 0.6),
        )
        return TelegramNotifier(telegram_config)

    return TelegramNotifier()
