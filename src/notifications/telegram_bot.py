"""
Interactive Telegram Bot for trading bot control.

Provides commands for:
- Viewing status and positions
- Starting/stopping trading
- Manual order placement
- Risk management controls
- Performance reports
"""

import asyncio
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Awaitable

import aiohttp
from loguru import logger


# =============================================================================
# Types
# =============================================================================

class BotState(Enum):
    """Bot trading state."""
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class TelegramBotConfig:
    """Telegram bot configuration."""
    bot_token: str
    allowed_users: List[int] = field(default_factory=list)  # User IDs allowed to use bot
    admin_users: List[int] = field(default_factory=list)  # Admin user IDs
    enable_commands: bool = True
    command_prefix: str = "/"


CommandHandler = Callable[['TelegramBotController', Dict[str, Any]], Awaitable[str]]


# =============================================================================
# Bot Controller
# =============================================================================

class TelegramBotController:
    """
    Interactive Telegram bot for trading control.

    Commands:
        /status - View current bot status
        /balance - View account balance
        /positions - View open positions
        /orders - View open orders
        /pnl - View P&L summary
        /trades - View recent trades
        /start_trading - Start trading
        /stop_trading - Stop trading
        /pause - Pause trading
        /resume - Resume trading
        /close [symbol] - Close position
        /cancel [order_id] - Cancel order
        /leverage [symbol] [value] - Set leverage
        /risk - View risk metrics
        /help - Show commands
    """

    API_URL = "https://api.telegram.org/bot{token}/{method}"

    def __init__(
        self,
        config: TelegramBotConfig = None,
        get_status: Callable[[], Dict[str, Any]] = None,
        get_balance: Callable[[], Awaitable[Dict[str, Any]]] = None,
        get_positions: Callable[[], Awaitable[List[Dict[str, Any]]]] = None,
        get_orders: Callable[[], Awaitable[List[Dict[str, Any]]]] = None,
        get_trades: Callable[[int], Awaitable[List[Dict[str, Any]]]] = None,
        get_pnl: Callable[[], Dict[str, Any]] = None,
        get_risk: Callable[[], Dict[str, Any]] = None,
        start_trading: Callable[[], Awaitable[bool]] = None,
        stop_trading: Callable[[], Awaitable[bool]] = None,
        close_position: Callable[[str], Awaitable[bool]] = None,
        cancel_order: Callable[[str], Awaitable[bool]] = None,
        set_leverage: Callable[[str, int], Awaitable[bool]] = None,
    ):
        if config:
            self.config = config
        else:
            self.config = TelegramBotConfig(
                bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
                allowed_users=self._parse_user_ids(os.getenv("TELEGRAM_ALLOWED_USERS", "")),
                admin_users=self._parse_user_ids(os.getenv("TELEGRAM_ADMIN_USERS", "")),
            )

        # Callbacks
        self._get_status = get_status or self._default_status
        self._get_balance = get_balance
        self._get_positions = get_positions
        self._get_orders = get_orders
        self._get_trades = get_trades
        self._get_pnl = get_pnl or self._default_pnl
        self._get_risk = get_risk or self._default_risk
        self._start_trading = start_trading
        self._stop_trading = stop_trading
        self._close_position = close_position
        self._cancel_order = cancel_order
        self._set_leverage = set_leverage

        # State
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._last_update_id = 0
        self._state = BotState.STOPPED
        self._paused = False

        # Command handlers
        self._commands: Dict[str, CommandHandler] = {
            "start": self._cmd_start,
            "help": self._cmd_help,
            "status": self._cmd_status,
            "balance": self._cmd_balance,
            "positions": self._cmd_positions,
            "pos": self._cmd_positions,
            "orders": self._cmd_orders,
            "pnl": self._cmd_pnl,
            "trades": self._cmd_trades,
            "risk": self._cmd_risk,
            "start_trading": self._cmd_start_trading,
            "stop_trading": self._cmd_stop_trading,
            "pause": self._cmd_pause,
            "resume": self._cmd_resume,
            "close": self._cmd_close,
            "cancel": self._cmd_cancel,
            "leverage": self._cmd_leverage,
            "lev": self._cmd_leverage,
        }

    def _parse_user_ids(self, user_str: str) -> List[int]:
        """Parse comma-separated user IDs."""
        if not user_str:
            return []
        try:
            return [int(u.strip()) for u in user_str.split(",") if u.strip()]
        except ValueError:
            return []

    @property
    def is_configured(self) -> bool:
        """Check if bot is configured."""
        return bool(self.config.bot_token)

    def _default_status(self) -> Dict[str, Any]:
        """Default status callback."""
        return {
            "state": self._state.value,
            "paused": self._paused,
            "uptime": "unknown",
        }

    def _default_pnl(self) -> Dict[str, Any]:
        """Default P&L callback."""
        return {"total_pnl": 0, "daily_pnl": 0, "trades": 0}

    def _default_risk(self) -> Dict[str, Any]:
        """Default risk callback."""
        return {"drawdown": 0, "exposure": 0}

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start the bot controller."""
        if not self.is_configured:
            logger.warning("Telegram bot not configured")
            return

        self._session = aiohttp.ClientSession()
        self._running = True
        self._state = BotState.RUNNING

        # Start polling
        asyncio.create_task(self._poll_updates())

        logger.info("Telegram bot controller started")

    async def stop(self) -> None:
        """Stop the bot controller."""
        self._running = False
        self._state = BotState.STOPPED

        if self._session:
            await self._session.close()
            self._session = None

        logger.info("Telegram bot controller stopped")

    # =========================================================================
    # API Methods
    # =========================================================================

    async def _api_call(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make Telegram API call."""
        if not self._session:
            return {}

        url = self.API_URL.format(token=self.config.bot_token, method=method)

        try:
            async with self._session.post(url, json=params or {}) as resp:
                data = await resp.json()
                if not data.get("ok"):
                    logger.error(f"Telegram API error: {data.get('description')}")
                return data
        except Exception as e:
            logger.error(f"Telegram API call failed: {e}")
            return {}

    async def send_message(
        self,
        chat_id: int,
        text: str,
        parse_mode: str = "HTML",
        reply_markup: Dict = None,
    ) -> bool:
        """Send a message."""
        params = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        if reply_markup:
            params["reply_markup"] = reply_markup

        result = await self._api_call("sendMessage", params)
        return result.get("ok", False)

    async def _poll_updates(self) -> None:
        """Poll for updates."""
        while self._running:
            try:
                result = await self._api_call("getUpdates", {
                    "offset": self._last_update_id + 1,
                    "timeout": 30,
                    "allowed_updates": ["message"],
                })

                updates = result.get("result", [])
                for update in updates:
                    self._last_update_id = update["update_id"]
                    await self._handle_update(update)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Polling error: {e}")
                await asyncio.sleep(5)

    async def _handle_update(self, update: Dict[str, Any]) -> None:
        """Handle incoming update."""
        message = update.get("message", {})
        if not message:
            return

        text = message.get("text", "")
        chat_id = message.get("chat", {}).get("id")
        user_id = message.get("from", {}).get("id")
        username = message.get("from", {}).get("username", "Unknown")

        if not text or not chat_id:
            return

        # Check authorization
        if not self._is_authorized(user_id):
            await self.send_message(
                chat_id,
                "Unauthorized. Your user ID is not in the allowed list."
            )
            logger.warning(f"Unauthorized access attempt from {username} (ID: {user_id})")
            return

        # Parse command
        if text.startswith("/"):
            await self._handle_command(chat_id, user_id, text)

    def _is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized."""
        # If no allowed users configured, allow all
        if not self.config.allowed_users and not self.config.admin_users:
            return True
        return user_id in self.config.allowed_users or user_id in self.config.admin_users

    def _is_admin(self, user_id: int) -> bool:
        """Check if user is admin."""
        if not self.config.admin_users:
            return True  # If no admins configured, all authorized users are admins
        return user_id in self.config.admin_users

    async def _handle_command(self, chat_id: int, user_id: int, text: str) -> None:
        """Handle a command."""
        parts = text[1:].split()  # Remove / and split
        command = parts[0].lower().split("@")[0]  # Handle @botname suffix
        args = parts[1:] if len(parts) > 1 else []

        context = {
            "chat_id": chat_id,
            "user_id": user_id,
            "args": args,
            "is_admin": self._is_admin(user_id),
        }

        handler = self._commands.get(command)
        if handler:
            try:
                response = await handler(context)
                await self.send_message(chat_id, response)
            except Exception as e:
                logger.error(f"Command error: {e}")
                await self.send_message(chat_id, f"Error: {str(e)}")
        else:
            await self.send_message(chat_id, f"Unknown command: /{command}\nUse /help for available commands.")

    # =========================================================================
    # Command Handlers
    # =========================================================================

    async def _cmd_start(self, ctx: Dict) -> str:
        """Handle /start command."""
        return """
Welcome to the Trading Bot Controller!

Use /help to see available commands.
Use /status to check bot status.
"""

    async def _cmd_help(self, ctx: Dict) -> str:
        """Handle /help command."""
        commands = """
<b>Available Commands</b>

📊 <b>Information</b>
/status - Bot status
/balance - Account balance
/positions - Open positions
/orders - Open orders
/pnl - Profit & Loss summary
/trades - Recent trades
/risk - Risk metrics

⚙️ <b>Control</b>
/start_trading - Start trading
/stop_trading - Stop trading
/pause - Pause trading
/resume - Resume trading

📝 <b>Actions</b>
/close [symbol] - Close position
/cancel [order_id] - Cancel order
/leverage [symbol] [value] - Set leverage
"""
        return commands.strip()

    async def _cmd_status(self, ctx: Dict) -> str:
        """Handle /status command."""
        status = self._get_status()

        state_emoji = {
            "running": "🟢",
            "paused": "🟡",
            "stopped": "🔴",
        }.get(status.get("state", ""), "⚪")

        return f"""
<b>Bot Status</b>

{state_emoji} State: {status.get('state', 'unknown').upper()}
⏱ Uptime: {status.get('uptime', 'N/A')}
📊 Active Strategies: {status.get('strategies', 'N/A')}
🔗 Exchange: {status.get('exchange', 'N/A')}
"""

    async def _cmd_balance(self, ctx: Dict) -> str:
        """Handle /balance command."""
        if not self._get_balance:
            return "Balance information not available."

        balance = await self._get_balance()

        return f"""
<b>Account Balance</b>

💰 Wallet: <code>${balance.get('wallet_balance', 0):,.2f}</code>
💵 Available: <code>${balance.get('available_balance', 0):,.2f}</code>
📊 Margin: <code>${balance.get('margin_balance', 0):,.2f}</code>
📈 Unrealized P&L: <code>${balance.get('unrealized_pnl', 0):,.2f}</code>
"""

    async def _cmd_positions(self, ctx: Dict) -> str:
        """Handle /positions command."""
        if not self._get_positions:
            return "Position information not available."

        positions = await self._get_positions()

        if not positions:
            return "No open positions."

        lines = ["<b>Open Positions</b>\n"]
        for pos in positions:
            side = pos.get("side", "").upper()
            emoji = "🟢" if side == "LONG" or side == "BUY" else "🔴"
            pnl = float(pos.get("unrealized_pnl", 0))
            pnl_emoji = "📈" if pnl > 0 else "📉"

            lines.append(f"""
{emoji} <b>{pos.get('symbol')}</b> {side}
   Size: {pos.get('quantity', 0)}
   Entry: ${float(pos.get('entry_price', 0)):,.2f}
   Mark: ${float(pos.get('mark_price', 0)):,.2f}
   {pnl_emoji} P&L: ${pnl:+,.2f}
""")

        return "\n".join(lines)

    async def _cmd_orders(self, ctx: Dict) -> str:
        """Handle /orders command."""
        if not self._get_orders:
            return "Order information not available."

        orders = await self._get_orders()

        if not orders:
            return "No open orders."

        lines = ["<b>Open Orders</b>\n"]
        for order in orders[:10]:  # Limit to 10
            side = order.get("side", "").upper()
            emoji = "🟢" if side == "BUY" else "🔴"

            lines.append(f"""
{emoji} {order.get('symbol')} {order.get('type', '')}
   {side} {order.get('quantity', 0)} @ ${float(order.get('price', 0)):,.2f}
   ID: <code>{order.get('order_id', '')[:8]}...</code>
""")

        return "\n".join(lines)

    async def _cmd_pnl(self, ctx: Dict) -> str:
        """Handle /pnl command."""
        pnl = self._get_pnl()

        total = pnl.get("total_pnl", 0)
        daily = pnl.get("daily_pnl", 0)
        total_emoji = "📈" if total > 0 else "📉" if total < 0 else "➡️"
        daily_emoji = "📈" if daily > 0 else "📉" if daily < 0 else "➡️"

        return f"""
<b>Profit & Loss</b>

{total_emoji} Total P&L: <code>${total:+,.2f}</code>
{daily_emoji} Daily P&L: <code>${daily:+,.2f}</code>
📊 Total Trades: {pnl.get('trades', 0)}
✅ Win Rate: {pnl.get('win_rate', 0):.1f}%
📈 Profit Factor: {pnl.get('profit_factor', 0):.2f}
"""

    async def _cmd_trades(self, ctx: Dict) -> str:
        """Handle /trades command."""
        if not self._get_trades:
            return "Trade history not available."

        limit = 5
        if ctx["args"]:
            try:
                limit = min(int(ctx["args"][0]), 20)
            except ValueError:
                pass

        trades = await self._get_trades(limit)

        if not trades:
            return "No recent trades."

        lines = ["<b>Recent Trades</b>\n"]
        for trade in trades:
            side = trade.get("side", "").upper()
            emoji = "🟢" if side == "BUY" else "🔴"
            pnl = float(trade.get("realized_pnl", 0))
            pnl_str = f"${pnl:+,.2f}" if pnl != 0 else ""

            lines.append(f"""
{emoji} {trade.get('symbol')} {side}
   {trade.get('quantity', 0)} @ ${float(trade.get('price', 0)):,.2f}
   {pnl_str}
""")

        return "\n".join(lines)

    async def _cmd_risk(self, ctx: Dict) -> str:
        """Handle /risk command."""
        risk = self._get_risk()

        dd = risk.get("drawdown", 0)
        dd_emoji = "🟢" if dd < 5 else "🟡" if dd < 10 else "🔴"

        return f"""
<b>Risk Metrics</b>

{dd_emoji} Drawdown: {dd:.2f}%
📊 Exposure: {risk.get('exposure', 0):.2f}%
⚡ Leverage Used: {risk.get('leverage_used', 0):.1f}x
🛡 Available Margin: ${risk.get('available_margin', 0):,.2f}
"""

    async def _cmd_start_trading(self, ctx: Dict) -> str:
        """Handle /start_trading command."""
        if not ctx["is_admin"]:
            return "Admin permission required."

        if not self._start_trading:
            return "Start trading not available."

        success = await self._start_trading()
        if success:
            self._state = BotState.RUNNING
            self._paused = False
            return "✅ Trading started successfully."
        return "❌ Failed to start trading."

    async def _cmd_stop_trading(self, ctx: Dict) -> str:
        """Handle /stop_trading command."""
        if not ctx["is_admin"]:
            return "Admin permission required."

        if not self._stop_trading:
            return "Stop trading not available."

        success = await self._stop_trading()
        if success:
            self._state = BotState.STOPPED
            return "🛑 Trading stopped."
        return "❌ Failed to stop trading."

    async def _cmd_pause(self, ctx: Dict) -> str:
        """Handle /pause command."""
        if not ctx["is_admin"]:
            return "Admin permission required."

        self._paused = True
        self._state = BotState.PAUSED
        return "⏸ Trading paused. Use /resume to continue."

    async def _cmd_resume(self, ctx: Dict) -> str:
        """Handle /resume command."""
        if not ctx["is_admin"]:
            return "Admin permission required."

        self._paused = False
        self._state = BotState.RUNNING
        return "▶️ Trading resumed."

    async def _cmd_close(self, ctx: Dict) -> str:
        """Handle /close command."""
        if not ctx["is_admin"]:
            return "Admin permission required."

        if not self._close_position:
            return "Close position not available."

        if not ctx["args"]:
            return "Usage: /close BTCUSDT"

        symbol = ctx["args"][0].upper()
        success = await self._close_position(symbol)

        if success:
            return f"✅ Position {symbol} closed."
        return f"❌ Failed to close {symbol} position."

    async def _cmd_cancel(self, ctx: Dict) -> str:
        """Handle /cancel command."""
        if not ctx["is_admin"]:
            return "Admin permission required."

        if not self._cancel_order:
            return "Cancel order not available."

        if not ctx["args"]:
            return "Usage: /cancel ORDER_ID"

        order_id = ctx["args"][0]
        success = await self._cancel_order(order_id)

        if success:
            return f"✅ Order {order_id} cancelled."
        return f"❌ Failed to cancel order {order_id}."

    async def _cmd_leverage(self, ctx: Dict) -> str:
        """Handle /leverage command."""
        if not ctx["is_admin"]:
            return "Admin permission required."

        if not self._set_leverage:
            return "Set leverage not available."

        if len(ctx["args"]) < 2:
            return "Usage: /leverage BTCUSDT 10"

        symbol = ctx["args"][0].upper()
        try:
            leverage = int(ctx["args"][1])
        except ValueError:
            return "Invalid leverage value."

        if leverage < 1 or leverage > 125:
            return "Leverage must be between 1 and 125."

        success = await self._set_leverage(symbol, leverage)

        if success:
            return f"✅ Leverage for {symbol} set to {leverage}x."
        return f"❌ Failed to set leverage for {symbol}."


# =============================================================================
# Factory
# =============================================================================

def create_telegram_bot_controller(**kwargs) -> TelegramBotController:
    """Create Telegram bot controller from kwargs or environment."""
    return TelegramBotController(**kwargs)
