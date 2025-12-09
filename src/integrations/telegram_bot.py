"""
Telegram Bot Integration

Full-featured Telegram bot for trading bot control and monitoring.
"""
import asyncio
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import json

try:
    from telegram import (
        Update, InlineKeyboardButton, InlineKeyboardMarkup,
        ReplyKeyboardMarkup, KeyboardButton
    )
    from telegram.ext import (
        Application, CommandHandler, CallbackQueryHandler,
        MessageHandler, ContextTypes, filters
    )
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False

from loguru import logger


class BotState(Enum):
    """Bot conversation states."""
    IDLE = "idle"
    AWAITING_SYMBOL = "awaiting_symbol"
    AWAITING_AMOUNT = "awaiting_amount"
    AWAITING_CONFIRMATION = "awaiting_confirmation"


@dataclass
class UserSession:
    """User session data."""
    user_id: int
    state: BotState = BotState.IDLE
    data: Dict[str, Any] = None
    last_activity: datetime = None

    def __post_init__(self):
        self.data = self.data or {}
        self.last_activity = self.last_activity or datetime.now()


class TelegramTradingBot:
    """Telegram bot for trading operations."""

    def __init__(
        self,
        token: str,
        allowed_users: Optional[List[int]] = None,
        admin_users: Optional[List[int]] = None,
    ):
        if not HAS_TELEGRAM:
            raise ImportError("python-telegram-bot not installed")

        self.token = token
        self.allowed_users = set(allowed_users or [])
        self.admin_users = set(admin_users or [])
        self.sessions: Dict[int, UserSession] = {}

        # Callbacks for trading operations
        self.callbacks: Dict[str, Callable] = {}

        # Trading state
        self.trading_enabled = True
        self.paper_mode = True

        self.app: Optional[Application] = None

    def set_callback(self, name: str, callback: Callable) -> None:
        """Set callback function for trading operations."""
        self.callbacks[name] = callback

    def _get_session(self, user_id: int) -> UserSession:
        """Get or create user session."""
        if user_id not in self.sessions:
            self.sessions[user_id] = UserSession(user_id=user_id)
        session = self.sessions[user_id]
        session.last_activity = datetime.now()
        return session

    def _is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized."""
        if not self.allowed_users:
            return True
        return user_id in self.allowed_users or user_id in self.admin_users

    def _is_admin(self, user_id: int) -> bool:
        """Check if user is admin."""
        return user_id in self.admin_users

    async def _check_auth(self, update: Update) -> bool:
        """Check authorization and send message if not authorized."""
        user_id = update.effective_user.id
        if not self._is_authorized(user_id):
            await update.message.reply_text(
                "You are not authorized to use this bot.\n"
                f"Your user ID: {user_id}"
            )
            return False
        return True

    def build_app(self) -> Application:
        """Build telegram application with handlers."""
        self.app = Application.builder().token(self.token).build()

        # Command handlers
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("balance", self.cmd_balance))
        self.app.add_handler(CommandHandler("positions", self.cmd_positions))
        self.app.add_handler(CommandHandler("trades", self.cmd_trades))
        self.app.add_handler(CommandHandler("pnl", self.cmd_pnl))
        self.app.add_handler(CommandHandler("buy", self.cmd_buy))
        self.app.add_handler(CommandHandler("sell", self.cmd_sell))
        self.app.add_handler(CommandHandler("close", self.cmd_close))
        self.app.add_handler(CommandHandler("closeall", self.cmd_closeall))
        self.app.add_handler(CommandHandler("cancel", self.cmd_cancel))
        self.app.add_handler(CommandHandler("orders", self.cmd_orders))
        self.app.add_handler(CommandHandler("start_trading", self.cmd_start_trading))
        self.app.add_handler(CommandHandler("stop_trading", self.cmd_stop_trading))
        self.app.add_handler(CommandHandler("stats", self.cmd_stats))
        self.app.add_handler(CommandHandler("settings", self.cmd_settings))

        # Callback query handler for inline buttons
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))

        # Message handler for text input
        self.app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self.handle_message
        ))

        return self.app

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        if not await self._check_auth(update):
            return

        keyboard = [
            [KeyboardButton("/status"), KeyboardButton("/balance")],
            [KeyboardButton("/positions"), KeyboardButton("/pnl")],
            [KeyboardButton("/trades"), KeyboardButton("/stats")],
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

        await update.message.reply_text(
            "*Welcome to Crypto Trading Bot*\n\n"
            "Use the keyboard below or type /help for commands.",
            parse_mode="Markdown",
            reply_markup=reply_markup
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        if not await self._check_auth(update):
            return

        help_text = """
*Trading Bot Commands*

*Info Commands*
/status - Bot status
/balance - Account balance
/positions - Open positions
/orders - Pending orders
/trades - Recent trades
/pnl - Today's P&L
/stats - Trading statistics

*Trading Commands*
/buy <symbol> <amount> - Buy
/sell <symbol> <amount> - Sell
/close <symbol> - Close position
/closeall - Close all positions
/cancel <order_id> - Cancel order

*Control Commands*
/start\\_trading - Enable trading
/stop\\_trading - Disable trading
/settings - Bot settings

*Example*
`/buy BTCUSDT 0.01`
`/sell ETHUSDT 0.5`
`/close BTCUSDT`
"""
        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        if not await self._check_auth(update):
            return

        callback = self.callbacks.get('get_status')
        if callback:
            status = await self._call_async(callback)
        else:
            status = {
                "trading_enabled": self.trading_enabled,
                "paper_mode": self.paper_mode,
                "uptime": "N/A",
                "connected_exchanges": ["Binance"],
            }

        mode = "PAPER" if status.get('paper_mode', True) else "LIVE"
        trading = "ON" if status.get('trading_enabled', False) else "OFF"

        text = f"""
*Bot Status*

Trading: `{trading}`
Mode: `{mode}`
Uptime: `{status.get('uptime', 'N/A')}`
Exchanges: `{', '.join(status.get('connected_exchanges', []))}`
Open Positions: `{status.get('open_positions', 0)}`
Today's Trades: `{status.get('today_trades', 0)}`
"""
        keyboard = [
            [
                InlineKeyboardButton(
                    "Stop Trading" if self.trading_enabled else "Start Trading",
                    callback_data="toggle_trading"
                )
            ],
            [InlineKeyboardButton("Refresh", callback_data="refresh_status")]
        ]

        await update.message.reply_text(
            text,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /balance command."""
        if not await self._check_auth(update):
            return

        callback = self.callbacks.get('get_balance')
        if callback:
            balance = await self._call_async(callback)
        else:
            balance = {
                "total": 10000.0,
                "available": 8500.0,
                "in_positions": 1500.0,
                "unrealized_pnl": 125.50,
            }

        pnl_emoji = "+" if balance.get('unrealized_pnl', 0) >= 0 else ""

        text = f"""
*Account Balance*

Total: `${balance.get('total', 0):,.2f}`
Available: `${balance.get('available', 0):,.2f}`
In Positions: `${balance.get('in_positions', 0):,.2f}`
Unrealized P&L: `{pnl_emoji}${balance.get('unrealized_pnl', 0):,.2f}`
"""
        await update.message.reply_text(text, parse_mode="Markdown")

    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command."""
        if not await self._check_auth(update):
            return

        callback = self.callbacks.get('get_positions')
        if callback:
            positions = await self._call_async(callback)
        else:
            positions = []

        if not positions:
            await update.message.reply_text("No open positions.")
            return

        text = "*Open Positions*\n\n"

        for pos in positions:
            pnl = pos.get('unrealized_pnl', 0)
            pnl_pct = pos.get('unrealized_pnl_pct', 0)
            emoji = "🟢" if pnl >= 0 else "🔴"

            text += f"""
{emoji} *{pos['symbol']}*
Side: `{pos['side'].upper()}`
Size: `{pos['quantity']}`
Entry: `${pos['entry_price']:,.2f}`
Current: `${pos['current_price']:,.2f}`
P&L: `${pnl:+,.2f} ({pnl_pct:+.2f}%)`
"""

        keyboard = [[
            InlineKeyboardButton(
                f"Close {pos['symbol']}",
                callback_data=f"close_{pos['symbol']}"
            )
        ] for pos in positions]
        keyboard.append([InlineKeyboardButton("Close All", callback_data="closeall")])

        await update.message.reply_text(
            text,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /trades command."""
        if not await self._check_auth(update):
            return

        callback = self.callbacks.get('get_trades')
        if callback:
            trades = await self._call_async(callback, limit=10)
        else:
            trades = []

        if not trades:
            await update.message.reply_text("No recent trades.")
            return

        text = "*Recent Trades*\n\n"

        for trade in trades[:10]:
            pnl = trade.get('pnl', 0)
            emoji = "✅" if pnl >= 0 else "❌"

            text += f"{emoji} {trade['symbol']} {trade['side'].upper()} ${pnl:+,.2f}\n"

        await update.message.reply_text(text, parse_mode="Markdown")

    async def cmd_pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pnl command."""
        if not await self._check_auth(update):
            return

        callback = self.callbacks.get('get_pnl')
        if callback:
            pnl_data = await self._call_async(callback)
        else:
            pnl_data = {
                "today": 250.50,
                "week": 1200.00,
                "month": 3500.00,
                "total": 15000.00,
            }

        def fmt(v):
            return f"{'🟢' if v >= 0 else '🔴'} ${v:+,.2f}"

        text = f"""
*Profit & Loss*

Today: {fmt(pnl_data.get('today', 0))}
This Week: {fmt(pnl_data.get('week', 0))}
This Month: {fmt(pnl_data.get('month', 0))}
All Time: {fmt(pnl_data.get('total', 0))}
"""
        await update.message.reply_text(text, parse_mode="Markdown")

    async def cmd_buy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /buy command."""
        if not await self._check_auth(update):
            return

        args = context.args
        if len(args) < 2:
            await update.message.reply_text(
                "Usage: /buy <symbol> <amount>\n"
                "Example: /buy BTCUSDT 0.01"
            )
            return

        symbol = args[0].upper()
        try:
            amount = float(args[1])
        except ValueError:
            await update.message.reply_text("Invalid amount.")
            return

        # Confirmation
        keyboard = [
            [
                InlineKeyboardButton("Confirm", callback_data=f"confirm_buy_{symbol}_{amount}"),
                InlineKeyboardButton("Cancel", callback_data="cancel_order")
            ]
        ]

        await update.message.reply_text(
            f"*Confirm Order*\n\n"
            f"Action: `BUY`\n"
            f"Symbol: `{symbol}`\n"
            f"Amount: `{amount}`",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def cmd_sell(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /sell command."""
        if not await self._check_auth(update):
            return

        args = context.args
        if len(args) < 2:
            await update.message.reply_text(
                "Usage: /sell <symbol> <amount>\n"
                "Example: /sell BTCUSDT 0.01"
            )
            return

        symbol = args[0].upper()
        try:
            amount = float(args[1])
        except ValueError:
            await update.message.reply_text("Invalid amount.")
            return

        keyboard = [
            [
                InlineKeyboardButton("Confirm", callback_data=f"confirm_sell_{symbol}_{amount}"),
                InlineKeyboardButton("Cancel", callback_data="cancel_order")
            ]
        ]

        await update.message.reply_text(
            f"*Confirm Order*\n\n"
            f"Action: `SELL`\n"
            f"Symbol: `{symbol}`\n"
            f"Amount: `{amount}`",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def cmd_close(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /close command."""
        if not await self._check_auth(update):
            return

        args = context.args
        if not args:
            await update.message.reply_text("Usage: /close <symbol>")
            return

        symbol = args[0].upper()

        keyboard = [
            [
                InlineKeyboardButton("Confirm", callback_data=f"confirm_close_{symbol}"),
                InlineKeyboardButton("Cancel", callback_data="cancel_order")
            ]
        ]

        await update.message.reply_text(
            f"Close position for *{symbol}*?",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def cmd_closeall(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /closeall command."""
        if not await self._check_auth(update):
            return

        keyboard = [
            [
                InlineKeyboardButton("YES, Close All", callback_data="confirm_closeall"),
                InlineKeyboardButton("Cancel", callback_data="cancel_order")
            ]
        ]

        await update.message.reply_text(
            "⚠️ *Close ALL positions?*\n\nThis action cannot be undone.",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def cmd_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /cancel command."""
        if not await self._check_auth(update):
            return

        args = context.args
        if not args:
            await update.message.reply_text("Usage: /cancel <order_id>")
            return

        order_id = args[0]

        callback = self.callbacks.get('cancel_order')
        if callback:
            result = await self._call_async(callback, order_id)
            if result.get('success'):
                await update.message.reply_text(f"Order {order_id} cancelled.")
            else:
                await update.message.reply_text(f"Failed: {result.get('error')}")
        else:
            await update.message.reply_text("Order cancellation not available.")

    async def cmd_orders(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /orders command."""
        if not await self._check_auth(update):
            return

        callback = self.callbacks.get('get_orders')
        if callback:
            orders = await self._call_async(callback)
        else:
            orders = []

        if not orders:
            await update.message.reply_text("No pending orders.")
            return

        text = "*Pending Orders*\n\n"
        for order in orders:
            text += (
                f"ID: `{order['id']}`\n"
                f"{order['side'].upper()} {order['symbol']} @ ${order['price']:,.2f}\n\n"
            )

        await update.message.reply_text(text, parse_mode="Markdown")

    async def cmd_start_trading(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start_trading command."""
        if not await self._check_auth(update):
            return

        if not self._is_admin(update.effective_user.id):
            await update.message.reply_text("Admin only command.")
            return

        self.trading_enabled = True
        callback = self.callbacks.get('start_trading')
        if callback:
            await self._call_async(callback)

        await update.message.reply_text("✅ Trading ENABLED")

    async def cmd_stop_trading(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop_trading command."""
        if not await self._check_auth(update):
            return

        if not self._is_admin(update.effective_user.id):
            await update.message.reply_text("Admin only command.")
            return

        self.trading_enabled = False
        callback = self.callbacks.get('stop_trading')
        if callback:
            await self._call_async(callback)

        await update.message.reply_text("🛑 Trading DISABLED")

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command."""
        if not await self._check_auth(update):
            return

        callback = self.callbacks.get('get_stats')
        if callback:
            stats = await self._call_async(callback)
        else:
            stats = {
                "total_trades": 150,
                "win_rate": 65.5,
                "profit_factor": 1.85,
                "sharpe_ratio": 1.42,
                "max_drawdown": 8.5,
                "avg_trade": 45.20,
            }

        text = f"""
*Trading Statistics*

Total Trades: `{stats.get('total_trades', 0)}`
Win Rate: `{stats.get('win_rate', 0):.1f}%`
Profit Factor: `{stats.get('profit_factor', 0):.2f}`
Sharpe Ratio: `{stats.get('sharpe_ratio', 0):.2f}`
Max Drawdown: `{stats.get('max_drawdown', 0):.1f}%`
Avg Trade: `${stats.get('avg_trade', 0):.2f}`
"""
        await update.message.reply_text(text, parse_mode="Markdown")

    async def cmd_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command."""
        if not await self._check_auth(update):
            return

        mode = "PAPER" if self.paper_mode else "LIVE"
        trading = "ON" if self.trading_enabled else "OFF"

        keyboard = [
            [InlineKeyboardButton(
                f"Mode: {mode}",
                callback_data="toggle_mode"
            )],
            [InlineKeyboardButton(
                f"Trading: {trading}",
                callback_data="toggle_trading"
            )],
        ]

        await update.message.reply_text(
            "*Settings*\n\nTap to toggle:",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callbacks."""
        query = update.callback_query
        await query.answer()

        user_id = query.from_user.id
        if not self._is_authorized(user_id):
            return

        data = query.data

        if data == "toggle_trading":
            if self._is_admin(user_id):
                self.trading_enabled = not self.trading_enabled
                status = "ENABLED" if self.trading_enabled else "DISABLED"
                await query.edit_message_text(f"Trading {status}")

        elif data == "toggle_mode":
            if self._is_admin(user_id):
                self.paper_mode = not self.paper_mode
                mode = "PAPER" if self.paper_mode else "LIVE"
                await query.edit_message_text(f"Mode: {mode}")

        elif data == "refresh_status":
            await query.edit_message_text("Refreshing...")
            # Re-run status command
            await self.cmd_status(update, context)

        elif data.startswith("confirm_buy_"):
            parts = data.split("_")
            symbol = parts[2]
            amount = float(parts[3])
            await self._execute_order("buy", symbol, amount, query)

        elif data.startswith("confirm_sell_"):
            parts = data.split("_")
            symbol = parts[2]
            amount = float(parts[3])
            await self._execute_order("sell", symbol, amount, query)

        elif data.startswith("confirm_close_"):
            symbol = data.replace("confirm_close_", "")
            await self._close_position(symbol, query)

        elif data == "confirm_closeall":
            await self._close_all_positions(query)

        elif data.startswith("close_"):
            symbol = data.replace("close_", "")
            await self._close_position(symbol, query)

        elif data == "closeall":
            await query.edit_message_text(
                "⚠️ Close ALL positions?",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("YES", callback_data="confirm_closeall"),
                    InlineKeyboardButton("NO", callback_data="cancel_order")
                ]])
            )

        elif data == "cancel_order":
            await query.edit_message_text("Cancelled.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages."""
        if not await self._check_auth(update):
            return

        session = self._get_session(update.effective_user.id)

        if session.state == BotState.AWAITING_SYMBOL:
            session.data['symbol'] = update.message.text.upper()
            session.state = BotState.AWAITING_AMOUNT
            await update.message.reply_text("Enter amount:")

        elif session.state == BotState.AWAITING_AMOUNT:
            try:
                amount = float(update.message.text)
                session.data['amount'] = amount
                session.state = BotState.IDLE
                # Process order
            except ValueError:
                await update.message.reply_text("Invalid amount. Try again:")

    async def _execute_order(self, side: str, symbol: str, amount: float, query):
        """Execute a trade order."""
        callback = self.callbacks.get('place_order')
        if callback:
            result = await self._call_async(callback, side=side, symbol=symbol, amount=amount)
            if result.get('success'):
                await query.edit_message_text(
                    f"✅ Order executed!\n\n"
                    f"Side: {side.upper()}\n"
                    f"Symbol: {symbol}\n"
                    f"Amount: {amount}\n"
                    f"Price: ${result.get('price', 0):,.2f}"
                )
            else:
                await query.edit_message_text(f"❌ Order failed: {result.get('error')}")
        else:
            await query.edit_message_text("Order execution not available.")

    async def _close_position(self, symbol: str, query):
        """Close a position."""
        callback = self.callbacks.get('close_position')
        if callback:
            result = await self._call_async(callback, symbol=symbol)
            if result.get('success'):
                await query.edit_message_text(f"✅ Position {symbol} closed!")
            else:
                await query.edit_message_text(f"❌ Failed: {result.get('error')}")
        else:
            await query.edit_message_text("Position close not available.")

    async def _close_all_positions(self, query):
        """Close all positions."""
        callback = self.callbacks.get('close_all_positions')
        if callback:
            result = await self._call_async(callback)
            if result.get('success'):
                await query.edit_message_text("✅ All positions closed!")
            else:
                await query.edit_message_text(f"❌ Failed: {result.get('error')}")
        else:
            await query.edit_message_text("Close all not available.")

    async def _call_async(self, callback: Callable, *args, **kwargs):
        """Call callback, handling both sync and async functions."""
        if asyncio.iscoroutinefunction(callback):
            return await callback(*args, **kwargs)
        return callback(*args, **kwargs)

    async def send_alert(self, chat_id: int, message: str, parse_mode: str = "Markdown"):
        """Send alert message to specific chat."""
        if self.app:
            await self.app.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=parse_mode
            )

    async def broadcast(self, message: str, parse_mode: str = "Markdown"):
        """Broadcast message to all allowed users."""
        for user_id in self.allowed_users:
            try:
                await self.send_alert(user_id, message, parse_mode)
            except Exception as e:
                logger.error(f"Failed to send to {user_id}: {e}")

    def run(self):
        """Run the bot (blocking)."""
        app = self.build_app()
        logger.info("Starting Telegram bot...")
        app.run_polling(allowed_updates=Update.ALL_TYPES)


# Notification helper
class TelegramNotifier:
    """Send notifications via Telegram."""

    def __init__(self, token: str, chat_ids: List[int]):
        self.token = token
        self.chat_ids = chat_ids
        self._bot = None

    async def _get_bot(self):
        if not HAS_TELEGRAM:
            raise ImportError("python-telegram-bot not installed")
        if self._bot is None:
            from telegram import Bot
            self._bot = Bot(token=self.token)
        return self._bot

    async def send_trade_alert(self, trade: Dict[str, Any]):
        """Send trade execution alert."""
        pnl = trade.get('pnl', 0)
        emoji = "✅" if pnl >= 0 else "❌"

        message = f"""
{emoji} *Trade Executed*

Symbol: `{trade.get('symbol')}`
Side: `{trade.get('side', '').upper()}`
Entry: `${trade.get('entry_price', 0):,.2f}`
Exit: `${trade.get('exit_price', 0):,.2f}`
P&L: `${pnl:+,.2f}`
"""
        await self._broadcast(message)

    async def send_signal_alert(self, signal: Dict[str, Any]):
        """Send signal alert."""
        action = signal.get('action', 'unknown').upper()
        emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"

        message = f"""
{emoji} *Signal Generated*

Symbol: `{signal.get('symbol')}`
Action: `{action}`
Price: `${signal.get('price', 0):,.2f}`
Strategy: `{signal.get('strategy', 'Unknown')}`
Confidence: `{signal.get('confidence', 0):.1f}%`
"""
        await self._broadcast(message)

    async def send_error_alert(self, error: str, details: str = ""):
        """Send error alert."""
        message = f"""
🚨 *Error Alert*

`{error}`

{details}
"""
        await self._broadcast(message)

    async def send_daily_report(self, report: Dict[str, Any]):
        """Send daily performance report."""
        pnl = report.get('pnl', 0)
        emoji = "📈" if pnl >= 0 else "📉"

        message = f"""
{emoji} *Daily Report*

Date: `{report.get('date', 'Today')}`
P&L: `${pnl:+,.2f}`
Trades: `{report.get('trades', 0)}`
Win Rate: `{report.get('win_rate', 0):.1f}%`
Balance: `${report.get('balance', 0):,.2f}`
"""
        await self._broadcast(message)

    async def _broadcast(self, message: str):
        """Send message to all chat IDs."""
        bot = await self._get_bot()
        for chat_id in self.chat_ids:
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode="Markdown"
                )
            except Exception as e:
                logger.error(f"Failed to send to {chat_id}: {e}")
