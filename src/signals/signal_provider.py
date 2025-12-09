"""
Signal Provider System.

Система надання торгових сигналів підписникам через різні канали:
- Telegram канал
- WebSocket API
- Webhook callbacks
- REST API

Features:
- Публікація сигналів з затримкою для преміум/безкоштовних підписників
- Статистика ефективності сигналів
- Підписки та рівні доступу
- Історія сигналів
"""

import asyncio
import hashlib
import hmac
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Set
import aiohttp
from loguru import logger


# =============================================================================
# Enums and Types
# =============================================================================

class SignalDirection(Enum):
    """Signal direction."""
    LONG = "long"
    SHORT = "short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


class SignalStatus(Enum):
    """Signal status."""
    PENDING = "pending"
    ACTIVE = "active"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class SubscriptionTier(Enum):
    """Subscription tiers."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    VIP = "vip"


class DeliveryChannel(Enum):
    """Signal delivery channels."""
    TELEGRAM = "telegram"
    WEBSOCKET = "websocket"
    WEBHOOK = "webhook"
    EMAIL = "email"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TradingSignal:
    """Trading signal data."""
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    # Signal details
    symbol: str = ""
    direction: SignalDirection = SignalDirection.LONG
    entry_price: Decimal = Decimal("0")

    # Targets
    stop_loss: Optional[Decimal] = None
    take_profit_1: Optional[Decimal] = None
    take_profit_2: Optional[Decimal] = None
    take_profit_3: Optional[Decimal] = None

    # Risk
    risk_reward_ratio: float = 0.0
    position_size_pct: float = 0.0  # Recommended % of portfolio
    leverage: int = 1

    # Metadata
    strategy: str = ""
    timeframe: str = ""
    confidence: float = 0.0  # 0.0 - 1.0
    notes: str = ""

    # Status
    status: SignalStatus = SignalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    activated_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    # Results
    exit_price: Optional[Decimal] = None
    pnl_pct: Optional[float] = None
    hit_target: Optional[int] = None  # Which TP was hit (1, 2, 3) or 0 for SL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "entry_price": str(self.entry_price),
            "stop_loss": str(self.stop_loss) if self.stop_loss else None,
            "take_profit_1": str(self.take_profit_1) if self.take_profit_1 else None,
            "take_profit_2": str(self.take_profit_2) if self.take_profit_2 else None,
            "take_profit_3": str(self.take_profit_3) if self.take_profit_3 else None,
            "risk_reward_ratio": self.risk_reward_ratio,
            "position_size_pct": self.position_size_pct,
            "leverage": self.leverage,
            "strategy": self.strategy,
            "timeframe": self.timeframe,
            "confidence": self.confidence,
            "notes": self.notes,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "exit_price": str(self.exit_price) if self.exit_price else None,
            "pnl_pct": self.pnl_pct,
            "hit_target": self.hit_target,
        }
        return data

    def to_telegram_message(self, include_entry: bool = True) -> str:
        """Format signal for Telegram."""
        emoji = "🟢" if self.direction in (SignalDirection.LONG, SignalDirection.CLOSE_SHORT) else "🔴"
        direction_text = self.direction.value.upper().replace("_", " ")

        msg = f"""
{emoji} **{direction_text}** #{self.symbol}

📊 **Деталі сигналу:**
"""
        if include_entry:
            msg += f"• Вхід: `{self.entry_price}`\n"

        if self.stop_loss:
            msg += f"• Stop Loss: `{self.stop_loss}`\n"

        if self.take_profit_1:
            msg += f"• Take Profit 1: `{self.take_profit_1}`\n"
        if self.take_profit_2:
            msg += f"• Take Profit 2: `{self.take_profit_2}`\n"
        if self.take_profit_3:
            msg += f"• Take Profit 3: `{self.take_profit_3}`\n"

        msg += f"""
⚙️ **Параметри:**
• R/R: `{self.risk_reward_ratio:.1f}`
• Плече: `{self.leverage}x`
• Впевненість: `{self.confidence*100:.0f}%`

📈 Стратегія: `{self.strategy}`
⏱ Таймфрейм: `{self.timeframe}`

🆔 `{self.signal_id}`
"""
        if self.notes:
            msg += f"\n💡 {self.notes}"

        return msg


@dataclass
class Subscriber:
    """Subscriber data."""
    subscriber_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Identity
    name: str = ""
    email: Optional[str] = None
    telegram_id: Optional[int] = None
    telegram_username: Optional[str] = None

    # Subscription
    tier: SubscriptionTier = SubscriptionTier.FREE
    channels: List[DeliveryChannel] = field(default_factory=list)

    # Webhook (for automated trading)
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None

    # Settings
    symbols_filter: List[str] = field(default_factory=list)  # Empty = all
    min_confidence: float = 0.0

    # Status
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    # Stats
    signals_received: int = 0
    last_signal_at: Optional[datetime] = None


@dataclass
class SignalPerformance:
    """Signal performance statistics."""
    total_signals: int = 0
    winning_signals: int = 0
    losing_signals: int = 0
    pending_signals: int = 0

    total_pnl_pct: float = 0.0
    avg_pnl_pct: float = 0.0

    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0

    best_signal_pnl: float = 0.0
    worst_signal_pnl: float = 0.0

    avg_holding_time_hours: float = 0.0

    # By strategy
    by_strategy: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # By symbol
    by_symbol: Dict[str, Dict[str, float]] = field(default_factory=dict)


# =============================================================================
# Signal Provider
# =============================================================================

class SignalProvider:
    """
    Signal Provider for distributing trading signals to subscribers.

    Usage:
        provider = SignalProvider()

        # Add subscriber
        subscriber = provider.add_subscriber(
            name="John",
            telegram_id=123456789,
            tier=SubscriptionTier.PREMIUM,
            channels=[DeliveryChannel.TELEGRAM],
        )

        # Publish signal
        signal = TradingSignal(
            symbol="BTCUSDT",
            direction=SignalDirection.LONG,
            entry_price=Decimal("50000"),
            stop_loss=Decimal("49500"),
            take_profit_1=Decimal("51000"),
        )
        await provider.publish_signal(signal)
    """

    def __init__(
        self,
        telegram_bot_token: Optional[str] = None,
        telegram_channel_id: Optional[str] = None,
        delay_free_minutes: int = 15,
        delay_basic_minutes: int = 5,
    ):
        """
        Initialize Signal Provider.

        Args:
            telegram_bot_token: Telegram bot token for sending signals
            telegram_channel_id: Telegram channel ID for public signals
            delay_free_minutes: Delay for free tier subscribers
            delay_basic_minutes: Delay for basic tier subscribers
        """
        self.telegram_bot_token = telegram_bot_token
        self.telegram_channel_id = telegram_channel_id

        # Delays by tier
        self.tier_delays = {
            SubscriptionTier.FREE: delay_free_minutes,
            SubscriptionTier.BASIC: delay_basic_minutes,
            SubscriptionTier.PREMIUM: 0,
            SubscriptionTier.VIP: 0,
        }

        # Storage
        self._signals: Dict[str, TradingSignal] = {}
        self._subscribers: Dict[str, Subscriber] = {}
        self._signal_history: List[TradingSignal] = []

        # Callbacks
        self._on_signal_callbacks: List[Callable] = []

        # WebSocket connections
        self._ws_connections: Dict[str, Set] = {}  # subscriber_id -> connections

        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info("Signal Provider initialized")

    async def start(self) -> None:
        """Start the signal provider."""
        self._session = aiohttp.ClientSession()
        logger.info("Signal Provider started")

    async def stop(self) -> None:
        """Stop the signal provider."""
        if self._session:
            await self._session.close()
        logger.info("Signal Provider stopped")

    # =========================================================================
    # Subscriber Management
    # =========================================================================

    def add_subscriber(
        self,
        name: str,
        tier: SubscriptionTier = SubscriptionTier.FREE,
        channels: List[DeliveryChannel] = None,
        telegram_id: Optional[int] = None,
        email: Optional[str] = None,
        webhook_url: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        **kwargs,
    ) -> Subscriber:
        """Add a new subscriber."""
        subscriber = Subscriber(
            name=name,
            tier=tier,
            channels=channels or [DeliveryChannel.TELEGRAM],
            telegram_id=telegram_id,
            email=email,
            webhook_url=webhook_url,
            expires_at=expires_at,
            **kwargs,
        )

        # Generate webhook secret if webhook is used
        if webhook_url:
            subscriber.webhook_secret = hashlib.sha256(
                f"{subscriber.subscriber_id}{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()[:32]

        self._subscribers[subscriber.subscriber_id] = subscriber
        logger.info(f"Added subscriber: {subscriber.subscriber_id} ({name}) - {tier.value}")

        return subscriber

    def get_subscriber(self, subscriber_id: str) -> Optional[Subscriber]:
        """Get subscriber by ID."""
        return self._subscribers.get(subscriber_id)

    def get_subscriber_by_telegram(self, telegram_id: int) -> Optional[Subscriber]:
        """Get subscriber by Telegram ID."""
        for sub in self._subscribers.values():
            if sub.telegram_id == telegram_id:
                return sub
        return None

    def update_subscriber(self, subscriber_id: str, **kwargs) -> Optional[Subscriber]:
        """Update subscriber details."""
        subscriber = self._subscribers.get(subscriber_id)
        if not subscriber:
            return None

        for key, value in kwargs.items():
            if hasattr(subscriber, key):
                setattr(subscriber, key, value)

        return subscriber

    def remove_subscriber(self, subscriber_id: str) -> bool:
        """Remove a subscriber."""
        if subscriber_id in self._subscribers:
            del self._subscribers[subscriber_id]
            logger.info(f"Removed subscriber: {subscriber_id}")
            return True
        return False

    def get_active_subscribers(self, tier: Optional[SubscriptionTier] = None) -> List[Subscriber]:
        """Get all active subscribers, optionally filtered by tier."""
        now = datetime.utcnow()
        subscribers = []

        for sub in self._subscribers.values():
            if not sub.is_active:
                continue
            if sub.expires_at and sub.expires_at < now:
                continue
            if tier and sub.tier != tier:
                continue
            subscribers.append(sub)

        return subscribers

    # =========================================================================
    # Signal Publishing
    # =========================================================================

    async def publish_signal(self, signal: TradingSignal) -> None:
        """
        Publish a signal to all subscribers.

        Premium/VIP get instant delivery, lower tiers get delayed.
        """
        signal.status = SignalStatus.ACTIVE
        signal.activated_at = datetime.utcnow()

        self._signals[signal.signal_id] = signal
        self._signal_history.append(signal)

        logger.info(f"Publishing signal: {signal.signal_id} - {signal.direction.value} {signal.symbol}")

        # Group subscribers by delay
        instant_subscribers = []
        delayed_subscribers: Dict[int, List[Subscriber]] = {}

        for subscriber in self.get_active_subscribers():
            # Check filters
            if subscriber.symbols_filter and signal.symbol not in subscriber.symbols_filter:
                continue
            if signal.confidence < subscriber.min_confidence:
                continue

            delay = self.tier_delays.get(subscriber.tier, 0)

            if delay == 0:
                instant_subscribers.append(subscriber)
            else:
                if delay not in delayed_subscribers:
                    delayed_subscribers[delay] = []
                delayed_subscribers[delay].append(subscriber)

        # Send to instant subscribers
        await self._deliver_signal(signal, instant_subscribers, include_entry=True)

        # Schedule delayed deliveries
        for delay_minutes, subscribers in delayed_subscribers.items():
            asyncio.create_task(
                self._delayed_delivery(signal, subscribers, delay_minutes)
            )

        # Publish to public channel (if configured)
        if self.telegram_channel_id:
            asyncio.create_task(
                self._publish_to_channel(signal)
            )

        # Call callbacks
        for callback in self._on_signal_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(signal)
                else:
                    callback(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")

    async def _delayed_delivery(
        self,
        signal: TradingSignal,
        subscribers: List[Subscriber],
        delay_minutes: int,
    ) -> None:
        """Deliver signal after delay."""
        await asyncio.sleep(delay_minutes * 60)

        # Check if signal is still active
        if signal.status not in (SignalStatus.ACTIVE, SignalStatus.PENDING):
            return

        await self._deliver_signal(signal, subscribers, include_entry=True)

    async def _deliver_signal(
        self,
        signal: TradingSignal,
        subscribers: List[Subscriber],
        include_entry: bool = True,
    ) -> None:
        """Deliver signal to subscribers via their preferred channels."""
        for subscriber in subscribers:
            for channel in subscriber.channels:
                try:
                    if channel == DeliveryChannel.TELEGRAM:
                        await self._send_telegram(subscriber, signal, include_entry)
                    elif channel == DeliveryChannel.WEBHOOK:
                        await self._send_webhook(subscriber, signal)
                    elif channel == DeliveryChannel.WEBSOCKET:
                        await self._send_websocket(subscriber, signal)

                    subscriber.signals_received += 1
                    subscriber.last_signal_at = datetime.utcnow()

                except Exception as e:
                    logger.error(f"Failed to deliver signal to {subscriber.subscriber_id} via {channel.value}: {e}")

    async def _send_telegram(
        self,
        subscriber: Subscriber,
        signal: TradingSignal,
        include_entry: bool = True,
    ) -> None:
        """Send signal via Telegram."""
        if not self.telegram_bot_token or not subscriber.telegram_id:
            return

        message = signal.to_telegram_message(include_entry)

        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": subscriber.telegram_id,
            "text": message,
            "parse_mode": "Markdown",
        }

        async with self._session.post(url, json=payload) as resp:
            if resp.status != 200:
                logger.error(f"Telegram send failed: {await resp.text()}")

    async def _send_webhook(self, subscriber: Subscriber, signal: TradingSignal) -> None:
        """Send signal via webhook."""
        if not subscriber.webhook_url:
            return

        payload = signal.to_dict()

        # Generate signature
        signature = hmac.new(
            subscriber.webhook_secret.encode(),
            json.dumps(payload).encode(),
            hashlib.sha256,
        ).hexdigest()

        headers = {
            "Content-Type": "application/json",
            "X-Signal-Signature": signature,
            "X-Signal-Id": signal.signal_id,
        }

        async with self._session.post(
            subscriber.webhook_url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                logger.error(f"Webhook send failed to {subscriber.webhook_url}: {resp.status}")

    async def _send_websocket(self, subscriber: Subscriber, signal: TradingSignal) -> None:
        """Send signal via WebSocket."""
        connections = self._ws_connections.get(subscriber.subscriber_id, set())

        message = json.dumps({
            "type": "signal",
            "data": signal.to_dict(),
        })

        for ws in connections:
            try:
                await ws.send_str(message)
            except Exception as e:
                logger.error(f"WebSocket send failed: {e}")

    async def _publish_to_channel(self, signal: TradingSignal) -> None:
        """Publish signal to public Telegram channel (without exact entry)."""
        if not self.telegram_bot_token or not self.telegram_channel_id:
            return

        # Public channel gets signal without exact entry price
        message = signal.to_telegram_message(include_entry=False)
        message += "\n\n💎 *Підпишіться на Premium для точних входів!*"

        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_channel_id,
            "text": message,
            "parse_mode": "Markdown",
        }

        async with self._session.post(url, json=payload) as resp:
            if resp.status != 200:
                logger.error(f"Channel publish failed: {await resp.text()}")

    # =========================================================================
    # Signal Updates
    # =========================================================================

    async def update_signal(
        self,
        signal_id: str,
        status: Optional[SignalStatus] = None,
        exit_price: Optional[Decimal] = None,
        pnl_pct: Optional[float] = None,
        hit_target: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> Optional[TradingSignal]:
        """Update signal status and notify subscribers."""
        signal = self._signals.get(signal_id)
        if not signal:
            return None

        if status:
            signal.status = status
            if status == SignalStatus.CLOSED:
                signal.closed_at = datetime.utcnow()

        if exit_price is not None:
            signal.exit_price = exit_price

        if pnl_pct is not None:
            signal.pnl_pct = pnl_pct

        if hit_target is not None:
            signal.hit_target = hit_target

        if notes:
            signal.notes = notes

        # Notify subscribers about update
        await self._notify_signal_update(signal)

        return signal

    async def close_signal(
        self,
        signal_id: str,
        exit_price: Decimal,
        hit_target: int = 0,  # 0 = SL, 1-3 = TP levels
    ) -> Optional[TradingSignal]:
        """Close a signal with results."""
        signal = self._signals.get(signal_id)
        if not signal:
            return None

        # Calculate PnL
        if signal.direction == SignalDirection.LONG:
            pnl_pct = float((exit_price - signal.entry_price) / signal.entry_price * 100)
        else:
            pnl_pct = float((signal.entry_price - exit_price) / signal.entry_price * 100)

        pnl_pct *= signal.leverage

        return await self.update_signal(
            signal_id,
            status=SignalStatus.CLOSED,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            hit_target=hit_target,
        )

    async def _notify_signal_update(self, signal: TradingSignal) -> None:
        """Notify subscribers about signal update."""
        if signal.status == SignalStatus.CLOSED:
            emoji = "✅" if signal.pnl_pct and signal.pnl_pct > 0 else "❌"
            target_text = f"TP{signal.hit_target}" if signal.hit_target else "SL"

            message = f"""
{emoji} **Сигнал закрито** #{signal.symbol}

🆔 `{signal.signal_id}`
📊 Вихід: `{signal.exit_price}`
💰 P&L: `{signal.pnl_pct:+.2f}%`
🎯 Досягнуто: `{target_text}`
"""

            # Send update to all subscribers who received this signal
            for subscriber in self.get_active_subscribers():
                if DeliveryChannel.TELEGRAM in subscriber.channels and subscriber.telegram_id:
                    await self._send_telegram_message(subscriber.telegram_id, message)

    async def _send_telegram_message(self, chat_id: int, message: str) -> None:
        """Send a simple Telegram message."""
        if not self.telegram_bot_token:
            return

        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }

        async with self._session.post(url, json=payload) as resp:
            pass  # Ignore errors for updates

    # =========================================================================
    # Performance Statistics
    # =========================================================================

    def get_performance(
        self,
        days: int = 30,
        strategy: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> SignalPerformance:
        """Calculate signal performance statistics."""
        cutoff = datetime.utcnow() - timedelta(days=days)

        # Filter signals
        signals = [
            s for s in self._signal_history
            if s.created_at >= cutoff
            and (not strategy or s.strategy == strategy)
            and (not symbol or s.symbol == symbol)
        ]

        if not signals:
            return SignalPerformance()

        # Calculate stats
        closed_signals = [s for s in signals if s.status == SignalStatus.CLOSED]
        winning = [s for s in closed_signals if s.pnl_pct and s.pnl_pct > 0]
        losing = [s for s in closed_signals if s.pnl_pct and s.pnl_pct < 0]

        total_pnl = sum(s.pnl_pct or 0 for s in closed_signals)

        gross_profit = sum(s.pnl_pct for s in winning) if winning else 0
        gross_loss = abs(sum(s.pnl_pct for s in losing)) if losing else 0

        perf = SignalPerformance(
            total_signals=len(signals),
            winning_signals=len(winning),
            losing_signals=len(losing),
            pending_signals=len([s for s in signals if s.status == SignalStatus.ACTIVE]),
            total_pnl_pct=total_pnl,
            avg_pnl_pct=total_pnl / len(closed_signals) if closed_signals else 0,
            win_rate=len(winning) / len(closed_signals) * 100 if closed_signals else 0,
            avg_win_pct=gross_profit / len(winning) if winning else 0,
            avg_loss_pct=gross_loss / len(losing) if losing else 0,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            best_signal_pnl=max((s.pnl_pct or 0) for s in closed_signals) if closed_signals else 0,
            worst_signal_pnl=min((s.pnl_pct or 0) for s in closed_signals) if closed_signals else 0,
        )

        # Calculate average holding time
        holding_times = []
        for s in closed_signals:
            if s.activated_at and s.closed_at:
                delta = (s.closed_at - s.activated_at).total_seconds() / 3600
                holding_times.append(delta)

        if holding_times:
            perf.avg_holding_time_hours = sum(holding_times) / len(holding_times)

        # Stats by strategy
        strategies = set(s.strategy for s in closed_signals if s.strategy)
        for strat in strategies:
            strat_signals = [s for s in closed_signals if s.strategy == strat]
            strat_winning = [s for s in strat_signals if s.pnl_pct and s.pnl_pct > 0]
            perf.by_strategy[strat] = {
                "total": len(strat_signals),
                "win_rate": len(strat_winning) / len(strat_signals) * 100 if strat_signals else 0,
                "total_pnl": sum(s.pnl_pct or 0 for s in strat_signals),
            }

        # Stats by symbol
        symbols = set(s.symbol for s in closed_signals)
        for sym in symbols:
            sym_signals = [s for s in closed_signals if s.symbol == sym]
            sym_winning = [s for s in sym_signals if s.pnl_pct and s.pnl_pct > 0]
            perf.by_symbol[sym] = {
                "total": len(sym_signals),
                "win_rate": len(sym_winning) / len(sym_signals) * 100 if sym_signals else 0,
                "total_pnl": sum(s.pnl_pct or 0 for s in sym_signals),
            }

        return perf

    # =========================================================================
    # WebSocket Support
    # =========================================================================

    def register_websocket(self, subscriber_id: str, ws) -> None:
        """Register a WebSocket connection for a subscriber."""
        if subscriber_id not in self._ws_connections:
            self._ws_connections[subscriber_id] = set()
        self._ws_connections[subscriber_id].add(ws)

    def unregister_websocket(self, subscriber_id: str, ws) -> None:
        """Unregister a WebSocket connection."""
        if subscriber_id in self._ws_connections:
            self._ws_connections[subscriber_id].discard(ws)

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_signal(self, callback: Callable) -> None:
        """Register callback for new signals."""
        self._on_signal_callbacks.append(callback)

    # =========================================================================
    # Signal History
    # =========================================================================

    def get_signals(
        self,
        status: Optional[SignalStatus] = None,
        symbol: Optional[str] = None,
        limit: int = 50,
    ) -> List[TradingSignal]:
        """Get signals with optional filters."""
        signals = self._signal_history.copy()

        if status:
            signals = [s for s in signals if s.status == status]

        if symbol:
            signals = [s for s in signals if s.symbol == symbol]

        signals.sort(key=lambda s: s.created_at, reverse=True)

        return signals[:limit]

    def get_active_signals(self) -> List[TradingSignal]:
        """Get all active signals."""
        return [s for s in self._signals.values() if s.status == SignalStatus.ACTIVE]
