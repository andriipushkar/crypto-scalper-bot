"""
Risk Manager.

Controls position sizing, maximum exposure, and trade validation.
Critical for protecting capital in scalping operations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any

from loguru import logger

from src.data.models import Signal, SignalType, Side, Position


@dataclass
class RiskConfig:
    """Risk management configuration."""

    # Capital
    total_capital: Decimal = Decimal("100")  # Total capital in USD
    max_position_pct: Decimal = Decimal("0.1")  # Max 10% per position

    # Risk per trade
    risk_per_trade_pct: Decimal = Decimal("0.01")  # Risk 1% per trade
    max_loss_per_trade: Decimal = Decimal("2")  # Max $2 loss per trade

    # Position limits
    max_positions: int = 1  # Max concurrent positions
    max_daily_trades: int = 50  # Max trades per day
    max_daily_loss: Decimal = Decimal("10")  # Max $10 daily loss

    # Stop loss / Take profit
    default_stop_loss_pct: Decimal = Decimal("0.002")  # 0.2% stop loss
    default_take_profit_pct: Decimal = Decimal("0.003")  # 0.3% take profit

    # Leverage
    max_leverage: int = 10
    default_leverage: int = 5

    # Cooldowns
    cooldown_after_loss: int = 60  # seconds
    cooldown_after_max_loss: int = 3600  # 1 hour


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: datetime = field(default_factory=lambda: datetime.utcnow().date())
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    peak_pnl: Decimal = Decimal("0")


class RiskManager:
    """
    Risk management for the trading bot.

    Responsibilities:
    - Position sizing based on risk parameters
    - Trade validation (can we trade?)
    - Exposure management
    - Daily loss limits
    - Cooldown enforcement
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk manager.

        Args:
            config: Risk configuration dictionary
        """
        self.config = self._parse_config(config)

        # State
        self._positions: Dict[str, Position] = {}
        self._daily_stats = DailyStats()
        self._last_loss_time: Optional[datetime] = None
        self._paused = False
        self._pause_until: Optional[datetime] = None

        # Trade history for analysis
        self._recent_trades: List[Dict] = []

        logger.info(
            f"Risk Manager initialized: capital=${self.config.total_capital}, "
            f"max_position={float(self.config.max_position_pct)*100}%, "
            f"risk_per_trade={float(self.config.risk_per_trade_pct)*100}%"
        )

    def _parse_config(self, config: Dict[str, Any]) -> RiskConfig:
        """Parse configuration dictionary into RiskConfig."""
        return RiskConfig(
            total_capital=Decimal(str(config.get("total_capital", 100))),
            max_position_pct=Decimal(str(config.get("max_position_pct", 0.1))),
            risk_per_trade_pct=Decimal(str(config.get("risk_per_trade_pct", 0.01))),
            max_loss_per_trade=Decimal(str(config.get("max_loss_per_trade", 2))),
            max_positions=config.get("max_positions", 1),
            max_daily_trades=config.get("max_daily_trades", 50),
            max_daily_loss=Decimal(str(config.get("max_daily_loss", 10))),
            default_stop_loss_pct=Decimal(str(config.get("default_stop_loss_pct", 0.002))),
            default_take_profit_pct=Decimal(str(config.get("default_take_profit_pct", 0.003))),
            max_leverage=config.get("max_leverage", 10),
            default_leverage=config.get("default_leverage", 5),
            cooldown_after_loss=config.get("cooldown_after_loss", 60),
            cooldown_after_max_loss=config.get("cooldown_after_max_loss", 3600),
        )

    @property
    def stats(self) -> Dict[str, Any]:
        """Get current risk statistics."""
        return {
            "total_capital": float(self.config.total_capital),
            "available_capital": float(self.available_capital),
            "position_count": len(self._positions),
            "daily_trades": self._daily_stats.trades,
            "daily_pnl": float(self._daily_stats.total_pnl),
            "daily_wins": self._daily_stats.wins,
            "daily_losses": self._daily_stats.losses,
            "paused": self._paused,
        }

    @property
    def available_capital(self) -> Decimal:
        """Calculate available capital (not in positions)."""
        used = sum(p.notional_value for p in self._positions.values())
        return max(self.config.total_capital - used, Decimal("0"))

    # =========================================================================
    # Trade Validation
    # =========================================================================

    def can_trade(self, signal: Signal = None) -> Tuple[bool, str]:
        """
        Check if we can execute a trade.

        Args:
            signal: Optional signal to validate

        Returns:
            Tuple of (can_trade, reason)
        """
        # Check if paused
        if self._paused:
            if self._pause_until and datetime.utcnow() >= self._pause_until:
                self._paused = False
                self._pause_until = None
            else:
                return False, "Trading paused"

        # Check daily date reset
        self._check_daily_reset()

        # Check daily trade limit
        if self._daily_stats.trades >= self.config.max_daily_trades:
            return False, f"Daily trade limit reached ({self.config.max_daily_trades})"

        # Check daily loss limit
        if self._daily_stats.total_pnl <= -self.config.max_daily_loss:
            self._pause_trading(self.config.cooldown_after_max_loss)
            return False, f"Daily loss limit reached (${self.config.max_daily_loss})"

        # Check position limit (for new entries)
        if signal and signal.is_entry:
            if len(self._positions) >= self.config.max_positions:
                return False, f"Max positions reached ({self.config.max_positions})"

            # Check if already have position in symbol
            if signal.symbol in self._positions:
                return False, f"Already have position in {signal.symbol}"

        # Check loss cooldown
        if self._last_loss_time:
            elapsed = (datetime.utcnow() - self._last_loss_time).total_seconds()
            if elapsed < self.config.cooldown_after_loss:
                remaining = self.config.cooldown_after_loss - elapsed
                return False, f"Loss cooldown ({remaining:.0f}s remaining)"

        # Check available capital (only for entry signals)
        if signal and signal.is_entry:
            if self.available_capital < self.config.total_capital * Decimal("0.05"):
                return False, "Insufficient available capital"

        return True, "OK"

    def _check_daily_reset(self) -> None:
        """Reset daily stats if new day."""
        today = datetime.utcnow().date()
        if self._daily_stats.date != today:
            logger.info(f"New trading day: {today}")
            self._daily_stats = DailyStats(date=today)

    def _pause_trading(self, seconds: int) -> None:
        """Pause trading for specified duration."""
        self._paused = True
        self._pause_until = datetime.utcnow() + timedelta(seconds=seconds)
        logger.warning(f"Trading paused for {seconds} seconds")

    # =========================================================================
    # Position Sizing
    # =========================================================================

    def calculate_position_size(
        self,
        signal: Signal,
        leverage: int = None,
    ) -> Decimal:
        """
        Calculate position size for a signal.

        Uses risk-based sizing: position size is calculated to risk
        a fixed percentage of capital on each trade.

        Args:
            signal: Trading signal
            leverage: Optional leverage override

        Returns:
            Position size in base currency (e.g., BTC)
        """
        leverage = leverage or self.config.default_leverage
        leverage = min(leverage, self.config.max_leverage)

        # Maximum position value
        max_position_value = self.config.total_capital * self.config.max_position_pct

        # Risk-based sizing
        # Risk amount = capital * risk_per_trade_pct
        risk_amount = self.config.total_capital * self.config.risk_per_trade_pct
        risk_amount = min(risk_amount, self.config.max_loss_per_trade)

        # Position value = risk_amount / stop_loss_pct
        stop_loss_pct = self.config.default_stop_loss_pct
        position_value = risk_amount / stop_loss_pct

        # Apply leverage
        position_value_leveraged = position_value * leverage

        # Cap at max position
        position_value_leveraged = min(position_value_leveraged, max_position_value)

        # Cap at available capital
        position_value_leveraged = min(position_value_leveraged, self.available_capital * leverage)

        # Convert to quantity
        if signal.price > 0:
            quantity = position_value_leveraged / signal.price
        else:
            quantity = Decimal("0")

        logger.debug(
            f"Position sizing: risk=${risk_amount:.2f}, "
            f"value=${position_value_leveraged:.2f}, "
            f"qty={quantity:.6f} @ {signal.price}"
        )

        return quantity

    def calculate_stop_loss(
        self,
        entry_price: Decimal,
        side: Side,
        stop_loss_pct: Decimal = None,
    ) -> Decimal:
        """
        Calculate stop loss price.

        Args:
            entry_price: Entry price
            side: Position side
            stop_loss_pct: Optional custom stop loss percentage

        Returns:
            Stop loss price
        """
        pct = stop_loss_pct or self.config.default_stop_loss_pct

        if side == Side.BUY:
            return entry_price * (1 - pct)
        else:
            return entry_price * (1 + pct)

    def calculate_take_profit(
        self,
        entry_price: Decimal,
        side: Side,
        take_profit_pct: Decimal = None,
    ) -> Decimal:
        """
        Calculate take profit price.

        Args:
            entry_price: Entry price
            side: Position side
            take_profit_pct: Optional custom take profit percentage

        Returns:
            Take profit price
        """
        pct = take_profit_pct or self.config.default_take_profit_pct

        if side == Side.BUY:
            return entry_price * (1 + pct)
        else:
            return entry_price * (1 - pct)

    # =========================================================================
    # Position Management
    # =========================================================================

    def register_position(self, position: Position) -> None:
        """Register a new position."""
        self._positions[position.symbol] = position
        self._daily_stats.trades += 1
        logger.info(f"Position registered: {position.symbol} {position.side.value} {position.size}")

    def update_position(self, position: Position) -> None:
        """Update an existing position."""
        self._positions[position.symbol] = position

    def close_position(self, symbol: str, pnl: Decimal) -> None:
        """
        Close a position and record P&L.

        Args:
            symbol: Position symbol
            pnl: Realized P&L
        """
        if symbol in self._positions:
            del self._positions[symbol]

        # Update stats
        self._daily_stats.total_pnl += pnl

        if pnl > 0:
            self._daily_stats.wins += 1
        else:
            self._daily_stats.losses += 1
            self._last_loss_time = datetime.utcnow()

        # Track drawdown
        if self._daily_stats.total_pnl > self._daily_stats.peak_pnl:
            self._daily_stats.peak_pnl = self._daily_stats.total_pnl

        drawdown = self._daily_stats.peak_pnl - self._daily_stats.total_pnl
        if drawdown > self._daily_stats.max_drawdown:
            self._daily_stats.max_drawdown = drawdown

        logger.info(
            f"Position closed: {symbol} PnL=${pnl:.2f}, "
            f"Daily PnL=${self._daily_stats.total_pnl:.2f}"
        )

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self._positions.values())

    # =========================================================================
    # Emergency Controls
    # =========================================================================

    def emergency_stop(self) -> None:
        """Emergency stop - pause all trading."""
        self._paused = True
        self._pause_until = None  # Indefinite pause
        logger.critical("EMERGENCY STOP - Trading halted")

    def resume_trading(self) -> None:
        """Resume trading after pause."""
        self._paused = False
        self._pause_until = None
        logger.info("Trading resumed")

    def reset_daily_stats(self) -> None:
        """Reset daily statistics (use with caution)."""
        self._daily_stats = DailyStats()
        logger.info("Daily stats reset")
