"""
Dollar Cost Averaging (DCA) Strategy.

Systematically buys/sells at regular intervals or on price dips.
Reduces impact of volatility by spreading entries over time.

Modes:
- Time-based: Buy/sell at fixed intervals
- Price-based: Buy on dips, sell on pumps
- Hybrid: Combine both approaches
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List, Deque

from loguru import logger

from src.data.models import OrderBookSnapshot, Trade, Signal, SignalType
from src.strategy.base import BaseStrategy


class DCAMode(Enum):
    """DCA execution mode."""
    TIME_BASED = "time_based"  # Fixed intervals
    PRICE_BASED = "price_based"  # Buy dips / sell pumps
    HYBRID = "hybrid"  # Both conditions


class DCADirection(Enum):
    """DCA direction."""
    ACCUMULATE = "accumulate"  # Buy over time (long-term bullish)
    DISTRIBUTE = "distribute"  # Sell over time (taking profits)
    BIDIRECTIONAL = "bidirectional"  # Both directions based on conditions


@dataclass
class SafetyOrder:
    """Safety order configuration."""
    level: int  # Order level (1, 2, 3, ...)
    price_deviation_pct: Decimal  # Price drop % from initial entry
    size_multiplier: Decimal  # Size multiplier relative to base order
    price: Optional[Decimal] = None  # Calculated price
    triggered: bool = False
    trigger_time: Optional[datetime] = None


@dataclass
class DCAState:
    """DCA strategy state."""
    total_invested: Decimal = Decimal("0")
    total_quantity: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    entries_count: int = 0
    last_entry_time: Optional[datetime] = None
    last_entry_price: Optional[Decimal] = None
    highest_price_seen: Decimal = Decimal("0")
    lowest_price_seen: Decimal = Decimal("999999999")
    # Trailing take profit
    trailing_tp_active: bool = False
    trailing_tp_highest: Decimal = Decimal("0")
    trailing_tp_trigger_price: Optional[Decimal] = None
    # Base strategy state fields
    last_signal_time: Optional[datetime] = None
    last_signal_type: Optional[SignalType] = None
    signals_generated: int = 0
    errors: int = 0


@dataclass
class DCAEntry:
    """Record of a DCA entry."""
    timestamp: datetime
    price: Decimal
    quantity: Decimal
    cumulative_quantity: Decimal
    cumulative_invested: Decimal
    average_price: Decimal


class DCAStrategy(BaseStrategy):
    """
    Dollar Cost Averaging Strategy.

    Systematically accumulates or distributes position over time,
    reducing timing risk and averaging entry/exit prices.

    Parameters:
    - mode: DCA mode (time_based, price_based, hybrid)
    - direction: accumulate, distribute, or bidirectional
    - interval_minutes: Time between entries (for time_based/hybrid)
    - dip_threshold: Price drop % to trigger buy (for price_based/hybrid)
    - pump_threshold: Price rise % to trigger sell (for price_based/hybrid)
    - quantity_per_entry: Amount per DCA entry
    - max_entries: Maximum number of entries
    - total_budget: Total budget for DCA campaign
    - scale_on_dip: Increase size on bigger dips
    - trailing_dip: Use trailing dip detection
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DCA strategy.

        Args:
            config: Strategy configuration
        """
        super().__init__(config)

        # Mode and direction
        mode_str = config.get("mode", "hybrid").lower()
        self.mode = DCAMode(mode_str)

        direction_str = config.get("direction", "accumulate").lower()
        self.direction = DCADirection(direction_str)

        # Time-based settings
        self.interval_minutes = config.get("interval_minutes", 60)
        self._interval_delta = timedelta(minutes=self.interval_minutes)

        # Price-based settings
        self.dip_threshold = Decimal(str(config.get("dip_threshold", 0.02)))  # 2%
        self.pump_threshold = Decimal(str(config.get("pump_threshold", 0.02)))  # 2%
        self.lookback_minutes = config.get("lookback_minutes", 60)

        # Position sizing
        self.quantity_per_entry = Decimal(str(config.get("quantity_per_entry", 0.001)))
        self.max_entries = config.get("max_entries", 100)
        self.total_budget = Decimal(str(config.get("total_budget", 1000)))

        # Advanced settings
        self.scale_on_dip = config.get("scale_on_dip", False)
        self.scale_factor = Decimal(str(config.get("scale_factor", 1.5)))
        self.trailing_dip = config.get("trailing_dip", False)
        self.trailing_window = config.get("trailing_window", 10)  # minutes

        # Take profit / Stop loss for accumulated position
        self.take_profit_pct = Decimal(str(config.get("take_profit_pct", 0.10)))  # 10%
        self.stop_loss_pct = Decimal(str(config.get("stop_loss_pct", 0.05)))  # 5%

        # Trailing take profit settings
        self.trailing_tp_enabled = config.get("trailing_tp_enabled", False)
        self.trailing_tp_activation_pct = Decimal(str(config.get("trailing_tp_activation_pct", 0.05)))  # 5%
        self.trailing_tp_deviation_pct = Decimal(str(config.get("trailing_tp_deviation_pct", 0.02)))  # 2%

        # Safety orders settings
        self.safety_orders_enabled = config.get("safety_orders_enabled", False)
        self.safety_orders_count = config.get("safety_orders_count", 5)
        self.safety_order_step_pct = Decimal(str(config.get("safety_order_step_pct", 0.01)))  # 1%
        self.safety_order_step_scale = Decimal(str(config.get("safety_order_step_scale", 1.5)))  # Step multiplier
        self.safety_order_size_scale = Decimal(str(config.get("safety_order_size_scale", 1.5)))  # Size multiplier

        # State
        self._state = DCAState()
        self._price_history: Deque[tuple] = deque(maxlen=1000)  # (timestamp, price)
        self._entries: List[DCAEntry] = []
        self._reference_price: Optional[Decimal] = None
        self._safety_orders: List[SafetyOrder] = []
        self._initial_entry_price: Optional[Decimal] = None

        logger.info(
            f"[{self.name}] Config: mode={self.mode.value}, "
            f"direction={self.direction.value}, "
            f"interval={self.interval_minutes}min, "
            f"dip={float(self.dip_threshold)*100:.1f}%"
        )

    def on_orderbook(self, snapshot: OrderBookSnapshot) -> Optional[Signal]:
        """
        Process order book update and check for DCA signals.

        Args:
            snapshot: Current order book snapshot

        Returns:
            Signal if DCA conditions are met
        """
        if not snapshot.best_bid or not snapshot.best_ask:
            return None

        current_price = snapshot.mid_price
        current_time = snapshot.timestamp

        # Update price history
        self._update_price_history(current_time, current_price)

        # Update reference price for dip/pump calculation
        self._update_reference_price(current_price)

        # Check if max entries reached
        if self._state.entries_count >= self.max_entries:
            return self._check_exit_conditions(snapshot)

        # Check budget
        if self._state.total_invested >= self.total_budget:
            return self._check_exit_conditions(snapshot)

        # Generate signal based on mode
        signal = None

        if self.mode == DCAMode.TIME_BASED:
            signal = self._check_time_signal(snapshot)
        elif self.mode == DCAMode.PRICE_BASED:
            signal = self._check_price_signal(snapshot)
        else:  # HYBRID
            signal = self._check_hybrid_signal(snapshot)

        # Also check exit conditions
        if signal is None:
            signal = self._check_exit_conditions(snapshot)

        return signal

    def _update_price_history(self, timestamp: datetime, price: Decimal) -> None:
        """Update price history."""
        self._price_history.append((timestamp, price))

        # Update high/low tracking
        if price > self._state.highest_price_seen:
            self._state.highest_price_seen = price
        if price < self._state.lowest_price_seen:
            self._state.lowest_price_seen = price

    def _update_reference_price(self, current_price: Decimal) -> None:
        """Update reference price for dip/pump calculations."""
        if self._reference_price is None:
            self._reference_price = current_price
            return

        if self.trailing_dip:
            # Trailing: reference moves up with price, but not down
            if self.direction in [DCADirection.ACCUMULATE, DCADirection.BIDIRECTIONAL]:
                if current_price > self._reference_price:
                    self._reference_price = current_price
            if self.direction in [DCADirection.DISTRIBUTE, DCADirection.BIDIRECTIONAL]:
                if current_price < self._reference_price:
                    self._reference_price = current_price

    def _get_recent_high(self) -> Optional[Decimal]:
        """Get recent high price."""
        if not self._price_history:
            return None

        cutoff = datetime.utcnow() - timedelta(minutes=self.lookback_minutes)
        recent_prices = [p for t, p in self._price_history if t >= cutoff]

        if not recent_prices:
            return None

        return max(recent_prices)

    def _get_recent_low(self) -> Optional[Decimal]:
        """Get recent low price."""
        if not self._price_history:
            return None

        cutoff = datetime.utcnow() - timedelta(minutes=self.lookback_minutes)
        recent_prices = [p for t, p in self._price_history if t >= cutoff]

        if not recent_prices:
            return None

        return min(recent_prices)

    def _check_time_signal(self, snapshot: OrderBookSnapshot) -> Optional[Signal]:
        """Check for time-based DCA signal."""
        if not self.can_signal():
            return None

        current_time = snapshot.timestamp

        # Check if interval has passed
        if self._state.last_entry_time is not None:
            elapsed = current_time - self._state.last_entry_time
            if elapsed < self._interval_delta:
                return None

        # Generate signal based on direction
        return self._create_dca_signal(snapshot, reason="time_interval")

    def _check_price_signal(self, snapshot: OrderBookSnapshot) -> Optional[Signal]:
        """Check for price-based DCA signal."""
        if not self.can_signal():
            return None

        current_price = snapshot.mid_price

        # For accumulation, look for dips
        if self.direction in [DCADirection.ACCUMULATE, DCADirection.BIDIRECTIONAL]:
            recent_high = self._get_recent_high()
            if recent_high and recent_high > 0:
                dip_pct = (recent_high - current_price) / recent_high
                if dip_pct >= self.dip_threshold:
                    return self._create_dca_signal(
                        snapshot,
                        reason="price_dip",
                        dip_pct=float(dip_pct),
                    )

        # For distribution, look for pumps
        if self.direction in [DCADirection.DISTRIBUTE, DCADirection.BIDIRECTIONAL]:
            recent_low = self._get_recent_low()
            if recent_low and recent_low > 0:
                pump_pct = (current_price - recent_low) / recent_low
                if pump_pct >= self.pump_threshold:
                    return self._create_dca_signal(
                        snapshot,
                        reason="price_pump",
                        pump_pct=float(pump_pct),
                        is_sell=True,
                    )

        return None

    def _check_hybrid_signal(self, snapshot: OrderBookSnapshot) -> Optional[Signal]:
        """Check for hybrid (time + price) DCA signal."""
        if not self.can_signal():
            return None

        current_time = snapshot.timestamp
        current_price = snapshot.mid_price

        # Check time condition
        time_ready = True
        if self._state.last_entry_time is not None:
            elapsed = current_time - self._state.last_entry_time
            time_ready = elapsed >= self._interval_delta

        if not time_ready:
            return None

        # Time is ready, now check if we should wait for better price
        # or execute now

        # Check for significant dip - execute immediately
        recent_high = self._get_recent_high()
        if recent_high and recent_high > 0:
            dip_pct = (recent_high - current_price) / recent_high
            if dip_pct >= self.dip_threshold:
                return self._create_dca_signal(
                    snapshot,
                    reason="hybrid_dip",
                    dip_pct=float(dip_pct),
                )

        # No significant dip, but time interval passed - execute anyway
        # (this is the "DCA regardless" behavior)
        return self._create_dca_signal(snapshot, reason="hybrid_time")

    def _create_dca_signal(
        self,
        snapshot: OrderBookSnapshot,
        reason: str,
        dip_pct: float = 0,
        pump_pct: float = 0,
        is_sell: bool = False,
    ) -> Optional[Signal]:
        """Create DCA entry signal."""
        current_price = snapshot.mid_price

        # Determine signal type
        if is_sell or self.direction == DCADirection.DISTRIBUTE:
            signal_type = SignalType.SHORT
        else:
            signal_type = SignalType.LONG

        # Calculate quantity (potentially scaled)
        quantity = self.quantity_per_entry
        if self.scale_on_dip and dip_pct > 0:
            # Increase size proportionally to dip
            scale = 1 + (Decimal(str(dip_pct)) / self.dip_threshold) * (self.scale_factor - 1)
            quantity = quantity * min(scale, self.scale_factor * 2)

        # Check budget
        entry_cost = quantity * current_price
        remaining_budget = self.total_budget - self._state.total_invested
        if entry_cost > remaining_budget:
            quantity = remaining_budget / current_price
            if quantity <= 0:
                return None

        # Calculate strength based on conditions
        strength = 0.5
        if dip_pct > 0:
            strength = min(0.5 + dip_pct * 5, 1.0)  # Bigger dip = stronger signal
        if pump_pct > 0:
            strength = min(0.5 + pump_pct * 5, 1.0)

        # Update state
        self._record_entry(current_price, quantity, snapshot.timestamp)

        return self.create_signal(
            signal_type=signal_type,
            symbol=snapshot.symbol,
            price=current_price,
            strength=strength,
            metadata={
                "dca_reason": reason,
                "entry_number": self._state.entries_count,
                "quantity": float(quantity),
                "dip_pct": dip_pct,
                "pump_pct": pump_pct,
                "average_price": float(self._state.average_price),
                "total_invested": float(self._state.total_invested),
                "total_quantity": float(self._state.total_quantity),
            },
        )

    def _record_entry(
        self,
        price: Decimal,
        quantity: Decimal,
        timestamp: datetime,
    ) -> None:
        """Record a DCA entry."""
        cost = price * quantity

        # Update state
        self._state.total_invested += cost
        self._state.total_quantity += quantity
        self._state.entries_count += 1
        self._state.last_entry_time = timestamp
        self._state.last_entry_price = price

        # Calculate new average price
        if self._state.total_quantity > 0:
            self._state.average_price = (
                self._state.total_invested / self._state.total_quantity
            )

        # Record entry
        self._entries.append(DCAEntry(
            timestamp=timestamp,
            price=price,
            quantity=quantity,
            cumulative_quantity=self._state.total_quantity,
            cumulative_invested=self._state.total_invested,
            average_price=self._state.average_price,
        ))

        logger.debug(
            f"[{self.name}] DCA entry #{self._state.entries_count}: "
            f"price={price:.2f}, qty={quantity:.6f}, "
            f"avg={self._state.average_price:.2f}"
        )

    def _check_exit_conditions(self, snapshot: OrderBookSnapshot) -> Optional[Signal]:
        """Check if position should be closed (TP/SL)."""
        if self._state.total_quantity <= 0:
            return None

        if self._state.average_price <= 0:
            return None

        current_price = snapshot.mid_price
        pnl_pct = (current_price - self._state.average_price) / self._state.average_price

        # Take profit
        if self.direction == DCADirection.ACCUMULATE:
            if pnl_pct >= self.take_profit_pct:
                return self._create_exit_signal(
                    snapshot,
                    reason="take_profit",
                    pnl_pct=float(pnl_pct),
                )

            # Stop loss
            if pnl_pct <= -self.stop_loss_pct:
                return self._create_exit_signal(
                    snapshot,
                    reason="stop_loss",
                    pnl_pct=float(pnl_pct),
                )

        return None

    def _create_exit_signal(
        self,
        snapshot: OrderBookSnapshot,
        reason: str,
        pnl_pct: float,
    ) -> Optional[Signal]:
        """Create exit signal to close DCA position."""
        # Exit is opposite of accumulation
        signal_type = SignalType.SHORT if self.direction == DCADirection.ACCUMULATE else SignalType.LONG

        return self.create_signal(
            signal_type=signal_type,
            symbol=snapshot.symbol,
            price=snapshot.mid_price,
            strength=0.9,  # Exit signals are strong
            metadata={
                "dca_exit": True,
                "exit_reason": reason,
                "pnl_pct": pnl_pct,
                "average_price": float(self._state.average_price),
                "total_quantity": float(self._state.total_quantity),
                "entries_count": self._state.entries_count,
            },
        )

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def get_dca_stats(self) -> Dict[str, Any]:
        """Get DCA statistics."""
        return {
            "mode": self.mode.value,
            "direction": self.direction.value,
            "entries_count": self._state.entries_count,
            "max_entries": self.max_entries,
            "total_invested": float(self._state.total_invested),
            "total_budget": float(self.total_budget),
            "budget_used_pct": float(self._state.total_invested / self.total_budget * 100)
            if self.total_budget > 0 else 0,
            "total_quantity": float(self._state.total_quantity),
            "average_price": float(self._state.average_price),
            "last_entry_time": self._state.last_entry_time,
            "last_entry_price": float(self._state.last_entry_price)
            if self._state.last_entry_price else None,
        }

    def get_entries(self) -> List[Dict[str, Any]]:
        """Get list of all DCA entries."""
        return [
            {
                "timestamp": e.timestamp,
                "price": float(e.price),
                "quantity": float(e.quantity),
                "cumulative_quantity": float(e.cumulative_quantity),
                "cumulative_invested": float(e.cumulative_invested),
                "average_price": float(e.average_price),
            }
            for e in self._entries
        ]

    def get_unrealized_pnl(self, current_price: Decimal) -> Dict[str, Any]:
        """Calculate unrealized P&L."""
        if self._state.total_quantity <= 0 or self._state.average_price <= 0:
            return {"pnl": 0, "pnl_pct": 0}

        pnl = (current_price - self._state.average_price) * self._state.total_quantity
        pnl_pct = (current_price - self._state.average_price) / self._state.average_price

        return {
            "pnl": float(pnl),
            "pnl_pct": float(pnl_pct * 100),
            "current_price": float(current_price),
            "average_price": float(self._state.average_price),
            "quantity": float(self._state.total_quantity),
        }

    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self._state = DCAState()
        self._price_history.clear()
        self._entries.clear()
        self._reference_price = None
        self._safety_orders.clear()
        self._initial_entry_price = None
        logger.info(f"[{self.name}] State reset")

    # =========================================================================
    # Safety Orders
    # =========================================================================

    def calculate_safety_orders(self, entry_price: Decimal) -> List[SafetyOrder]:
        """
        Calculate safety order levels based on entry price.

        Safety orders are placed at progressively lower prices with increasing sizes.

        Args:
            entry_price: Initial entry price

        Returns:
            List of SafetyOrder objects with calculated prices
        """
        safety_orders = []
        current_step = self.safety_order_step_pct
        current_size = self.safety_order_size_scale

        for level in range(1, self.safety_orders_count + 1):
            # Calculate price deviation from entry
            total_deviation = current_step
            for _ in range(level - 1):
                current_step *= self.safety_order_step_scale

            # Calculate safety order price (below entry for accumulation)
            if self.direction == DCADirection.ACCUMULATE:
                price = entry_price * (1 - total_deviation)
            else:
                price = entry_price * (1 + total_deviation)

            # Calculate size multiplier
            size_mult = Decimal("1")
            for _ in range(level):
                size_mult *= self.safety_order_size_scale

            safety_orders.append(SafetyOrder(
                level=level,
                price_deviation_pct=total_deviation,
                size_multiplier=size_mult,
                price=price,
                triggered=False,
            ))

            # Update step for next level
            current_step = self.safety_order_step_pct * (self.safety_order_step_scale ** level)

        return safety_orders

    def get_safety_order_size(self, level: int) -> Decimal:
        """
        Calculate size for a specific safety order level.

        Args:
            level: Safety order level (1-based)

        Returns:
            Order size in base currency
        """
        base_size = self.quantity_per_entry
        multiplier = self.safety_order_size_scale ** level
        return base_size * multiplier

    def check_safety_order_triggers(
        self,
        current_price: Decimal,
        timestamp: datetime,
    ) -> Optional[SafetyOrder]:
        """
        Check if any safety orders should be triggered.

        Args:
            current_price: Current market price
            timestamp: Current timestamp

        Returns:
            Triggered SafetyOrder or None
        """
        if not self.safety_orders_enabled or not self._safety_orders:
            return None

        for so in self._safety_orders:
            if so.triggered:
                continue

            # Check if price has reached safety order level
            if self.direction == DCADirection.ACCUMULATE:
                if current_price <= so.price:
                    so.triggered = True
                    so.trigger_time = timestamp
                    logger.info(
                        f"[{self.name}] Safety order #{so.level} triggered at {current_price}"
                    )
                    return so
            else:
                if current_price >= so.price:
                    so.triggered = True
                    so.trigger_time = timestamp
                    logger.info(
                        f"[{self.name}] Safety order #{so.level} triggered at {current_price}"
                    )
                    return so

        return None

    def get_safety_orders_status(self) -> List[Dict[str, Any]]:
        """Get status of all safety orders."""
        return [
            {
                "level": so.level,
                "price": float(so.price) if so.price else None,
                "price_deviation_pct": float(so.price_deviation_pct * 100),
                "size_multiplier": float(so.size_multiplier),
                "triggered": so.triggered,
                "trigger_time": so.trigger_time,
            }
            for so in self._safety_orders
        ]

    # =========================================================================
    # Take Profit Price Calculation
    # =========================================================================

    def calculate_take_profit_price(self, entry_price: Optional[Decimal] = None) -> Decimal:
        """
        Calculate take profit price based on average entry or specified price.

        Args:
            entry_price: Optional specific entry price, uses average if not provided

        Returns:
            Take profit price
        """
        base_price = entry_price or self._state.average_price
        if base_price <= 0:
            return Decimal("0")

        if self.direction == DCADirection.ACCUMULATE:
            # Long position - TP is above entry
            return base_price * (1 + self.take_profit_pct)
        else:
            # Short position - TP is below entry
            return base_price * (1 - self.take_profit_pct)

    def calculate_stop_loss_price(self, entry_price: Optional[Decimal] = None) -> Decimal:
        """
        Calculate stop loss price based on average entry or specified price.

        Args:
            entry_price: Optional specific entry price, uses average if not provided

        Returns:
            Stop loss price
        """
        base_price = entry_price or self._state.average_price
        if base_price <= 0:
            return Decimal("0")

        if self.direction == DCADirection.ACCUMULATE:
            # Long position - SL is below entry
            return base_price * (1 - self.stop_loss_pct)
        else:
            # Short position - SL is above entry
            return base_price * (1 + self.stop_loss_pct)

    def get_tp_sl_prices(self) -> Dict[str, Any]:
        """
        Get take profit and stop loss prices.

        Returns:
            Dict with TP and SL prices and percentages
        """
        tp_price = self.calculate_take_profit_price()
        sl_price = self.calculate_stop_loss_price()

        return {
            "take_profit_price": float(tp_price),
            "stop_loss_price": float(sl_price),
            "take_profit_pct": float(self.take_profit_pct * 100),
            "stop_loss_pct": float(self.stop_loss_pct * 100),
            "average_entry_price": float(self._state.average_price),
        }

    # =========================================================================
    # Trailing Take Profit
    # =========================================================================

    def update_trailing_take_profit(self, current_price: Decimal) -> Optional[Signal]:
        """
        Update trailing take profit state and check for trigger.

        Trailing TP activates when price reaches activation threshold,
        then follows price up. Triggers when price drops by deviation %.

        Args:
            current_price: Current market price

        Returns:
            Exit signal if trailing TP triggered, None otherwise
        """
        if not self.trailing_tp_enabled:
            return None

        if self._state.total_quantity <= 0 or self._state.average_price <= 0:
            return None

        pnl_pct = (current_price - self._state.average_price) / self._state.average_price

        # Check if trailing TP should be activated
        if not self._state.trailing_tp_active:
            if self.direction == DCADirection.ACCUMULATE:
                if pnl_pct >= self.trailing_tp_activation_pct:
                    self._state.trailing_tp_active = True
                    self._state.trailing_tp_highest = current_price
                    self._state.trailing_tp_trigger_price = current_price * (1 - self.trailing_tp_deviation_pct)
                    logger.info(
                        f"[{self.name}] Trailing TP activated at {current_price}, "
                        f"trigger at {self._state.trailing_tp_trigger_price}"
                    )
            else:
                # For distribution, activate when price drops
                if pnl_pct <= -self.trailing_tp_activation_pct:
                    self._state.trailing_tp_active = True
                    self._state.trailing_tp_highest = current_price
                    self._state.trailing_tp_trigger_price = current_price * (1 + self.trailing_tp_deviation_pct)
                    logger.info(f"[{self.name}] Trailing TP activated at {current_price}")
            return None

        # Trailing TP is active - update or trigger
        if self.direction == DCADirection.ACCUMULATE:
            # Update highest price and trigger level
            if current_price > self._state.trailing_tp_highest:
                self._state.trailing_tp_highest = current_price
                self._state.trailing_tp_trigger_price = current_price * (1 - self.trailing_tp_deviation_pct)
                logger.debug(
                    f"[{self.name}] Trailing TP updated: highest={current_price}, "
                    f"trigger={self._state.trailing_tp_trigger_price}"
                )

            # Check if trigger hit
            if current_price <= self._state.trailing_tp_trigger_price:
                pnl_pct_final = (current_price - self._state.average_price) / self._state.average_price
                logger.info(
                    f"[{self.name}] Trailing TP triggered at {current_price}, "
                    f"PnL: {float(pnl_pct_final)*100:.2f}%"
                )
                return True  # Signal to exit
        else:
            # For distribution (shorts)
            if current_price < self._state.trailing_tp_highest:
                self._state.trailing_tp_highest = current_price
                self._state.trailing_tp_trigger_price = current_price * (1 + self.trailing_tp_deviation_pct)

            if current_price >= self._state.trailing_tp_trigger_price:
                return True

        return None

    def get_trailing_tp_status(self) -> Dict[str, Any]:
        """Get trailing take profit status."""
        return {
            "enabled": self.trailing_tp_enabled,
            "active": self._state.trailing_tp_active,
            "activation_pct": float(self.trailing_tp_activation_pct * 100),
            "deviation_pct": float(self.trailing_tp_deviation_pct * 100),
            "highest_price": float(self._state.trailing_tp_highest),
            "trigger_price": float(self._state.trailing_tp_trigger_price)
            if self._state.trailing_tp_trigger_price else None,
        }
