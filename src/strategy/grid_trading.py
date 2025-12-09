"""
Grid Trading Strategy.

Creates a grid of buy and sell orders at predefined price levels.
Profits from price oscillation within a range (sideways market).

Logic:
- Define upper and lower price bounds
- Create evenly spaced grid levels
- Buy at lower levels, sell at higher levels
- Each filled buy order triggers a sell order at the next level up
- Each filled sell order triggers a buy order at the next level down
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List, Deque, Set

from loguru import logger

from src.data.models import OrderBookSnapshot, Trade, Signal, SignalType
from src.strategy.base import BaseStrategy


class GridDirection(Enum):
    """Grid trading direction."""
    NEUTRAL = "neutral"  # Both buy and sell
    LONG = "long"  # Only buy orders (expecting price to rise)
    SHORT = "short"  # Only sell orders (expecting price to fall)


@dataclass
class GridLevel:
    """Single grid level."""
    price: Decimal
    is_buy: bool  # True = buy level, False = sell level
    filled: bool = False
    order_id: Optional[str] = None


@dataclass
class GridState:
    """Grid trading state."""
    levels: List[GridLevel] = field(default_factory=list)
    active_position_size: Decimal = Decimal("0")
    total_profit: Decimal = Decimal("0")
    trades_count: int = 0
    last_fill_price: Optional[Decimal] = None
    last_fill_time: Optional[datetime] = None


class GridTradingStrategy(BaseStrategy):
    """
    Grid Trading Strategy.

    Creates a grid of orders and profits from price oscillation.
    Best suited for ranging/sideways markets.

    Parameters:
    - upper_price: Upper boundary of the grid
    - lower_price: Lower boundary of the grid
    - grid_levels: Number of grid levels
    - quantity_per_grid: Position size per grid level
    - direction: Grid direction (neutral, long, short)
    - take_profit_grids: Number of grids for take profit (default: 1)
    - trailing_up: Enable trailing grid upward (default: False)
    - trailing_down: Enable trailing grid downward (default: False)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize grid trading strategy.

        Args:
            config: Strategy configuration
        """
        super().__init__(config)

        # Grid parameters
        self.upper_price = Decimal(str(config.get("upper_price", 0)))
        self.lower_price = Decimal(str(config.get("lower_price", 0)))
        self.grid_levels = config.get("grid_levels", 10)
        self.quantity_per_grid = Decimal(str(config.get("quantity_per_grid", 0.001)))

        # Direction
        direction_str = config.get("direction", "neutral").lower()
        self.direction = GridDirection(direction_str)

        # Additional settings
        self.take_profit_grids = config.get("take_profit_grids", 1)
        self.trailing_up = config.get("trailing_up", False)
        self.trailing_down = config.get("trailing_down", False)

        # Dynamic grid (auto-adjust based on price)
        self.auto_range = config.get("auto_range", False)
        self.range_percent = Decimal(str(config.get("range_percent", 0.05)))  # 5%

        # State
        self._grid_state = GridState()
        self._initialized = False
        self._price_history: Deque[Decimal] = deque(maxlen=100)

        # Validate config
        if not self.auto_range and (self.upper_price <= 0 or self.lower_price <= 0):
            logger.warning(
                f"[{self.name}] Invalid price range. Enable auto_range or set upper/lower prices."
            )

        logger.info(
            f"[{self.name}] Config: levels={self.grid_levels}, "
            f"qty={self.quantity_per_grid}, direction={self.direction.value}"
        )

    def on_orderbook(self, snapshot: OrderBookSnapshot) -> Optional[Signal]:
        """
        Process order book update and manage grid.

        Args:
            snapshot: Current order book snapshot

        Returns:
            Signal if grid action needed, None otherwise
        """
        if not snapshot.best_bid or not snapshot.best_ask:
            return None

        mid_price = snapshot.mid_price
        self._price_history.append(mid_price)

        # Initialize grid on first update
        if not self._initialized:
            self._initialize_grid(mid_price)
            self._initialized = True

        # Check if price is outside grid range
        if self._should_reset_grid(mid_price):
            self._reset_grid(mid_price)

        # Check for grid level triggers
        signal = self._check_grid_signals(snapshot)

        return signal

    def on_trade(self, trade: Trade) -> Optional[Signal]:
        """
        Process trade and check for grid fills.

        Args:
            trade: Trade event

        Returns:
            Signal if grid action needed
        """
        # Update price history
        self._price_history.append(trade.price)

        # Check if any grid level was triggered
        return self._check_trade_fills(trade)

    def _initialize_grid(self, current_price: Decimal) -> None:
        """
        Initialize grid levels around current price.

        Args:
            current_price: Current market price
        """
        if self.auto_range:
            # Auto-calculate range based on current price
            self.upper_price = current_price * (1 + self.range_percent)
            self.lower_price = current_price * (1 - self.range_percent)

        if self.upper_price <= self.lower_price:
            logger.error(f"[{self.name}] Invalid grid range: {self.lower_price} - {self.upper_price}")
            return

        # Calculate grid step
        grid_step = (self.upper_price - self.lower_price) / self.grid_levels

        # Create grid levels
        levels = []
        for i in range(self.grid_levels + 1):
            price = self.lower_price + (grid_step * i)

            # Determine if this is a buy or sell level
            if self.direction == GridDirection.NEUTRAL:
                # Below current price = buy, above = sell
                is_buy = price < current_price
            elif self.direction == GridDirection.LONG:
                is_buy = True
            else:  # SHORT
                is_buy = False

            levels.append(GridLevel(price=price, is_buy=is_buy))

        self._grid_state.levels = levels

        logger.info(
            f"[{self.name}] Grid initialized: {len(levels)} levels from "
            f"{self.lower_price:.2f} to {self.upper_price:.2f}"
        )

    def _should_reset_grid(self, current_price: Decimal) -> bool:
        """Check if grid should be reset (price outside range)."""
        if not self._grid_state.levels:
            return True

        # Check if price is significantly outside range
        buffer = (self.upper_price - self.lower_price) * Decimal("0.1")

        if current_price > self.upper_price + buffer:
            return self.trailing_up
        if current_price < self.lower_price - buffer:
            return self.trailing_down

        return False

    def _reset_grid(self, current_price: Decimal) -> None:
        """Reset grid around new price level."""
        logger.info(f"[{self.name}] Resetting grid around {current_price:.2f}")

        # Clear existing grid
        self._grid_state = GridState()

        # Reinitialize
        self._initialize_grid(current_price)

    def _check_grid_signals(self, snapshot: OrderBookSnapshot) -> Optional[Signal]:
        """
        Check if any grid level should trigger a signal.

        Args:
            snapshot: Order book snapshot

        Returns:
            Signal if action needed
        """
        if not self.can_signal():
            return None

        mid_price = snapshot.mid_price
        best_bid = snapshot.best_bid
        best_ask = snapshot.best_ask

        # Find nearest unfilled levels
        nearest_buy = None
        nearest_sell = None

        for level in self._grid_state.levels:
            if level.filled:
                continue

            if level.is_buy and best_bid and level.price >= best_bid.price:
                if nearest_buy is None or level.price > nearest_buy.price:
                    nearest_buy = level
            elif not level.is_buy and best_ask and level.price <= best_ask.price:
                if nearest_sell is None or level.price < nearest_sell.price:
                    nearest_sell = level

        # Decide which signal to generate
        if nearest_buy and self._should_execute_buy(nearest_buy, mid_price):
            return self._create_grid_signal(
                signal_type=SignalType.LONG,
                snapshot=snapshot,
                grid_level=nearest_buy,
            )

        if nearest_sell and self._should_execute_sell(nearest_sell, mid_price):
            return self._create_grid_signal(
                signal_type=SignalType.SHORT,
                snapshot=snapshot,
                grid_level=nearest_sell,
            )

        return None

    def _should_execute_buy(self, level: GridLevel, current_price: Decimal) -> bool:
        """Check if buy level should execute."""
        # Price should be at or below the grid level
        return current_price <= level.price * Decimal("1.001")  # 0.1% tolerance

    def _should_execute_sell(self, level: GridLevel, current_price: Decimal) -> bool:
        """Check if sell level should execute."""
        # Price should be at or above the grid level
        return current_price >= level.price * Decimal("0.999")  # 0.1% tolerance

    def _create_grid_signal(
        self,
        signal_type: SignalType,
        snapshot: OrderBookSnapshot,
        grid_level: GridLevel,
    ) -> Optional[Signal]:
        """Create signal for grid execution."""
        # Calculate strength based on grid position
        total_levels = len(self._grid_state.levels)
        level_index = self._grid_state.levels.index(grid_level)

        if signal_type == SignalType.LONG:
            # Lower levels are stronger buys
            strength = 1.0 - (level_index / total_levels)
        else:
            # Higher levels are stronger sells
            strength = level_index / total_levels

        strength = max(0.5, min(1.0, strength))

        # Mark level as pending
        grid_level.filled = True

        return self.create_signal(
            signal_type=signal_type,
            symbol=snapshot.symbol,
            price=grid_level.price,
            strength=strength,
            metadata={
                "grid_level": float(grid_level.price),
                "grid_index": level_index,
                "total_levels": total_levels,
                "quantity": float(self.quantity_per_grid),
                "direction": self.direction.value,
                "active_position": float(self._grid_state.active_position_size),
            },
        )

    def _check_trade_fills(self, trade: Trade) -> Optional[Signal]:
        """
        Check if a trade filled any grid level.

        Args:
            trade: Executed trade

        Returns:
            Signal for counter-order if needed
        """
        # Find the level that was just filled
        for level in self._grid_state.levels:
            if not level.filled:
                continue

            # Check if this trade is near the level price
            price_diff = abs(trade.price - level.price) / level.price
            if price_diff < Decimal("0.002"):  # 0.2% tolerance
                # Update state
                if level.is_buy:
                    self._grid_state.active_position_size += self.quantity_per_grid
                else:
                    self._grid_state.active_position_size -= self.quantity_per_grid

                self._grid_state.trades_count += 1
                self._grid_state.last_fill_price = trade.price
                self._grid_state.last_fill_time = trade.timestamp

                # Flip the level for the counter-order
                level.is_buy = not level.is_buy
                level.filled = False

                logger.debug(
                    f"[{self.name}] Grid fill at {trade.price:.2f}, "
                    f"position: {self._grid_state.active_position_size}"
                )
                break

        return None

    # =========================================================================
    # Grid Management
    # =========================================================================

    def get_grid_levels(self) -> List[Dict[str, Any]]:
        """Get current grid levels."""
        return [
            {
                "price": float(level.price),
                "is_buy": level.is_buy,
                "filled": level.filled,
            }
            for level in self._grid_state.levels
        ]

    def get_unfilled_levels(self) -> Dict[str, List[Decimal]]:
        """Get unfilled buy and sell levels."""
        buy_levels = []
        sell_levels = []

        for level in self._grid_state.levels:
            if level.filled:
                continue
            if level.is_buy:
                buy_levels.append(level.price)
            else:
                sell_levels.append(level.price)

        return {"buy": buy_levels, "sell": sell_levels}

    def get_grid_stats(self) -> Dict[str, Any]:
        """Get grid statistics."""
        return {
            "total_levels": len(self._grid_state.levels),
            "filled_levels": sum(1 for l in self._grid_state.levels if l.filled),
            "active_position": float(self._grid_state.active_position_size),
            "total_profit": float(self._grid_state.total_profit),
            "trades_count": self._grid_state.trades_count,
            "upper_price": float(self.upper_price),
            "lower_price": float(self.lower_price),
            "direction": self.direction.value,
        }

    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self._grid_state = GridState()
        self._initialized = False
        self._price_history.clear()
        logger.info(f"[{self.name}] Grid reset")
