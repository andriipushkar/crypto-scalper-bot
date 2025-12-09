"""
Volume Spike Strategy.

Generates signals based on unusual volume activity.

Logic:
- Detect volume spikes (X times average)
- Determine direction from trade flow (buy vs sell volume)
- Generate signal in direction of dominant flow
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, Deque, List

from loguru import logger

from src.data.models import (
    OrderBookSnapshot,
    Trade,
    Signal,
    SignalType,
    Side,
)
from src.strategy.base import BaseStrategy


@dataclass
class VolumeBar:
    """Aggregated volume for a time period."""
    timestamp: datetime
    buy_volume: Decimal = Decimal("0")
    sell_volume: Decimal = Decimal("0")
    buy_value: Decimal = Decimal("0")
    sell_value: Decimal = Decimal("0")
    trade_count: int = 0
    vwap: Decimal = Decimal("0")

    @property
    def total_volume(self) -> Decimal:
        return self.buy_volume + self.sell_volume

    @property
    def total_value(self) -> Decimal:
        return self.buy_value + self.sell_value

    @property
    def net_flow(self) -> Decimal:
        """Net volume flow (positive = buying pressure)."""
        return self.buy_value - self.sell_value

    @property
    def flow_ratio(self) -> Decimal:
        """Buy/sell ratio."""
        if self.sell_value == 0:
            return Decimal("999")
        return self.buy_value / self.sell_value


class VolumeSpikeStrategy(BaseStrategy):
    """
    Strategy based on volume spikes.

    Detects unusual volume activity and generates signals
    based on the direction of the volume flow.

    Parameters:
    - volume_multiplier: Volume must be X times average to trigger
    - lookback_seconds: Period for calculating average volume
    - min_volume_usd: Minimum spike volume in USD
    - flow_threshold: Min buy/sell ratio for directional signal
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy.

        Args:
            config: Strategy configuration
        """
        super().__init__(config)

        # Strategy-specific parameters
        self.volume_multiplier = config.get("volume_multiplier", 3.0)
        self.lookback_seconds = config.get("lookback_seconds", 60)
        self.min_volume_usd = Decimal(str(config.get("min_volume_usd", 10000)))
        self.flow_threshold = Decimal(str(config.get("flow_threshold", 1.5)))

        # Volume history (1-second bars)
        self._volume_history: Deque[VolumeBar] = deque(maxlen=self.lookback_seconds)
        self._current_bar: Optional[VolumeBar] = None
        self._last_bar_time: Optional[datetime] = None

        # Last known price for order book signals
        self._last_mid_price: Optional[Decimal] = None
        self._last_symbol: str = "BTCUSDT"

        logger.info(
            f"[{self.name}] Config: multiplier={self.volume_multiplier}x, "
            f"lookback={self.lookback_seconds}s, min_volume=${self.min_volume_usd}"
        )

    def on_orderbook(self, snapshot: OrderBookSnapshot) -> Optional[Signal]:
        """
        Update price reference and check for volume spike signals.

        Args:
            snapshot: Current order book snapshot

        Returns:
            Signal if conditions are met, None otherwise
        """
        # Update price reference
        self._last_mid_price = snapshot.mid_price
        self._last_symbol = snapshot.symbol

        # Check for volume spike (based on accumulated trade data)
        return self._check_volume_spike()

    def on_trade(self, trade: Trade) -> Optional[Signal]:
        """
        Process trade and accumulate volume.

        Args:
            trade: Trade event

        Returns:
            Signal if spike detected, None otherwise
        """
        # Get or create current bar
        current_time = trade.timestamp.replace(microsecond=0)

        if self._current_bar is None or current_time != self._last_bar_time:
            # Save previous bar
            if self._current_bar is not None:
                self._volume_history.append(self._current_bar)

            # Create new bar
            self._current_bar = VolumeBar(timestamp=current_time)
            self._last_bar_time = current_time

        # Accumulate trade
        value = trade.value
        quantity = trade.quantity

        if trade.side == Side.BUY:
            self._current_bar.buy_volume += quantity
            self._current_bar.buy_value += value
        else:
            self._current_bar.sell_volume += quantity
            self._current_bar.sell_value += value

        self._current_bar.trade_count += 1

        # Update VWAP
        total_value = self._current_bar.total_value
        total_volume = self._current_bar.total_volume
        if total_volume > 0:
            self._current_bar.vwap = total_value / total_volume

        # Update price reference
        self._last_mid_price = trade.price
        self._last_symbol = trade.symbol

        # Check for spike
        return self._check_volume_spike()

    def _check_volume_spike(self) -> Optional[Signal]:
        """
        Check if current volume constitutes a spike.

        Returns:
            Signal if spike detected, None otherwise
        """
        if not self.can_signal():
            return None

        if self._current_bar is None:
            return None

        if self._last_mid_price is None:
            return None

        # Need some history for average
        if len(self._volume_history) < 10:
            return None

        # Calculate average volume
        avg_value = sum(bar.total_value for bar in self._volume_history) / len(self._volume_history)

        if avg_value == 0:
            return None

        # Current bar volume
        current_value = self._current_bar.total_value

        # Check for spike
        volume_ratio = float(current_value / avg_value)

        if volume_ratio < self.volume_multiplier:
            return None

        # Check minimum volume
        if current_value < self.min_volume_usd:
            return None

        # Determine direction from flow
        flow_ratio = self._current_bar.flow_ratio

        # Calculate signal strength based on spike magnitude
        # strength = 0.5 at threshold, 1.0 at 2x threshold
        excess = volume_ratio - self.volume_multiplier
        strength = min(0.5 + (excess / self.volume_multiplier) * 0.5, 1.0)

        # Adjust strength based on flow clarity
        flow_clarity = abs(float(flow_ratio) - 1.0)  # How one-sided is the flow
        strength = min(strength * (1 + flow_clarity * 0.2), 1.0)

        # Generate signal based on flow direction
        if flow_ratio >= self.flow_threshold:
            # Strong buying pressure
            return self.create_signal(
                signal_type=SignalType.LONG,
                symbol=self._last_symbol,
                price=self._last_mid_price,
                strength=strength,
                metadata={
                    "volume_ratio": volume_ratio,
                    "flow_ratio": float(flow_ratio),
                    "spike_value_usd": float(current_value),
                    "avg_value_usd": float(avg_value),
                    "buy_value": float(self._current_bar.buy_value),
                    "sell_value": float(self._current_bar.sell_value),
                    "trade_count": self._current_bar.trade_count,
                },
            )

        elif flow_ratio <= 1 / self.flow_threshold:
            # Strong selling pressure
            return self.create_signal(
                signal_type=SignalType.SHORT,
                symbol=self._last_symbol,
                price=self._last_mid_price,
                strength=strength,
                metadata={
                    "volume_ratio": volume_ratio,
                    "flow_ratio": float(flow_ratio),
                    "spike_value_usd": float(current_value),
                    "avg_value_usd": float(avg_value),
                    "buy_value": float(self._current_bar.buy_value),
                    "sell_value": float(self._current_bar.sell_value),
                    "trade_count": self._current_bar.trade_count,
                },
            )

        return None

    def get_average_volume(self) -> Optional[Decimal]:
        """Get average volume from history."""
        if not self._volume_history:
            return None
        return sum(bar.total_value for bar in self._volume_history) / len(self._volume_history)

    def get_current_volume(self) -> Optional[Decimal]:
        """Get current bar volume."""
        if self._current_bar:
            return self._current_bar.total_value
        return None

    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self._volume_history.clear()
        self._current_bar = None
        self._last_bar_time = None
