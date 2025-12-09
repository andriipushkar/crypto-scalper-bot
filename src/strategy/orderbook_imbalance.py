"""
Order Book Imbalance Strategy.

Generates signals based on bid/ask volume imbalance in the order book.

Logic:
- When bid volume >> ask volume -> bullish pressure -> LONG signal
- When ask volume >> bid volume -> bearish pressure -> SHORT signal
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, Deque

from loguru import logger

from src.data.models import OrderBookSnapshot, Signal, SignalType
from src.strategy.base import BaseStrategy


@dataclass
class ImbalanceSnapshot:
    """Snapshot of order book imbalance."""
    timestamp: datetime
    imbalance: Decimal
    mid_price: Decimal
    spread_bps: Decimal
    bid_volume: Decimal
    ask_volume: Decimal


class OrderBookImbalanceStrategy(BaseStrategy):
    """
    Strategy based on order book imbalance.

    Parameters:
    - imbalance_threshold: Ratio threshold for signal (e.g., 1.5 = 50% more bids)
    - max_spread: Maximum spread in decimal (e.g., 0.0005 = 0.05%)
    - levels: Number of order book levels to analyze
    - min_volume_usd: Minimum combined volume in USD
    - lookback: Number of snapshots to consider for trend
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy.

        Args:
            config: Strategy configuration
        """
        super().__init__(config)

        # Strategy-specific parameters
        self.imbalance_threshold = Decimal(str(config.get("imbalance_threshold", 1.5)))
        self.max_spread = Decimal(str(config.get("max_spread", 0.0005)))  # 0.05%
        self.levels = config.get("levels", 5)
        self.min_volume_usd = Decimal(str(config.get("min_volume_usd", 5000)))
        self.lookback = config.get("lookback", 10)

        # History for trend analysis
        self._history: Deque[ImbalanceSnapshot] = deque(maxlen=self.lookback)

        logger.info(
            f"[{self.name}] Config: threshold={self.imbalance_threshold}, "
            f"max_spread={float(self.max_spread)*100:.3f}%, levels={self.levels}"
        )

    def on_orderbook(self, snapshot: OrderBookSnapshot) -> Optional[Signal]:
        """
        Process order book update and generate signal if conditions met.

        Args:
            snapshot: Current order book snapshot

        Returns:
            Signal if conditions are met, None otherwise
        """
        # Basic validation
        if not snapshot.best_bid or not snapshot.best_ask:
            return None

        # Calculate metrics
        imbalance = snapshot.imbalance(self.levels)
        spread_bps = snapshot.spread_bps or Decimal("0")
        spread_pct = spread_bps / 10000  # Convert bps to decimal
        mid_price = snapshot.mid_price

        bid_volume = snapshot.bid_volume(self.levels)
        ask_volume = snapshot.ask_volume(self.levels)
        total_volume_usd = (bid_volume + ask_volume) * mid_price

        # Store in history
        self._history.append(ImbalanceSnapshot(
            timestamp=snapshot.timestamp,
            imbalance=imbalance,
            mid_price=mid_price,
            spread_bps=spread_bps,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
        ))

        # Check basic conditions
        if not self.can_signal():
            return None

        # Check spread (don't trade if spread too wide)
        if spread_pct > self.max_spread:
            return None

        # Check minimum volume
        if total_volume_usd < self.min_volume_usd:
            return None

        # Generate signal based on imbalance
        signal = self._evaluate_imbalance(snapshot, imbalance)

        return signal

    def _evaluate_imbalance(
        self,
        snapshot: OrderBookSnapshot,
        imbalance: Decimal,
    ) -> Optional[Signal]:
        """
        Evaluate imbalance and generate signal.

        Args:
            snapshot: Order book snapshot
            imbalance: Current imbalance ratio

        Returns:
            Signal if threshold met, None otherwise
        """
        # Calculate signal strength based on how much imbalance exceeds threshold
        # strength = 0.5 at threshold, 1.0 at 2x threshold
        if imbalance >= self.imbalance_threshold:
            # Bullish: more bids than asks
            excess = float(imbalance - self.imbalance_threshold)
            threshold_float = float(self.imbalance_threshold)
            strength = min(0.5 + (excess / threshold_float) * 0.5, 1.0)

            # Check trend confirmation
            if self._confirm_trend(bullish=True):
                strength = min(strength + 0.1, 1.0)

            return self.create_signal(
                signal_type=SignalType.LONG,
                symbol=snapshot.symbol,
                price=snapshot.mid_price,
                strength=strength,
                metadata={
                    "imbalance": float(imbalance),
                    "bid_volume": float(snapshot.bid_volume(self.levels)),
                    "ask_volume": float(snapshot.ask_volume(self.levels)),
                    "spread_bps": float(snapshot.spread_bps),
                    "trend_confirmed": self._confirm_trend(bullish=True),
                },
            )

        elif imbalance <= 1 / self.imbalance_threshold:
            # Bearish: more asks than bids
            inverse_imbalance = 1 / imbalance if imbalance > 0 else Decimal("999")
            excess = float(inverse_imbalance - self.imbalance_threshold)
            threshold_float = float(self.imbalance_threshold)
            strength = min(0.5 + (excess / threshold_float) * 0.5, 1.0)

            # Check trend confirmation
            if self._confirm_trend(bullish=False):
                strength = min(strength + 0.1, 1.0)

            return self.create_signal(
                signal_type=SignalType.SHORT,
                symbol=snapshot.symbol,
                price=snapshot.mid_price,
                strength=strength,
                metadata={
                    "imbalance": float(imbalance),
                    "bid_volume": float(snapshot.bid_volume(self.levels)),
                    "ask_volume": float(snapshot.ask_volume(self.levels)),
                    "spread_bps": float(snapshot.spread_bps),
                    "trend_confirmed": self._confirm_trend(bullish=False),
                },
            )

        return None

    def _confirm_trend(self, bullish: bool) -> bool:
        """
        Check if recent history confirms the trend.

        Args:
            bullish: True for bullish trend, False for bearish

        Returns:
            True if trend is confirmed
        """
        if len(self._history) < 3:
            return False

        recent = list(self._history)[-3:]

        if bullish:
            # Check if imbalance has been consistently bullish
            return all(h.imbalance > 1 for h in recent)
        else:
            # Check if imbalance has been consistently bearish
            return all(h.imbalance < 1 for h in recent)

    def get_current_imbalance(self) -> Optional[Decimal]:
        """Get the most recent imbalance value."""
        if self._history:
            return self._history[-1].imbalance
        return None

    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self._history.clear()
