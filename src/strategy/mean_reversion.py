"""
Mean Reversion Strategy.

Trades on the assumption that prices tend to revert to their mean.
Uses statistical indicators to identify overbought/oversold conditions.

Logic:
- Calculate moving average and standard deviation (Bollinger Bands style)
- When price deviates significantly below mean -> LONG (expect reversion up)
- When price deviates significantly above mean -> SHORT (expect reversion down)
- Use RSI and Z-score for confirmation
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List, Deque
import statistics

from loguru import logger

from src.data.models import OrderBookSnapshot, Trade, Signal, SignalType
from src.strategy.base import BaseStrategy


@dataclass
class PricePoint:
    """Price data point with timestamp."""
    timestamp: datetime
    price: Decimal
    volume: Decimal = Decimal("0")


@dataclass
class MeanReversionMetrics:
    """Calculated metrics for mean reversion."""
    mean: Decimal
    std_dev: Decimal
    z_score: Decimal
    upper_band: Decimal
    lower_band: Decimal
    rsi: Optional[Decimal] = None
    deviation_percent: Decimal = Decimal("0")


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy.

    Identifies overbought/oversold conditions and trades on reversion to mean.

    Parameters:
    - lookback_period: Number of price points for mean calculation (default: 20)
    - std_dev_multiplier: Standard deviation multiplier for bands (default: 2.0)
    - entry_z_score: Z-score threshold for entry (default: 2.0)
    - exit_z_score: Z-score threshold for exit (default: 0.5)
    - rsi_period: RSI calculation period (default: 14)
    - rsi_oversold: RSI oversold threshold (default: 30)
    - rsi_overbought: RSI overbought threshold (default: 70)
    - use_rsi_confirmation: Require RSI confirmation (default: True)
    - min_volume: Minimum volume for signal (default: 0)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize mean reversion strategy.

        Args:
            config: Strategy configuration
        """
        super().__init__(config)

        # Mean reversion parameters
        self.lookback_period = config.get("lookback_period", 20)
        self.std_dev_multiplier = Decimal(str(config.get("std_dev_multiplier", 2.0)))
        self.entry_z_score = Decimal(str(config.get("entry_z_score", 2.0)))
        self.exit_z_score = Decimal(str(config.get("exit_z_score", 0.5)))

        # RSI parameters
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_oversold = Decimal(str(config.get("rsi_oversold", 30)))
        self.rsi_overbought = Decimal(str(config.get("rsi_overbought", 70)))
        self.use_rsi_confirmation = config.get("use_rsi_confirmation", True)

        # Volume filter
        self.min_volume = Decimal(str(config.get("min_volume", 0)))

        # Maximum position time (to limit exposure)
        self.max_position_bars = config.get("max_position_bars", 50)

        # History
        self._price_history: Deque[PricePoint] = deque(maxlen=max(self.lookback_period * 2, 100))
        self._gains: Deque[Decimal] = deque(maxlen=self.rsi_period)
        self._losses: Deque[Decimal] = deque(maxlen=self.rsi_period)

        # State
        self._last_price: Optional[Decimal] = None
        self._in_position = False
        self._position_side: Optional[SignalType] = None
        self._position_bars = 0

        logger.info(
            f"[{self.name}] Config: lookback={self.lookback_period}, "
            f"z_entry={self.entry_z_score}, z_exit={self.exit_z_score}, "
            f"rsi_confirmation={self.use_rsi_confirmation}"
        )

    def on_orderbook(self, snapshot: OrderBookSnapshot) -> Optional[Signal]:
        """
        Process order book update and check for mean reversion signals.

        Args:
            snapshot: Current order book snapshot

        Returns:
            Signal if conditions are met, None otherwise
        """
        if not snapshot.best_bid or not snapshot.best_ask:
            return None

        mid_price = snapshot.mid_price
        timestamp = snapshot.timestamp

        # Calculate volume from orderbook
        bid_volume = snapshot.bid_volume(5)
        ask_volume = snapshot.ask_volume(5)
        total_volume = bid_volume + ask_volume

        # Update price history
        self._update_price_history(PricePoint(
            timestamp=timestamp,
            price=mid_price,
            volume=total_volume,
        ))

        # Update RSI data
        self._update_rsi_data(mid_price)

        # Need enough history
        if len(self._price_history) < self.lookback_period:
            return None

        # Calculate metrics
        metrics = self._calculate_metrics(mid_price)

        if metrics is None:
            return None

        # Check for signals
        signal = self._evaluate_signal(snapshot, metrics)

        return signal

    def on_trade(self, trade: Trade) -> Optional[Signal]:
        """
        Process trade event.

        Args:
            trade: Trade event

        Returns:
            Signal if conditions are met
        """
        # Update price history with trade data
        self._update_price_history(PricePoint(
            timestamp=trade.timestamp,
            price=trade.price,
            volume=trade.quantity,
        ))

        self._update_rsi_data(trade.price)

        return None

    def _update_price_history(self, point: PricePoint) -> None:
        """Update price history with new data point."""
        self._price_history.append(point)

    def _update_rsi_data(self, current_price: Decimal) -> None:
        """Update RSI calculation data."""
        if self._last_price is not None:
            change = current_price - self._last_price
            if change > 0:
                self._gains.append(change)
                self._losses.append(Decimal("0"))
            else:
                self._gains.append(Decimal("0"))
                self._losses.append(abs(change))

        self._last_price = current_price

    def _calculate_metrics(self, current_price: Decimal) -> Optional[MeanReversionMetrics]:
        """
        Calculate mean reversion metrics.

        Args:
            current_price: Current market price

        Returns:
            MeanReversionMetrics or None if insufficient data
        """
        prices = [float(p.price) for p in self._price_history][-self.lookback_period:]

        if len(prices) < self.lookback_period:
            return None

        # Calculate mean and standard deviation
        mean = Decimal(str(statistics.mean(prices)))
        std_dev = Decimal(str(statistics.stdev(prices))) if len(prices) > 1 else Decimal("0.0001")

        # Prevent division by zero
        if std_dev == 0:
            std_dev = Decimal("0.0001")

        # Calculate Z-score
        z_score = (current_price - mean) / std_dev

        # Calculate Bollinger Bands
        upper_band = mean + (self.std_dev_multiplier * std_dev)
        lower_band = mean - (self.std_dev_multiplier * std_dev)

        # Calculate RSI
        rsi = self._calculate_rsi()

        # Deviation from mean
        deviation_percent = ((current_price - mean) / mean) * 100

        return MeanReversionMetrics(
            mean=mean,
            std_dev=std_dev,
            z_score=z_score,
            upper_band=upper_band,
            lower_band=lower_band,
            rsi=rsi,
            deviation_percent=deviation_percent,
        )

    def _calculate_rsi(self) -> Optional[Decimal]:
        """Calculate RSI (Relative Strength Index)."""
        if len(self._gains) < self.rsi_period:
            return None

        avg_gain = sum(self._gains) / len(self._gains)
        avg_loss = sum(self._losses) / len(self._losses)

        if avg_loss == 0:
            return Decimal("100")

        rs = avg_gain / avg_loss
        rsi = Decimal("100") - (Decimal("100") / (1 + rs))

        return rsi

    def _evaluate_signal(
        self,
        snapshot: OrderBookSnapshot,
        metrics: MeanReversionMetrics,
    ) -> Optional[Signal]:
        """
        Evaluate metrics and generate signal.

        Args:
            snapshot: Order book snapshot
            metrics: Calculated metrics

        Returns:
            Signal if conditions are met
        """
        if not self.can_signal():
            return None

        current_price = snapshot.mid_price

        # Track position duration
        if self._in_position:
            self._position_bars += 1

            # Check for exit signal (reversion to mean)
            if abs(metrics.z_score) <= self.exit_z_score:
                return self._generate_exit_signal(snapshot, metrics)

            # Check for max position time
            if self._position_bars >= self.max_position_bars:
                return self._generate_exit_signal(snapshot, metrics, reason="timeout")

        # Check for entry signals
        # Oversold condition (price significantly below mean)
        if metrics.z_score <= -self.entry_z_score:
            if self._check_rsi_confirmation(oversold=True, metrics=metrics):
                return self._generate_entry_signal(
                    signal_type=SignalType.LONG,
                    snapshot=snapshot,
                    metrics=metrics,
                )

        # Overbought condition (price significantly above mean)
        elif metrics.z_score >= self.entry_z_score:
            if self._check_rsi_confirmation(oversold=False, metrics=metrics):
                return self._generate_entry_signal(
                    signal_type=SignalType.SHORT,
                    snapshot=snapshot,
                    metrics=metrics,
                )

        return None

    def _check_rsi_confirmation(self, oversold: bool, metrics: MeanReversionMetrics) -> bool:
        """
        Check RSI confirmation for signal.

        Args:
            oversold: True if checking oversold, False for overbought
            metrics: Calculated metrics

        Returns:
            True if RSI confirms the signal
        """
        if not self.use_rsi_confirmation:
            return True

        if metrics.rsi is None:
            return True  # No RSI data, allow signal

        if oversold:
            return metrics.rsi <= self.rsi_oversold
        else:
            return metrics.rsi >= self.rsi_overbought

    def _generate_entry_signal(
        self,
        signal_type: SignalType,
        snapshot: OrderBookSnapshot,
        metrics: MeanReversionMetrics,
    ) -> Optional[Signal]:
        """Generate entry signal."""
        # Calculate signal strength based on deviation
        z_abs = abs(float(metrics.z_score))
        entry_threshold = float(self.entry_z_score)

        # Strength increases with deviation (capped at 1.0)
        strength = min(0.5 + (z_abs - entry_threshold) / entry_threshold * 0.5, 1.0)

        # Boost strength if RSI confirms
        if metrics.rsi is not None:
            if signal_type == SignalType.LONG and metrics.rsi < self.rsi_oversold:
                strength = min(strength + 0.1, 1.0)
            elif signal_type == SignalType.SHORT and metrics.rsi > self.rsi_overbought:
                strength = min(strength + 0.1, 1.0)

        # Update state
        self._in_position = True
        self._position_side = signal_type
        self._position_bars = 0

        return self.create_signal(
            signal_type=signal_type,
            symbol=snapshot.symbol,
            price=snapshot.mid_price,
            strength=strength,
            metadata={
                "entry_type": "mean_reversion",
                "z_score": float(metrics.z_score),
                "mean": float(metrics.mean),
                "std_dev": float(metrics.std_dev),
                "upper_band": float(metrics.upper_band),
                "lower_band": float(metrics.lower_band),
                "rsi": float(metrics.rsi) if metrics.rsi else None,
                "deviation_percent": float(metrics.deviation_percent),
            },
        )

    def _generate_exit_signal(
        self,
        snapshot: OrderBookSnapshot,
        metrics: MeanReversionMetrics,
        reason: str = "reversion",
    ) -> Optional[Signal]:
        """Generate exit signal (close position)."""
        if not self._in_position or self._position_side is None:
            return None

        # Exit is opposite of entry
        exit_type = SignalType.SHORT if self._position_side == SignalType.LONG else SignalType.LONG

        # Reset state
        self._in_position = False
        position_side = self._position_side
        self._position_side = None
        bars_held = self._position_bars
        self._position_bars = 0

        return self.create_signal(
            signal_type=exit_type,
            symbol=snapshot.symbol,
            price=snapshot.mid_price,
            strength=0.8,  # Exit signals are fairly strong
            metadata={
                "exit_type": "mean_reversion_exit",
                "exit_reason": reason,
                "z_score": float(metrics.z_score),
                "bars_held": bars_held,
                "original_side": position_side.name if position_side else None,
            },
        )

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def get_current_metrics(self) -> Optional[MeanReversionMetrics]:
        """Get current metrics without generating signal."""
        if len(self._price_history) < self.lookback_period:
            return None

        current_price = self._price_history[-1].price
        return self._calculate_metrics(current_price)

    def get_band_distance(self) -> Optional[Dict[str, Decimal]]:
        """Get distance to upper and lower bands."""
        metrics = self.get_current_metrics()
        if metrics is None:
            return None

        current_price = self._price_history[-1].price

        return {
            "to_upper": metrics.upper_band - current_price,
            "to_lower": current_price - metrics.lower_band,
            "to_mean": metrics.mean - current_price,
        }

    def is_oversold(self) -> bool:
        """Check if currently in oversold territory."""
        metrics = self.get_current_metrics()
        if metrics is None:
            return False

        z_check = metrics.z_score <= -self.entry_z_score

        if self.use_rsi_confirmation and metrics.rsi is not None:
            return z_check and metrics.rsi <= self.rsi_oversold

        return z_check

    def is_overbought(self) -> bool:
        """Check if currently in overbought territory."""
        metrics = self.get_current_metrics()
        if metrics is None:
            return False

        z_check = metrics.z_score >= self.entry_z_score

        if self.use_rsi_confirmation and metrics.rsi is not None:
            return z_check and metrics.rsi >= self.rsi_overbought

        return z_check

    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state."""
        metrics = self.get_current_metrics()

        return {
            "in_position": self._in_position,
            "position_side": self._position_side.name if self._position_side else None,
            "position_bars": self._position_bars,
            "price_history_length": len(self._price_history),
            "current_z_score": float(metrics.z_score) if metrics else None,
            "current_rsi": float(metrics.rsi) if metrics and metrics.rsi else None,
            "is_oversold": self.is_oversold(),
            "is_overbought": self.is_overbought(),
        }

    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self._price_history.clear()
        self._gains.clear()
        self._losses.clear()
        self._last_price = None
        self._in_position = False
        self._position_side = None
        self._position_bars = 0
        logger.info(f"[{self.name}] State reset")
