"""
Base strategy class.

All trading strategies inherit from this base class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List

from loguru import logger

from src.data.models import (
    OrderBookSnapshot,
    Trade,
    MarkPrice,
    Signal,
    SignalType,
    Side,
)


@dataclass
class StrategyConfig:
    """Base configuration for strategies."""
    enabled: bool = True
    signal_cooldown: float = 10.0  # seconds between signals
    min_strength: float = 0.5  # minimum signal strength to emit


@dataclass
class StrategyState:
    """Runtime state for a strategy."""
    last_signal_time: Optional[datetime] = None
    last_signal_type: Optional[SignalType] = None
    signals_generated: int = 0
    errors: int = 0


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    Subclasses must implement:
    - on_orderbook(): Process order book updates and generate signals
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy.

        Args:
            config: Strategy-specific configuration
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.signal_cooldown = config.get("signal_cooldown", 10.0)
        self.min_strength = config.get("min_strength", 0.5)

        self._state = StrategyState()
        self._name = self.__class__.__name__

        logger.info(f"Initialized strategy: {self._name}")

    @property
    def name(self) -> str:
        """Get strategy name."""
        return self._name

    @property
    def state(self) -> StrategyState:
        """Get strategy state."""
        return self._state

    @property
    def stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        return {
            "name": self._name,
            "enabled": self.enabled,
            "signals_generated": self._state.signals_generated,
            "errors": self._state.errors,
            "last_signal_time": self._state.last_signal_time,
        }

    # =========================================================================
    # Main Interface
    # =========================================================================

    @abstractmethod
    def on_orderbook(self, snapshot: OrderBookSnapshot) -> Optional[Signal]:
        """
        Process order book update.

        This is the main entry point for the strategy.
        Called on every order book update.

        Args:
            snapshot: Current order book snapshot

        Returns:
            Signal if conditions are met, None otherwise
        """
        pass

    def on_trade(self, trade: Trade) -> Optional[Signal]:
        """
        Process trade event (optional).

        Override if strategy needs trade data.

        Args:
            trade: Trade event

        Returns:
            Signal if conditions are met, None otherwise
        """
        return None

    def on_mark_price(self, mark_price: MarkPrice) -> Optional[Signal]:
        """
        Process mark price event (optional).

        Override if strategy uses funding rate.

        Args:
            mark_price: Mark price event

        Returns:
            Signal if conditions are met, None otherwise
        """
        return None

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def can_signal(self) -> bool:
        """Check if strategy can emit a signal (cooldown check)."""
        if not self.enabled:
            return False

        if self._state.last_signal_time is None:
            return True

        elapsed = (datetime.utcnow() - self._state.last_signal_time).total_seconds()
        return elapsed >= self.signal_cooldown

    def create_signal(
        self,
        signal_type: SignalType,
        symbol: str,
        price: Decimal,
        strength: float,
        metadata: Dict = None,
    ) -> Optional[Signal]:
        """
        Create a signal if conditions are met.

        Args:
            signal_type: Type of signal
            symbol: Trading symbol
            price: Current price
            strength: Signal strength (0.0 to 1.0)
            metadata: Additional signal metadata

        Returns:
            Signal if valid, None otherwise
        """
        # Check cooldown
        if not self.can_signal():
            return None

        # Check minimum strength
        if strength < self.min_strength:
            return None

        # Create signal
        signal = Signal(
            strategy=self._name,
            signal_type=signal_type,
            symbol=symbol,
            timestamp=datetime.utcnow(),
            strength=min(max(strength, 0.0), 1.0),  # Clamp to [0, 1]
            price=price,
            metadata=metadata or {},
        )

        # Update state
        self._state.last_signal_time = signal.timestamp
        self._state.last_signal_type = signal_type
        self._state.signals_generated += 1

        logger.debug(
            f"[{self._name}] Signal: {signal_type.name} {symbol} "
            f"@ {price} (strength={strength:.2f})"
        )

        return signal

    def reset(self) -> None:
        """Reset strategy state."""
        self._state = StrategyState()
        logger.info(f"[{self._name}] State reset")

    # =========================================================================
    # Backtest Interface
    # =========================================================================

    def generate_signal(
        self,
        symbol: str,
        trades: List[Trade] = None,
        orderbook: OrderBookSnapshot = None,
    ) -> Optional[Signal]:
        """
        Generate signal from market data (for backtesting).

        Args:
            symbol: Trading symbol
            trades: Recent trades
            orderbook: Order book snapshot

        Returns:
            Signal if conditions are met
        """
        # If orderbook provided, use standard method
        if orderbook:
            return self.on_orderbook(orderbook)

        # If trades provided, use trade-based analysis
        if trades:
            # Default implementation: use last trade
            last_trade = trades[-1]
            return self.on_trade(last_trade)

        return None

    def generate_signal_from_bars(
        self,
        symbol: str,
        bars: List[Any],  # List[OHLCV]
    ) -> Optional[Signal]:
        """
        Generate signal from OHLCV bars (for backtesting).

        Override this method for bar-based strategies.

        Args:
            symbol: Trading symbol
            bars: List of OHLCV bars (oldest to newest)

        Returns:
            Signal if conditions are met
        """
        # Default implementation - no signal
        # Subclasses should override for bar-based logic
        return None


class CompositeStrategy(BaseStrategy):
    """
    Strategy that combines multiple sub-strategies.

    Aggregates signals from child strategies using weighted voting.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        strategies: List[BaseStrategy],
        weights: Dict[str, float] = None,
    ):
        """
        Initialize composite strategy.

        Args:
            config: Strategy configuration
            strategies: List of child strategies
            weights: Weight for each strategy (by name), defaults to equal
        """
        super().__init__(config)

        self.strategies = strategies
        self.weights = weights or {s.name: 1.0 for s in strategies}

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def on_orderbook(self, snapshot: OrderBookSnapshot) -> Optional[Signal]:
        """
        Process order book with all child strategies.

        Returns weighted combination of signals.
        """
        if not self.can_signal():
            return None

        long_score = 0.0
        short_score = 0.0
        signals = []

        # Collect signals from each strategy
        for strategy in self.strategies:
            if not strategy.enabled:
                continue

            try:
                signal = strategy.on_orderbook(snapshot)

                if signal and signal.signal_type != SignalType.NO_ACTION:
                    weight = self.weights.get(strategy.name, 0.0)
                    signals.append((signal, weight))

                    if signal.signal_type == SignalType.LONG:
                        long_score += signal.strength * weight
                    elif signal.signal_type == SignalType.SHORT:
                        short_score += signal.strength * weight

            except Exception as e:
                logger.error(f"[{strategy.name}] Error: {e}")
                strategy._state.errors += 1

        # No signals from any strategy
        if not signals:
            return None

        # Determine combined signal
        threshold = self.config.get("combined_threshold", 0.5)

        if long_score > short_score and long_score >= threshold:
            return self.create_signal(
                signal_type=SignalType.LONG,
                symbol=snapshot.symbol,
                price=snapshot.mid_price,
                strength=long_score,
                metadata={
                    "component_signals": len(signals),
                    "long_score": long_score,
                    "short_score": short_score,
                },
            )

        elif short_score > long_score and short_score >= threshold:
            return self.create_signal(
                signal_type=SignalType.SHORT,
                symbol=snapshot.symbol,
                price=snapshot.mid_price,
                strength=short_score,
                metadata={
                    "component_signals": len(signals),
                    "long_score": long_score,
                    "short_score": short_score,
                },
            )

        return None
