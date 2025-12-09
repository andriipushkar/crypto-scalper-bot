"""
Smart DCA Strategy with AI/ML.

Використовує машинне навчання для визначення оптимальних точок входу:
- Fear & Greed Index
- Технічні індикатори (RSI, Bollinger Bands, MACD)
- On-chain метрики
- Sentiment analysis
- Volume analysis

Features:
- ML модель для прогнозування оптимальності входу
- Динамічне коригування розміру позиції
- Адаптивний DCA на основі ринкових умов
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from loguru import logger

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from src.strategy.base import BaseStrategy, StrategyConfig, StrategyState
from src.data.models import Signal, SignalType, OrderBookSnapshot


# =============================================================================
# Enums and Types
# =============================================================================

class MarketCondition(Enum):
    """Market condition classification."""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


class EntryQuality(Enum):
    """Entry point quality."""
    EXCELLENT = "excellent"    # Score >= 0.8
    GOOD = "good"              # Score >= 0.6
    AVERAGE = "average"        # Score >= 0.4
    POOR = "poor"              # Score >= 0.2
    BAD = "bad"                # Score < 0.2


class DCAMode(Enum):
    """DCA operation mode."""
    AGGRESSIVE = "aggressive"  # Buy on every dip
    BALANCED = "balanced"      # Buy on good opportunities
    CONSERVATIVE = "conservative"  # Only buy on excellent opportunities


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MarketMetrics:
    """Market metrics for ML model."""
    # Price metrics
    price: Decimal = Decimal("0")
    price_change_1h: float = 0.0
    price_change_24h: float = 0.0
    price_change_7d: float = 0.0

    # Technical indicators
    rsi_14: float = 50.0
    rsi_7: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    bb_position: float = 0.5  # 0 = lower band, 1 = upper band
    ema_9_distance: float = 0.0  # % distance from EMA9
    ema_21_distance: float = 0.0
    sma_50_distance: float = 0.0
    sma_200_distance: float = 0.0

    # Volume metrics
    volume_24h: Decimal = Decimal("0")
    volume_change: float = 0.0
    volume_sma_ratio: float = 1.0

    # Volatility
    atr_14: float = 0.0
    volatility_24h: float = 0.0

    # Fear & Greed
    fear_greed_index: int = 50
    fear_greed_change: int = 0

    # Funding rate (for futures)
    funding_rate: float = 0.0

    # On-chain (if available)
    exchange_inflow: float = 0.0
    exchange_outflow: float = 0.0
    whale_transactions: int = 0

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_feature_vector(self) -> List[float]:
        """Convert to feature vector for ML model."""
        return [
            self.price_change_1h,
            self.price_change_24h,
            self.price_change_7d,
            self.rsi_14,
            self.rsi_7,
            self.macd,
            self.macd_signal,
            self.macd_histogram,
            self.bb_position,
            self.ema_9_distance,
            self.ema_21_distance,
            self.sma_50_distance,
            self.sma_200_distance,
            self.volume_change,
            self.volume_sma_ratio,
            self.atr_14,
            self.volatility_24h,
            self.fear_greed_index / 100,  # Normalize
            self.funding_rate * 100,  # Scale
            self.exchange_inflow - self.exchange_outflow,  # Net flow
        ]


@dataclass
class DCAEntry:
    """Single DCA entry record."""
    entry_id: str
    timestamp: datetime
    price: Decimal
    quantity: Decimal
    entry_score: float
    market_condition: MarketCondition
    notes: str = ""


@dataclass
class SmartDCAConfig:
    """Smart DCA configuration."""
    # Base DCA settings
    base_amount: Decimal = Decimal("100")  # Base amount per entry
    max_entries: int = 10
    min_interval_hours: int = 4  # Minimum hours between entries

    # Entry thresholds
    min_entry_score: float = 0.4  # Minimum ML score to enter
    excellent_score_multiplier: float = 2.0  # Multiply amount on excellent entries
    good_score_multiplier: float = 1.5
    average_score_multiplier: float = 1.0

    # Price drop levels
    dip_threshold_pct: float = 0.03  # 3% drop triggers evaluation
    major_dip_pct: float = 0.10  # 10% drop = aggressive entry

    # Take profit
    take_profit_pct: float = 0.15  # 15% total gain
    partial_tp_pct: float = 0.10  # Take partial at 10%
    partial_tp_amount: float = 0.5  # Take 50% at partial TP

    # Stop loss
    stop_loss_pct: float = 0.20  # 20% total loss

    # Mode
    mode: DCAMode = DCAMode.BALANCED

    # ML model
    model_path: Optional[str] = None
    retrain_interval_days: int = 7


@dataclass
class SmartDCAState:
    """Smart DCA strategy state."""
    # Position
    entries: List[DCAEntry] = field(default_factory=list)
    total_invested: Decimal = Decimal("0")
    total_quantity: Decimal = Decimal("0")
    average_entry_price: Decimal = Decimal("0")

    # Status
    is_active: bool = False
    last_entry_time: Optional[datetime] = None
    last_evaluation_time: Optional[datetime] = None

    # Performance
    unrealized_pnl: Decimal = Decimal("0")
    unrealized_pnl_pct: float = 0.0
    highest_price_since_entry: Decimal = Decimal("0")

    # ML model state
    model_trained: bool = False
    last_training_time: Optional[datetime] = None


# =============================================================================
# Smart DCA Strategy
# =============================================================================

class SmartDCAStrategy(BaseStrategy):
    """
    Smart DCA Strategy with AI/ML optimization.

    Uses machine learning to identify optimal entry points based on:
    - Technical indicators
    - Market sentiment (Fear & Greed)
    - Volume analysis
    - On-chain metrics

    Usage:
        strategy = SmartDCAStrategy(
            symbol="BTCUSDT",
            config=SmartDCAConfig(
                base_amount=Decimal("100"),
                mode=DCAMode.BALANCED,
            ),
        )

        # Evaluate entry opportunity
        score = await strategy.evaluate_entry(market_metrics)
        if score.quality in (EntryQuality.EXCELLENT, EntryQuality.GOOD):
            signal = strategy.generate_entry_signal(score)
    """

    def __init__(
        self,
        symbol: str,
        config: SmartDCAConfig = None,
    ):
        self.symbol = symbol
        self.config = config or SmartDCAConfig()
        self._state = SmartDCAState()  # Use _state to avoid property conflict

        # ML components
        self._model: Optional[Any] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_history: List[Tuple[List[float], int]] = []  # (features, label)

        # Price history for indicators
        self._price_history: List[Tuple[datetime, Decimal]] = []
        self._volume_history: List[Tuple[datetime, Decimal]] = []

        # Load model if exists
        if self.config.model_path and HAS_SKLEARN:
            self._load_model()

        logger.info(f"Smart DCA Strategy initialized for {symbol}")

    # =========================================================================
    # Model Management
    # =========================================================================

    def _load_model(self) -> bool:
        """Load trained ML model."""
        if not HAS_SKLEARN:
            logger.warning("scikit-learn not installed, ML features disabled")
            return False

        try:
            data = joblib.load(self.config.model_path)
            self._model = data["model"]
            self._scaler = data["scaler"]
            self.state.model_trained = True
            self.state.last_training_time = data.get("trained_at")
            logger.info(f"Loaded ML model from {self.config.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _save_model(self, path: str) -> None:
        """Save trained model."""
        if not self._model:
            return

        joblib.dump({
            "model": self._model,
            "scaler": self._scaler,
            "trained_at": datetime.utcnow(),
        }, path)
        logger.info(f"Saved ML model to {path}")

    def train_model(
        self,
        features: List[List[float]],
        labels: List[int],  # 1 = good entry, 0 = bad entry
    ) -> float:
        """
        Train the ML model on historical data.

        Args:
            features: List of feature vectors
            labels: List of labels (1 = profitable entry, 0 = unprofitable)

        Returns:
            Model accuracy score
        """
        if not HAS_SKLEARN:
            logger.error("scikit-learn required for training")
            return 0.0

        if len(features) < 100:
            logger.warning("Insufficient data for training (need at least 100 samples)")
            return 0.0

        # Initialize scaler and model
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(features)

        # Use Gradient Boosting for better performance
        self._model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )

        # Train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]

        self._model.fit(X_train, y_train)

        # Calculate accuracy
        accuracy = self._model.score(X_test, y_test)

        self.state.model_trained = True
        self.state.last_training_time = datetime.utcnow()

        logger.info(f"Model trained with accuracy: {accuracy:.2%}")

        # Save model if path configured
        if self.config.model_path:
            self._save_model(self.config.model_path)

        return accuracy

    # =========================================================================
    # Entry Evaluation
    # =========================================================================

    def evaluate_entry(self, metrics: MarketMetrics) -> Tuple[float, EntryQuality]:
        """
        Evaluate entry opportunity using ML model.

        Args:
            metrics: Current market metrics

        Returns:
            Tuple of (score 0-1, quality classification)
        """
        # If no model, use rule-based scoring
        if not self._model or not self.state.model_trained:
            score = self._rule_based_score(metrics)
        else:
            score = self._ml_score(metrics)

        # Apply mode adjustments
        if self.config.mode == DCAMode.CONSERVATIVE:
            score *= 0.8  # More strict
        elif self.config.mode == DCAMode.AGGRESSIVE:
            score *= 1.2  # More lenient
            score = min(score, 1.0)

        # Classify quality
        if score >= 0.8:
            quality = EntryQuality.EXCELLENT
        elif score >= 0.6:
            quality = EntryQuality.GOOD
        elif score >= 0.4:
            quality = EntryQuality.AVERAGE
        elif score >= 0.2:
            quality = EntryQuality.POOR
        else:
            quality = EntryQuality.BAD

        return score, quality

    def _ml_score(self, metrics: MarketMetrics) -> float:
        """Calculate entry score using ML model."""
        features = np.array([metrics.to_feature_vector()])
        features_scaled = self._scaler.transform(features)

        # Get probability of good entry
        proba = self._model.predict_proba(features_scaled)[0]
        score = proba[1]  # Probability of class 1 (good entry)

        return float(score)

    def _rule_based_score(self, metrics: MarketMetrics) -> float:
        """Calculate entry score using rule-based approach."""
        score = 0.5  # Start neutral

        # RSI scoring
        if metrics.rsi_14 < 30:
            score += 0.2
        elif metrics.rsi_14 < 40:
            score += 0.1
        elif metrics.rsi_14 > 70:
            score -= 0.2
        elif metrics.rsi_14 > 60:
            score -= 0.1

        # Fear & Greed scoring
        if metrics.fear_greed_index < 20:
            score += 0.15  # Extreme fear = good to buy
        elif metrics.fear_greed_index < 40:
            score += 0.1
        elif metrics.fear_greed_index > 80:
            score -= 0.15  # Extreme greed = bad to buy
        elif metrics.fear_greed_index > 60:
            score -= 0.1

        # Bollinger Bands position
        if metrics.bb_position < 0.2:
            score += 0.1  # Near lower band
        elif metrics.bb_position > 0.8:
            score -= 0.1  # Near upper band

        # Price drop bonus
        if metrics.price_change_24h < -self.config.major_dip_pct * 100:
            score += 0.2
        elif metrics.price_change_24h < -self.config.dip_threshold_pct * 100:
            score += 0.1

        # Volume spike (high volume on dip = accumulation)
        if metrics.volume_sma_ratio > 2.0 and metrics.price_change_24h < 0:
            score += 0.1

        # Funding rate (negative = shorts paying longs)
        if metrics.funding_rate < -0.01:
            score += 0.1
        elif metrics.funding_rate > 0.01:
            score -= 0.05

        # MACD
        if metrics.macd_histogram > 0 and metrics.macd < 0:
            score += 0.1  # Bullish divergence
        elif metrics.macd_histogram < 0 and metrics.macd > 0:
            score -= 0.1  # Bearish divergence

        # Clamp score
        return max(0.0, min(1.0, score))

    def get_market_condition(self, metrics: MarketMetrics) -> MarketCondition:
        """Classify current market condition."""
        fg = metrics.fear_greed_index

        if fg < 20:
            return MarketCondition.EXTREME_FEAR
        elif fg < 40:
            return MarketCondition.FEAR
        elif fg < 60:
            return MarketCondition.NEUTRAL
        elif fg < 80:
            return MarketCondition.GREED
        else:
            return MarketCondition.EXTREME_GREED

    # =========================================================================
    # Entry Generation
    # =========================================================================

    def calculate_entry_amount(self, score: float, quality: EntryQuality) -> Decimal:
        """Calculate entry amount based on score and quality."""
        base = self.config.base_amount

        if quality == EntryQuality.EXCELLENT:
            multiplier = self.config.excellent_score_multiplier
        elif quality == EntryQuality.GOOD:
            multiplier = self.config.good_score_multiplier
        else:
            multiplier = self.config.average_score_multiplier

        # Apply score-based fine-tuning
        amount = base * Decimal(str(multiplier * (0.8 + score * 0.4)))

        return amount.quantize(Decimal("0.01"))

    def should_enter(self, score: float, quality: EntryQuality, metrics: MarketMetrics) -> Tuple[bool, str]:
        """
        Determine if we should enter based on all criteria.

        Returns:
            Tuple of (should_enter, reason)
        """
        # Check max entries
        if len(self.state.entries) >= self.config.max_entries:
            return False, "Max entries reached"

        # Check minimum score
        if score < self.config.min_entry_score:
            return False, f"Score {score:.2f} below threshold {self.config.min_entry_score}"

        # Check time since last entry
        if self.state.last_entry_time:
            hours_since = (datetime.utcnow() - self.state.last_entry_time).total_seconds() / 3600
            if hours_since < self.config.min_interval_hours:
                return False, f"Too soon since last entry ({hours_since:.1f}h < {self.config.min_interval_hours}h)"

        # Mode-specific checks
        if self.config.mode == DCAMode.CONSERVATIVE:
            if quality not in (EntryQuality.EXCELLENT, EntryQuality.GOOD):
                return False, f"Conservative mode requires GOOD or better (got {quality.value})"

        elif self.config.mode == DCAMode.BALANCED:
            if quality == EntryQuality.BAD:
                return False, "Quality too low"

        # Check market condition
        condition = self.get_market_condition(metrics)
        if condition in (MarketCondition.EXTREME_GREED, MarketCondition.GREED):
            if quality != EntryQuality.EXCELLENT:
                return False, f"Market too greedy for non-excellent entry"

        return True, "Entry approved"

    def generate_entry_signal(
        self,
        metrics: MarketMetrics,
        score: float,
        quality: EntryQuality,
    ) -> Optional[Signal]:
        """Generate entry signal."""
        should, reason = self.should_enter(score, quality, metrics)

        if not should:
            logger.debug(f"Entry rejected: {reason}")
            return None

        amount = self.calculate_entry_amount(score, quality)
        quantity = amount / metrics.price

        signal = Signal(
            signal_type=SignalType.LONG,
            symbol=self.symbol,
            timestamp=datetime.utcnow(),
            price=metrics.price,
            strength=score,
            strategy=f"smart_dca_{self.config.mode.value}",
            metadata={
                "entry_quality": quality.value,
                "entry_score": score,
                "amount_usd": float(amount),
                "market_condition": self.get_market_condition(metrics).value,
                "fear_greed": metrics.fear_greed_index,
                "rsi": metrics.rsi_14,
            },
        )

        return signal

    # =========================================================================
    # Position Management
    # =========================================================================

    def record_entry(
        self,
        price: Decimal,
        quantity: Decimal,
        score: float,
        metrics: MarketMetrics,
    ) -> DCAEntry:
        """Record a new DCA entry."""
        import uuid

        entry = DCAEntry(
            entry_id=str(uuid.uuid4())[:8],
            timestamp=datetime.utcnow(),
            price=price,
            quantity=quantity,
            entry_score=score,
            market_condition=self.get_market_condition(metrics),
        )

        self.state.entries.append(entry)
        self.state.total_invested += price * quantity
        self.state.total_quantity += quantity
        self.state.last_entry_time = datetime.utcnow()
        self.state.is_active = True

        # Update average entry price
        self.state.average_entry_price = (
            self.state.total_invested / self.state.total_quantity
        )

        logger.info(
            f"DCA Entry #{len(self.state.entries)}: "
            f"{quantity} @ {price} (score: {score:.2f})"
        )

        return entry

    def update_position(self, current_price: Decimal) -> None:
        """Update position with current price."""
        if not self.state.is_active or self.state.total_quantity == 0:
            return

        # Update PnL
        current_value = current_price * self.state.total_quantity
        self.state.unrealized_pnl = current_value - self.state.total_invested
        self.state.unrealized_pnl_pct = float(
            self.state.unrealized_pnl / self.state.total_invested * 100
        )

        # Track highest price
        if current_price > self.state.highest_price_since_entry:
            self.state.highest_price_since_entry = current_price

    def should_take_profit(self, current_price: Decimal) -> Tuple[bool, float]:
        """
        Check if we should take profit.

        Returns:
            Tuple of (should_tp, amount_pct to close)
        """
        if not self.state.is_active:
            return False, 0.0

        self.update_position(current_price)

        # Full take profit
        if self.state.unrealized_pnl_pct >= self.config.take_profit_pct * 100:
            return True, 1.0

        # Partial take profit
        if self.state.unrealized_pnl_pct >= self.config.partial_tp_pct * 100:
            return True, self.config.partial_tp_amount

        return False, 0.0

    def should_stop_loss(self, current_price: Decimal) -> bool:
        """Check if we should stop loss."""
        if not self.state.is_active:
            return False

        self.update_position(current_price)

        return self.state.unrealized_pnl_pct <= -self.config.stop_loss_pct * 100

    def close_position(self, exit_price: Decimal, amount_pct: float = 1.0) -> Dict[str, Any]:
        """Close position (fully or partially)."""
        if not self.state.is_active:
            return {}

        quantity_to_close = self.state.total_quantity * Decimal(str(amount_pct))
        invested_to_close = self.state.total_invested * Decimal(str(amount_pct))

        pnl = exit_price * quantity_to_close - invested_to_close
        pnl_pct = float(pnl / invested_to_close * 100)

        result = {
            "quantity_closed": float(quantity_to_close),
            "exit_price": float(exit_price),
            "pnl": float(pnl),
            "pnl_pct": pnl_pct,
            "entries_count": len(self.state.entries),
            "average_entry": float(self.state.average_entry_price),
        }

        # Update state
        self.state.total_quantity -= quantity_to_close
        self.state.total_invested -= invested_to_close

        if self.state.total_quantity <= 0:
            self.state.is_active = False
            self.state.entries = []
            self.state.total_quantity = Decimal("0")
            self.state.total_invested = Decimal("0")
            self.state.average_entry_price = Decimal("0")

        logger.info(f"Position closed: {result}")

        return result

    # =========================================================================
    # BaseStrategy Implementation
    # =========================================================================

    def on_orderbook(self, snapshot: OrderBookSnapshot) -> Optional[Signal]:
        """Process orderbook update (required by BaseStrategy)."""
        # Smart DCA doesn't use orderbook directly
        # It uses evaluate_entry() with market metrics
        return None

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        return {
            "symbol": self.symbol,
            "mode": self.config.mode.value,
            "is_active": self.state.is_active,
            "entries_count": len(self.state.entries),
            "total_invested": float(self.state.total_invested),
            "total_quantity": float(self.state.total_quantity),
            "average_entry_price": float(self.state.average_entry_price),
            "unrealized_pnl": float(self.state.unrealized_pnl),
            "unrealized_pnl_pct": self.state.unrealized_pnl_pct,
            "model_trained": self.state.model_trained,
            "last_entry_time": self.state.last_entry_time.isoformat() if self.state.last_entry_time else None,
        }


# =============================================================================
# Fear & Greed Index Fetcher
# =============================================================================

class FearGreedFetcher:
    """Fetch Fear & Greed Index from alternative.me API."""

    API_URL = "https://api.alternative.me/fng/"

    def __init__(self):
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=30)

    async def fetch(self) -> Dict[str, Any]:
        """Fetch current Fear & Greed Index."""
        # Check cache
        if self._cache and self._cache_time:
            if datetime.utcnow() - self._cache_time < self._cache_duration:
                return self._cache

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(self.API_URL) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = {
                            "value": int(data["data"][0]["value"]),
                            "classification": data["data"][0]["value_classification"],
                            "timestamp": datetime.utcnow(),
                        }
                        self._cache = result
                        self._cache_time = datetime.utcnow()
                        return result

        except Exception as e:
            logger.error(f"Failed to fetch Fear & Greed Index: {e}")

        # Return cached or default
        return self._cache or {"value": 50, "classification": "Neutral", "timestamp": datetime.utcnow()}

    def get_cached_value(self) -> int:
        """Get cached Fear & Greed value."""
        if self._cache:
            return self._cache["value"]
        return 50
