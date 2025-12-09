"""
Machine Learning based trading strategy.

Uses a neural network to predict price direction based on:
- Order book features (imbalance, spread, depth)
- Recent trade features (volume, direction, VWAP)
- Technical indicators
"""

import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from loguru import logger

from src.data.models import OrderBookSnapshot, Trade, Signal, SignalType
from src.strategy.base import BaseStrategy

# Try to import ML libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. ML strategy disabled.")

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed. ML strategy may have limited functionality.")


# =============================================================================
# Neural Network Model
# =============================================================================

if TORCH_AVAILABLE:
    class TradingMLP(nn.Module):
        """
        Multi-Layer Perceptron for trading signal prediction.

        Input: Feature vector (order book + trade features)
        Output: Probabilities for [LONG, SHORT, NO_ACTION]
        """

        def __init__(
            self,
            input_size: int = 20,
            hidden_sizes: List[int] = None,
            dropout: float = 0.3,
        ):
            super().__init__()

            hidden_sizes = hidden_sizes or [64, 32, 16]

            layers = []
            prev_size = input_size

            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev_size = hidden_size

            # Output layer: 3 classes (LONG, SHORT, NO_ACTION)
            layers.append(nn.Linear(prev_size, 3))

            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)

        def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
            """Get softmax probabilities."""
            with torch.no_grad():
                logits = self.forward(x)
                return F.softmax(logits, dim=-1)


    class TradingLSTM(nn.Module):
        """
        LSTM network for sequence-based prediction.

        Processes sequences of market data to predict direction.
        """

        def __init__(
            self,
            input_size: int = 20,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.3,
        ):
            super().__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
            )

            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 3),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch, seq_len, features)
            lstm_out, _ = self.lstm(x)
            # Use last output
            last_out = lstm_out[:, -1, :]
            return self.fc(last_out)

        def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                logits = self.forward(x)
                return F.softmax(logits, dim=-1)


# =============================================================================
# Feature Engineering
# =============================================================================

@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    # Order book features
    orderbook_levels: int = 5
    include_imbalance: bool = True
    include_spread: bool = True
    include_depth: bool = True

    # Trade features
    trade_window: int = 100  # Number of recent trades
    include_vwap: bool = True
    include_volume_profile: bool = True

    # Technical features
    include_momentum: bool = True
    momentum_periods: List[int] = field(default_factory=lambda: [5, 10, 20])


class FeatureExtractor:
    """
    Extract features from market data for ML model.
    """

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self._is_fitted = False

        # Feature history for sequence models
        self._feature_history: List[np.ndarray] = []
        self._max_history = 100

    def extract_orderbook_features(self, snapshot: OrderBookSnapshot) -> np.ndarray:
        """Extract features from order book snapshot."""
        features = []

        # Best bid/ask
        if snapshot.best_bid and snapshot.best_ask:
            mid_price = float(snapshot.mid_price)
            spread_bps = float(snapshot.spread_bps)

            features.extend([
                spread_bps,
                float(snapshot.best_bid.quantity),
                float(snapshot.best_ask.quantity),
            ])

            # Imbalance at different levels
            if self.config.include_imbalance:
                for i in range(min(self.config.orderbook_levels, len(snapshot.bids))):
                    bid_qty = float(snapshot.bids[i].quantity) if i < len(snapshot.bids) else 0
                    ask_qty = float(snapshot.asks[i].quantity) if i < len(snapshot.asks) else 0
                    total = bid_qty + ask_qty
                    imbalance = (bid_qty - ask_qty) / total if total > 0 else 0
                    features.append(imbalance)

            # Depth
            if self.config.include_depth:
                bid_depth = sum(float(b.quantity) for b in snapshot.bids[:5])
                ask_depth = sum(float(a.quantity) for a in snapshot.asks[:5])
                features.extend([bid_depth, ask_depth])

        else:
            # Pad with zeros if no data
            n_features = 3 + self.config.orderbook_levels + 2
            features = [0.0] * n_features

        return np.array(features, dtype=np.float32)

    def extract_trade_features(self, trades: List[Trade]) -> np.ndarray:
        """Extract features from recent trades."""
        features = []

        if not trades:
            return np.zeros(10, dtype=np.float32)

        # Basic stats
        prices = [float(t.price) for t in trades]
        quantities = [float(t.quantity) for t in trades]

        features.extend([
            np.mean(prices),
            np.std(prices) if len(prices) > 1 else 0,
            np.sum(quantities),
        ])

        # Buy/sell ratio
        buy_vol = sum(float(t.quantity) for t in trades if not t.is_buyer_maker)
        sell_vol = sum(float(t.quantity) for t in trades if t.is_buyer_maker)
        total_vol = buy_vol + sell_vol
        buy_ratio = buy_vol / total_vol if total_vol > 0 else 0.5
        features.append(buy_ratio)

        # VWAP
        if self.config.include_vwap:
            vwap = sum(float(t.price) * float(t.quantity) for t in trades)
            vwap = vwap / total_vol if total_vol > 0 else prices[-1]
            features.append(vwap)

        # Price momentum
        if self.config.include_momentum and len(prices) > 1:
            returns = np.diff(prices) / np.array(prices[:-1])
            features.append(np.mean(returns))
            features.append(np.std(returns) if len(returns) > 1 else 0)
        else:
            features.extend([0.0, 0.0])

        # Trade intensity
        if len(trades) >= 2:
            time_span = (trades[-1].timestamp - trades[0].timestamp).total_seconds()
            intensity = len(trades) / time_span if time_span > 0 else 0
            features.append(intensity)
        else:
            features.append(0.0)

        # Pad to fixed size
        while len(features) < 10:
            features.append(0.0)

        return np.array(features[:10], dtype=np.float32)

    def extract_features(
        self,
        orderbook: OrderBookSnapshot = None,
        trades: List[Trade] = None,
    ) -> np.ndarray:
        """Extract combined feature vector."""
        ob_features = self.extract_orderbook_features(orderbook) if orderbook else np.zeros(10)
        trade_features = self.extract_trade_features(trades) if trades else np.zeros(10)

        features = np.concatenate([ob_features, trade_features])

        # Store in history
        self._feature_history.append(features)
        if len(self._feature_history) > self._max_history:
            self._feature_history.pop(0)

        return features

    def get_sequence(self, length: int = 20) -> np.ndarray:
        """Get sequence of recent features for LSTM."""
        if len(self._feature_history) < length:
            # Pad with zeros
            padding = [np.zeros_like(self._feature_history[0])] * (length - len(self._feature_history))
            seq = padding + self._feature_history
        else:
            seq = self._feature_history[-length:]

        return np.array(seq, dtype=np.float32)

    def fit_scaler(self, features: np.ndarray) -> None:
        """Fit the scaler on training data."""
        if self.scaler is not None:
            self.scaler.fit(features)
            self._is_fitted = True

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply scaling to features."""
        if self.scaler is not None and self._is_fitted:
            return self.scaler.transform(features.reshape(1, -1)).flatten()
        return features


# =============================================================================
# ML Strategy
# =============================================================================

class MLStrategy(BaseStrategy):
    """
    Machine Learning based trading strategy.

    Uses a trained neural network to predict trade direction.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for ML strategy")

        # Model config
        self.model_type = config.get("model_type", "mlp")  # mlp or lstm
        self.model_path = config.get("model_path", "data/models/trading_model.pt")
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        self.use_gpu = config.get("use_gpu", False) and torch.cuda.is_available()

        # Feature extraction
        self.feature_config = FeatureConfig(
            orderbook_levels=config.get("orderbook_levels", 5),
            trade_window=config.get("trade_window", 100),
        )
        self.feature_extractor = FeatureExtractor(self.feature_config)

        # Initialize model
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.model = self._load_or_create_model()
        self.model.to(self.device)
        self.model.eval()

        # Trade buffer
        self._trade_buffer: List[Trade] = []
        self._buffer_size = config.get("trade_buffer_size", 100)

        logger.info(f"ML Strategy initialized (model: {self.model_type}, device: {self.device})")

    def _load_or_create_model(self) -> nn.Module:
        """Load existing model or create new one."""
        model_path = Path(self.model_path)

        if model_path.exists():
            logger.info(f"Loading model from {model_path}")
            model = torch.load(model_path, map_location=self.device)
            return model

        # Create new model
        logger.info(f"Creating new {self.model_type} model")

        if self.model_type == "lstm":
            return TradingLSTM(input_size=20)
        else:
            return TradingMLP(input_size=20)

    def save_model(self, path: str = None) -> None:
        """Save model to file."""
        save_path = Path(path or self.model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model, save_path)
        logger.info(f"Model saved to {save_path}")

    def on_orderbook(self, snapshot: OrderBookSnapshot) -> Optional[Signal]:
        """Generate signal from order book."""
        if not self.can_signal():
            return None

        # Extract features
        features = self.feature_extractor.extract_features(
            orderbook=snapshot,
            trades=self._trade_buffer[-self._buffer_size:] if self._trade_buffer else None,
        )

        # Get prediction
        if self.model_type == "lstm":
            sequence = self.feature_extractor.get_sequence(length=20)
            x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        else:
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        probs = self.model.predict_proba(x).cpu().numpy()[0]

        # Classes: [LONG, SHORT, NO_ACTION]
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]

        # Only signal if confidence exceeds threshold
        if confidence < self.confidence_threshold:
            return None

        # Map class to signal type
        signal_map = {
            0: SignalType.LONG,
            1: SignalType.SHORT,
            2: SignalType.NO_ACTION,
        }

        signal_type = signal_map[predicted_class]

        if signal_type == SignalType.NO_ACTION:
            return None

        return self.create_signal(
            signal_type=signal_type,
            symbol=snapshot.symbol,
            price=snapshot.mid_price,
            strength=float(confidence),
            metadata={
                "model": self.model_type,
                "probabilities": {
                    "long": float(probs[0]),
                    "short": float(probs[1]),
                    "no_action": float(probs[2]),
                },
            },
        )

    def on_trade(self, trade: Trade) -> Optional[Signal]:
        """Process trade and update buffer."""
        self._trade_buffer.append(trade)

        # Keep buffer size limited
        if len(self._trade_buffer) > self._buffer_size * 2:
            self._trade_buffer = self._trade_buffer[-self._buffer_size:]

        return None

    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self._trade_buffer = []
        self.feature_extractor._feature_history = []


# =============================================================================
# Training Utilities
# =============================================================================

def prepare_training_data(
    trades: List[Trade],
    orderbook_snapshots: List[OrderBookSnapshot],
    lookahead: int = 10,
    threshold_pct: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data from historical market data.

    Args:
        trades: Historical trades
        orderbook_snapshots: Historical order book snapshots
        lookahead: Number of snapshots to look ahead for labeling
        threshold_pct: Price change threshold for labeling

    Returns:
        X (features), y (labels)
    """
    extractor = FeatureExtractor()

    X = []
    y = []

    # Group trades by timestamp
    trades_by_time = {}
    for trade in trades:
        key = trade.timestamp.replace(microsecond=0)
        if key not in trades_by_time:
            trades_by_time[key] = []
        trades_by_time[key].append(trade)

    for i in range(len(orderbook_snapshots) - lookahead):
        snapshot = orderbook_snapshots[i]
        future_snapshot = orderbook_snapshots[i + lookahead]

        # Get relevant trades
        key = snapshot.timestamp.replace(microsecond=0)
        recent_trades = trades_by_time.get(key, [])

        # Extract features
        features = extractor.extract_features(
            orderbook=snapshot,
            trades=recent_trades,
        )

        # Calculate future return
        if snapshot.mid_price and future_snapshot.mid_price:
            current_price = float(snapshot.mid_price)
            future_price = float(future_snapshot.mid_price)
            return_pct = (future_price - current_price) / current_price * 100

            # Label based on return
            if return_pct > threshold_pct:
                label = 0  # LONG
            elif return_pct < -threshold_pct:
                label = 1  # SHORT
            else:
                label = 2  # NO_ACTION

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "mlp",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    validation_split: float = 0.2,
) -> nn.Module:
    """
    Train the ML model.

    Args:
        X: Feature matrix
        y: Labels
        model_type: "mlp" or "lstm"
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        validation_split: Validation data fraction

    Returns:
        Trained model
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for training")

    # Split data
    n_val = int(len(X) * validation_split)
    X_train, X_val = X[:-n_val], X[-n_val:]
    y_train, y_val = y[:-n_val], y[-n_val:]

    # Create model
    input_size = X.shape[-1]
    if model_type == "lstm":
        model = TradingLSTM(input_size=input_size)
    else:
        model = TradingMLP(input_size=input_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()

        # Shuffle
        perm = torch.randperm(len(X_train_t))
        X_train_t = X_train_t[perm]
        y_train_t = y_train_t[perm]

        total_loss = 0
        n_batches = 0

        for i in range(0, len(X_train_t), batch_size):
            batch_x = X_train_t[i:i + batch_size]
            batch_y = y_train_t[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()

            # Accuracy
            val_preds = torch.argmax(val_outputs, dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model
