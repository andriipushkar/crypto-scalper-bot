"""
Machine Learning Signal Generator

ML-based trading signal generation using various models.
"""
import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import pickle
import json
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from loguru import logger


class SignalType(Enum):
    """Signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class ModelType(Enum):
    """ML model types."""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LSTM = "lstm"
    TRANSFORMER = "transformer"


@dataclass
class MLSignal:
    """ML-generated trading signal."""
    symbol: str
    signal_type: SignalType
    confidence: float
    model_name: str
    features_used: List[str]
    price: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "signal": self.signal_type.value,
            "confidence": self.confidence,
            "model": self.model_name,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
        }


class FeatureExtractor:
    """Extract features from market data."""

    def __init__(self, lookback_periods: List[int] = None):
        self.lookback_periods = lookback_periods or [5, 10, 20, 50, 100]

    def extract_features(self, prices: List[float], volumes: Optional[List[float]] = None) -> Dict[str, float]:
        """Extract features from price/volume data."""
        if not HAS_NUMPY:
            raise ImportError("numpy required for feature extraction")

        prices = np.array(prices)
        features = {}

        # Price features
        features["price_current"] = prices[-1]
        features["price_mean"] = np.mean(prices)
        features["price_std"] = np.std(prices)

        # Returns
        returns = np.diff(prices) / prices[:-1]
        features["return_mean"] = np.mean(returns)
        features["return_std"] = np.std(returns)
        features["return_skew"] = self._skewness(returns)
        features["return_kurtosis"] = self._kurtosis(returns)

        # Moving averages
        for period in self.lookback_periods:
            if len(prices) >= period:
                ma = np.mean(prices[-period:])
                features[f"ma_{period}"] = ma
                features[f"price_to_ma_{period}"] = prices[-1] / ma

        # Momentum
        for period in [5, 10, 20]:
            if len(prices) > period:
                features[f"momentum_{period}"] = prices[-1] / prices[-period-1] - 1

        # Volatility
        for period in [10, 20]:
            if len(returns) >= period:
                features[f"volatility_{period}"] = np.std(returns[-period:])

        # RSI
        if len(prices) >= 14:
            features["rsi_14"] = self._calculate_rsi(prices, 14)

        # MACD
        if len(prices) >= 26:
            ema12 = self._ema(prices, 12)
            ema26 = self._ema(prices, 26)
            macd = ema12 - ema26
            signal = self._ema([macd], 9) if len([macd]) >= 9 else macd
            features["macd"] = macd
            features["macd_signal"] = signal
            features["macd_histogram"] = macd - signal

        # Bollinger Bands
        if len(prices) >= 20:
            ma20 = np.mean(prices[-20:])
            std20 = np.std(prices[-20:])
            features["bb_upper"] = ma20 + 2 * std20
            features["bb_lower"] = ma20 - 2 * std20
            features["bb_position"] = (prices[-1] - features["bb_lower"]) / (features["bb_upper"] - features["bb_lower"])

        # Volume features
        if volumes:
            volumes = np.array(volumes)
            features["volume_current"] = volumes[-1]
            features["volume_mean"] = np.mean(volumes)
            features["volume_ratio"] = volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else 1

        return features

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _ema(self, data: List[float], period: int) -> float:
        """Calculate EMA."""
        data = np.array(data)
        alpha = 2 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness."""
        n = len(data)
        if n < 3:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)

    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis."""
        n = len(data)
        if n < 4:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3


class MLModel:
    """Base ML model interface."""

    def __init__(self, name: str, model_type: ModelType):
        self.name = name
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Train the model."""
        raise NotImplementedError

    def predict(self, features: Dict[str, float]) -> Tuple[SignalType, float]:
        """Make prediction."""
        raise NotImplementedError

    def save(self, path: str):
        """Save model to file."""
        raise NotImplementedError

    def load(self, path: str):
        """Load model from file."""
        raise NotImplementedError


class RandomForestModel(MLModel):
    """Random Forest classifier for trading signals."""

    def __init__(
        self,
        name: str = "rf_model",
        n_estimators: int = 100,
        max_depth: int = 10,
    ):
        super().__init__(name, ModelType.RANDOM_FOREST)

        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Train the model."""
        self.feature_names = feature_names

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Train
        self.model.fit(X_train, y_train)

        # Evaluate
        accuracy = self.model.score(X_test, y_test)
        logger.info(f"Model {self.name} trained with accuracy: {accuracy:.2%}")

        self.is_trained = True
        return accuracy

    def predict(self, features: Dict[str, float]) -> Tuple[SignalType, float]:
        """Make prediction."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        # Prepare features
        X = np.array([[features.get(f, 0) for f in self.feature_names]])
        X_scaled = self.scaler.transform(X)

        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        confidence = max(probabilities)

        signal_map = {0: SignalType.SELL, 1: SignalType.HOLD, 2: SignalType.BUY}
        return signal_map.get(prediction, SignalType.HOLD), confidence

    def save(self, path: str):
        """Save model to file."""
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.is_trained = data["is_trained"]
        logger.info(f"Model loaded from {path}")


class LSTMModel(MLModel):
    """LSTM model for time series prediction."""

    def __init__(
        self,
        name: str = "lstm_model",
        hidden_size: int = 64,
        num_layers: int = 2,
        sequence_length: int = 50,
    ):
        super().__init__(name, ModelType.LSTM)

        if not HAS_TORCH:
            raise ImportError("PyTorch required for LSTM model")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, input_size: int, output_size: int = 3):
        """Build LSTM network."""

        class LSTMNetwork(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers,
                    batch_first=True, dropout=0.2
                )
                self.fc = nn.Linear(hidden_size, output_size)
                self.softmax = nn.Softmax(dim=1)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                out = self.fc(lstm_out[:, -1, :])
                return self.softmax(out)

        self.model = LSTMNetwork(
            input_size, self.hidden_size, self.num_layers, output_size
        ).to(self.device)

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Train the LSTM model."""
        self.feature_names = feature_names
        self._build_model(len(feature_names))

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Train
        epochs = 100
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        self.is_trained = True
        return loss.item()

    def predict(self, features: Dict[str, float]) -> Tuple[SignalType, float]:
        """Make prediction."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        self.model.eval()
        X = np.array([[features.get(f, 0) for f in self.feature_names]])
        X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(X_tensor)
            prediction = torch.argmax(output, dim=1).item()
            confidence = output[0][prediction].item()

        signal_map = {0: SignalType.SELL, 1: SignalType.HOLD, 2: SignalType.BUY}
        return signal_map.get(prediction, SignalType.HOLD), confidence

    def save(self, path: str):
        """Save model."""
        torch.save({
            "model_state": self.model.state_dict(),
            "feature_names": self.feature_names,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }, path)

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.feature_names = checkpoint["feature_names"]
        self._build_model(len(self.feature_names))
        self.model.load_state_dict(checkpoint["model_state"])
        self.is_trained = True


class MLSignalGenerator:
    """Generate trading signals using ML models."""

    def __init__(
        self,
        models: Optional[List[MLModel]] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        confidence_threshold: float = 0.6,
    ):
        self.models = models or []
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.confidence_threshold = confidence_threshold

        # Price history
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}
        self.history_length = 200

        # Signal history
        self.signals: List[MLSignal] = []

        # Callbacks
        self.on_signal: Optional[Callable] = None

    def add_model(self, model: MLModel):
        """Add ML model."""
        self.models.append(model)

    def update_data(self, symbol: str, price: float, volume: float = 0):
        """Update price/volume data."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.volume_history[symbol] = []

        self.price_history[symbol].append(price)
        self.volume_history[symbol].append(volume)

        # Trim history
        if len(self.price_history[symbol]) > self.history_length:
            self.price_history[symbol] = self.price_history[symbol][-self.history_length:]
            self.volume_history[symbol] = self.volume_history[symbol][-self.history_length:]

    def generate_signal(self, symbol: str) -> Optional[MLSignal]:
        """Generate trading signal for symbol."""
        prices = self.price_history.get(symbol, [])
        volumes = self.volume_history.get(symbol, [])

        if len(prices) < 50:  # Need minimum history
            return None

        # Extract features
        features = self.feature_extractor.extract_features(prices, volumes)

        # Get predictions from all models
        predictions = []
        for model in self.models:
            if not model.is_trained:
                continue
            try:
                signal_type, confidence = model.predict(features)
                predictions.append((model.name, signal_type, confidence))
            except Exception as e:
                logger.warning(f"Model {model.name} prediction failed: {e}")

        if not predictions:
            return None

        # Ensemble: vote or average
        signal = self._ensemble_predictions(predictions)

        if signal and signal.confidence >= self.confidence_threshold:
            signal.symbol = symbol
            signal.price = prices[-1]
            signal.features_used = list(features.keys())

            self.signals.append(signal)

            if self.on_signal:
                self.on_signal(signal)

            return signal

        return None

    def _ensemble_predictions(
        self,
        predictions: List[Tuple[str, SignalType, float]],
    ) -> Optional[MLSignal]:
        """Combine predictions from multiple models."""
        if not predictions:
            return None

        # Weighted voting by confidence
        votes = {SignalType.BUY: 0, SignalType.SELL: 0, SignalType.HOLD: 0}
        total_confidence = 0

        for model_name, signal_type, confidence in predictions:
            votes[signal_type] += confidence
            total_confidence += confidence

        # Find winner
        winner = max(votes, key=votes.get)
        avg_confidence = votes[winner] / len(predictions)

        return MLSignal(
            symbol="",
            signal_type=winner,
            confidence=avg_confidence,
            model_name="ensemble",
            features_used=[],
            price=0,
        )

    def get_recent_signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent signals."""
        return [s.to_dict() for s in self.signals[-limit:]]

    async def run_loop(self, symbols: List[str], interval_seconds: float = 1):
        """Run signal generation loop."""
        while True:
            for symbol in symbols:
                signal = self.generate_signal(symbol)
                if signal:
                    logger.info(
                        f"ML Signal: {signal.signal_type.value} {symbol} "
                        f"confidence={signal.confidence:.2%}"
                    )
            await asyncio.sleep(interval_seconds)


def create_training_data(
    prices: List[float],
    volumes: List[float],
    lookahead: int = 5,
    threshold: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Create training data from historical prices."""
    if not HAS_NUMPY:
        raise ImportError("numpy required")

    extractor = FeatureExtractor()
    X_list = []
    y_list = []

    for i in range(100, len(prices) - lookahead):
        # Extract features
        features = extractor.extract_features(
            prices[:i+1],
            volumes[:i+1] if volumes else None
        )

        # Calculate label (future return)
        future_return = (prices[i + lookahead] - prices[i]) / prices[i]

        if future_return > threshold:
            label = 2  # BUY
        elif future_return < -threshold:
            label = 0  # SELL
        else:
            label = 1  # HOLD

        X_list.append(list(features.values()))
        y_list.append(label)

    feature_names = list(features.keys())
    return np.array(X_list), np.array(y_list), feature_names
