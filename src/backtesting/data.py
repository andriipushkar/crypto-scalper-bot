"""
Historical Data Management

Load, store, and manage historical market data for backtesting.
"""
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Iterator
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
from pathlib import Path
import json
import csv

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from loguru import logger


@dataclass
class OHLCV:
    """Single OHLCV candle."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OHLCV":
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts / 1000)  # Assume milliseconds

        return cls(
            timestamp=ts,
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data["volume"]),
        )

    @classmethod
    def from_binance(cls, kline: List) -> "OHLCV":
        """Parse from Binance kline format."""
        return cls(
            timestamp=datetime.fromtimestamp(kline[0] / 1000),
            open=float(kline[1]),
            high=float(kline[2]),
            low=float(kline[3]),
            close=float(kline[4]),
            volume=float(kline[5]),
        )


class HistoricalDataLoader:
    """Load and manage historical market data."""

    def __init__(self, data_dir: str = "data/historical"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, List[OHLCV]] = {}

    def get_cache_path(self, symbol: str, timeframe: str) -> Path:
        """Get cache file path for symbol/timeframe."""
        return self.data_dir / f"{symbol}_{timeframe}.json"

    async def fetch_from_exchange(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        exchange: str = "binance",
    ) -> List[OHLCV]:
        """Fetch historical data from exchange."""
        import aiohttp

        all_candles = []
        current_start = start_time

        # Timeframe to milliseconds
        tf_ms = {
            "1m": 60000, "5m": 300000, "15m": 900000,
            "1h": 3600000, "4h": 14400000, "1d": 86400000,
        }
        interval_ms = tf_ms.get(timeframe, 60000)

        async with aiohttp.ClientSession() as session:
            while current_start < end_time:
                url = "https://api.binance.com/api/v3/klines"
                params = {
                    "symbol": symbol,
                    "interval": timeframe,
                    "startTime": int(current_start.timestamp() * 1000),
                    "endTime": int(end_time.timestamp() * 1000),
                    "limit": 1000,
                }

                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Failed to fetch data: {response.status}")
                        break

                    data = await response.json()
                    if not data:
                        break

                    for kline in data:
                        candle = OHLCV.from_binance(kline)
                        all_candles.append(candle)

                    # Move to next batch
                    last_ts = data[-1][0]
                    current_start = datetime.fromtimestamp((last_ts + interval_ms) / 1000)

                    # Rate limit
                    await asyncio.sleep(0.1)

        logger.info(f"Fetched {len(all_candles)} candles for {symbol} {timeframe}")
        return all_candles

    def save_to_cache(self, symbol: str, timeframe: str, candles: List[OHLCV]):
        """Save candles to cache file."""
        path = self.get_cache_path(symbol, timeframe)

        data = [c.to_dict() for c in candles]
        with open(path, "w") as f:
            json.dump(data, f)

        # Update memory cache
        cache_key = f"{symbol}_{timeframe}"
        self._cache[cache_key] = candles

        logger.info(f"Saved {len(candles)} candles to {path}")

    def load_from_cache(self, symbol: str, timeframe: str) -> Optional[List[OHLCV]]:
        """Load candles from cache file."""
        cache_key = f"{symbol}_{timeframe}"

        # Check memory cache first
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self.get_cache_path(symbol, timeframe)
        if not path.exists():
            return None

        with open(path, "r") as f:
            data = json.load(f)

        candles = [OHLCV.from_dict(d) for d in data]
        self._cache[cache_key] = candles

        logger.info(f"Loaded {len(candles)} candles from {path}")
        return candles

    async def get_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        use_cache: bool = True,
    ) -> List[OHLCV]:
        """Get historical data, fetching from exchange if needed."""
        if use_cache:
            cached = self.load_from_cache(symbol, timeframe)
            if cached:
                # Filter by time range
                return [
                    c for c in cached
                    if start_time <= c.timestamp <= end_time
                ]

        # Fetch from exchange
        candles = await self.fetch_from_exchange(
            symbol, timeframe, start_time, end_time
        )

        if use_cache and candles:
            self.save_to_cache(symbol, timeframe, candles)

        return candles

    def load_from_csv(self, filepath: str) -> List[OHLCV]:
        """Load data from CSV file."""
        candles = []

        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                candle = OHLCV(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
                candles.append(candle)

        return candles

    def to_dataframe(self, candles: List[OHLCV]) -> "pd.DataFrame":
        """Convert candles to pandas DataFrame."""
        if not HAS_PANDAS:
            raise ImportError("pandas not installed")

        data = [c.to_dict() for c in candles]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df

    def resample(
        self,
        candles: List[OHLCV],
        target_timeframe: str,
    ) -> List[OHLCV]:
        """Resample candles to different timeframe."""
        if not HAS_PANDAS:
            raise ImportError("pandas not installed")

        df = self.to_dataframe(candles)

        # Timeframe mapping
        tf_map = {
            "1m": "1T", "5m": "5T", "15m": "15T",
            "1h": "1H", "4h": "4H", "1d": "1D",
        }
        rule = tf_map.get(target_timeframe, "1H")

        resampled = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        result = []
        for ts, row in resampled.iterrows():
            result.append(OHLCV(
                timestamp=ts.to_pydatetime(),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            ))

        return result


class DataIterator:
    """Iterator for streaming historical data during backtest."""

    def __init__(
        self,
        candles: List[OHLCV],
        warmup_period: int = 0,
    ):
        self.candles = candles
        self.warmup_period = warmup_period
        self.current_index = warmup_period

    def __iter__(self) -> Iterator[Tuple[OHLCV, List[OHLCV]]]:
        """Iterate through candles, yielding current and historical."""
        for i in range(self.warmup_period, len(self.candles)):
            current = self.candles[i]
            history = self.candles[:i]
            yield current, history

    def reset(self):
        """Reset iterator to beginning."""
        self.current_index = self.warmup_period

    @property
    def progress(self) -> float:
        """Get progress percentage."""
        total = len(self.candles) - self.warmup_period
        current = self.current_index - self.warmup_period
        return (current / total) * 100 if total > 0 else 0
