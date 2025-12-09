"""
Historical data loader for backtesting.

Loads data from SQLite database or CSV files.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any

import pandas as pd
from loguru import logger

from src.data.models import Trade, OrderBookSnapshot, OrderBookLevel, MarkPrice


@dataclass
class OHLCV:
    """OHLCV bar data."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    trades: int = 0
    buy_volume: Decimal = Decimal("0")
    sell_volume: Decimal = Decimal("0")


class HistoricalDataLoader:
    """
    Load historical data for backtesting.

    Supports:
    - SQLite database (from collect_data.py)
    - CSV files
    - Binance API (for downloading)
    """

    def __init__(self, data_path: str = "data/raw/market_data.db"):
        """
        Initialize data loader.

        Args:
            data_path: Path to SQLite database or data directory
        """
        self.data_path = Path(data_path)
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Connect to database."""
        if self.data_path.suffix == ".db":
            self._conn = sqlite3.connect(str(self.data_path))
            logger.debug(f"Connected to {self.data_path}")

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # =========================================================================
    # Data Info
    # =========================================================================

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols in database."""
        if not self._conn:
            self.connect()

        cursor = self._conn.execute(
            "SELECT DISTINCT symbol FROM trades ORDER BY symbol"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_date_range(self, symbol: str) -> tuple:
        """
        Get available date range for a symbol.

        Returns:
            Tuple of (start_date, end_date)
        """
        if not self._conn:
            self.connect()

        cursor = self._conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM trades WHERE symbol = ?",
            (symbol,)
        )
        row = cursor.fetchone()

        if row and row[0]:
            return (
                datetime.fromisoformat(row[0]),
                datetime.fromisoformat(row[1]),
            )
        return None, None

    def get_trade_count(self, symbol: str, start: datetime = None, end: datetime = None) -> int:
        """Get number of trades for a symbol in date range."""
        if not self._conn:
            self.connect()

        query = "SELECT COUNT(*) FROM trades WHERE symbol = ?"
        params = [symbol]

        if start:
            query += " AND timestamp >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND timestamp <= ?"
            params.append(end.isoformat())

        cursor = self._conn.execute(query, params)
        return cursor.fetchone()[0]

    # =========================================================================
    # Data Loading
    # =========================================================================

    def load_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: int = None,
    ) -> List[Trade]:
        """
        Load trades from database.

        Args:
            symbol: Trading symbol
            start: Start datetime
            end: End datetime
            limit: Maximum number of trades

        Returns:
            List of Trade objects
        """
        if not self._conn:
            self.connect()

        query = """
            SELECT symbol, trade_id, price, quantity, timestamp, is_buyer_maker
            FROM trades
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        """
        params = [symbol, start.isoformat(), end.isoformat()]

        if limit:
            query += f" LIMIT {limit}"

        cursor = self._conn.execute(query, params)

        trades = []
        for row in cursor.fetchall():
            trades.append(Trade(
                symbol=row[0],
                trade_id=row[1],
                price=Decimal(row[2]),
                quantity=Decimal(row[3]),
                timestamp=datetime.fromisoformat(row[4]),
                is_buyer_maker=bool(row[5]),
            ))

        logger.debug(f"Loaded {len(trades)} trades for {symbol}")
        return trades

    def load_trades_iterator(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        batch_size: int = 10000,
    ) -> Iterator[Trade]:
        """
        Load trades as an iterator (memory efficient).

        Args:
            symbol: Trading symbol
            start: Start datetime
            end: End datetime
            batch_size: Number of trades per batch

        Yields:
            Trade objects
        """
        if not self._conn:
            self.connect()

        offset = 0
        while True:
            query = """
                SELECT symbol, trade_id, price, quantity, timestamp, is_buyer_maker
                FROM trades
                WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
                LIMIT ? OFFSET ?
            """
            cursor = self._conn.execute(
                query,
                (symbol, start.isoformat(), end.isoformat(), batch_size, offset)
            )

            rows = cursor.fetchall()
            if not rows:
                break

            for row in rows:
                yield Trade(
                    symbol=row[0],
                    trade_id=row[1],
                    price=Decimal(row[2]),
                    quantity=Decimal(row[3]),
                    timestamp=datetime.fromisoformat(row[4]),
                    is_buyer_maker=bool(row[5]),
                )

            offset += batch_size

    def load_orderbook_snapshots(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: int = None,
    ) -> pd.DataFrame:
        """
        Load order book snapshots as DataFrame.

        Args:
            symbol: Trading symbol
            start: Start datetime
            end: End datetime
            limit: Maximum rows

        Returns:
            DataFrame with order book data
        """
        if not self._conn:
            self.connect()

        query = """
            SELECT timestamp, mid_price, spread_bps, imbalance_5,
                   bid_volume_5, ask_volume_5
            FROM orderbook_snapshots
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        """
        params = [symbol, start.isoformat(), end.isoformat()]

        if limit:
            query += f" LIMIT {limit}"

        df = pd.read_sql_query(query, self._conn, params=params)

        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            for col in ['mid_price', 'spread_bps', 'imbalance_5', 'bid_volume_5', 'ask_volume_5']:
                df[col] = df[col].astype(float)

        logger.debug(f"Loaded {len(df)} order book snapshots for {symbol}")
        return df

    def load_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1s",
    ) -> List[OHLCV]:
        """
        Aggregate trades into OHLCV bars.

        Args:
            symbol: Trading symbol
            start: Start datetime
            end: End datetime
            timeframe: Bar timeframe (1s, 5s, 1m, etc.)

        Returns:
            List of OHLCV bars
        """
        # Parse timeframe
        tf_map = {
            "1s": timedelta(seconds=1),
            "5s": timedelta(seconds=5),
            "10s": timedelta(seconds=10),
            "30s": timedelta(seconds=30),
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
        }

        interval = tf_map.get(timeframe, timedelta(seconds=1))

        # Load trades and aggregate
        trades = self.load_trades(symbol, start, end)

        if not trades:
            return []

        bars = []
        current_bar_start = trades[0].timestamp.replace(microsecond=0)
        current_trades = []

        for trade in trades:
            trade_bar_start = trade.timestamp.replace(microsecond=0)

            # Check if trade belongs to current bar
            if trade_bar_start >= current_bar_start + interval:
                # Finalize current bar
                if current_trades:
                    bars.append(self._create_ohlcv(current_bar_start, current_trades))

                # Start new bar
                current_bar_start = trade_bar_start
                current_trades = [trade]
            else:
                current_trades.append(trade)

        # Don't forget last bar
        if current_trades:
            bars.append(self._create_ohlcv(current_bar_start, current_trades))

        logger.debug(f"Created {len(bars)} {timeframe} bars for {symbol}")
        return bars

    def _create_ohlcv(self, timestamp: datetime, trades: List[Trade]) -> OHLCV:
        """Create OHLCV bar from trades."""
        prices = [t.price for t in trades]
        quantities = [t.quantity for t in trades]

        buy_vol = sum(t.quantity for t in trades if not t.is_buyer_maker)
        sell_vol = sum(t.quantity for t in trades if t.is_buyer_maker)

        return OHLCV(
            timestamp=timestamp,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=sum(quantities),
            trades=len(trades),
            buy_volume=buy_vol,
            sell_volume=sell_vol,
        )

    # =========================================================================
    # Data Export
    # =========================================================================

    def export_to_csv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        output_path: str,
        timeframe: str = "1s",
    ) -> None:
        """
        Export data to CSV file.

        Args:
            symbol: Trading symbol
            start: Start datetime
            end: End datetime
            output_path: Output file path
            timeframe: Bar timeframe
        """
        bars = self.load_ohlcv(symbol, start, end, timeframe)

        if not bars:
            logger.warning("No data to export")
            return

        df = pd.DataFrame([
            {
                "timestamp": b.timestamp,
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": float(b.volume),
                "trades": b.trades,
                "buy_volume": float(b.buy_volume),
                "sell_volume": float(b.sell_volume),
            }
            for b in bars
        ])

        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} bars to {output_path}")
