"""
Data storage for market data.

Stores trades, order book snapshots, and other market data
to SQLite for later analysis.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional, List, Dict, Any

import aiosqlite
from loguru import logger

from src.data.models import (
    Trade,
    OrderBookSnapshot,
    MarkPrice,
    TradeEvent,
    OrderBookEvent,
    MarkPriceEvent,
)


class MarketDataStorage:
    """
    Async SQLite storage for market data.

    Buffers data and flushes periodically for performance.
    """

    def __init__(
        self,
        database_path: str = "data/raw/market_data.db",
        flush_interval: int = 60,
        buffer_size: int = 1000,
    ):
        """
        Initialize storage.

        Args:
            database_path: Path to SQLite database
            flush_interval: Seconds between auto-flushes
            buffer_size: Max items in buffer before forced flush
        """
        self.database_path = Path(database_path)
        self.flush_interval = flush_interval
        self.buffer_size = buffer_size

        self._db: Optional[aiosqlite.Connection] = None
        self._running = False

        # Buffers for batch inserts
        self._trade_buffer: List[Trade] = []
        self._orderbook_buffer: List[Dict] = []
        self._mark_price_buffer: List[MarkPrice] = []

        # Stats
        self._trades_stored = 0
        self._orderbooks_stored = 0
        self._mark_prices_stored = 0
        self._last_flush_time: Optional[datetime] = None

    @property
    def stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "trades_stored": self._trades_stored,
            "orderbooks_stored": self._orderbooks_stored,
            "mark_prices_stored": self._mark_prices_stored,
            "last_flush_time": self._last_flush_time,
            "buffer_sizes": {
                "trades": len(self._trade_buffer),
                "orderbooks": len(self._orderbook_buffer),
                "mark_prices": len(self._mark_price_buffer),
            },
        }

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start storage service."""
        logger.info(f"Starting storage: {self.database_path}")

        # Ensure directory exists
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self._db = await aiosqlite.connect(str(self.database_path))
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")

        # Create tables
        await self._create_tables()

        # Start flush loop
        self._running = True
        asyncio.create_task(self._flush_loop())

        logger.info("Storage started")

    async def stop(self) -> None:
        """Stop storage service."""
        logger.info("Stopping storage")
        self._running = False

        # Final flush
        await self.flush()

        # Close database
        if self._db:
            await self._db.close()
            self._db = None

        logger.info("Storage stopped")

    async def _create_tables(self) -> None:
        """Create database tables with optimized indexes."""
        await self._db.executescript("""
            -- Trades table
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                trade_id INTEGER NOT NULL,
                price TEXT NOT NULL,
                quantity TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                is_buyer_maker INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, trade_id)
            );

            -- Primary query index: symbol + timestamp for date range queries
            CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp
            ON trades(symbol, timestamp);

            -- Index for trade_id lookups
            CREATE INDEX IF NOT EXISTS idx_trades_trade_id
            ON trades(trade_id);

            -- Covering index for common aggregation queries
            CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp_price
            ON trades(symbol, timestamp, price, quantity);

            -- Order book snapshots table
            CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                last_update_id INTEGER NOT NULL,
                best_bid_price TEXT,
                best_bid_qty TEXT,
                best_ask_price TEXT,
                best_ask_qty TEXT,
                mid_price TEXT,
                spread_bps TEXT,
                bid_volume_5 TEXT,
                ask_volume_5 TEXT,
                imbalance_5 TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            -- Primary query index
            CREATE INDEX IF NOT EXISTS idx_orderbook_symbol_timestamp
            ON orderbook_snapshots(symbol, timestamp);

            -- Index for imbalance analysis
            CREATE INDEX IF NOT EXISTS idx_orderbook_imbalance
            ON orderbook_snapshots(symbol, timestamp, imbalance_5);

            -- Mark price / funding rate table
            CREATE TABLE IF NOT EXISTS mark_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                mark_price TEXT NOT NULL,
                index_price TEXT NOT NULL,
                funding_rate TEXT NOT NULL,
                next_funding_time TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            -- Primary query index
            CREATE INDEX IF NOT EXISTS idx_mark_prices_symbol_timestamp
            ON mark_prices(symbol, timestamp);

            -- Index for funding rate analysis
            CREATE INDEX IF NOT EXISTS idx_mark_prices_funding
            ON mark_prices(symbol, next_funding_time, funding_rate);
        """)
        await self._db.commit()

    async def optimize_database(self) -> Dict[str, Any]:
        """
        Run database optimizations.

        Call periodically (e.g., daily) to maintain performance.

        Returns:
            Optimization statistics
        """
        if not self._db:
            return {"error": "Database not connected"}

        stats = {}

        # Analyze tables for query optimizer
        await self._db.execute("ANALYZE trades")
        await self._db.execute("ANALYZE orderbook_snapshots")
        await self._db.execute("ANALYZE mark_prices")
        stats["analyzed"] = True

        # Vacuum to reclaim space and defragment
        await self._db.execute("VACUUM")
        stats["vacuumed"] = True

        # Get database stats
        cursor = await self._db.execute(
            "SELECT name, SUM(pgsize) as size FROM dbstat GROUP BY name"
        )
        table_sizes = {}
        async for row in cursor:
            table_sizes[row[0]] = row[1]
        stats["table_sizes"] = table_sizes

        # Get row counts
        for table in ["trades", "orderbook_snapshots", "mark_prices"]:
            cursor = await self._db.execute(f"SELECT COUNT(*) FROM {table}")
            row = await cursor.fetchone()
            stats[f"{table}_count"] = row[0]

        logger.info(f"Database optimized: {stats}")
        return stats

    async def cleanup_old_data(self, retention_days: int = 30) -> int:
        """
        Remove data older than retention period.

        Args:
            retention_days: Days to retain data

        Returns:
            Number of rows deleted
        """
        if not self._db:
            return 0

        cutoff = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()
        total_deleted = 0

        for table in ["trades", "orderbook_snapshots", "mark_prices"]:
            cursor = await self._db.execute(
                f"DELETE FROM {table} WHERE timestamp < ?",
                (cutoff,)
            )
            total_deleted += cursor.rowcount

        await self._db.commit()
        logger.info(f"Cleaned up {total_deleted} old records (>{retention_days} days)")

        return total_deleted

    # =========================================================================
    # Data Ingestion
    # =========================================================================

    async def store_trade(self, event: TradeEvent) -> None:
        """Store trade from event."""
        self._trade_buffer.append(event.trade)

        if len(self._trade_buffer) >= self.buffer_size:
            await self.flush()

    async def store_orderbook(self, event: OrderBookEvent) -> None:
        """Store order book snapshot from event."""
        snapshot = event.snapshot

        record = {
            "symbol": snapshot.symbol,
            "timestamp": snapshot.timestamp.isoformat(),
            "last_update_id": snapshot.last_update_id,
            "best_bid_price": str(snapshot.best_bid.price) if snapshot.best_bid else None,
            "best_bid_qty": str(snapshot.best_bid.quantity) if snapshot.best_bid else None,
            "best_ask_price": str(snapshot.best_ask.price) if snapshot.best_ask else None,
            "best_ask_qty": str(snapshot.best_ask.quantity) if snapshot.best_ask else None,
            "mid_price": str(snapshot.mid_price) if snapshot.mid_price else None,
            "spread_bps": str(snapshot.spread_bps) if snapshot.spread_bps else None,
            "bid_volume_5": str(snapshot.bid_volume(5)),
            "ask_volume_5": str(snapshot.ask_volume(5)),
            "imbalance_5": str(snapshot.imbalance(5)),
        }

        self._orderbook_buffer.append(record)

        if len(self._orderbook_buffer) >= self.buffer_size:
            await self.flush()

    async def store_mark_price(self, event: MarkPriceEvent) -> None:
        """Store mark price from event."""
        self._mark_price_buffer.append(event.mark_price)

        if len(self._mark_price_buffer) >= self.buffer_size:
            await self.flush()

    # =========================================================================
    # Flushing
    # =========================================================================

    async def flush(self) -> None:
        """Flush all buffers to database."""
        if not self._db:
            return

        try:
            await self._flush_trades()
            await self._flush_orderbooks()
            await self._flush_mark_prices()
            await self._db.commit()

            self._last_flush_time = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error flushing data: {e}")

    async def _flush_trades(self) -> None:
        """Flush trade buffer."""
        if not self._trade_buffer:
            return

        trades = self._trade_buffer
        self._trade_buffer = []

        await self._db.executemany(
            """
            INSERT OR IGNORE INTO trades
            (symbol, trade_id, price, quantity, timestamp, is_buyer_maker)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    t.symbol,
                    t.trade_id,
                    str(t.price),
                    str(t.quantity),
                    t.timestamp.isoformat(),
                    1 if t.is_buyer_maker else 0,
                )
                for t in trades
            ],
        )

        self._trades_stored += len(trades)
        logger.debug(f"Flushed {len(trades)} trades")

    async def _flush_orderbooks(self) -> None:
        """Flush order book buffer."""
        if not self._orderbook_buffer:
            return

        records = self._orderbook_buffer
        self._orderbook_buffer = []

        await self._db.executemany(
            """
            INSERT INTO orderbook_snapshots
            (symbol, timestamp, last_update_id, best_bid_price, best_bid_qty,
             best_ask_price, best_ask_qty, mid_price, spread_bps,
             bid_volume_5, ask_volume_5, imbalance_5)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["symbol"],
                    r["timestamp"],
                    r["last_update_id"],
                    r["best_bid_price"],
                    r["best_bid_qty"],
                    r["best_ask_price"],
                    r["best_ask_qty"],
                    r["mid_price"],
                    r["spread_bps"],
                    r["bid_volume_5"],
                    r["ask_volume_5"],
                    r["imbalance_5"],
                )
                for r in records
            ],
        )

        self._orderbooks_stored += len(records)
        logger.debug(f"Flushed {len(records)} order book snapshots")

    async def _flush_mark_prices(self) -> None:
        """Flush mark price buffer."""
        if not self._mark_price_buffer:
            return

        prices = self._mark_price_buffer
        self._mark_price_buffer = []

        await self._db.executemany(
            """
            INSERT INTO mark_prices
            (symbol, mark_price, index_price, funding_rate, next_funding_time, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    p.symbol,
                    str(p.mark_price),
                    str(p.index_price),
                    str(p.funding_rate),
                    p.next_funding_time.isoformat(),
                    p.timestamp.isoformat(),
                )
                for p in prices
            ],
        )

        self._mark_prices_stored += len(prices)
        logger.debug(f"Flushed {len(prices)} mark prices")

    async def _flush_loop(self) -> None:
        """Background loop for periodic flushing."""
        while self._running:
            await asyncio.sleep(self.flush_interval)
            if self._running:
                await self.flush()

    # =========================================================================
    # Data Retrieval (for analysis)
    # =========================================================================

    async def get_trades(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 10000,
    ) -> List[Dict]:
        """
        Get trades for a symbol in time range.

        Args:
            symbol: Trading symbol
            start: Start time
            end: End time (default: now)
            limit: Max records to return

        Returns:
            List of trade dictionaries
        """
        if not self._db:
            return []

        end = end or datetime.utcnow()

        cursor = await self._db.execute(
            """
            SELECT symbol, trade_id, price, quantity, timestamp, is_buyer_maker
            FROM trades
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (symbol, start.isoformat(), end.isoformat(), limit),
        )

        rows = await cursor.fetchall()
        return [
            {
                "symbol": row[0],
                "trade_id": row[1],
                "price": Decimal(row[2]),
                "quantity": Decimal(row[3]),
                "timestamp": datetime.fromisoformat(row[4]),
                "is_buyer_maker": bool(row[5]),
            }
            for row in rows
        ]

    async def get_orderbook_snapshots(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: int = 10000,
    ) -> List[Dict]:
        """Get order book snapshots for a symbol in time range."""
        if not self._db:
            return []

        end = end or datetime.utcnow()

        cursor = await self._db.execute(
            """
            SELECT symbol, timestamp, mid_price, spread_bps, imbalance_5
            FROM orderbook_snapshots
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (symbol, start.isoformat(), end.isoformat(), limit),
        )

        rows = await cursor.fetchall()
        return [
            {
                "symbol": row[0],
                "timestamp": datetime.fromisoformat(row[1]),
                "mid_price": Decimal(row[2]) if row[2] else None,
                "spread_bps": Decimal(row[3]) if row[3] else None,
                "imbalance_5": Decimal(row[4]) if row[4] else None,
            }
            for row in rows
        ]

    # =========================================================================
    # Maintenance
    # =========================================================================

    async def cleanup_old_data(self, retention_days: int = 30) -> int:
        """
        Delete data older than retention period.

        Args:
            retention_days: Days to keep data

        Returns:
            Number of records deleted
        """
        if not self._db:
            return 0

        cutoff = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()
        total_deleted = 0

        for table in ["trades", "orderbook_snapshots", "mark_prices"]:
            cursor = await self._db.execute(
                f"DELETE FROM {table} WHERE timestamp < ?",
                (cutoff,),
            )
            total_deleted += cursor.rowcount

        await self._db.commit()
        logger.info(f"Cleaned up {total_deleted} old records")

        return total_deleted

    async def vacuum(self) -> None:
        """Reclaim unused space in database."""
        if self._db:
            await self._db.execute("VACUUM")
            logger.info("Database vacuumed")
