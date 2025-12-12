"""
Trade history database storage.

Stores completed trades (positions) with P&L for analysis and history.
Uses SQLite for persistent storage.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from loguru import logger


class TradeHistoryDB:
    """
    SQLite storage for trade history.

    Stores completed trades with full P&L information.
    """

    def __init__(self, database_path: str = "data/trade_history.db"):
        """
        Initialize trade history database.

        Args:
            database_path: Path to SQLite database
        """
        self.database_path = Path(database_path)
        self._db: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Connect to database and create tables."""
        logger.info(f"Connecting to trade history DB: {self.database_path}")

        # Ensure directory exists
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self._db = sqlite3.connect(str(self.database_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row

        # Create tables
        self._create_tables()

        logger.info("Trade history DB connected")

    def close(self) -> None:
        """Close database connection."""
        if self._db:
            self._db.close()
            self._db = None
            logger.info("Trade history DB closed")

    def _create_tables(self) -> None:
        """Create database tables."""
        self._db.executescript("""
            -- Completed trades table
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity REAL NOT NULL,
                pnl REAL NOT NULL,
                pnl_pct REAL NOT NULL,
                strategy TEXT DEFAULT 'manual',
                exchange TEXT DEFAULT 'binance',
                leverage INTEGER DEFAULT 1,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                fees REAL DEFAULT 0,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            -- Indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_trade_history_symbol
            ON trade_history(symbol);

            CREATE INDEX IF NOT EXISTS idx_trade_history_exit_time
            ON trade_history(exit_time DESC);

            CREATE INDEX IF NOT EXISTS idx_trade_history_strategy
            ON trade_history(strategy);

            CREATE INDEX IF NOT EXISTS idx_trade_history_pnl
            ON trade_history(pnl);

            -- Daily P&L summary table (for fast stats)
            CREATE TABLE IF NOT EXISTS daily_pnl (
                date TEXT PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                gross_profit REAL DEFAULT 0,
                gross_loss REAL DEFAULT 0,
                net_pnl REAL DEFAULT 0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_daily_pnl_date
            ON daily_pnl(date DESC);
        """)
        self._db.commit()

    def save_trade(self, trade: Dict[str, Any]) -> int:
        """
        Save a completed trade to database.

        Args:
            trade: Trade dictionary with fields:
                - id or trade_id: Unique trade ID
                - symbol: Trading pair
                - side: BUY or SELL
                - entry_price: Entry price
                - exit_price: Exit price
                - quantity: Trade size
                - pnl: Profit/Loss in USD
                - pnl_pct: Profit/Loss percentage
                - strategy: Strategy name (optional)
                - entry_time: Entry timestamp
                - exit_time: Exit timestamp

        Returns:
            Database row ID
        """
        if not self._db:
            self.connect()

        trade_id = trade.get('trade_id') or trade.get('id') or f"t_{datetime.utcnow().timestamp()}"

        cursor = self._db.execute("""
            INSERT OR REPLACE INTO trade_history
            (trade_id, symbol, side, entry_price, exit_price, quantity,
             pnl, pnl_pct, strategy, exchange, leverage,
             entry_time, exit_time, stop_loss, take_profit, fees, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id,
            trade.get('symbol', ''),
            trade.get('side', ''),
            trade.get('entry_price', 0),
            trade.get('exit_price', 0),
            trade.get('quantity', 0),
            trade.get('pnl', 0),
            trade.get('pnl_pct', 0),
            trade.get('strategy', 'manual'),
            trade.get('exchange', 'binance'),
            trade.get('leverage', 1),
            trade.get('entry_time', datetime.utcnow().isoformat()),
            trade.get('exit_time', datetime.utcnow().isoformat()),
            trade.get('stop_loss'),
            trade.get('take_profit'),
            trade.get('fees', 0),
            trade.get('notes'),
        ))

        self._db.commit()

        # Update daily P&L
        self._update_daily_pnl(trade)

        logger.debug(f"Trade saved: {trade_id} {trade.get('symbol')} P&L: ${trade.get('pnl', 0):.2f}")

        return cursor.lastrowid

    def _update_daily_pnl(self, trade: Dict[str, Any]) -> None:
        """Update daily P&L summary."""
        exit_time = trade.get('exit_time', '')
        if isinstance(exit_time, str):
            date = exit_time[:10]  # Get YYYY-MM-DD
        else:
            date = exit_time.strftime('%Y-%m-%d') if exit_time else datetime.utcnow().strftime('%Y-%m-%d')

        pnl = trade.get('pnl', 0)
        is_win = pnl > 0

        # Insert or update daily record
        self._db.execute("""
            INSERT INTO daily_pnl (date, total_trades, winning_trades, losing_trades,
                                   gross_profit, gross_loss, net_pnl, updated_at)
            VALUES (?, 1, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_trades = total_trades + 1,
                winning_trades = winning_trades + excluded.winning_trades,
                losing_trades = losing_trades + excluded.losing_trades,
                gross_profit = gross_profit + excluded.gross_profit,
                gross_loss = gross_loss + excluded.gross_loss,
                net_pnl = net_pnl + excluded.net_pnl,
                updated_at = excluded.updated_at
        """, (
            date,
            1 if is_win else 0,
            0 if is_win else 1,
            pnl if pnl > 0 else 0,
            abs(pnl) if pnl < 0 else 0,
            pnl,
            datetime.utcnow().isoformat(),
        ))
        self._db.commit()

    def get_trades(
        self,
        limit: int = 50,
        offset: int = 0,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get trade history with filters.

        Args:
            limit: Max trades to return
            offset: Skip first N trades
            symbol: Filter by symbol
            strategy: Filter by strategy
            start_date: Filter from date (YYYY-MM-DD)
            end_date: Filter to date (YYYY-MM-DD)

        Returns:
            List of trade dictionaries
        """
        if not self._db:
            self.connect()

        query = "SELECT * FROM trade_history WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)

        if start_date:
            query += " AND exit_time >= ?"
            params.append(start_date)

        if end_date:
            query += " AND exit_time <= ?"
            params.append(end_date + "T23:59:59")

        query += " ORDER BY exit_time DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = self._db.execute(query, params)
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_trade_count(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> int:
        """Get total trade count with optional filters."""
        if not self._db:
            self.connect()

        query = "SELECT COUNT(*) FROM trade_history WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)

        cursor = self._db.execute(query, params)
        return cursor.fetchone()[0]

    def get_stats(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get overall trading statistics.

        Returns:
            Statistics dictionary with win rate, P&L, etc.
        """
        if not self._db:
            self.connect()

        query = """
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                SUM(pnl) as total_pnl,
                SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
                SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as gross_loss,
                AVG(pnl) as avg_pnl,
                MAX(pnl) as best_trade,
                MIN(pnl) as worst_trade,
                AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss
            FROM trade_history
            WHERE 1=1
        """
        params = []

        if start_date:
            query += " AND exit_time >= ?"
            params.append(start_date)

        if end_date:
            query += " AND exit_time <= ?"
            params.append(end_date + "T23:59:59")

        cursor = self._db.execute(query, params)
        row = cursor.fetchone()

        total = row['total_trades'] or 0
        wins = row['winning_trades'] or 0
        losses = row['losing_trades'] or 0
        gross_profit = row['gross_profit'] or 0
        gross_loss = row['gross_loss'] or 0

        return {
            'total_trades': total,
            'winning_trades': wins,
            'losing_trades': losses,
            'win_rate': (wins / total * 100) if total > 0 else 0,
            'total_pnl': row['total_pnl'] or 0,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': (gross_profit / gross_loss) if gross_loss > 0 else 0,
            'avg_pnl': row['avg_pnl'] or 0,
            'best_trade': row['best_trade'] or 0,
            'worst_trade': row['worst_trade'] or 0,
            'avg_win': row['avg_win'] or 0,
            'avg_loss': row['avg_loss'] or 0,
        }

    def get_daily_pnl(
        self,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get daily P&L for last N days.

        Args:
            days: Number of days

        Returns:
            List of daily P&L dictionaries
        """
        if not self._db:
            self.connect()

        cursor = self._db.execute("""
            SELECT * FROM daily_pnl
            ORDER BY date DESC
            LIMIT ?
        """, (days,))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_pnl_by_symbol(self) -> List[Dict[str, Any]]:
        """Get P&L breakdown by symbol."""
        if not self._db:
            self.connect()

        cursor = self._db.execute("""
            SELECT
                symbol,
                COUNT(*) as trades,
                SUM(pnl) as total_pnl,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                AVG(pnl) as avg_pnl
            FROM trade_history
            GROUP BY symbol
            ORDER BY total_pnl DESC
        """)

        return [dict(row) for row in cursor.fetchall()]

    def get_pnl_by_strategy(self) -> List[Dict[str, Any]]:
        """Get P&L breakdown by strategy."""
        if not self._db:
            self.connect()

        cursor = self._db.execute("""
            SELECT
                strategy,
                COUNT(*) as trades,
                SUM(pnl) as total_pnl,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                AVG(pnl) as avg_pnl
            FROM trade_history
            GROUP BY strategy
            ORDER BY total_pnl DESC
        """)

        return [dict(row) for row in cursor.fetchall()]

    def delete_trade(self, trade_id: str) -> bool:
        """Delete a trade by ID."""
        if not self._db:
            return False

        cursor = self._db.execute(
            "DELETE FROM trade_history WHERE trade_id = ?",
            (trade_id,)
        )
        self._db.commit()
        return cursor.rowcount > 0

    def clear_all(self) -> int:
        """Clear all trade history. Use with caution!"""
        if not self._db:
            return 0

        cursor = self._db.execute("DELETE FROM trade_history")
        self._db.execute("DELETE FROM daily_pnl")
        self._db.commit()

        logger.warning(f"Cleared {cursor.rowcount} trades from history")
        return cursor.rowcount


# Global instance
trade_history_db = TradeHistoryDB()
