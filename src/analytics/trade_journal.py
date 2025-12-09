"""
Trade Journal for tracking and analyzing trades.

Features:
- Trade logging with notes and tags
- Performance analysis by various dimensions
- Trade review and learning
- Export capabilities
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional

from loguru import logger


# =============================================================================
# Models
# =============================================================================

class TradeOutcome(Enum):
    """Trade outcome classification."""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


class TradeQuality(Enum):
    """Trade execution quality."""
    EXCELLENT = "excellent"  # Perfect execution
    GOOD = "good"           # Good entry/exit
    AVERAGE = "average"     # Room for improvement
    POOR = "poor"          # Mistakes made
    TERRIBLE = "terrible"   # Major errors


class EmotionalState(Enum):
    """Trader emotional state."""
    CALM = "calm"
    CONFIDENT = "confident"
    ANXIOUS = "anxious"
    FEARFUL = "fearful"
    GREEDY = "greedy"
    FRUSTRATED = "frustrated"
    EUPHORIC = "euphoric"


@dataclass
class TradeEntry:
    """A single trade journal entry."""
    # Identifiers
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    order_id: Optional[str] = None

    # Trade details
    symbol: str = ""
    side: str = ""  # LONG or SHORT
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    quantity: float = 0.0
    leverage: int = 1

    # Results
    pnl: float = 0.0
    pnl_percent: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0
    outcome: TradeOutcome = TradeOutcome.BREAKEVEN

    # Strategy
    strategy: str = ""
    signal_strength: float = 0.0
    timeframe: str = ""

    # Analysis
    setup_type: str = ""  # e.g., "breakout", "reversal", "trend"
    market_condition: str = ""  # e.g., "trending", "ranging", "volatile"
    quality: TradeQuality = TradeQuality.AVERAGE

    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_planned: Optional[float] = None
    risk_reward_actual: Optional[float] = None
    position_size_percent: float = 0.0

    # Notes
    entry_reason: str = ""
    exit_reason: str = ""
    notes: str = ""
    lessons: str = ""
    mistakes: str = ""

    # Tags
    tags: List[str] = field(default_factory=list)

    # Emotional tracking
    emotional_state_entry: Optional[EmotionalState] = None
    emotional_state_exit: Optional[EmotionalState] = None

    # Screenshots/charts (paths)
    chart_entry: Optional[str] = None
    chart_exit: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert enums
        data["outcome"] = self.outcome.value if self.outcome else None
        data["quality"] = self.quality.value if self.quality else None
        data["emotional_state_entry"] = self.emotional_state_entry.value if self.emotional_state_entry else None
        data["emotional_state_exit"] = self.emotional_state_exit.value if self.emotional_state_exit else None
        # Convert datetimes
        data["entry_time"] = self.entry_time.isoformat() if self.entry_time else None
        data["exit_time"] = self.exit_time.isoformat() if self.exit_time else None
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeEntry":
        """Create from dictionary."""
        # Convert enums
        if data.get("outcome"):
            data["outcome"] = TradeOutcome(data["outcome"])
        if data.get("quality"):
            data["quality"] = TradeQuality(data["quality"])
        if data.get("emotional_state_entry"):
            data["emotional_state_entry"] = EmotionalState(data["emotional_state_entry"])
        if data.get("emotional_state_exit"):
            data["emotional_state_exit"] = EmotionalState(data["emotional_state_exit"])
        # Convert datetimes
        for field_name in ["entry_time", "exit_time", "created_at", "updated_at"]:
            if data.get(field_name) and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name])
        return cls(**data)


# =============================================================================
# Trade Journal
# =============================================================================

class TradeJournal:
    """
    Trade journal for logging and analyzing trades.

    Usage:
        journal = TradeJournal("data/journal.db")

        # Log a trade
        entry = journal.create_entry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000,
            quantity=0.01,
            strategy="orderbook_imbalance",
        )

        # Update when closed
        journal.close_trade(
            entry.trade_id,
            exit_price=51000,
            pnl=10.0,
            notes="Clean breakout trade",
        )

        # Analyze
        stats = journal.get_statistics()
        print(f"Win rate: {stats['win_rate']:.1%}")
    """

    def __init__(self, db_path: str = "data/trade_journal.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    order_id TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    leverage INTEGER DEFAULT 1,
                    pnl REAL DEFAULT 0,
                    pnl_percent REAL DEFAULT 0,
                    commission REAL DEFAULT 0,
                    net_pnl REAL DEFAULT 0,
                    outcome TEXT,
                    strategy TEXT,
                    signal_strength REAL,
                    timeframe TEXT,
                    setup_type TEXT,
                    market_condition TEXT,
                    quality TEXT,
                    stop_loss REAL,
                    take_profit REAL,
                    risk_reward_planned REAL,
                    risk_reward_actual REAL,
                    position_size_percent REAL,
                    entry_reason TEXT,
                    exit_reason TEXT,
                    notes TEXT,
                    lessons TEXT,
                    mistakes TEXT,
                    tags TEXT,
                    emotional_state_entry TEXT,
                    emotional_state_exit TEXT,
                    chart_entry TEXT,
                    chart_exit TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)
            """)

    def create_entry(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        **kwargs,
    ) -> TradeEntry:
        """Create a new trade entry."""
        entry = TradeEntry(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            **kwargs,
        )

        self._save_entry(entry)
        logger.info(f"Trade journal entry created: {entry.trade_id}")
        return entry

    def _save_entry(self, entry: TradeEntry) -> None:
        """Save entry to database."""
        data = entry.to_dict()

        # Convert tags list to JSON string
        data["tags"] = json.dumps(data.get("tags", []))

        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO trades ({columns}) VALUES ({placeholders})",
                list(data.values()),
            )

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        pnl: float,
        **kwargs,
    ) -> Optional[TradeEntry]:
        """Close a trade and update the entry."""
        entry = self.get_entry(trade_id)
        if not entry:
            logger.warning(f"Trade not found: {trade_id}")
            return None

        entry.exit_time = datetime.now()
        entry.exit_price = exit_price
        entry.pnl = pnl
        entry.net_pnl = pnl - entry.commission
        entry.updated_at = datetime.now()

        # Calculate P&L percent
        if entry.entry_price > 0:
            if entry.side == "LONG":
                entry.pnl_percent = ((exit_price - entry.entry_price) / entry.entry_price) * 100
            else:
                entry.pnl_percent = ((entry.entry_price - exit_price) / entry.entry_price) * 100
            entry.pnl_percent *= entry.leverage

        # Determine outcome
        if pnl > 0:
            entry.outcome = TradeOutcome.WIN
        elif pnl < 0:
            entry.outcome = TradeOutcome.LOSS
        else:
            entry.outcome = TradeOutcome.BREAKEVEN

        # Calculate actual R:R
        if entry.stop_loss and entry.entry_price:
            risk = abs(entry.entry_price - entry.stop_loss)
            if risk > 0:
                reward = abs(exit_price - entry.entry_price)
                entry.risk_reward_actual = reward / risk

        # Update additional fields
        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)

        self._save_entry(entry)
        logger.info(f"Trade closed: {trade_id} P&L: {pnl:+.2f}")
        return entry

    def get_entry(self, trade_id: str) -> Optional[TradeEntry]:
        """Get a trade entry by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM trades WHERE trade_id = ?",
                (trade_id,),
            )
            row = cursor.fetchone()

            if row:
                data = dict(row)
                data["tags"] = json.loads(data.get("tags", "[]"))
                return TradeEntry.from_dict(data)

        return None

    def update_entry(self, trade_id: str, **kwargs) -> Optional[TradeEntry]:
        """Update a trade entry."""
        entry = self.get_entry(trade_id)
        if not entry:
            return None

        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)

        entry.updated_at = datetime.now()
        self._save_entry(entry)
        return entry

    def add_note(self, trade_id: str, note: str) -> Optional[TradeEntry]:
        """Add a note to a trade."""
        entry = self.get_entry(trade_id)
        if not entry:
            return None

        if entry.notes:
            entry.notes += f"\n\n{datetime.now().strftime('%Y-%m-%d %H:%M')}: {note}"
        else:
            entry.notes = f"{datetime.now().strftime('%Y-%m-%d %H:%M')}: {note}"

        entry.updated_at = datetime.now()
        self._save_entry(entry)
        return entry

    def add_tag(self, trade_id: str, tag: str) -> Optional[TradeEntry]:
        """Add a tag to a trade."""
        entry = self.get_entry(trade_id)
        if not entry:
            return None

        if tag not in entry.tags:
            entry.tags.append(tag)
            entry.updated_at = datetime.now()
            self._save_entry(entry)

        return entry

    def get_trades(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        outcome: Optional[TradeOutcome] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[TradeEntry]:
        """Query trades with filters."""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)

        if outcome:
            query += " AND outcome = ?"
            params.append(outcome.value)

        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND entry_time <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY entry_time DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)

            trades = []
            for row in cursor:
                data = dict(row)
                data["tags"] = json.loads(data.get("tags", "[]"))

                # Filter by tags if specified
                if tags:
                    if not any(t in data["tags"] for t in tags):
                        continue

                trades.append(TradeEntry.from_dict(data))

            return trades

    def get_statistics(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get trading statistics."""
        trades = self.get_trades(
            symbol=symbol,
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            limit=10000,
        )

        if not trades:
            return {"total_trades": 0}

        closed_trades = [t for t in trades if t.exit_time]

        wins = [t for t in closed_trades if t.outcome == TradeOutcome.WIN]
        losses = [t for t in closed_trades if t.outcome == TradeOutcome.LOSS]

        total_pnl = sum(t.net_pnl for t in closed_trades)
        gross_profit = sum(t.net_pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.net_pnl for t in losses)) if losses else 0

        avg_win = sum(t.net_pnl for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t.net_pnl for t in losses) / len(losses)) if losses else 0

        return {
            "total_trades": len(trades),
            "closed_trades": len(closed_trades),
            "open_trades": len(trades) - len(closed_trades),
            "wins": len(wins),
            "losses": len(losses),
            "breakeven": len([t for t in closed_trades if t.outcome == TradeOutcome.BREAKEVEN]),
            "win_rate": len(wins) / len(closed_trades) if closed_trades else 0,
            "total_pnl": total_pnl,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_trade": total_pnl / len(closed_trades) if closed_trades else 0,
            "expectancy": (len(wins) / len(closed_trades) * avg_win -
                          len(losses) / len(closed_trades) * avg_loss) if closed_trades else 0,
            "largest_win": max((t.net_pnl for t in wins), default=0),
            "largest_loss": min((t.net_pnl for t in losses), default=0),
            "avg_hold_time_hours": self._avg_hold_time(closed_trades),
        }

    def _avg_hold_time(self, trades: List[TradeEntry]) -> float:
        """Calculate average hold time in hours."""
        hold_times = []
        for t in trades:
            if t.entry_time and t.exit_time:
                delta = t.exit_time - t.entry_time
                hold_times.append(delta.total_seconds() / 3600)

        return sum(hold_times) / len(hold_times) if hold_times else 0

    def get_by_setup(self, setup_type: str) -> Dict[str, Any]:
        """Get statistics for a specific setup type."""
        trades = self.get_trades(limit=10000)
        setup_trades = [t for t in trades if t.setup_type == setup_type and t.exit_time]

        if not setup_trades:
            return {"setup": setup_type, "trades": 0}

        wins = len([t for t in setup_trades if t.outcome == TradeOutcome.WIN])

        return {
            "setup": setup_type,
            "trades": len(setup_trades),
            "win_rate": wins / len(setup_trades),
            "avg_pnl": sum(t.net_pnl for t in setup_trades) / len(setup_trades),
            "total_pnl": sum(t.net_pnl for t in setup_trades),
        }

    def get_by_emotional_state(self) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by emotional state."""
        trades = self.get_trades(limit=10000)
        closed = [t for t in trades if t.exit_time]

        results = {}

        for state in EmotionalState:
            state_trades = [t for t in closed if t.emotional_state_entry == state]
            if state_trades:
                wins = len([t for t in state_trades if t.outcome == TradeOutcome.WIN])
                results[state.value] = {
                    "trades": len(state_trades),
                    "win_rate": wins / len(state_trades),
                    "avg_pnl": sum(t.net_pnl for t in state_trades) / len(state_trades),
                }

        return results

    def export_json(self, path: str) -> None:
        """Export all trades to JSON."""
        trades = self.get_trades(limit=100000)
        data = [t.to_dict() for t in trades]

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Exported {len(trades)} trades to {path}")

    def export_csv(self, path: str) -> None:
        """Export trades to CSV."""
        import csv

        trades = self.get_trades(limit=100000)

        if not trades:
            return

        fieldnames = list(trades[0].to_dict().keys())

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trade in trades:
                writer.writerow(trade.to_dict())

        logger.info(f"Exported {len(trades)} trades to {path}")
