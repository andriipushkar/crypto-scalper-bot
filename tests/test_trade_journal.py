"""
Comprehensive tests for TradeJournal and Analytics.

Tests cover:
- TradeEntry model
- TradeJournal CRUD operations
- Statistics calculation
- Filtering and queries
- Export functionality
- Emotional state analysis
"""

import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, '/home/sssmmmddd/Code/pro/crypto-scalper-bot')

from src.analytics.trade_journal import (
    TradeJournal,
    TradeEntry,
    TradeOutcome,
    TradeQuality,
    EmotionalState,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_db():
    """Create temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def journal(temp_db):
    """Create TradeJournal with temp database."""
    return TradeJournal(db_path=temp_db)


@pytest.fixture
def sample_entry():
    """Create sample trade entry."""
    return TradeEntry(
        trade_id="test123",
        symbol="BTCUSDT",
        side="LONG",
        entry_price=50000.0,
        quantity=0.1,
        leverage=10,
        strategy="orderbook_imbalance",
        entry_time=datetime(2024, 1, 15, 10, 0, 0),
    )


@pytest.fixture
def closed_entry():
    """Create closed trade entry."""
    return TradeEntry(
        trade_id="closed123",
        symbol="BTCUSDT",
        side="LONG",
        entry_price=50000.0,
        exit_price=51000.0,
        quantity=0.1,
        leverage=10,
        strategy="orderbook_imbalance",
        entry_time=datetime(2024, 1, 15, 10, 0, 0),
        exit_time=datetime(2024, 1, 15, 11, 0, 0),
        pnl=100.0,
        net_pnl=96.0,
        outcome=TradeOutcome.WIN,
    )


# =============================================================================
# TradeEntry Model Tests
# =============================================================================

class TestTradeEntry:
    """Tests for TradeEntry model."""

    def test_create_entry(self):
        """Test creating trade entry."""
        entry = TradeEntry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            quantity=0.1,
        )

        assert entry.symbol == "BTCUSDT"
        assert entry.side == "LONG"
        assert entry.entry_price == 50000.0
        assert entry.quantity == 0.1
        assert entry.trade_id is not None

    def test_entry_defaults(self):
        """Test default values."""
        entry = TradeEntry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            quantity=0.1,
        )

        assert entry.leverage == 1
        assert entry.pnl == 0.0
        assert entry.commission == 0.0
        assert entry.outcome == TradeOutcome.BREAKEVEN
        assert entry.quality == TradeQuality.AVERAGE
        assert entry.tags == []

    def test_entry_to_dict(self, sample_entry):
        """Test converting entry to dictionary."""
        data = sample_entry.to_dict()

        assert data["symbol"] == "BTCUSDT"
        assert data["side"] == "LONG"
        assert data["entry_price"] == 50000.0
        assert data["outcome"] == "breakeven"
        assert isinstance(data["entry_time"], str)

    def test_entry_from_dict(self):
        """Test creating entry from dictionary."""
        data = {
            "trade_id": "test456",
            "symbol": "ETHUSDT",
            "side": "SHORT",
            "entry_price": 3000.0,
            "quantity": 1.0,
            "outcome": "win",
            "quality": "good",
            "entry_time": "2024-01-15T10:00:00",
            "created_at": "2024-01-15T10:00:00",
            "updated_at": "2024-01-15T10:00:00",
            "tags": [],
        }

        entry = TradeEntry.from_dict(data)

        assert entry.trade_id == "test456"
        assert entry.symbol == "ETHUSDT"
        assert entry.outcome == TradeOutcome.WIN
        assert entry.quality == TradeQuality.GOOD

    def test_entry_with_emotional_state(self):
        """Test entry with emotional state."""
        entry = TradeEntry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            quantity=0.1,
            emotional_state_entry=EmotionalState.CONFIDENT,
        )

        assert entry.emotional_state_entry == EmotionalState.CONFIDENT

        data = entry.to_dict()
        assert data["emotional_state_entry"] == "confident"

    def test_entry_with_tags(self):
        """Test entry with tags."""
        entry = TradeEntry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            quantity=0.1,
            tags=["breakout", "high_volume"],
        )

        assert entry.tags == ["breakout", "high_volume"]


# =============================================================================
# TradeJournal CRUD Tests
# =============================================================================

class TestTradeJournalCRUD:
    """Tests for TradeJournal CRUD operations."""

    def test_create_entry(self, journal):
        """Test creating journal entry."""
        entry = journal.create_entry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            quantity=0.1,
            strategy="test_strategy",
        )

        assert entry.trade_id is not None
        assert entry.symbol == "BTCUSDT"

    def test_get_entry(self, journal):
        """Test retrieving entry by ID."""
        created = journal.create_entry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            quantity=0.1,
        )

        retrieved = journal.get_entry(created.trade_id)

        assert retrieved is not None
        assert retrieved.trade_id == created.trade_id
        assert retrieved.symbol == created.symbol

    def test_get_nonexistent_entry(self, journal):
        """Test retrieving non-existent entry."""
        retrieved = journal.get_entry("nonexistent_id")
        assert retrieved is None

    def test_update_entry(self, journal):
        """Test updating entry."""
        entry = journal.create_entry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            quantity=0.1,
        )

        updated = journal.update_entry(
            entry.trade_id,
            notes="Updated note",
            quality=TradeQuality.GOOD,
        )

        assert updated.notes == "Updated note"
        assert updated.quality == TradeQuality.GOOD

    def test_update_nonexistent_entry(self, journal):
        """Test updating non-existent entry."""
        result = journal.update_entry("nonexistent", notes="test")
        assert result is None

    def test_close_trade(self, journal):
        """Test closing trade."""
        entry = journal.create_entry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            quantity=0.1,
        )

        closed = journal.close_trade(
            entry.trade_id,
            exit_price=51000.0,
            pnl=100.0,
            exit_reason="take_profit",
        )

        assert closed.exit_time is not None
        assert closed.exit_price == 51000.0
        assert closed.pnl == 100.0
        assert closed.outcome == TradeOutcome.WIN

    def test_close_trade_loss(self, journal):
        """Test closing trade with loss."""
        entry = journal.create_entry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            quantity=0.1,
        )

        closed = journal.close_trade(
            entry.trade_id,
            exit_price=49000.0,
            pnl=-100.0,
        )

        assert closed.outcome == TradeOutcome.LOSS
        assert closed.pnl_percent < 0

    def test_close_trade_breakeven(self, journal):
        """Test closing trade at breakeven."""
        entry = journal.create_entry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            quantity=0.1,
        )

        closed = journal.close_trade(
            entry.trade_id,
            exit_price=50000.0,
            pnl=0.0,
        )

        assert closed.outcome == TradeOutcome.BREAKEVEN

    def test_close_nonexistent_trade(self, journal):
        """Test closing non-existent trade."""
        result = journal.close_trade("nonexistent", exit_price=50000.0, pnl=0.0)
        assert result is None

    def test_add_note(self, journal):
        """Test adding note to trade."""
        entry = journal.create_entry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            quantity=0.1,
        )

        journal.add_note(entry.trade_id, "First note")
        journal.add_note(entry.trade_id, "Second note")

        updated = journal.get_entry(entry.trade_id)

        assert "First note" in updated.notes
        assert "Second note" in updated.notes

    def test_add_tag(self, journal):
        """Test adding tag to trade."""
        entry = journal.create_entry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            quantity=0.1,
        )

        journal.add_tag(entry.trade_id, "breakout")
        journal.add_tag(entry.trade_id, "high_volume")
        journal.add_tag(entry.trade_id, "breakout")  # Duplicate

        updated = journal.get_entry(entry.trade_id)

        assert "breakout" in updated.tags
        assert "high_volume" in updated.tags
        assert len(updated.tags) == 2  # No duplicate

    def test_calculate_risk_reward(self, journal):
        """Test risk/reward calculation on close."""
        entry = journal.create_entry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            quantity=0.1,
            stop_loss=49500.0,
        )

        closed = journal.close_trade(
            entry.trade_id,
            exit_price=51500.0,
            pnl=150.0,
        )

        # Risk = 500, Reward = 1500, R:R = 3
        assert closed.risk_reward_actual == 3.0


# =============================================================================
# Query Tests
# =============================================================================

class TestTradeJournalQueries:
    """Tests for TradeJournal query operations."""

    def test_get_trades_all(self, journal):
        """Test getting all trades."""
        for i in range(5):
            journal.create_entry(
                symbol="BTCUSDT",
                side="LONG",
                entry_price=50000.0 + i * 100,
                quantity=0.1,
            )

        trades = journal.get_trades()
        assert len(trades) == 5

    def test_get_trades_by_symbol(self, journal):
        """Test filtering trades by symbol."""
        journal.create_entry(symbol="BTCUSDT", side="LONG", entry_price=50000.0, quantity=0.1)
        journal.create_entry(symbol="ETHUSDT", side="LONG", entry_price=3000.0, quantity=1.0)
        journal.create_entry(symbol="BTCUSDT", side="SHORT", entry_price=51000.0, quantity=0.1)

        btc_trades = journal.get_trades(symbol="BTCUSDT")
        eth_trades = journal.get_trades(symbol="ETHUSDT")

        assert len(btc_trades) == 2
        assert len(eth_trades) == 1

    def test_get_trades_by_strategy(self, journal):
        """Test filtering trades by strategy."""
        journal.create_entry(symbol="BTCUSDT", side="LONG", entry_price=50000.0, quantity=0.1, strategy="imbalance")
        journal.create_entry(symbol="BTCUSDT", side="LONG", entry_price=50000.0, quantity=0.1, strategy="volume_spike")
        journal.create_entry(symbol="BTCUSDT", side="LONG", entry_price=50000.0, quantity=0.1, strategy="imbalance")

        imbalance_trades = journal.get_trades(strategy="imbalance")
        volume_trades = journal.get_trades(strategy="volume_spike")

        assert len(imbalance_trades) == 2
        assert len(volume_trades) == 1

    def test_get_trades_by_outcome(self, journal):
        """Test filtering trades by outcome."""
        entry1 = journal.create_entry(symbol="BTCUSDT", side="LONG", entry_price=50000.0, quantity=0.1)
        entry2 = journal.create_entry(symbol="BTCUSDT", side="LONG", entry_price=50000.0, quantity=0.1)

        journal.close_trade(entry1.trade_id, exit_price=51000.0, pnl=100.0)
        journal.close_trade(entry2.trade_id, exit_price=49000.0, pnl=-100.0)

        wins = journal.get_trades(outcome=TradeOutcome.WIN)
        losses = journal.get_trades(outcome=TradeOutcome.LOSS)

        assert len(wins) == 1
        assert len(losses) == 1

    def test_get_trades_by_date_range(self, journal):
        """Test filtering trades by date range."""
        entry1 = journal.create_entry(symbol="BTCUSDT", side="LONG", entry_price=50000.0, quantity=0.1)
        entry2 = journal.create_entry(symbol="BTCUSDT", side="LONG", entry_price=50000.0, quantity=0.1)

        # Manually set dates for testing
        journal.update_entry(entry1.trade_id, entry_time=datetime(2024, 1, 10))
        journal.update_entry(entry2.trade_id, entry_time=datetime(2024, 1, 20))

        trades = journal.get_trades(
            start_date=datetime(2024, 1, 15),
            end_date=datetime(2024, 1, 25),
        )

        assert len(trades) == 1

    def test_get_trades_limit(self, journal):
        """Test limiting trade results."""
        for i in range(20):
            journal.create_entry(symbol="BTCUSDT", side="LONG", entry_price=50000.0, quantity=0.1)

        trades = journal.get_trades(limit=10)
        assert len(trades) == 10


# =============================================================================
# Statistics Tests
# =============================================================================

class TestTradeJournalStatistics:
    """Tests for TradeJournal statistics."""

    def test_get_statistics_empty(self, journal):
        """Test statistics with no trades."""
        stats = journal.get_statistics()
        assert stats["total_trades"] == 0

    def test_get_statistics_basic(self, journal):
        """Test basic statistics calculation."""
        # Create and close trades
        entry1 = journal.create_entry(symbol="BTCUSDT", side="LONG", entry_price=50000.0, quantity=0.1)
        entry2 = journal.create_entry(symbol="BTCUSDT", side="LONG", entry_price=50000.0, quantity=0.1)
        entry3 = journal.create_entry(symbol="BTCUSDT", side="LONG", entry_price=50000.0, quantity=0.1)

        journal.close_trade(entry1.trade_id, exit_price=51000.0, pnl=100.0)
        journal.close_trade(entry2.trade_id, exit_price=49000.0, pnl=-50.0)
        journal.close_trade(entry3.trade_id, exit_price=51500.0, pnl=150.0)

        stats = journal.get_statistics()

        assert stats["total_trades"] == 3
        assert stats["closed_trades"] == 3
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert stats["win_rate"] == pytest.approx(2/3, rel=0.01)
        assert stats["total_pnl"] == 200.0

    def test_get_statistics_profit_factor(self, journal):
        """Test profit factor calculation."""
        entry1 = journal.create_entry(symbol="BTCUSDT", side="LONG", entry_price=50000.0, quantity=0.1)
        entry2 = journal.create_entry(symbol="BTCUSDT", side="LONG", entry_price=50000.0, quantity=0.1)

        journal.close_trade(entry1.trade_id, exit_price=51000.0, pnl=100.0)
        journal.close_trade(entry2.trade_id, exit_price=49000.0, pnl=-50.0)

        stats = journal.get_statistics()

        # Profit factor = gross_profit / gross_loss = 100 / 50 = 2
        assert stats["profit_factor"] == 2.0

    def test_get_statistics_averages(self, journal):
        """Test average calculations."""
        entries = []
        for i in range(4):
            entry = journal.create_entry(symbol="BTCUSDT", side="LONG", entry_price=50000.0, quantity=0.1)
            entries.append(entry)

        # 2 wins: 100, 50
        # 2 losses: -30, -20
        journal.close_trade(entries[0].trade_id, exit_price=51000.0, pnl=100.0)
        journal.close_trade(entries[1].trade_id, exit_price=50500.0, pnl=50.0)
        journal.close_trade(entries[2].trade_id, exit_price=49700.0, pnl=-30.0)
        journal.close_trade(entries[3].trade_id, exit_price=49800.0, pnl=-20.0)

        stats = journal.get_statistics()

        assert stats["avg_win"] == 75.0  # (100+50)/2
        assert stats["avg_loss"] == 25.0  # (30+20)/2

    def test_get_statistics_by_symbol(self, journal):
        """Test statistics filtered by symbol."""
        # BTC trades
        btc1 = journal.create_entry(symbol="BTCUSDT", side="LONG", entry_price=50000.0, quantity=0.1)
        btc2 = journal.create_entry(symbol="BTCUSDT", side="LONG", entry_price=50000.0, quantity=0.1)

        # ETH trade
        eth1 = journal.create_entry(symbol="ETHUSDT", side="LONG", entry_price=3000.0, quantity=1.0)

        journal.close_trade(btc1.trade_id, exit_price=51000.0, pnl=100.0)
        journal.close_trade(btc2.trade_id, exit_price=49000.0, pnl=-50.0)
        journal.close_trade(eth1.trade_id, exit_price=3100.0, pnl=100.0)

        btc_stats = journal.get_statistics(symbol="BTCUSDT")
        eth_stats = journal.get_statistics(symbol="ETHUSDT")

        assert btc_stats["total_trades"] == 2
        assert btc_stats["total_pnl"] == 50.0

        assert eth_stats["total_trades"] == 1
        assert eth_stats["total_pnl"] == 100.0

    def test_get_by_setup(self, journal):
        """Test statistics by setup type."""
        entry1 = journal.create_entry(
            symbol="BTCUSDT", side="LONG", entry_price=50000.0, quantity=0.1,
            setup_type="breakout",
        )
        entry2 = journal.create_entry(
            symbol="BTCUSDT", side="LONG", entry_price=50000.0, quantity=0.1,
            setup_type="breakout",
        )

        journal.close_trade(entry1.trade_id, exit_price=51000.0, pnl=100.0)
        journal.close_trade(entry2.trade_id, exit_price=49000.0, pnl=-50.0)

        setup_stats = journal.get_by_setup("breakout")

        assert setup_stats["setup"] == "breakout"
        assert setup_stats["trades"] == 2
        assert setup_stats["win_rate"] == 0.5

    def test_get_by_emotional_state(self, journal):
        """Test statistics by emotional state."""
        entry1 = journal.create_entry(
            symbol="BTCUSDT", side="LONG", entry_price=50000.0, quantity=0.1,
            emotional_state_entry=EmotionalState.CONFIDENT,
        )
        entry2 = journal.create_entry(
            symbol="BTCUSDT", side="LONG", entry_price=50000.0, quantity=0.1,
            emotional_state_entry=EmotionalState.FEARFUL,
        )

        journal.close_trade(entry1.trade_id, exit_price=51000.0, pnl=100.0)
        journal.close_trade(entry2.trade_id, exit_price=49000.0, pnl=-50.0)

        emotional_stats = journal.get_by_emotional_state()

        assert "confident" in emotional_stats
        assert emotional_stats["confident"]["win_rate"] == 1.0

        assert "fearful" in emotional_stats
        assert emotional_stats["fearful"]["win_rate"] == 0.0


# =============================================================================
# Export Tests
# =============================================================================

class TestTradeJournalExport:
    """Tests for TradeJournal export functionality."""

    def test_export_json(self, journal, temp_db):
        """Test exporting to JSON."""
        entry = journal.create_entry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            quantity=0.1,
        )
        journal.close_trade(entry.trade_id, exit_price=51000.0, pnl=100.0)

        export_path = temp_db.replace(".db", ".json")

        try:
            journal.export_json(export_path)

            assert os.path.exists(export_path)

            with open(export_path) as f:
                data = json.load(f)

            assert len(data) == 1
            assert data[0]["symbol"] == "BTCUSDT"
        finally:
            if os.path.exists(export_path):
                os.remove(export_path)

    def test_export_csv(self, journal, temp_db):
        """Test exporting to CSV."""
        entry = journal.create_entry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            quantity=0.1,
        )
        journal.close_trade(entry.trade_id, exit_price=51000.0, pnl=100.0)

        export_path = temp_db.replace(".db", ".csv")

        try:
            journal.export_csv(export_path)

            assert os.path.exists(export_path)

            with open(export_path) as f:
                content = f.read()

            assert "BTCUSDT" in content
            assert "symbol" in content  # Header
        finally:
            if os.path.exists(export_path):
                os.remove(export_path)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_pnl_percent_long(self, journal):
        """Test PnL percent calculation for long."""
        entry = journal.create_entry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            quantity=0.1,
            leverage=10,
        )

        closed = journal.close_trade(entry.trade_id, exit_price=51000.0, pnl=100.0)

        # (51000-50000)/50000 * 100 * 10 = 20%
        assert closed.pnl_percent == pytest.approx(20.0, rel=0.01)

    def test_pnl_percent_short(self, journal):
        """Test PnL percent calculation for short."""
        entry = journal.create_entry(
            symbol="BTCUSDT",
            side="SHORT",
            entry_price=50000.0,
            quantity=0.1,
            leverage=10,
        )

        closed = journal.close_trade(entry.trade_id, exit_price=49000.0, pnl=100.0)

        # (50000-49000)/50000 * 100 * 10 = 20%
        assert closed.pnl_percent == pytest.approx(20.0, rel=0.01)

    def test_zero_entry_price(self, journal):
        """Test handling zero entry price."""
        entry = journal.create_entry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=0.0,
            quantity=0.1,
        )

        closed = journal.close_trade(entry.trade_id, exit_price=1000.0, pnl=100.0)

        # Should not crash, pnl_percent will be handled
        assert closed is not None

    def test_very_large_values(self, journal):
        """Test handling very large values."""
        entry = journal.create_entry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=100000000.0,
            quantity=1000.0,
        )

        closed = journal.close_trade(entry.trade_id, exit_price=100000001.0, pnl=1000.0)

        assert closed is not None
        assert closed.pnl == 1000.0

    def test_concurrent_entries(self, journal):
        """Test multiple entries in quick succession."""
        entries = []
        for i in range(100):
            entry = journal.create_entry(
                symbol="BTCUSDT",
                side="LONG",
                entry_price=50000.0 + i,
                quantity=0.1,
            )
            entries.append(entry)

        all_trades = journal.get_trades(limit=1000)
        assert len(all_trades) == 100

    def test_hold_time_calculation(self, journal):
        """Test hold time calculation."""
        entry = journal.create_entry(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000.0,
            quantity=0.1,
        )

        # Manually set times
        journal.update_entry(
            entry.trade_id,
            entry_time=datetime(2024, 1, 15, 10, 0, 0),
        )

        journal.close_trade(
            entry.trade_id,
            exit_price=51000.0,
            pnl=100.0,
        )

        # Update exit time manually to ensure correct calculation
        updated = journal.get_entry(entry.trade_id)
        journal.update_entry(
            entry.trade_id,
            exit_time=datetime(2024, 1, 15, 12, 0, 0),  # 2 hours later
        )

        stats = journal.get_statistics()

        # Average hold time should be ~2 hours
        assert stats["avg_hold_time_hours"] == pytest.approx(2.0, rel=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
