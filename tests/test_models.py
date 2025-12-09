"""
Unit tests for data models.
"""

import pytest
from datetime import datetime
from decimal import Decimal

from src.data.models import (
    Side,
    OrderType,
    SignalType,
    Trade,
    OrderBookLevel,
    OrderBookSnapshot,
    DepthUpdate,
    MarkPrice,
    Signal,
)


class TestSide:
    """Tests for Side enum."""

    def test_opposite(self):
        assert Side.BUY.opposite == Side.SELL
        assert Side.SELL.opposite == Side.BUY


class TestTrade:
    """Tests for Trade model."""

    @pytest.fixture
    def sample_trade(self):
        return Trade(
            symbol="BTCUSDT",
            trade_id=123456,
            price=Decimal("50000.00"),
            quantity=Decimal("0.1"),
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            is_buyer_maker=False,
        )

    def test_side_buy(self, sample_trade):
        # is_buyer_maker=False means buyer took the order = BUY
        assert sample_trade.side == Side.BUY

    def test_side_sell(self):
        trade = Trade(
            symbol="BTCUSDT",
            trade_id=123456,
            price=Decimal("50000.00"),
            quantity=Decimal("0.1"),
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            is_buyer_maker=True,  # buyer was maker = SELL
        )
        assert trade.side == Side.SELL

    def test_value(self, sample_trade):
        assert sample_trade.value == Decimal("5000.00")

    def test_from_binance(self):
        binance_data = {
            "s": "BTCUSDT",
            "a": 123456,
            "p": "50000.00",
            "q": "0.1",
            "T": 1704110400000,  # 2024-01-01 12:00:00 UTC
            "m": False,
        }
        trade = Trade.from_binance(binance_data)

        assert trade.symbol == "BTCUSDT"
        assert trade.trade_id == 123456
        assert trade.price == Decimal("50000.00")
        assert trade.quantity == Decimal("0.1")
        assert trade.is_buyer_maker == False


class TestOrderBookLevel:
    """Tests for OrderBookLevel model."""

    def test_value(self):
        level = OrderBookLevel(
            price=Decimal("50000.00"),
            quantity=Decimal("1.5"),
        )
        assert level.value == Decimal("75000.00")

    def test_from_binance(self):
        level = OrderBookLevel.from_binance(["50000.00", "1.5"])
        assert level.price == Decimal("50000.00")
        assert level.quantity == Decimal("1.5")


class TestOrderBookSnapshot:
    """Tests for OrderBookSnapshot model."""

    @pytest.fixture
    def sample_snapshot(self):
        bids = [
            OrderBookLevel(Decimal("50000"), Decimal("1.0")),
            OrderBookLevel(Decimal("49999"), Decimal("2.0")),
            OrderBookLevel(Decimal("49998"), Decimal("3.0")),
        ]
        asks = [
            OrderBookLevel(Decimal("50001"), Decimal("1.5")),
            OrderBookLevel(Decimal("50002"), Decimal("2.5")),
            OrderBookLevel(Decimal("50003"), Decimal("3.5")),
        ]
        return OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            bids=bids,
            asks=asks,
            last_update_id=12345,
        )

    def test_best_bid(self, sample_snapshot):
        assert sample_snapshot.best_bid.price == Decimal("50000")
        assert sample_snapshot.best_bid.quantity == Decimal("1.0")

    def test_best_ask(self, sample_snapshot):
        assert sample_snapshot.best_ask.price == Decimal("50001")
        assert sample_snapshot.best_ask.quantity == Decimal("1.5")

    def test_mid_price(self, sample_snapshot):
        expected = (Decimal("50000") + Decimal("50001")) / 2
        assert sample_snapshot.mid_price == expected

    def test_spread(self, sample_snapshot):
        assert sample_snapshot.spread == Decimal("1")

    def test_spread_bps(self, sample_snapshot):
        # spread / mid_price * 10000
        mid = Decimal("50000.5")
        expected = (Decimal("1") / mid) * 10000
        assert abs(sample_snapshot.spread_bps - expected) < Decimal("0.001")

    def test_bid_volume(self, sample_snapshot):
        # Top 2 levels: 1.0 + 2.0 = 3.0
        assert sample_snapshot.bid_volume(2) == Decimal("3.0")
        # All 3 levels: 1.0 + 2.0 + 3.0 = 6.0
        assert sample_snapshot.bid_volume(3) == Decimal("6.0")

    def test_ask_volume(self, sample_snapshot):
        # Top 2 levels: 1.5 + 2.5 = 4.0
        assert sample_snapshot.ask_volume(2) == Decimal("4.0")

    def test_imbalance(self, sample_snapshot):
        # bid_volume(3) = 6.0, ask_volume(3) = 7.5
        expected = Decimal("6.0") / Decimal("7.5")
        assert sample_snapshot.imbalance(3) == expected

    def test_empty_snapshot(self):
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            bids=[],
            asks=[],
            last_update_id=0,
        )
        assert snapshot.best_bid is None
        assert snapshot.best_ask is None
        assert snapshot.mid_price is None
        assert snapshot.spread is None


class TestDepthUpdate:
    """Tests for DepthUpdate model."""

    def test_from_binance(self):
        binance_data = {
            "s": "BTCUSDT",
            "E": 1704110400000,
            "U": 100,
            "u": 110,
            "b": [["50000.00", "1.0"], ["49999.00", "0"]],
            "a": [["50001.00", "1.5"]],
        }
        update = DepthUpdate.from_binance(binance_data)

        assert update.symbol == "BTCUSDT"
        assert update.first_update_id == 100
        assert update.final_update_id == 110
        assert len(update.bids) == 2
        assert len(update.asks) == 1
        assert update.bids[0].price == Decimal("50000.00")
        assert update.bids[1].quantity == Decimal("0")  # Remove signal


class TestMarkPrice:
    """Tests for MarkPrice model."""

    def test_from_binance(self):
        binance_data = {
            "s": "BTCUSDT",
            "p": "50000.00",
            "i": "50001.00",
            "P": "49999.00",
            "r": "0.0001",
            "T": 1704139200000,  # Next funding time
            "E": 1704110400000,  # Event time
        }
        mp = MarkPrice.from_binance(binance_data)

        assert mp.symbol == "BTCUSDT"
        assert mp.mark_price == Decimal("50000.00")
        assert mp.index_price == Decimal("50001.00")
        assert mp.funding_rate == Decimal("0.0001")


class TestSignal:
    """Tests for Signal model."""

    def test_is_entry_long(self):
        signal = Signal(
            strategy="test",
            signal_type=SignalType.LONG,
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            strength=0.8,
            price=Decimal("50000"),
        )
        assert signal.is_entry == True
        assert signal.is_exit == False
        assert signal.side == Side.BUY

    def test_is_entry_short(self):
        signal = Signal(
            strategy="test",
            signal_type=SignalType.SHORT,
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            strength=0.8,
            price=Decimal("50000"),
        )
        assert signal.is_entry == True
        assert signal.side == Side.SELL

    def test_is_exit_close_long(self):
        signal = Signal(
            strategy="test",
            signal_type=SignalType.CLOSE_LONG,
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            strength=0.8,
            price=Decimal("50000"),
        )
        assert signal.is_entry == False
        assert signal.is_exit == True
        assert signal.side == Side.SELL  # Close long = sell

    def test_no_action(self):
        signal = Signal(
            strategy="test",
            signal_type=SignalType.NO_ACTION,
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            strength=0.0,
            price=Decimal("50000"),
        )
        assert signal.is_entry == False
        assert signal.is_exit == False
        assert signal.side is None
