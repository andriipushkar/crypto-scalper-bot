"""
Unit tests for OrderBook.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch

from src.data.orderbook import OrderBook, OrderBookManager
from src.data.models import OrderBookLevel, DepthUpdate, DepthUpdateEvent


class TestOrderBook:
    """Tests for OrderBook class."""

    @pytest.fixture
    def orderbook(self):
        return OrderBook(symbol="BTCUSDT", depth=10, testnet=True)

    def test_init(self, orderbook):
        assert orderbook.symbol == "BTCUSDT"
        assert orderbook.depth == 10
        assert orderbook.testnet == True
        assert orderbook.is_synced == False

    def test_api_url_testnet(self, orderbook):
        assert "testnet" in orderbook.api_url

    def test_api_url_mainnet(self):
        book = OrderBook(symbol="BTCUSDT", testnet=False)
        assert "fapi.binance.com" in book.api_url

    def test_apply_snapshot(self, orderbook):
        snapshot = {
            "lastUpdateId": 12345,
            "bids": [
                ["50000.00", "1.0"],
                ["49999.00", "2.0"],
            ],
            "asks": [
                ["50001.00", "1.5"],
                ["50002.00", "2.5"],
            ],
        }

        orderbook._apply_snapshot(snapshot)

        assert orderbook.last_update_id == 12345
        assert len(orderbook._bids) == 2
        assert len(orderbook._asks) == 2
        assert orderbook._bids[Decimal("50000.00")] == Decimal("1.0")
        assert orderbook._asks[Decimal("50001.00")] == Decimal("1.5")

    def test_get_best_bid(self, orderbook):
        orderbook._bids = {
            Decimal("50000"): Decimal("1.0"),
            Decimal("49999"): Decimal("2.0"),
        }

        best_bid = orderbook.get_best_bid()
        assert best_bid == (Decimal("50000"), Decimal("1.0"))

    def test_get_best_ask(self, orderbook):
        orderbook._asks = {
            Decimal("50001"): Decimal("1.5"),
            Decimal("50002"): Decimal("2.5"),
        }

        best_ask = orderbook.get_best_ask()
        assert best_ask == (Decimal("50001"), Decimal("1.5"))

    def test_get_mid_price(self, orderbook):
        orderbook._bids = {Decimal("50000"): Decimal("1.0")}
        orderbook._asks = {Decimal("50002"): Decimal("1.0")}

        mid = orderbook.get_mid_price()
        assert mid == Decimal("50001")

    def test_get_spread(self, orderbook):
        orderbook._bids = {Decimal("50000"): Decimal("1.0")}
        orderbook._asks = {Decimal("50002"): Decimal("1.0")}

        spread = orderbook.get_spread()
        assert spread == Decimal("2")

    def test_get_spread_bps(self, orderbook):
        orderbook._bids = {Decimal("50000"): Decimal("1.0")}
        orderbook._asks = {Decimal("50010"): Decimal("1.0")}

        spread_bps = orderbook.get_spread_bps()
        # spread = 10, mid = 50005, bps = (10/50005) * 10000 ≈ 2.0
        assert spread_bps is not None
        assert float(spread_bps) == pytest.approx(2.0, rel=0.01)

    def test_apply_update_add_levels(self, orderbook):
        orderbook._synced = True
        orderbook._last_update_id = 100

        update = DepthUpdate(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            first_update_id=101,
            final_update_id=105,
            bids=[OrderBookLevel(Decimal("50000"), Decimal("1.0"))],
            asks=[OrderBookLevel(Decimal("50001"), Decimal("1.5"))],
        )

        orderbook._apply_update(update)

        assert orderbook._bids[Decimal("50000")] == Decimal("1.0")
        assert orderbook._asks[Decimal("50001")] == Decimal("1.5")
        assert orderbook.last_update_id == 105

    def test_apply_update_remove_levels(self, orderbook):
        orderbook._synced = True
        orderbook._last_update_id = 100
        orderbook._bids = {Decimal("50000"): Decimal("1.0")}
        orderbook._asks = {Decimal("50001"): Decimal("1.5")}

        # Quantity 0 = remove level
        update = DepthUpdate(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            first_update_id=101,
            final_update_id=105,
            bids=[OrderBookLevel(Decimal("50000"), Decimal("0"))],  # Remove
            asks=[],
        )

        orderbook._apply_update(update)

        assert Decimal("50000") not in orderbook._bids
        assert Decimal("50001") in orderbook._asks

    def test_get_snapshot(self, orderbook):
        orderbook._bids = {
            Decimal("50000"): Decimal("1.0"),
            Decimal("49999"): Decimal("2.0"),
        }
        orderbook._asks = {
            Decimal("50001"): Decimal("1.5"),
            Decimal("50002"): Decimal("2.5"),
        }
        orderbook._last_update_id = 12345
        orderbook._last_update_time = datetime.utcnow()

        snapshot = orderbook.get_snapshot()

        assert snapshot.symbol == "BTCUSDT"
        assert len(snapshot.bids) == 2
        assert len(snapshot.asks) == 2
        assert snapshot.bids[0].price == Decimal("50000")  # Best bid first
        assert snapshot.asks[0].price == Decimal("50001")  # Best ask first

    def test_get_imbalance(self, orderbook):
        orderbook._bids = {
            Decimal("50000"): Decimal("10.0"),
            Decimal("49999"): Decimal("10.0"),
        }
        orderbook._asks = {
            Decimal("50001"): Decimal("5.0"),
            Decimal("50002"): Decimal("5.0"),
        }

        # bid_volume = 20, ask_volume = 10, imbalance = 2.0
        imbalance = orderbook.get_imbalance(levels=2)
        assert imbalance == Decimal("2.0")

    def test_empty_orderbook_returns_none(self, orderbook):
        assert orderbook.get_best_bid() is None
        assert orderbook.get_best_ask() is None
        assert orderbook.get_mid_price() is None
        assert orderbook.get_spread() is None


class TestOrderBookManager:
    """Tests for OrderBookManager class."""

    def test_get_book_creates_new(self):
        manager = OrderBookManager(depth=20, testnet=True)

        book = manager.get_book("BTCUSDT")

        assert book.symbol == "BTCUSDT"
        assert book.depth == 20
        assert book.testnet == True

    def test_get_book_returns_existing(self):
        manager = OrderBookManager()

        book1 = manager.get_book("BTCUSDT")
        book2 = manager.get_book("BTCUSDT")

        assert book1 is book2

    def test_get_book_case_insensitive(self):
        manager = OrderBookManager()

        book1 = manager.get_book("btcusdt")
        book2 = manager.get_book("BTCUSDT")

        assert book1 is book2

    def test_get_all_snapshots(self):
        manager = OrderBookManager()

        # Create some books
        manager.get_book("BTCUSDT")
        manager.get_book("ETHUSDT")

        snapshots = manager.get_all_snapshots()

        assert "BTCUSDT" in snapshots
        assert "ETHUSDT" in snapshots


@pytest.mark.asyncio
class TestOrderBookAsync:
    """Async tests for OrderBook."""

    async def test_handle_depth_update_not_synced(self):
        book = OrderBook(symbol="BTCUSDT", testnet=True)

        update = DepthUpdate(
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            first_update_id=100,
            final_update_id=110,
            bids=[],
            asks=[],
        )
        event = DepthUpdateEvent(update=update)

        await book.handle_depth_update(event)

        # Should be queued, not applied
        assert len(book._pending_updates) == 1
        assert book.is_synced == False

    async def test_handle_depth_update_wrong_symbol(self):
        book = OrderBook(symbol="BTCUSDT", testnet=True)
        book._synced = True
        book._last_update_id = 100

        update = DepthUpdate(
            symbol="ETHUSDT",  # Different symbol
            timestamp=datetime.utcnow(),
            first_update_id=101,
            final_update_id=110,
            bids=[],
            asks=[],
        )
        event = DepthUpdateEvent(update=update)

        await book.handle_depth_update(event)

        # Should be ignored
        assert book.last_update_id == 100
