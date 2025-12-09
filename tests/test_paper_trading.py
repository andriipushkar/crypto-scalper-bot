"""
Tests for Paper Trading Module

Comprehensive tests for paper trading simulation.
"""
import pytest
import asyncio
from datetime import datetime
from decimal import Decimal

from src.trading.paper_trading import (
    PaperTradingEngine, PaperOrder, PaperPosition, PaperTrade,
    OrderType, OrderStatus
)


class TestPaperTradingEngine:
    """Tests for paper trading engine."""

    @pytest.fixture
    def engine(self):
        """Create paper trading engine."""
        return PaperTradingEngine(
            initial_balance=Decimal("10000"),
            commission_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            leverage=1,
        )

    @pytest.fixture
    def leveraged_engine(self):
        """Create leveraged paper trading engine."""
        return PaperTradingEngine(
            initial_balance=Decimal("10000"),
            commission_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            leverage=10,
        )

    def test_initial_state(self, engine):
        """Test initial engine state."""
        assert engine.balance == Decimal("10000")
        assert engine.equity == Decimal("10000")
        assert len(engine.positions) == 0
        assert len(engine.orders) == 0
        assert len(engine.trades) == 0

    def test_update_price(self, engine):
        """Test price update."""
        engine.update_price("BTCUSDT", Decimal("50000"))

        assert engine.current_prices["BTCUSDT"] == Decimal("50000")
        assert "BTCUSDT" in engine.bid_prices
        assert "BTCUSDT" in engine.ask_prices

    def test_bid_ask_spread(self, engine):
        """Test bid/ask spread calculation."""
        engine.update_price("BTCUSDT", Decimal("50000"))

        # Ask should be higher than mid price
        assert engine.ask_prices["BTCUSDT"] > Decimal("50000")
        # Bid should be lower than mid price
        assert engine.bid_prices["BTCUSDT"] < Decimal("50000")

    @pytest.mark.asyncio
    async def test_place_market_order(self, engine):
        """Test placing market order."""
        engine.update_price("BTCUSDT", Decimal("50000"))

        order = await engine.place_order(
            symbol="BTCUSDT",
            side="buy",
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )

        assert order.status == OrderStatus.FILLED
        assert "BTCUSDT" in engine.positions
        assert engine.stats["total_orders"] == 1
        assert engine.stats["filled_orders"] == 1

    @pytest.mark.asyncio
    async def test_place_limit_order(self, engine):
        """Test placing limit order."""
        engine.update_price("BTCUSDT", Decimal("50000"))

        order = await engine.place_order(
            symbol="BTCUSDT",
            side="buy",
            quantity=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            price=Decimal("49000"),  # Below current price
        )

        # Should be pending, not filled yet
        assert order.status == OrderStatus.PENDING
        assert order.id in engine.orders
        assert "BTCUSDT" not in engine.positions

    @pytest.mark.asyncio
    async def test_limit_order_fill(self, engine):
        """Test limit order getting filled."""
        engine.update_price("BTCUSDT", Decimal("50000"))

        await engine.place_order(
            symbol="BTCUSDT",
            side="buy",
            quantity=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            price=Decimal("49000"),
        )

        # Price drops to limit price
        engine.update_price("BTCUSDT", Decimal("48500"))

        # Order should now be filled
        assert len(engine.orders) == 0
        assert "BTCUSDT" in engine.positions

    @pytest.mark.asyncio
    async def test_stop_order(self, engine):
        """Test stop order."""
        engine.update_price("BTCUSDT", Decimal("50000"))

        await engine.place_order(
            symbol="BTCUSDT",
            side="buy",
            quantity=Decimal("0.1"),
            order_type=OrderType.STOP_MARKET,
            stop_price=Decimal("51000"),
        )

        # Price rises to stop price
        engine.update_price("BTCUSDT", Decimal("51500"))

        # Order should be filled
        assert len(engine.orders) == 0
        assert "BTCUSDT" in engine.positions

    @pytest.mark.asyncio
    async def test_position_pnl_update(self, engine):
        """Test position P&L calculation."""
        engine.update_price("BTCUSDT", Decimal("50000"))

        await engine.place_order(
            symbol="BTCUSDT",
            side="buy",
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )

        # Price goes up
        engine.update_price("BTCUSDT", Decimal("51000"))

        pos = engine.positions["BTCUSDT"]
        assert pos.unrealized_pnl > 0

    @pytest.mark.asyncio
    async def test_close_position(self, engine):
        """Test closing a position."""
        engine.update_price("BTCUSDT", Decimal("50000"))

        await engine.place_order(
            symbol="BTCUSDT",
            side="buy",
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )

        engine.update_price("BTCUSDT", Decimal("51000"))

        result = await engine.close_position("BTCUSDT")

        assert result is True
        assert "BTCUSDT" not in engine.positions
        assert len(engine.trades) == 1
        assert engine.trades[0].pnl > 0

    @pytest.mark.asyncio
    async def test_cancel_order(self, engine):
        """Test canceling an order."""
        engine.update_price("BTCUSDT", Decimal("50000"))

        order = await engine.place_order(
            symbol="BTCUSDT",
            side="buy",
            quantity=Decimal("0.1"),
            order_type=OrderType.LIMIT,
            price=Decimal("49000"),
        )

        result = await engine.cancel_order(order.id)

        assert result is True
        assert order.id not in engine.orders
        assert engine.stats["cancelled_orders"] == 1

    @pytest.mark.asyncio
    async def test_commission_deduction(self, engine):
        """Test commission is deducted."""
        initial_balance = engine.balance
        engine.update_price("BTCUSDT", Decimal("50000"))

        await engine.place_order(
            symbol="BTCUSDT",
            side="buy",
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )

        # Balance should be reduced by margin + commission
        assert engine.stats["total_commission"] > 0

    @pytest.mark.asyncio
    async def test_leverage(self, leveraged_engine):
        """Test leveraged trading."""
        leveraged_engine.update_price("BTCUSDT", Decimal("50000"))

        await leveraged_engine.place_order(
            symbol="BTCUSDT",
            side="buy",
            quantity=Decimal("1"),  # 1 BTC = $50,000 notional
            order_type=OrderType.MARKET,
        )

        # With 10x leverage, should only use $5,000 margin
        # Plus commission
        assert leveraged_engine.available_balance > Decimal("4000")

    @pytest.mark.asyncio
    async def test_short_position(self, engine):
        """Test short position."""
        engine.update_price("BTCUSDT", Decimal("50000"))

        await engine.place_order(
            symbol="BTCUSDT",
            side="sell",
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET,
        )

        pos = engine.positions.get("BTCUSDT")
        assert pos is not None
        assert pos.side == "short"

        # Price drops - should be profitable
        engine.update_price("BTCUSDT", Decimal("49000"))
        assert pos.unrealized_pnl > 0

    @pytest.mark.asyncio
    async def test_close_all_positions(self, engine):
        """Test closing all positions."""
        engine.update_price("BTCUSDT", Decimal("50000"))
        engine.update_price("ETHUSDT", Decimal("3000"))

        await engine.place_order("BTCUSDT", "buy", Decimal("0.1"), OrderType.MARKET)
        await engine.place_order("ETHUSDT", "buy", Decimal("1"), OrderType.MARKET)

        closed = await engine.close_all_positions()

        assert closed == 2
        assert len(engine.positions) == 0

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, engine):
        """Test canceling all orders."""
        engine.update_price("BTCUSDT", Decimal("50000"))

        await engine.place_order("BTCUSDT", "buy", Decimal("0.1"), OrderType.LIMIT, price=Decimal("49000"))
        await engine.place_order("BTCUSDT", "buy", Decimal("0.1"), OrderType.LIMIT, price=Decimal("48000"))
        await engine.place_order("BTCUSDT", "buy", Decimal("0.1"), OrderType.LIMIT, price=Decimal("47000"))

        cancelled = await engine.cancel_all_orders()

        assert cancelled == 3
        assert len(engine.orders) == 0

    def test_get_balance(self, engine):
        """Test get balance information."""
        balance = engine.get_balance()

        assert "total" in balance
        assert "available" in balance
        assert "balance" in balance
        assert balance["total"] == 10000.0

    def test_get_stats(self, engine):
        """Test get statistics."""
        stats = engine.get_stats()

        assert "total_orders" in stats
        assert "filled_orders" in stats
        assert "win_rate" in stats

    @pytest.mark.asyncio
    async def test_trade_callback(self, engine):
        """Test trade completion callback."""
        callback_called = False
        received_trade = None

        def on_trade(trade):
            nonlocal callback_called, received_trade
            callback_called = True
            received_trade = trade

        engine.on_trade_completed = on_trade

        engine.update_price("BTCUSDT", Decimal("50000"))
        await engine.place_order("BTCUSDT", "buy", Decimal("0.1"), OrderType.MARKET)

        engine.update_price("BTCUSDT", Decimal("51000"))
        await engine.close_position("BTCUSDT")

        assert callback_called
        assert received_trade is not None
        assert received_trade.pnl > 0

    def test_reset(self, engine):
        """Test engine reset."""
        engine.update_price("BTCUSDT", Decimal("50000"))
        engine.stats["total_orders"] = 10

        engine.reset()

        assert engine.balance == Decimal("10000")
        assert len(engine.positions) == 0
        assert engine.stats["total_orders"] == 0


class TestPaperOrder:
    """Tests for paper order."""

    def test_order_creation(self):
        """Test order creation."""
        order = PaperOrder(
            id="order_1",
            symbol="BTCUSDT",
            side="buy",
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        assert order.id == "order_1"
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == Decimal("0")


class TestPaperPosition:
    """Tests for paper position."""

    def test_position_creation(self):
        """Test position creation."""
        position = PaperPosition(
            symbol="BTCUSDT",
            side="long",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
        )

        assert position.symbol == "BTCUSDT"
        assert position.unrealized_pnl == Decimal("0")
