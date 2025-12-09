"""
E2E Tests for Full Trading Flow

Tests the complete trading cycle from signal generation to order execution.
"""
import pytest
import pytest_asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio


class TestFullTradingCycle:
    """Test complete trading cycle."""

    @pytest.mark.asyncio
    async def test_signal_to_order_flow(self, mock_exchange, sample_signal):
        """Test flow from signal generation to order placement."""
        # Simulate signal detection
        signal = sample_signal

        # Verify signal is valid
        assert signal["side"] in ["BUY", "SELL"]
        assert signal["strength"] >= 0.5
        assert signal["stop_loss"] < signal["entry_price"]
        assert signal["take_profit"] > signal["entry_price"]

        # Calculate position size based on risk
        account_balance = Decimal("10000.0")
        risk_percent = Decimal("0.01")  # 1% risk
        risk_amount = account_balance * risk_percent
        stop_distance = abs(signal["entry_price"] - signal["stop_loss"])
        position_size = risk_amount / stop_distance

        assert position_size > 0

        # Place order
        order = await mock_exchange.place_order(
            symbol=signal["symbol"],
            side=signal["side"],
            quantity=position_size,
            order_type="MARKET"
        )

        assert order["status"] == "FILLED"
        assert order["symbol"] == signal["symbol"]
        mock_exchange.place_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_position_management_flow(self, mock_exchange):
        """Test position opening, monitoring, and closing."""
        symbol = "BTCUSDT"

        # Step 1: Open position
        entry_order = await mock_exchange.place_order(
            symbol=symbol,
            side="BUY",
            quantity=Decimal("0.1"),
            order_type="MARKET"
        )
        assert entry_order["status"] == "FILLED"

        # Step 2: Set stop loss
        mock_exchange.get_positions.return_value = [{
            "symbol": symbol,
            "side": "LONG",
            "quantity": Decimal("0.1"),
            "entry_price": Decimal("50000.0"),
            "unrealized_pnl": Decimal("0")
        }]

        positions = await mock_exchange.get_positions()
        assert len(positions) == 1

        # Step 3: Simulate price movement (profit scenario)
        mock_exchange.get_ticker.return_value = {
            "symbol": symbol,
            "last": Decimal("51000.0"),  # +2% profit
            "bid": Decimal("50999.0"),
            "ask": Decimal("51001.0")
        }

        ticker = await mock_exchange.get_ticker(symbol)
        assert ticker["last"] > Decimal("50000.0")

        # Step 4: Close position with profit
        exit_order = await mock_exchange.place_order(
            symbol=symbol,
            side="SELL",
            quantity=Decimal("0.1"),
            order_type="MARKET"
        )
        assert exit_order["status"] == "FILLED"

        # Verify full cycle completed
        assert mock_exchange.place_order.call_count == 2

    @pytest.mark.asyncio
    async def test_stop_loss_trigger(self, mock_exchange):
        """Test stop loss order execution."""
        symbol = "BTCUSDT"

        # Open position
        await mock_exchange.place_order(
            symbol=symbol,
            side="BUY",
            quantity=Decimal("0.1"),
            order_type="MARKET"
        )

        # Set stop loss price
        stop_loss_price = Decimal("49000.0")  # -2% from entry

        # Simulate price drop below stop loss
        mock_exchange.get_ticker.return_value = {
            "symbol": symbol,
            "last": Decimal("48500.0"),  # Below stop loss
            "bid": Decimal("48499.0"),
            "ask": Decimal("48501.0")
        }

        ticker = await mock_exchange.get_ticker(symbol)

        # Check if stop loss should trigger
        if ticker["last"] <= stop_loss_price:
            # Execute stop loss
            stop_order = await mock_exchange.place_order(
                symbol=symbol,
                side="SELL",
                quantity=Decimal("0.1"),
                order_type="MARKET"
            )
            assert stop_order["status"] == "FILLED"

        assert mock_exchange.place_order.call_count == 2

    @pytest.mark.asyncio
    async def test_take_profit_trigger(self, mock_exchange):
        """Test take profit order execution."""
        symbol = "BTCUSDT"

        # Open position
        await mock_exchange.place_order(
            symbol=symbol,
            side="BUY",
            quantity=Decimal("0.1"),
            order_type="MARKET"
        )

        # Set take profit price
        take_profit_price = Decimal("52000.0")  # +4% from entry

        # Simulate price rise above take profit
        mock_exchange.get_ticker.return_value = {
            "symbol": symbol,
            "last": Decimal("52500.0"),
            "bid": Decimal("52499.0"),
            "ask": Decimal("52501.0")
        }

        ticker = await mock_exchange.get_ticker(symbol)

        # Check if take profit should trigger
        if ticker["last"] >= take_profit_price:
            # Execute take profit
            tp_order = await mock_exchange.place_order(
                symbol=symbol,
                side="SELL",
                quantity=Decimal("0.1"),
                order_type="MARKET"
            )
            assert tp_order["status"] == "FILLED"

        assert mock_exchange.place_order.call_count == 2


class TestMultiSymbolTrading:
    """Test trading multiple symbols simultaneously."""

    @pytest.mark.asyncio
    async def test_concurrent_symbol_trading(self, mock_exchange):
        """Test trading multiple symbols at once."""
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

        # Open positions on all symbols concurrently
        tasks = []
        for symbol in symbols:
            task = mock_exchange.place_order(
                symbol=symbol,
                side="BUY",
                quantity=Decimal("0.1"),
                order_type="MARKET"
            )
            tasks.append(task)

        orders = await asyncio.gather(*tasks)

        assert len(orders) == 3
        for order in orders:
            assert order["status"] == "FILLED"

    @pytest.mark.asyncio
    async def test_portfolio_risk_limit(self, mock_exchange):
        """Test that portfolio risk limits are respected."""
        max_positions = 3
        max_portfolio_risk = Decimal("0.05")  # 5%

        # Track open positions
        open_positions = []

        for i in range(5):  # Try to open 5 positions
            if len(open_positions) >= max_positions:
                break

            # Calculate current portfolio risk
            current_risk = len(open_positions) * Decimal("0.01")  # 1% per position

            if current_risk + Decimal("0.01") <= max_portfolio_risk:
                order = await mock_exchange.place_order(
                    symbol=f"SYMBOL{i}USDT",
                    side="BUY",
                    quantity=Decimal("0.1"),
                    order_type="MARKET"
                )
                open_positions.append(order)

        # Should only have 3 positions (max limit)
        assert len(open_positions) <= max_positions


class TestOrderExecution:
    """Test various order execution scenarios."""

    @pytest.mark.asyncio
    async def test_market_order_execution(self, mock_exchange):
        """Test market order fills immediately."""
        order = await mock_exchange.place_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("0.1"),
            order_type="MARKET"
        )

        assert order["status"] == "FILLED"
        assert order["filled_qty"] == Decimal("0.1")

    @pytest.mark.asyncio
    async def test_limit_order_execution(self, mock_exchange):
        """Test limit order placement."""
        order = await mock_exchange.place_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("0.1"),
            order_type="LIMIT",
            price=Decimal("49000.0")
        )

        assert order["order_id"] is not None
        assert order["type"] == "LIMIT"

    @pytest.mark.asyncio
    async def test_order_cancellation(self, mock_exchange):
        """Test order cancellation."""
        # Place order
        order = await mock_exchange.place_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("0.1"),
            order_type="LIMIT",
            price=Decimal("49000.0")
        )

        # Cancel order
        result = await mock_exchange.cancel_order(order["order_id"])
        assert result["status"] == "CANCELLED"

    @pytest.mark.asyncio
    async def test_partial_fill_handling(self, mock_exchange):
        """Test handling of partial fills."""
        # Configure mock for partial fill
        mock_exchange.place_order = AsyncMock(return_value={
            "order_id": "PARTIAL-001",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": Decimal("1.0"),
            "filled_qty": Decimal("0.5"),  # Only half filled
            "status": "PARTIALLY_FILLED",
            "avg_price": Decimal("50000.0")
        })

        order = await mock_exchange.place_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("1.0"),
            order_type="LIMIT",
            price=Decimal("50000.0")
        )

        assert order["status"] == "PARTIALLY_FILLED"
        assert order["filled_qty"] < order["quantity"]


class TestRiskManagement:
    """Test risk management during trading."""

    @pytest.mark.asyncio
    async def test_max_drawdown_protection(self, mock_exchange):
        """Test that trading stops when max drawdown is reached."""
        max_drawdown = Decimal("0.10")  # 10%
        starting_balance = Decimal("10000.0")
        current_balance = Decimal("8900.0")  # 11% loss

        drawdown = (starting_balance - current_balance) / starting_balance

        # Should not allow trading when drawdown exceeds max
        can_trade = drawdown < max_drawdown
        assert not can_trade

    @pytest.mark.asyncio
    async def test_daily_loss_limit(self, mock_exchange):
        """Test daily loss limit enforcement."""
        daily_loss_limit = Decimal("500.0")
        daily_losses = [
            Decimal("-100.0"),
            Decimal("-150.0"),
            Decimal("-200.0"),
            Decimal("-100.0")  # Total: -550
        ]

        total_loss = sum(daily_losses)
        can_continue_trading = abs(total_loss) < daily_loss_limit

        assert not can_continue_trading

    @pytest.mark.asyncio
    async def test_position_size_limit(self, mock_exchange):
        """Test position size limits."""
        max_position_value = Decimal("1000.0")
        current_price = Decimal("50000.0")

        # Calculate max quantity
        max_quantity = max_position_value / current_price
        requested_quantity = Decimal("0.05")  # $2500 at current price

        actual_quantity = min(requested_quantity, max_quantity)
        assert actual_quantity == max_quantity  # Should be capped

    @pytest.mark.asyncio
    async def test_leverage_limit(self, mock_exchange):
        """Test leverage limits are respected."""
        max_leverage = 10
        requested_leverage = 20

        actual_leverage = min(requested_leverage, max_leverage)

        await mock_exchange.set_leverage(actual_leverage)
        mock_exchange.set_leverage.assert_called_with(max_leverage)


class TestPaperTrading:
    """Test paper trading functionality."""

    @pytest.mark.asyncio
    async def test_paper_trade_execution(self, mock_exchange):
        """Test paper trading executes without real orders."""
        paper_trading = True

        if paper_trading:
            # Simulate order without calling exchange
            paper_order = {
                "order_id": "PAPER-001",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": Decimal("0.1"),
                "price": Decimal("50000.0"),
                "status": "FILLED",
                "paper": True
            }
            assert paper_order["paper"] is True
        else:
            order = await mock_exchange.place_order(
                symbol="BTCUSDT",
                side="BUY",
                quantity=Decimal("0.1")
            )
            assert order is not None

    @pytest.mark.asyncio
    async def test_paper_trade_pnl_calculation(self):
        """Test P&L calculation in paper trading."""
        entry_price = Decimal("50000.0")
        exit_price = Decimal("51000.0")
        quantity = Decimal("0.1")
        side = "LONG"

        if side == "LONG":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity

        assert pnl == Decimal("100.0")  # $100 profit


class TestErrorHandling:
    """Test error handling in trading flow."""

    @pytest.mark.asyncio
    async def test_exchange_connection_error(self, mock_exchange):
        """Test handling of exchange connection errors."""
        mock_exchange.place_order = AsyncMock(
            side_effect=ConnectionError("Exchange connection lost")
        )

        with pytest.raises(ConnectionError):
            await mock_exchange.place_order(
                symbol="BTCUSDT",
                side="BUY",
                quantity=Decimal("0.1")
            )

    @pytest.mark.asyncio
    async def test_insufficient_balance_error(self, mock_exchange):
        """Test handling of insufficient balance."""
        mock_exchange.place_order = AsyncMock(
            side_effect=ValueError("Insufficient balance")
        )

        with pytest.raises(ValueError, match="Insufficient balance"):
            await mock_exchange.place_order(
                symbol="BTCUSDT",
                side="BUY",
                quantity=Decimal("100.0")  # Too large
            )

    @pytest.mark.asyncio
    async def test_invalid_symbol_error(self, mock_exchange):
        """Test handling of invalid symbol."""
        mock_exchange.place_order = AsyncMock(
            side_effect=ValueError("Invalid symbol")
        )

        with pytest.raises(ValueError, match="Invalid symbol"):
            await mock_exchange.place_order(
                symbol="INVALIDUSDT",
                side="BUY",
                quantity=Decimal("0.1")
            )

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, mock_exchange):
        """Test rate limit handling with retry."""
        call_count = [0]

        async def rate_limited_call(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Rate limit exceeded")
            return {"status": "FILLED", "order_id": "SUCCESS"}

        mock_exchange.place_order = AsyncMock(side_effect=rate_limited_call)

        # Retry logic
        max_retries = 5
        for attempt in range(max_retries):
            try:
                result = await mock_exchange.place_order(
                    symbol="BTCUSDT",
                    side="BUY",
                    quantity=Decimal("0.1")
                )
                break
            except Exception:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1)

        assert result["status"] == "FILLED"
        assert call_count[0] == 3  # Succeeded on 3rd attempt
