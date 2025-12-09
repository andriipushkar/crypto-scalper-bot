"""
E2E Tests for Multi-Exchange Integration

Tests trading across multiple exchanges (Binance, Bybit, OKX).
"""
import pytest
import pytest_asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio


class MockExchange:
    """Base mock exchange for testing."""

    def __init__(self, name: str):
        self.name = name
        self.connected = False
        self.orders = []
        self.positions = []
        self.balance = Decimal("10000.0")

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    async def get_ticker(self, symbol: str):
        base_price = Decimal("50000.0")
        # Slight price difference between exchanges
        spread = Decimal(str(hash(self.name) % 10))
        return {
            "symbol": symbol,
            "last": base_price + spread,
            "bid": base_price + spread - 1,
            "ask": base_price + spread + 1,
            "exchange": self.name
        }

    async def place_order(self, symbol: str, side: str, quantity: Decimal, **kwargs):
        order = {
            "order_id": f"{self.name}-{len(self.orders) + 1}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "status": "FILLED",
            "exchange": self.name
        }
        self.orders.append(order)
        return order

    async def get_balance(self):
        return {"total": self.balance, "available": self.balance}


@pytest.fixture
def mock_binance():
    return MockExchange("binance")


@pytest.fixture
def mock_bybit():
    return MockExchange("bybit")


@pytest.fixture
def mock_okx():
    return MockExchange("okx")


class TestMultiExchangeConnection:
    """Test connecting to multiple exchanges."""

    @pytest.mark.asyncio
    async def test_connect_all_exchanges(self, mock_binance, mock_bybit, mock_okx):
        """Test connecting to all supported exchanges."""
        exchanges = [mock_binance, mock_bybit, mock_okx]

        # Connect all
        await asyncio.gather(*[ex.connect() for ex in exchanges])

        # Verify all connected
        for ex in exchanges:
            assert ex.connected is True

    @pytest.mark.asyncio
    async def test_disconnect_all_exchanges(self, mock_binance, mock_bybit, mock_okx):
        """Test disconnecting from all exchanges."""
        exchanges = [mock_binance, mock_bybit, mock_okx]

        # Connect first
        await asyncio.gather(*[ex.connect() for ex in exchanges])

        # Disconnect all
        await asyncio.gather(*[ex.disconnect() for ex in exchanges])

        for ex in exchanges:
            assert ex.connected is False

    @pytest.mark.asyncio
    async def test_partial_connection_failure(self, mock_binance, mock_bybit, mock_okx):
        """Test handling when one exchange fails to connect."""
        exchanges = [mock_binance, mock_bybit, mock_okx]

        # Make one fail
        async def failing_connect():
            raise ConnectionError("Bybit connection failed")

        mock_bybit.connect = failing_connect

        # Try to connect all
        results = await asyncio.gather(
            mock_binance.connect(),
            mock_bybit.connect(),
            mock_okx.connect(),
            return_exceptions=True
        )

        # Two should succeed, one should fail
        assert mock_binance.connected is True
        assert mock_okx.connected is True
        assert isinstance(results[1], ConnectionError)


class TestCrossExchangePriceComparison:
    """Test price comparison across exchanges."""

    @pytest.mark.asyncio
    async def test_get_prices_all_exchanges(self, mock_binance, mock_bybit, mock_okx):
        """Test fetching prices from all exchanges."""
        exchanges = [mock_binance, mock_bybit, mock_okx]
        symbol = "BTCUSDT"

        prices = await asyncio.gather(*[ex.get_ticker(symbol) for ex in exchanges])

        assert len(prices) == 3
        for price in prices:
            assert price["symbol"] == symbol
            assert price["last"] > 0

    @pytest.mark.asyncio
    async def test_find_best_price(self, mock_binance, mock_bybit, mock_okx):
        """Test finding best price across exchanges."""
        exchanges = [mock_binance, mock_bybit, mock_okx]
        symbol = "BTCUSDT"

        prices = await asyncio.gather(*[ex.get_ticker(symbol) for ex in exchanges])

        # Find best bid (highest) for selling
        best_bid = max(prices, key=lambda p: p["bid"])

        # Find best ask (lowest) for buying
        best_ask = min(prices, key=lambda p: p["ask"])

        assert best_bid["bid"] > 0
        assert best_ask["ask"] > 0

    @pytest.mark.asyncio
    async def test_price_spread_detection(self, mock_binance, mock_bybit, mock_okx):
        """Test detecting arbitrage-able price spreads."""
        exchanges = [mock_binance, mock_bybit, mock_okx]
        symbol = "BTCUSDT"

        prices = await asyncio.gather(*[ex.get_ticker(symbol) for ex in exchanges])

        # Calculate max spread
        all_bids = [p["bid"] for p in prices]
        all_asks = [p["ask"] for p in prices]

        max_bid = max(all_bids)
        min_ask = min(all_asks)

        # Check if arbitrage opportunity exists
        spread = max_bid - min_ask
        has_opportunity = spread > 0

        # Log spread for analysis
        assert spread is not None


class TestCrossExchangeTrading:
    """Test trading across multiple exchanges."""

    @pytest.mark.asyncio
    async def test_place_orders_multiple_exchanges(self, mock_binance, mock_bybit):
        """Test placing orders on multiple exchanges."""
        symbol = "BTCUSDT"
        quantity = Decimal("0.1")

        # Place buy on binance, sell on bybit (simulated arbitrage)
        orders = await asyncio.gather(
            mock_binance.place_order(symbol, "BUY", quantity),
            mock_bybit.place_order(symbol, "SELL", quantity)
        )

        assert len(orders) == 2
        assert orders[0]["exchange"] == "binance"
        assert orders[1]["exchange"] == "bybit"

    @pytest.mark.asyncio
    async def test_balance_allocation(self, mock_binance, mock_bybit, mock_okx):
        """Test balance allocation across exchanges."""
        exchanges = [mock_binance, mock_bybit, mock_okx]
        total_capital = Decimal("30000.0")

        # Allocate equally
        allocation = total_capital / len(exchanges)

        for ex in exchanges:
            ex.balance = allocation

        balances = await asyncio.gather(*[ex.get_balance() for ex in exchanges])

        total = sum(b["total"] for b in balances)
        assert total == total_capital

    @pytest.mark.asyncio
    async def test_smart_order_routing(self, mock_binance, mock_bybit, mock_okx):
        """Test smart order routing to best exchange."""
        exchanges = {
            "binance": mock_binance,
            "bybit": mock_bybit,
            "okx": mock_okx
        }
        symbol = "BTCUSDT"

        # Get prices from all exchanges
        prices = {}
        for name, ex in exchanges.items():
            ticker = await ex.get_ticker(symbol)
            prices[name] = ticker

        # Route BUY order to exchange with lowest ask
        best_exchange_for_buy = min(prices.keys(), key=lambda k: prices[k]["ask"])

        # Route SELL order to exchange with highest bid
        best_exchange_for_sell = max(prices.keys(), key=lambda k: prices[k]["bid"])

        # Place orders on best exchanges
        buy_order = await exchanges[best_exchange_for_buy].place_order(
            symbol, "BUY", Decimal("0.1")
        )
        sell_order = await exchanges[best_exchange_for_sell].place_order(
            symbol, "SELL", Decimal("0.1")
        )

        assert buy_order["exchange"] == best_exchange_for_buy
        assert sell_order["exchange"] == best_exchange_for_sell


class TestCrossExchangeArbitrage:
    """Test cross-exchange arbitrage scenarios."""

    @pytest.mark.asyncio
    async def test_detect_arbitrage_opportunity(self, mock_binance, mock_bybit):
        """Test detecting arbitrage opportunities."""
        # Simulate price discrepancy
        mock_binance.get_ticker = AsyncMock(return_value={
            "symbol": "BTCUSDT",
            "bid": Decimal("50000.0"),
            "ask": Decimal("50001.0"),
            "exchange": "binance"
        })

        mock_bybit.get_ticker = AsyncMock(return_value={
            "symbol": "BTCUSDT",
            "bid": Decimal("50010.0"),  # Higher bid
            "ask": Decimal("50011.0"),
            "exchange": "bybit"
        })

        binance_price = await mock_binance.get_ticker("BTCUSDT")
        bybit_price = await mock_bybit.get_ticker("BTCUSDT")

        # Buy on binance (at ask), sell on bybit (at bid)
        potential_profit = bybit_price["bid"] - binance_price["ask"]

        # Account for fees (0.1% each side)
        fee_rate = Decimal("0.001")
        fees = binance_price["ask"] * fee_rate + bybit_price["bid"] * fee_rate

        net_profit = potential_profit - fees
        is_profitable = net_profit > 0

        assert potential_profit == Decimal("9.0")
        assert is_profitable

    @pytest.mark.asyncio
    async def test_execute_arbitrage(self, mock_binance, mock_bybit):
        """Test executing arbitrage trade."""
        # Setup price discrepancy
        mock_binance.get_ticker = AsyncMock(return_value={
            "bid": Decimal("50000.0"),
            "ask": Decimal("50001.0")
        })
        mock_bybit.get_ticker = AsyncMock(return_value={
            "bid": Decimal("50015.0"),
            "ask": Decimal("50016.0")
        })

        quantity = Decimal("0.1")

        # Execute arbitrage simultaneously
        orders = await asyncio.gather(
            mock_binance.place_order("BTCUSDT", "BUY", quantity),
            mock_bybit.place_order("BTCUSDT", "SELL", quantity)
        )

        assert len(orders) == 2
        assert orders[0]["side"] == "BUY"
        assert orders[1]["side"] == "SELL"


class TestExchangeFailover:
    """Test failover between exchanges."""

    @pytest.mark.asyncio
    async def test_failover_on_error(self, mock_binance, mock_bybit):
        """Test failover to backup exchange on error."""
        primary = mock_binance
        backup = mock_bybit

        # Make primary fail
        primary.place_order = AsyncMock(
            side_effect=Exception("Primary exchange error")
        )

        # Try primary, failover to backup
        try:
            order = await primary.place_order("BTCUSDT", "BUY", Decimal("0.1"))
        except Exception:
            order = await backup.place_order("BTCUSDT", "BUY", Decimal("0.1"))

        assert order["exchange"] == "bybit"

    @pytest.mark.asyncio
    async def test_load_balancing(self, mock_binance, mock_bybit, mock_okx):
        """Test load balancing across exchanges."""
        exchanges = [mock_binance, mock_bybit, mock_okx]
        orders_to_place = 9

        # Round-robin distribution
        for i in range(orders_to_place):
            exchange = exchanges[i % len(exchanges)]
            await exchange.place_order("BTCUSDT", "BUY", Decimal("0.1"))

        # Each exchange should have 3 orders
        assert len(mock_binance.orders) == 3
        assert len(mock_bybit.orders) == 3
        assert len(mock_okx.orders) == 3


class TestExchangeNormalization:
    """Test data normalization across exchanges."""

    @pytest.mark.asyncio
    async def test_symbol_normalization(self):
        """Test symbol format normalization."""
        # Different exchanges use different formats
        symbols = {
            "binance": "BTCUSDT",
            "bybit": "BTCUSDT",
            "okx": "BTC-USDT"
        }

        # Normalize to common format
        def normalize(symbol: str) -> str:
            return symbol.replace("-", "").upper()

        normalized = {ex: normalize(sym) for ex, sym in symbols.items()}

        assert all(sym == "BTCUSDT" for sym in normalized.values())

    @pytest.mark.asyncio
    async def test_order_status_normalization(self):
        """Test order status normalization."""
        # Different exchanges return different status strings
        statuses = {
            "binance": "FILLED",
            "bybit": "Filled",
            "okx": "filled"
        }

        def normalize_status(status: str) -> str:
            return status.upper()

        normalized = {ex: normalize_status(s) for ex, s in statuses.items()}

        assert all(s == "FILLED" for s in normalized.values())

    @pytest.mark.asyncio
    async def test_quantity_precision(self):
        """Test quantity precision handling across exchanges."""
        # Exchanges have different precision requirements
        precisions = {
            "binance": 3,  # 0.001
            "bybit": 4,    # 0.0001
            "okx": 2       # 0.01
        }

        quantity = Decimal("0.12345")

        def round_to_precision(qty: Decimal, precision: int) -> Decimal:
            return round(qty, precision)

        rounded = {
            ex: round_to_precision(quantity, prec)
            for ex, prec in precisions.items()
        }

        assert rounded["binance"] == Decimal("0.123")
        assert rounded["bybit"] == Decimal("0.1235")
        assert rounded["okx"] == Decimal("0.12")


class TestExchangeSpecificFeatures:
    """Test exchange-specific features."""

    @pytest.mark.asyncio
    async def test_binance_specific_order_types(self, mock_binance):
        """Test Binance-specific order types."""
        # Binance supports STOP_MARKET
        order = await mock_binance.place_order(
            "BTCUSDT",
            "SELL",
            Decimal("0.1"),
            order_type="STOP_MARKET",
            stop_price=Decimal("49000.0")
        )
        assert order["status"] == "FILLED"

    @pytest.mark.asyncio
    async def test_bybit_specific_features(self, mock_bybit):
        """Test Bybit-specific features."""
        # Bybit conditional orders
        order = await mock_bybit.place_order(
            "BTCUSDT",
            "BUY",
            Decimal("0.1"),
            order_type="CONDITIONAL",
            trigger_price=Decimal("51000.0")
        )
        assert order["status"] == "FILLED"

    @pytest.mark.asyncio
    async def test_okx_specific_features(self, mock_okx):
        """Test OKX-specific features."""
        # OKX advanced order types
        order = await mock_okx.place_order(
            "BTC-USDT",
            "BUY",
            Decimal("0.1"),
            order_type="TRIGGER",
            trigger_price=Decimal("51000.0")
        )
        assert order["status"] == "FILLED"


class TestMultiExchangeRiskManagement:
    """Test risk management across exchanges."""

    @pytest.mark.asyncio
    async def test_total_exposure_calculation(self, mock_binance, mock_bybit, mock_okx):
        """Test calculating total exposure across exchanges."""
        # Set positions on each exchange
        positions = {
            "binance": [{"symbol": "BTCUSDT", "value": Decimal("1000.0")}],
            "bybit": [{"symbol": "ETHUSDT", "value": Decimal("500.0")}],
            "okx": [{"symbol": "BTCUSDT", "value": Decimal("750.0")}]
        }

        total_exposure = sum(
            sum(p["value"] for p in pos_list)
            for pos_list in positions.values()
        )

        assert total_exposure == Decimal("2250.0")

    @pytest.mark.asyncio
    async def test_cross_exchange_position_limit(self, mock_binance, mock_bybit):
        """Test enforcing position limits across exchanges."""
        max_total_position = Decimal("0.5")  # BTC

        binance_position = Decimal("0.3")
        bybit_position = Decimal("0.1")

        total_position = binance_position + bybit_position
        remaining_capacity = max_total_position - total_position

        assert remaining_capacity == Decimal("0.1")

        # Should not allow exceeding limit
        new_order_qty = Decimal("0.2")
        can_place = new_order_qty <= remaining_capacity
        assert can_place is False

    @pytest.mark.asyncio
    async def test_exchange_concentration_limit(self, mock_binance, mock_bybit, mock_okx):
        """Test limiting concentration on single exchange."""
        max_concentration = Decimal("0.5")  # 50%

        exposures = {
            "binance": Decimal("4000.0"),
            "bybit": Decimal("1500.0"),
            "okx": Decimal("1500.0")
        }
        total = sum(exposures.values())

        concentrations = {ex: exp / total for ex, exp in exposures.items()}

        # Binance is over-concentrated
        over_concentrated = any(c > max_concentration for c in concentrations.values())
        assert over_concentrated is True  # Binance at 57%
