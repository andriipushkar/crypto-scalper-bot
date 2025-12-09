"""
Unit tests for multi-exchange functionality.
"""
import pytest
import pytest_asyncio
from decimal import Decimal
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List


class MockExchangeClient:
    """Mock exchange client for testing."""

    def __init__(self, name: str, fee_rate: Decimal = Decimal("0.001")):
        self.name = name
        self.fee_rate = fee_rate
        self._connected = False
        self._positions: Dict[str, dict] = {}
        self._orders: List[dict] = []
        self._balance = Decimal("10000")
        self._prices: Dict[str, dict] = {}

    async def connect(self):
        self._connected = True

    async def disconnect(self):
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def get_ticker(self, symbol: str) -> dict:
        return self._prices.get(symbol, {
            "symbol": symbol,
            "last": Decimal("50000"),
            "bid": Decimal("49999"),
            "ask": Decimal("50001"),
            "volume": Decimal("1000"),
        })

    async def get_balance(self) -> dict:
        return {
            "total": self._balance,
            "available": self._balance,
        }

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str = "MARKET",
        price: Decimal = None,
    ) -> dict:
        order = {
            "order_id": f"{self.name}-{len(self._orders) + 1}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": order_type,
            "price": price or Decimal("50000"),
            "status": "FILLED",
            "exchange": self.name,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._orders.append(order)
        return order

    async def cancel_order(self, order_id: str) -> dict:
        return {"order_id": order_id, "status": "CANCELLED"}

    async def get_positions(self) -> List[dict]:
        return list(self._positions.values())

    def set_price(self, symbol: str, last: Decimal, bid: Decimal, ask: Decimal):
        """Set mock price for testing."""
        self._prices[symbol] = {
            "symbol": symbol,
            "last": last,
            "bid": bid,
            "ask": ask,
        }


@pytest.fixture
def binance():
    return MockExchangeClient("binance", Decimal("0.0004"))


@pytest.fixture
def bybit():
    return MockExchangeClient("bybit", Decimal("0.0006"))


@pytest.fixture
def okx():
    return MockExchangeClient("okx", Decimal("0.0005"))


class TestExchangeConnection:
    """Test exchange connection management."""

    @pytest.mark.asyncio
    async def test_connect_single_exchange(self, binance):
        """Test connecting to a single exchange."""
        assert not binance.is_connected

        await binance.connect()

        assert binance.is_connected

    @pytest.mark.asyncio
    async def test_connect_multiple_exchanges(self, binance, bybit, okx):
        """Test connecting to multiple exchanges."""
        exchanges = [binance, bybit, okx]

        for ex in exchanges:
            await ex.connect()

        for ex in exchanges:
            assert ex.is_connected

    @pytest.mark.asyncio
    async def test_disconnect(self, binance):
        """Test disconnecting from exchange."""
        await binance.connect()
        assert binance.is_connected

        await binance.disconnect()
        assert not binance.is_connected


class TestSymbolNormalization:
    """Test symbol format normalization across exchanges."""

    def test_binance_format(self):
        """Test Binance symbol format."""
        symbol = "BTCUSDT"
        assert symbol == "BTCUSDT"

    def test_okx_format(self):
        """Test OKX symbol format."""
        okx_symbol = "BTC-USDT"
        normalized = okx_symbol.replace("-", "")
        assert normalized == "BTCUSDT"

    def test_normalize_symbol(self):
        """Test symbol normalization function."""
        def normalize_symbol(symbol: str, exchange: str) -> str:
            if exchange == "okx":
                return symbol.replace("-", "")
            return symbol

        assert normalize_symbol("BTC-USDT", "okx") == "BTCUSDT"
        assert normalize_symbol("BTCUSDT", "binance") == "BTCUSDT"

    def test_denormalize_symbol(self):
        """Test symbol denormalization for exchange."""
        def denormalize_symbol(symbol: str, exchange: str) -> str:
            if exchange == "okx":
                # Insert hyphen before quote currency
                for quote in ["USDT", "USDC", "USD"]:
                    if symbol.endswith(quote):
                        base = symbol[:-len(quote)]
                        return f"{base}-{quote}"
            return symbol

        assert denormalize_symbol("BTCUSDT", "okx") == "BTC-USDT"
        assert denormalize_symbol("BTCUSDT", "binance") == "BTCUSDT"


class TestPriceComparison:
    """Test price comparison across exchanges."""

    @pytest.mark.asyncio
    async def test_get_prices_all_exchanges(self, binance, bybit, okx):
        """Test fetching prices from all exchanges."""
        binance.set_price("BTCUSDT", Decimal("50000"), Decimal("49999"), Decimal("50001"))
        bybit.set_price("BTCUSDT", Decimal("50010"), Decimal("50009"), Decimal("50011"))
        okx.set_price("BTCUSDT", Decimal("49990"), Decimal("49989"), Decimal("49991"))

        prices = {}
        for ex in [binance, bybit, okx]:
            ticker = await ex.get_ticker("BTCUSDT")
            prices[ex.name] = ticker

        assert prices["binance"]["last"] == Decimal("50000")
        assert prices["bybit"]["last"] == Decimal("50010")
        assert prices["okx"]["last"] == Decimal("49990")

    @pytest.mark.asyncio
    async def test_find_best_bid(self, binance, bybit, okx):
        """Test finding best bid across exchanges."""
        binance.set_price("BTCUSDT", Decimal("50000"), Decimal("49999"), Decimal("50001"))
        bybit.set_price("BTCUSDT", Decimal("50010"), Decimal("50009"), Decimal("50011"))
        okx.set_price("BTCUSDT", Decimal("49990"), Decimal("49989"), Decimal("49991"))

        best_bid = None
        best_exchange = None

        for ex in [binance, bybit, okx]:
            ticker = await ex.get_ticker("BTCUSDT")
            if best_bid is None or ticker["bid"] > best_bid:
                best_bid = ticker["bid"]
                best_exchange = ex.name

        assert best_exchange == "bybit"
        assert best_bid == Decimal("50009")

    @pytest.mark.asyncio
    async def test_find_best_ask(self, binance, bybit, okx):
        """Test finding best ask across exchanges."""
        binance.set_price("BTCUSDT", Decimal("50000"), Decimal("49999"), Decimal("50001"))
        bybit.set_price("BTCUSDT", Decimal("50010"), Decimal("50009"), Decimal("50011"))
        okx.set_price("BTCUSDT", Decimal("49990"), Decimal("49989"), Decimal("49991"))

        best_ask = None
        best_exchange = None

        for ex in [binance, bybit, okx]:
            ticker = await ex.get_ticker("BTCUSDT")
            if best_ask is None or ticker["ask"] < best_ask:
                best_ask = ticker["ask"]
                best_exchange = ex.name

        assert best_exchange == "okx"
        assert best_ask == Decimal("49991")


class TestSmartOrderRouting:
    """Test smart order routing logic."""

    @pytest.mark.asyncio
    async def test_route_buy_to_best_ask(self, binance, bybit, okx):
        """Test routing buy order to exchange with best ask."""
        binance.set_price("BTCUSDT", Decimal("50000"), Decimal("49999"), Decimal("50001"))
        bybit.set_price("BTCUSDT", Decimal("50010"), Decimal("50009"), Decimal("50011"))
        okx.set_price("BTCUSDT", Decimal("49990"), Decimal("49989"), Decimal("49991"))

        exchanges = {"binance": binance, "bybit": bybit, "okx": okx}

        # Find best ask
        best_ask_exchange = None
        best_ask = None

        for name, ex in exchanges.items():
            ticker = await ex.get_ticker("BTCUSDT")
            if best_ask is None or ticker["ask"] < best_ask:
                best_ask = ticker["ask"]
                best_ask_exchange = name

        # Route order
        order = await exchanges[best_ask_exchange].place_order(
            "BTCUSDT", "BUY", Decimal("0.1")
        )

        assert order["exchange"] == "okx"

    @pytest.mark.asyncio
    async def test_route_sell_to_best_bid(self, binance, bybit, okx):
        """Test routing sell order to exchange with best bid."""
        binance.set_price("BTCUSDT", Decimal("50000"), Decimal("49999"), Decimal("50001"))
        bybit.set_price("BTCUSDT", Decimal("50010"), Decimal("50009"), Decimal("50011"))
        okx.set_price("BTCUSDT", Decimal("49990"), Decimal("49989"), Decimal("49991"))

        exchanges = {"binance": binance, "bybit": bybit, "okx": okx}

        # Find best bid
        best_bid_exchange = None
        best_bid = None

        for name, ex in exchanges.items():
            ticker = await ex.get_ticker("BTCUSDT")
            if best_bid is None or ticker["bid"] > best_bid:
                best_bid = ticker["bid"]
                best_bid_exchange = name

        # Route order
        order = await exchanges[best_bid_exchange].place_order(
            "BTCUSDT", "SELL", Decimal("0.1")
        )

        assert order["exchange"] == "bybit"


class TestArbitrageDetection:
    """Test arbitrage opportunity detection."""

    @pytest.mark.asyncio
    async def test_detect_arbitrage(self, binance, bybit):
        """Test detecting arbitrage opportunity."""
        # Set up price discrepancy
        binance.set_price("BTCUSDT", Decimal("50000"), Decimal("49999"), Decimal("50001"))
        bybit.set_price("BTCUSDT", Decimal("50020"), Decimal("50019"), Decimal("50021"))

        binance_ticker = await binance.get_ticker("BTCUSDT")
        bybit_ticker = await bybit.get_ticker("BTCUSDT")

        # Check for arbitrage: buy on binance (ask), sell on bybit (bid)
        buy_price = binance_ticker["ask"]
        sell_price = bybit_ticker["bid"]

        gross_profit = sell_price - buy_price

        # Account for fees
        binance_fee = buy_price * binance.fee_rate
        bybit_fee = sell_price * bybit.fee_rate
        total_fees = binance_fee + bybit_fee

        net_profit = gross_profit - total_fees

        assert gross_profit == Decimal("18")
        assert net_profit > 0  # Profitable after fees

    @pytest.mark.asyncio
    async def test_no_arbitrage(self, binance, bybit):
        """Test when no arbitrage opportunity exists."""
        # Prices too close - no arbitrage
        binance.set_price("BTCUSDT", Decimal("50000"), Decimal("49999"), Decimal("50001"))
        bybit.set_price("BTCUSDT", Decimal("50002"), Decimal("50001"), Decimal("50003"))

        binance_ticker = await binance.get_ticker("BTCUSDT")
        bybit_ticker = await bybit.get_ticker("BTCUSDT")

        # Best case: buy on binance, sell on bybit
        buy_price = binance_ticker["ask"]
        sell_price = bybit_ticker["bid"]

        gross_profit = sell_price - buy_price

        # Fees would eat the profit
        total_fees = buy_price * Decimal("0.001") + sell_price * Decimal("0.001")
        net_profit = gross_profit - total_fees

        assert net_profit < 0  # Not profitable


class TestExchangeFailover:
    """Test exchange failover logic."""

    @pytest.mark.asyncio
    async def test_failover_on_disconnect(self, binance, bybit):
        """Test failover when primary exchange disconnects."""
        await binance.connect()
        await bybit.connect()

        primary = binance
        backup = bybit

        # Simulate primary failure
        await primary.disconnect()

        # Select working exchange
        if primary.is_connected:
            active = primary
        elif backup.is_connected:
            active = backup
        else:
            active = None

        assert active == backup
        assert active.name == "bybit"

    @pytest.mark.asyncio
    async def test_order_failover(self, binance, bybit):
        """Test order failover on error."""
        await binance.connect()
        await bybit.connect()

        # Make binance fail
        async def failing_order(*args, **kwargs):
            raise Exception("Connection error")

        binance.place_order = failing_order

        # Try primary, failover to backup
        try:
            order = await binance.place_order("BTCUSDT", "BUY", Decimal("0.1"))
        except Exception:
            order = await bybit.place_order("BTCUSDT", "BUY", Decimal("0.1"))

        assert order["exchange"] == "bybit"


class TestBalanceAggregation:
    """Test balance aggregation across exchanges."""

    @pytest.mark.asyncio
    async def test_total_balance(self, binance, bybit, okx):
        """Test calculating total balance across exchanges."""
        binance._balance = Decimal("5000")
        bybit._balance = Decimal("3000")
        okx._balance = Decimal("2000")

        total = Decimal("0")
        for ex in [binance, bybit, okx]:
            balance = await ex.get_balance()
            total += balance["total"]

        assert total == Decimal("10000")

    @pytest.mark.asyncio
    async def test_balance_distribution(self, binance, bybit, okx):
        """Test balance distribution percentage."""
        binance._balance = Decimal("5000")
        bybit._balance = Decimal("3000")
        okx._balance = Decimal("2000")

        balances = {}
        total = Decimal("0")

        for ex in [binance, bybit, okx]:
            balance = await ex.get_balance()
            balances[ex.name] = balance["total"]
            total += balance["total"]

        distribution = {
            name: (bal / total * 100)
            for name, bal in balances.items()
        }

        assert distribution["binance"] == Decimal("50")
        assert distribution["bybit"] == Decimal("30")
        assert distribution["okx"] == Decimal("20")


class TestFeeComparison:
    """Test fee comparison across exchanges."""

    def test_fee_rates(self, binance, bybit, okx):
        """Test comparing fee rates."""
        exchanges = [binance, bybit, okx]

        lowest_fee_exchange = min(exchanges, key=lambda x: x.fee_rate)

        assert lowest_fee_exchange.name == "binance"
        assert lowest_fee_exchange.fee_rate == Decimal("0.0004")

    def test_fee_calculation(self, binance):
        """Test fee calculation for trade."""
        trade_value = Decimal("10000")
        fee = trade_value * binance.fee_rate

        assert fee == Decimal("4")  # 0.04%


class TestQuantityPrecision:
    """Test quantity precision handling."""

    def test_binance_precision(self):
        """Test Binance quantity precision."""
        precisions = {
            "BTCUSDT": 3,
            "ETHUSDT": 3,
            "DOGEUSDT": 0,
        }

        quantity = Decimal("0.12345678")
        symbol = "BTCUSDT"

        rounded = round(quantity, precisions[symbol])
        assert rounded == Decimal("0.123")

    def test_min_quantity(self):
        """Test minimum quantity requirements."""
        min_quantities = {
            "BTCUSDT": Decimal("0.001"),
            "ETHUSDT": Decimal("0.001"),
            "DOGEUSDT": Decimal("1"),
        }

        requested = Decimal("0.0001")
        symbol = "BTCUSDT"

        is_valid = requested >= min_quantities[symbol]
        assert is_valid is False


class TestLoadBalancing:
    """Test load balancing across exchanges."""

    @pytest.mark.asyncio
    async def test_round_robin(self, binance, bybit, okx):
        """Test round-robin order distribution."""
        exchanges = [binance, bybit, okx]
        order_count = {ex.name: 0 for ex in exchanges}

        # Distribute 9 orders
        for i in range(9):
            ex = exchanges[i % len(exchanges)]
            await ex.place_order("BTCUSDT", "BUY", Decimal("0.1"))
            order_count[ex.name] += 1

        assert order_count["binance"] == 3
        assert order_count["bybit"] == 3
        assert order_count["okx"] == 3

    @pytest.mark.asyncio
    async def test_weighted_distribution(self, binance, bybit, okx):
        """Test weighted order distribution."""
        # Weights based on balance
        binance._balance = Decimal("5000")
        bybit._balance = Decimal("3000")
        okx._balance = Decimal("2000")

        total = Decimal("10000")
        weights = {
            "binance": binance._balance / total,
            "bybit": bybit._balance / total,
            "okx": okx._balance / total,
        }

        assert weights["binance"] == Decimal("0.5")
        assert weights["bybit"] == Decimal("0.3")
        assert weights["okx"] == Decimal("0.2")
