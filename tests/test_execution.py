"""
Unit tests for Execution module.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from src.execution.binance_api import (
    BinanceFuturesAPI,
    BinanceAPIError,
    OrderExecutor,
)
from src.data.models import (
    Order,
    OrderType,
    OrderStatus,
    TimeInForce,
    Side,
    Position,
    Signal,
    SignalType,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def api():
    """Create API instance without real credentials."""
    api = BinanceFuturesAPI(
        api_key="test_key",
        api_secret="test_secret",
        testnet=True,
    )
    return api


@pytest.fixture
def mock_api():
    """Create mocked API for testing."""
    api = BinanceFuturesAPI(
        api_key="test_key",
        api_secret="test_secret",
        testnet=True,
    )
    api._request = AsyncMock()
    return api


@pytest.fixture
def sample_order_response():
    return {
        "orderId": 12345,
        "clientOrderId": "test_client_order",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "type": "LIMIT",
        "origQty": "0.001",
        "price": "50000.00",
        "status": "NEW",
        "executedQty": "0",
        "avgPrice": "0",
        "time": 1704110400000,
        "updateTime": 1704110400000,
        "timeInForce": "GTC",
    }


@pytest.fixture
def sample_position_response():
    return {
        "symbol": "BTCUSDT",
        "positionAmt": "0.001",
        "entryPrice": "50000.00",
        "markPrice": "50100.00",
        "liquidationPrice": "45000.00",
        "unrealizedProfit": "0.10",
        "leverage": "5",
        "marginType": "crossed",
    }


@pytest.fixture
def long_signal():
    return Signal(
        strategy="test",
        signal_type=SignalType.LONG,
        symbol="BTCUSDT",
        timestamp=datetime.utcnow(),
        strength=0.8,
        price=Decimal("50000"),
        metadata={"position_size": Decimal("0.001")},
    )


# =============================================================================
# BinanceFuturesAPI Tests
# =============================================================================

class TestBinanceFuturesAPI:
    """Tests for BinanceFuturesAPI class."""

    def test_init_testnet(self, api):
        assert api.testnet == True
        assert "testnet" in api.base_url

    def test_init_mainnet(self):
        api = BinanceFuturesAPI(
            api_key="test",
            api_secret="secret",
            testnet=False,
        )
        assert api.testnet == False
        assert "fapi.binance.com" in api.base_url

    def test_has_credentials(self, api):
        assert api.has_credentials == True

    def test_no_credentials(self):
        api = BinanceFuturesAPI(testnet=True)
        # Will be False unless env vars are set
        # Just test that property works
        assert isinstance(api.has_credentials, bool)

    def test_sign(self, api):
        params = {"symbol": "BTCUSDT", "timestamp": 1704110400000}
        signature = api._sign(params)

        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 hex length


class TestOrderParsing:
    """Tests for order response parsing."""

    def test_parse_order(self, api, sample_order_response):
        order = api._parse_order(sample_order_response)

        assert isinstance(order, Order)
        assert order.order_id == "12345"
        assert order.symbol == "BTCUSDT"
        assert order.side == Side.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == Decimal("0.001")
        assert order.price == Decimal("50000.00")
        assert order.status == OrderStatus.NEW

    def test_parse_order_filled(self, api):
        response = {
            "orderId": 12345,
            "clientOrderId": "test",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "MARKET",
            "origQty": "0.001",
            "price": "0",
            "status": "FILLED",
            "executedQty": "0.001",
            "avgPrice": "50050.00",
            "time": 1704110400000,
            "updateTime": 1704110401000,
            "timeInForce": "GTC",
        }

        order = api._parse_order(response)

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == Decimal("0.001")
        assert order.average_price == Decimal("50050.00")


@pytest.mark.asyncio
class TestAPIRequests:
    """Tests for API request handling."""

    async def test_get_exchange_info(self, mock_api):
        mock_api._request.return_value = {
            "symbols": [{"symbol": "BTCUSDT"}]
        }

        info = await mock_api.get_exchange_info()

        assert "symbols" in info
        mock_api._request.assert_called_once()

    async def test_get_ticker_price(self, mock_api):
        mock_api._request.return_value = {
            "symbol": "BTCUSDT",
            "price": "50000.00"
        }

        price = await mock_api.get_ticker_price("BTCUSDT")

        assert price == Decimal("50000.00")

    async def test_get_balance(self, mock_api):
        mock_api._request.return_value = {
            "assets": [
                {"asset": "USDT", "availableBalance": "100.00"},
                {"asset": "BTC", "availableBalance": "0.001"},
            ]
        }

        balance = await mock_api.get_balance("USDT")

        assert balance == Decimal("100.00")

    async def test_get_positions(self, mock_api, sample_position_response):
        mock_api._request.return_value = {
            "positions": [sample_position_response]
        }

        positions = await mock_api.get_positions()

        assert len(positions) == 1
        assert positions[0].symbol == "BTCUSDT"
        assert positions[0].side == Side.BUY
        assert positions[0].size == Decimal("0.001")

    async def test_get_positions_no_positions(self, mock_api):
        mock_api._request.return_value = {
            "positions": [
                {"symbol": "BTCUSDT", "positionAmt": "0"}  # Zero position
            ]
        }

        positions = await mock_api.get_positions()

        assert len(positions) == 0

    async def test_place_order(self, mock_api, sample_order_response):
        mock_api._request.return_value = sample_order_response
        mock_api._symbol_info = {
            "BTCUSDT": {"quantityPrecision": 3, "pricePrecision": 2}
        }

        order = await mock_api.place_order(
            symbol="BTCUSDT",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
        )

        assert order.order_id == "12345"
        assert order.side == Side.BUY

    async def test_place_market_order(self, mock_api, sample_order_response):
        sample_order_response["type"] = "MARKET"
        mock_api._request.return_value = sample_order_response
        mock_api._symbol_info = {
            "BTCUSDT": {"quantityPrecision": 3, "pricePrecision": 2}
        }

        order = await mock_api.place_market_order(
            symbol="BTCUSDT",
            side=Side.BUY,
            quantity=Decimal("0.001"),
        )

        assert order.order_type == OrderType.MARKET

    async def test_cancel_order(self, mock_api, sample_order_response):
        sample_order_response["status"] = "CANCELED"
        mock_api._request.return_value = sample_order_response

        order = await mock_api.cancel_order("BTCUSDT", "12345")

        assert order.status == OrderStatus.CANCELED

    async def test_set_leverage(self, mock_api):
        mock_api._request.return_value = {
            "leverage": 10,
            "symbol": "BTCUSDT"
        }

        result = await mock_api.set_leverage("BTCUSDT", 10)

        assert result["leverage"] == 10

    async def test_api_error_handling(self, mock_api):
        mock_api._request.side_effect = BinanceAPIError(-1000, "Test error")

        with pytest.raises(BinanceAPIError) as exc_info:
            await mock_api.get_ticker_price("BTCUSDT")

        assert exc_info.value.code == -1000


# =============================================================================
# BinanceAPIError Tests
# =============================================================================

class TestBinanceAPIError:
    """Tests for BinanceAPIError."""

    def test_error_message(self):
        error = BinanceAPIError(-1000, "Test error message")

        assert error.code == -1000
        assert error.message == "Test error message"
        assert "-1000" in str(error)
        assert "Test error" in str(error)


# =============================================================================
# OrderExecutor Tests
# =============================================================================

class TestOrderExecutor:
    """Tests for OrderExecutor class."""

    @pytest.fixture
    def executor(self, mock_api):
        return OrderExecutor(
            api=mock_api,
            risk_manager=None,
            use_limit_orders=False,
        )

    @pytest.fixture
    def executor_with_limit(self, mock_api):
        return OrderExecutor(
            api=mock_api,
            risk_manager=None,
            use_limit_orders=True,
            limit_offset_ticks=1,
        )

    @pytest.mark.asyncio
    async def test_execute_signal_market(self, executor, long_signal, sample_order_response):
        executor.api._request.return_value = sample_order_response
        executor.api._symbol_info = {
            "BTCUSDT": {"quantityPrecision": 3, "pricePrecision": 2}
        }

        order = await executor.execute_signal(long_signal)

        assert order is not None
        assert order.symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_execute_signal_limit(self, executor_with_limit, long_signal, sample_order_response):
        executor_with_limit.api._request.return_value = sample_order_response
        executor_with_limit.api._symbol_info = {
            "BTCUSDT": {"quantityPrecision": 3, "pricePrecision": 2}
        }

        order = await executor_with_limit.execute_signal(long_signal)

        assert order is not None

    @pytest.mark.asyncio
    async def test_execute_no_action_signal(self, executor):
        signal = Signal(
            strategy="test",
            signal_type=SignalType.NO_ACTION,
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            strength=0.0,
            price=Decimal("50000"),
        )

        order = await executor.execute_signal(signal)

        assert order is None

    @pytest.mark.asyncio
    async def test_execute_no_position_size(self, executor):
        signal = Signal(
            strategy="test",
            signal_type=SignalType.LONG,
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            strength=0.8,
            price=Decimal("50000"),
            metadata={},  # No position_size
        )

        order = await executor.execute_signal(signal)

        assert order is None

    @pytest.mark.asyncio
    async def test_execute_zero_position_size(self, executor):
        signal = Signal(
            strategy="test",
            signal_type=SignalType.LONG,
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            strength=0.8,
            price=Decimal("50000"),
            metadata={"position_size": Decimal("0")},
        )

        order = await executor.execute_signal(signal)

        assert order is None

    @pytest.mark.asyncio
    async def test_close_position(self, executor, sample_order_response):
        executor.api._request.return_value = sample_order_response
        executor.api._symbol_info = {
            "BTCUSDT": {"quantityPrecision": 3, "pricePrecision": 2}
        }

        position = Position(
            symbol="BTCUSDT",
            side=Side.BUY,
            size=Decimal("0.001"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50100"),
            liquidation_price=None,
            unrealized_pnl=Decimal("0.1"),
            realized_pnl=Decimal("0"),
            leverage=5,
            margin_type="CROSSED",
            updated_at=datetime.utcnow(),
        )

        order = await executor.close_position(position)

        assert order is not None

    @pytest.mark.asyncio
    async def test_execute_handles_api_error(self, executor, long_signal):
        executor.api._request.side_effect = BinanceAPIError(-1000, "Test error")
        executor.api._symbol_info = {
            "BTCUSDT": {"quantityPrecision": 3, "pricePrecision": 2}
        }

        order = await executor.execute_signal(long_signal)

        assert order is None  # Should handle error gracefully


class TestOrderExecutorWithRiskManager:
    """Tests for OrderExecutor with RiskManager."""

    @pytest.mark.asyncio
    async def test_execute_with_risk_manager(self, mock_api, sample_order_response):
        from src.risk.manager import RiskManager

        risk_manager = RiskManager({
            "total_capital": 100,
            "max_position_pct": 0.1,
        })

        executor = OrderExecutor(
            api=mock_api,
            risk_manager=risk_manager,
            use_limit_orders=False,
        )

        mock_api._request.return_value = sample_order_response
        mock_api._symbol_info = {
            "BTCUSDT": {"quantityPrecision": 3, "pricePrecision": 2}
        }

        signal = Signal(
            strategy="test",
            signal_type=SignalType.LONG,
            symbol="BTCUSDT",
            timestamp=datetime.utcnow(),
            strength=0.8,
            price=Decimal("50000"),
            metadata={},  # Risk manager will calculate size
        )

        # Risk manager calculates position size, order should be placed
        order = await executor.execute_signal(signal)

        assert order is not None
        assert order.symbol == "BTCUSDT"
        assert order.side == Side.BUY
