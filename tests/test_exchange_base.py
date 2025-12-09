"""
Comprehensive tests for Exchange Base classes.

Tests cover:
- ExchangeCredentials
- ExchangeSymbolInfo
- ExchangeOrder
- ExchangePosition
- ExchangeBalance
- ExchangeTicker
- BaseExchangeAPI (abstract methods)
"""

import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from abc import ABC

import sys
sys.path.insert(0, '/home/sssmmmddd/Code/pro/crypto-scalper-bot')

from src.execution.exchange_base import (
    Exchange,
    ExchangeCredentials,
    ExchangeSymbolInfo,
    ExchangeOrder,
    ExchangePosition,
    ExchangeBalance,
    ExchangeTicker,
    BaseExchangeAPI,
)
from src.data.models import Side, OrderType, OrderStatus


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def credentials():
    """Create exchange credentials."""
    return ExchangeCredentials(
        api_key="test_api_key",
        api_secret="test_api_secret",
        passphrase="test_passphrase",
        testnet=True,
    )


@pytest.fixture
def symbol_info():
    """Create symbol info."""
    return ExchangeSymbolInfo(
        symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        price_precision=2,
        quantity_precision=6,
        min_quantity=Decimal("0.001"),
        max_quantity=Decimal("1000"),
        min_notional=Decimal("10"),
        tick_size=Decimal("0.01"),
        step_size=Decimal("0.000001"),
    )


@pytest.fixture
def exchange_order():
    """Create exchange order."""
    return ExchangeOrder(
        order_id="order123",
        client_order_id="client123",
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        status=OrderStatus.FILLED,
        quantity=Decimal("0.1"),
        filled_quantity=Decimal("0.1"),
        price=Decimal("50000"),
        avg_fill_price=Decimal("49995"),
        commission=Decimal("0.02"),
        created_at=1704067200000,
        updated_at=1704067201000,
    )


@pytest.fixture
def exchange_position():
    """Create exchange position."""
    return ExchangePosition(
        symbol="BTCUSDT",
        side=Side.BUY,
        quantity=Decimal("0.1"),
        entry_price=Decimal("50000"),
        mark_price=Decimal("51000"),
        unrealized_pnl=Decimal("100"),
        leverage=10,
        margin_type="cross",
        liquidation_price=Decimal("45000"),
    )


@pytest.fixture
def exchange_balance():
    """Create exchange balance."""
    return ExchangeBalance(
        asset="USDT",
        wallet_balance=Decimal("10000"),
        available_balance=Decimal("9500"),
        unrealized_pnl=Decimal("100"),
        margin_balance=Decimal("500"),
    )


@pytest.fixture
def exchange_ticker():
    """Create exchange ticker."""
    return ExchangeTicker(
        symbol="BTCUSDT",
        last_price=Decimal("50500"),
        bid_price=Decimal("50499"),
        ask_price=Decimal("50501"),
        volume_24h=Decimal("50000"),
        price_change_24h=Decimal("500"),
        timestamp=1704067200000,
    )


# =============================================================================
# Exchange Enum Tests
# =============================================================================

class TestExchangeEnum:
    """Tests for Exchange enum."""

    def test_exchange_values(self):
        """Test exchange enum values."""
        assert Exchange.BINANCE.value == "binance"
        assert Exchange.BYBIT.value == "bybit"
        assert Exchange.OKX.value == "okx"
        assert Exchange.KRAKEN.value == "kraken"
        assert Exchange.KUCOIN.value == "kucoin"
        assert Exchange.GATEIO.value == "gateio"

    def test_exchange_from_string(self):
        """Test creating exchange from string."""
        assert Exchange("binance") == Exchange.BINANCE
        assert Exchange("bybit") == Exchange.BYBIT


# =============================================================================
# ExchangeCredentials Tests
# =============================================================================

class TestExchangeCredentials:
    """Tests for ExchangeCredentials."""

    def test_create_credentials(self, credentials):
        """Test creating credentials."""
        assert credentials.api_key == "test_api_key"
        assert credentials.api_secret == "test_api_secret"
        assert credentials.passphrase == "test_passphrase"
        assert credentials.testnet is True

    def test_credentials_defaults(self):
        """Test credential defaults."""
        creds = ExchangeCredentials(
            api_key="key",
            api_secret="secret",
        )

        assert creds.passphrase is None
        assert creds.testnet is True

    def test_credentials_mainnet(self):
        """Test mainnet credentials."""
        creds = ExchangeCredentials(
            api_key="key",
            api_secret="secret",
            testnet=False,
        )

        assert creds.testnet is False


# =============================================================================
# ExchangeSymbolInfo Tests
# =============================================================================

class TestExchangeSymbolInfo:
    """Tests for ExchangeSymbolInfo."""

    def test_create_symbol_info(self, symbol_info):
        """Test creating symbol info."""
        assert symbol_info.symbol == "BTCUSDT"
        assert symbol_info.base_asset == "BTC"
        assert symbol_info.quote_asset == "USDT"
        assert symbol_info.price_precision == 2
        assert symbol_info.quantity_precision == 6

    def test_symbol_constraints(self, symbol_info):
        """Test symbol trading constraints."""
        assert symbol_info.min_quantity == Decimal("0.001")
        assert symbol_info.max_quantity == Decimal("1000")
        assert symbol_info.min_notional == Decimal("10")
        assert symbol_info.tick_size == Decimal("0.01")
        assert symbol_info.step_size == Decimal("0.000001")


# =============================================================================
# ExchangeOrder Tests
# =============================================================================

class TestExchangeOrder:
    """Tests for ExchangeOrder."""

    def test_create_order(self, exchange_order):
        """Test creating order."""
        assert exchange_order.order_id == "order123"
        assert exchange_order.client_order_id == "client123"
        assert exchange_order.symbol == "BTCUSDT"
        assert exchange_order.side == Side.BUY
        assert exchange_order.order_type == OrderType.LIMIT
        assert exchange_order.status == OrderStatus.FILLED

    def test_order_quantities(self, exchange_order):
        """Test order quantities."""
        assert exchange_order.quantity == Decimal("0.1")
        assert exchange_order.filled_quantity == Decimal("0.1")
        assert exchange_order.price == Decimal("50000")
        assert exchange_order.avg_fill_price == Decimal("49995")

    def test_order_timestamps(self, exchange_order):
        """Test order timestamps."""
        assert exchange_order.created_at == 1704067200000
        assert exchange_order.updated_at == 1704067201000

    def test_market_order(self):
        """Test market order (no price)."""
        order = ExchangeOrder(
            order_id="market123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            price=None,
            avg_fill_price=Decimal("50000"),
            commission=Decimal("0.02"),
            created_at=1704067200000,
            updated_at=1704067200000,
        )

        assert order.price is None
        assert order.order_type == OrderType.MARKET


# =============================================================================
# ExchangePosition Tests
# =============================================================================

class TestExchangePosition:
    """Tests for ExchangePosition."""

    def test_create_position(self, exchange_position):
        """Test creating position."""
        assert exchange_position.symbol == "BTCUSDT"
        assert exchange_position.side == Side.BUY
        assert exchange_position.quantity == Decimal("0.1")
        assert exchange_position.entry_price == Decimal("50000")

    def test_position_pnl(self, exchange_position):
        """Test position P&L."""
        assert exchange_position.mark_price == Decimal("51000")
        assert exchange_position.unrealized_pnl == Decimal("100")

    def test_position_leverage(self, exchange_position):
        """Test position leverage."""
        assert exchange_position.leverage == 10
        assert exchange_position.margin_type == "cross"

    def test_position_liquidation(self, exchange_position):
        """Test liquidation price."""
        assert exchange_position.liquidation_price == Decimal("45000")

    def test_short_position(self):
        """Test short position."""
        position = ExchangePosition(
            symbol="BTCUSDT",
            side=Side.SELL,
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("49000"),
            unrealized_pnl=Decimal("100"),
            leverage=10,
            margin_type="isolated",
            liquidation_price=Decimal("55000"),
        )

        assert position.side == Side.SELL
        assert position.margin_type == "isolated"


# =============================================================================
# ExchangeBalance Tests
# =============================================================================

class TestExchangeBalance:
    """Tests for ExchangeBalance."""

    def test_create_balance(self, exchange_balance):
        """Test creating balance."""
        assert exchange_balance.asset == "USDT"
        assert exchange_balance.wallet_balance == Decimal("10000")
        assert exchange_balance.available_balance == Decimal("9500")

    def test_balance_components(self, exchange_balance):
        """Test balance components."""
        assert exchange_balance.unrealized_pnl == Decimal("100")
        assert exchange_balance.margin_balance == Decimal("500")

    def test_multiple_assets(self):
        """Test multiple asset balances."""
        usdt = ExchangeBalance(
            asset="USDT",
            wallet_balance=Decimal("10000"),
            available_balance=Decimal("9500"),
            unrealized_pnl=Decimal("0"),
            margin_balance=Decimal("500"),
        )

        btc = ExchangeBalance(
            asset="BTC",
            wallet_balance=Decimal("1.5"),
            available_balance=Decimal("1.0"),
            unrealized_pnl=Decimal("0"),
            margin_balance=Decimal("0.5"),
        )

        assert usdt.asset == "USDT"
        assert btc.asset == "BTC"


# =============================================================================
# ExchangeTicker Tests
# =============================================================================

class TestExchangeTicker:
    """Tests for ExchangeTicker."""

    def test_create_ticker(self, exchange_ticker):
        """Test creating ticker."""
        assert exchange_ticker.symbol == "BTCUSDT"
        assert exchange_ticker.last_price == Decimal("50500")
        assert exchange_ticker.bid_price == Decimal("50499")
        assert exchange_ticker.ask_price == Decimal("50501")

    def test_ticker_volume(self, exchange_ticker):
        """Test ticker volume."""
        assert exchange_ticker.volume_24h == Decimal("50000")
        assert exchange_ticker.price_change_24h == Decimal("500")

    def test_ticker_spread(self, exchange_ticker):
        """Test bid-ask spread."""
        spread = exchange_ticker.ask_price - exchange_ticker.bid_price
        assert spread == Decimal("2")


# =============================================================================
# Mock Exchange Implementation for Testing
# =============================================================================

class MockExchangeAPI(BaseExchangeAPI):
    """Mock implementation of BaseExchangeAPI for testing."""

    @property
    def exchange(self) -> Exchange:
        return Exchange.BINANCE

    @property
    def base_url(self) -> str:
        return "https://testnet.binancefuture.com"

    @property
    def ws_url(self) -> str:
        return "wss://testnet.binancefuture.com/ws"

    async def connect(self) -> None:
        self._session = MagicMock()

    async def close(self) -> None:
        self._session = None

    async def ping(self) -> bool:
        return True

    async def get_exchange_info(self):
        return {}

    async def get_ticker(self, symbol: str):
        return ExchangeTicker(
            symbol=symbol,
            last_price=Decimal("50000"),
            bid_price=Decimal("49999"),
            ask_price=Decimal("50001"),
            volume_24h=Decimal("10000"),
            price_change_24h=Decimal("100"),
            timestamp=1704067200000,
        )

    async def get_orderbook(self, symbol: str, limit: int = 20):
        return {"bids": [], "asks": []}

    async def get_recent_trades(self, symbol: str, limit: int = 100):
        return []

    async def get_klines(self, symbol: str, interval: str, limit: int = 100):
        return []

    async def get_balance(self):
        return [ExchangeBalance(
            asset="USDT",
            wallet_balance=Decimal("10000"),
            available_balance=Decimal("10000"),
            unrealized_pnl=Decimal("0"),
            margin_balance=Decimal("0"),
        )]

    async def get_positions(self, symbol=None):
        return []

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        return True

    async def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        return True

    async def place_order(
        self,
        symbol: str,
        side: Side,
        order_type: OrderType,
        quantity: Decimal,
        price=None,
        stop_price=None,
        reduce_only: bool = False,
        client_order_id=None,
    ):
        return ExchangeOrder(
            order_id="mock123",
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            status=OrderStatus.NEW,
            quantity=quantity,
            filled_quantity=Decimal("0"),
            price=price,
            avg_fill_price=None,
            commission=Decimal("0"),
            created_at=1704067200000,
            updated_at=1704067200000,
        )

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        return True

    async def cancel_all_orders(self, symbol: str) -> int:
        return 0

    async def get_order(self, symbol: str, order_id: str):
        return ExchangeOrder(
            order_id=order_id,
            client_order_id=None,
            symbol=symbol,
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            price=Decimal("50000"),
            avg_fill_price=Decimal("50000"),
            commission=Decimal("0.02"),
            created_at=1704067200000,
            updated_at=1704067200000,
        )

    async def get_open_orders(self, symbol=None):
        return []

    async def get_order_history(self, symbol: str, limit: int = 50):
        return []

    async def get_funding_rate(self, symbol: str) -> Decimal:
        return Decimal("0.0001")

    async def get_funding_history(self, symbol: str, limit: int = 100):
        return []


# =============================================================================
# BaseExchangeAPI Tests
# =============================================================================

class TestBaseExchangeAPI:
    """Tests for BaseExchangeAPI."""

    @pytest.fixture
    def mock_api(self, credentials):
        """Create mock API instance."""
        return MockExchangeAPI(credentials)

    @pytest.mark.asyncio
    async def test_initialize_api(self, mock_api, credentials):
        """Test API initialization."""
        assert mock_api.credentials == credentials
        assert mock_api.testnet is True

    @pytest.mark.asyncio
    async def test_connect(self, mock_api):
        """Test connecting to exchange."""
        await mock_api.connect()
        assert mock_api._session is not None

    @pytest.mark.asyncio
    async def test_close(self, mock_api):
        """Test closing connection."""
        await mock_api.connect()
        await mock_api.close()
        assert mock_api._session is None

    @pytest.mark.asyncio
    async def test_ping(self, mock_api):
        """Test ping."""
        result = await mock_api.ping()
        assert result is True

    @pytest.mark.asyncio
    async def test_get_ticker(self, mock_api):
        """Test getting ticker."""
        ticker = await mock_api.get_ticker("BTCUSDT")

        assert ticker.symbol == "BTCUSDT"
        assert ticker.last_price == Decimal("50000")

    @pytest.mark.asyncio
    async def test_get_balance(self, mock_api):
        """Test getting balance."""
        balances = await mock_api.get_balance()

        assert len(balances) == 1
        assert balances[0].asset == "USDT"

    @pytest.mark.asyncio
    async def test_place_order(self, mock_api):
        """Test placing order."""
        order = await mock_api.place_order(
            symbol="BTCUSDT",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )

        assert order.order_id == "mock123"
        assert order.symbol == "BTCUSDT"
        assert order.side == Side.BUY

    @pytest.mark.asyncio
    async def test_place_market_order(self, mock_api):
        """Test placing market order helper."""
        order = await mock_api.place_market_order(
            symbol="BTCUSDT",
            side=Side.BUY,
            quantity=Decimal("0.1"),
        )

        assert order.order_type == OrderType.MARKET

    @pytest.mark.asyncio
    async def test_place_limit_order(self, mock_api):
        """Test placing limit order helper."""
        order = await mock_api.place_limit_order(
            symbol="BTCUSDT",
            side=Side.SELL,
            quantity=Decimal("0.1"),
            price=Decimal("51000"),
        )

        assert order.order_type == OrderType.LIMIT
        assert order.side == Side.SELL

    @pytest.mark.asyncio
    async def test_cancel_order(self, mock_api):
        """Test cancelling order."""
        result = await mock_api.cancel_order("BTCUSDT", "order123")
        assert result is True

    @pytest.mark.asyncio
    async def test_set_leverage(self, mock_api):
        """Test setting leverage."""
        result = await mock_api.set_leverage("BTCUSDT", 10)
        assert result is True

    @pytest.mark.asyncio
    async def test_get_funding_rate(self, mock_api):
        """Test getting funding rate."""
        rate = await mock_api.get_funding_rate("BTCUSDT")
        assert rate == Decimal("0.0001")

    def test_normalize_symbol(self, mock_api):
        """Test symbol normalization."""
        assert mock_api.normalize_symbol("btc-usdt") == "BTCUSDT"
        assert mock_api.normalize_symbol("BTC/USDT") == "BTCUSDT"
        assert mock_api.normalize_symbol("BTCUSDT") == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_close_position_no_position(self, mock_api):
        """Test closing position when none exists."""
        result = await mock_api.close_position("BTCUSDT")
        assert result is None


# =============================================================================
# Exchange Properties Tests
# =============================================================================

class TestExchangeProperties:
    """Tests for exchange properties."""

    @pytest.fixture
    def mock_api(self, credentials):
        """Create mock API instance."""
        return MockExchangeAPI(credentials)

    def test_exchange_property(self, mock_api):
        """Test exchange property."""
        assert mock_api.exchange == Exchange.BINANCE

    def test_base_url_property(self, mock_api):
        """Test base URL property."""
        assert "binancefuture.com" in mock_api.base_url

    def test_ws_url_property(self, mock_api):
        """Test WebSocket URL property."""
        assert mock_api.ws_url.startswith("wss://")


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_quantity_order(self):
        """Test order with zero quantity."""
        order = ExchangeOrder(
            order_id="zero123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.CANCELED,
            quantity=Decimal("0"),
            filled_quantity=Decimal("0"),
            price=Decimal("50000"),
            avg_fill_price=None,
            commission=Decimal("0"),
            created_at=1704067200000,
            updated_at=1704067200000,
        )

        assert order.quantity == Decimal("0")

    def test_very_small_quantity(self):
        """Test very small quantity."""
        order = ExchangeOrder(
            order_id="small123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            quantity=Decimal("0.000001"),
            filled_quantity=Decimal("0.000001"),
            price=None,
            avg_fill_price=Decimal("50000"),
            commission=Decimal("0.00000001"),
            created_at=1704067200000,
            updated_at=1704067200000,
        )

        assert order.quantity == Decimal("0.000001")

    def test_position_without_liquidation(self):
        """Test position without liquidation price."""
        position = ExchangePosition(
            symbol="BTCUSDT",
            side=Side.BUY,
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            leverage=1,
            margin_type="cross",
            liquidation_price=None,
        )

        assert position.liquidation_price is None
        assert position.leverage == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
