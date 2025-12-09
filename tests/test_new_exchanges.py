"""
Tests for new exchange integrations (Kraken and KuCoin).
"""

import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp

from src.execution.exchange_base import (
    Exchange,
    ExchangeCredentials,
    ExchangeSymbolInfo,
    ExchangeOrder,
    ExchangePosition,
    ExchangeBalance,
    ExchangeTicker,
)
from src.execution.kraken_api import KrakenFuturesAPI, KrakenAPIError
from src.execution.kucoin_api import KuCoinFuturesAPI, KuCoinAPIError
from src.execution.exchange_factory import (
    ExchangeFactory,
    create_exchange,
    EXCHANGE_CLASSES,
)
from src.data.models import Side, OrderType, OrderStatus


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def kraken_credentials():
    """Create Kraken test credentials."""
    return ExchangeCredentials(
        api_key="test_api_key",
        api_secret="dGVzdF9hcGlfc2VjcmV0",  # base64 encoded
        testnet=True,
    )


@pytest.fixture
def kucoin_credentials():
    """Create KuCoin test credentials."""
    return ExchangeCredentials(
        api_key="test_api_key",
        api_secret="test_api_secret",
        passphrase="test_passphrase",
        testnet=True,
    )


@pytest.fixture
def kraken_api(kraken_credentials):
    """Create Kraken API instance."""
    return KrakenFuturesAPI(kraken_credentials)


@pytest.fixture
def kucoin_api(kucoin_credentials):
    """Create KuCoin API instance."""
    return KuCoinFuturesAPI(kucoin_credentials)


# =============================================================================
# Kraken API Tests
# =============================================================================

class TestKrakenFuturesAPI:
    """Tests for KrakenFuturesAPI."""

    def test_initialization(self, kraken_credentials):
        """Test API initialization."""
        api = KrakenFuturesAPI(kraken_credentials)
        assert api.exchange == Exchange.KRAKEN
        assert api.testnet is True

    def test_base_url_testnet(self, kraken_api):
        """Test testnet URL."""
        assert "demo-futures" in kraken_api.base_url

    def test_base_url_mainnet(self, kraken_credentials):
        """Test mainnet URL."""
        kraken_credentials.testnet = False
        api = KrakenFuturesAPI(kraken_credentials)
        assert "demo" not in api.base_url
        assert "futures.kraken.com" in api.base_url

    def test_ws_url(self, kraken_api):
        """Test WebSocket URL."""
        assert "ws" in kraken_api.ws_url

    def test_symbol_normalization(self, kraken_api):
        """Test symbol normalization."""
        assert kraken_api.normalize_symbol("BTCUSD") == "PI_XBTUSD"
        assert kraken_api.normalize_symbol("ETHUSD") == "PI_ETHUSD"
        assert kraken_api.normalize_symbol("PI_XBTUSD") == "PI_XBTUSD"

    def test_nonce_generation(self, kraken_api):
        """Test nonce is unique and increasing."""
        nonce1 = kraken_api._get_nonce()
        nonce2 = kraken_api._get_nonce()
        assert nonce2 > nonce1

    @pytest.mark.asyncio
    async def test_connect(self, kraken_api):
        """Test connection initialization."""
        await kraken_api.connect()
        assert kraken_api._session is not None
        await kraken_api.close()

    @pytest.mark.asyncio
    async def test_close(self, kraken_api):
        """Test connection close."""
        await kraken_api.connect()
        await kraken_api.close()
        # Session should be closed

    def test_parse_order(self, kraken_api):
        """Test order parsing."""
        order_data = {
            "order_id": "test123",
            "symbol": "PI_XBTUSD",
            "side": "buy",
            "orderType": "lmt",
            "size": "0.01",
            "status": "placed",
            "limitPrice": "50000",
            "receivedTime": 1700000000000,
        }
        order = kraken_api._parse_order(order_data)

        assert order.order_id == "test123"
        assert order.symbol == "PI_XBTUSD"
        assert order.side == Side.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.status == OrderStatus.NEW


class TestKrakenAPIError:
    """Tests for KrakenAPIError."""

    def test_error_creation(self):
        """Test error creation."""
        error = KrakenAPIError("TEST_CODE", "Test message")
        assert error.code == "TEST_CODE"
        assert error.message == "Test message"
        assert "TEST_CODE" in str(error)


# =============================================================================
# KuCoin API Tests
# =============================================================================

class TestKuCoinFuturesAPI:
    """Tests for KuCoinFuturesAPI."""

    def test_initialization(self, kucoin_credentials):
        """Test API initialization."""
        api = KuCoinFuturesAPI(kucoin_credentials)
        assert api.exchange == Exchange.KUCOIN
        assert api.testnet is True

    def test_base_url_testnet(self, kucoin_api):
        """Test testnet URL."""
        assert "sandbox" in kucoin_api.base_url

    def test_base_url_mainnet(self, kucoin_credentials):
        """Test mainnet URL."""
        kucoin_credentials.testnet = False
        api = KuCoinFuturesAPI(kucoin_credentials)
        assert "sandbox" not in api.base_url

    def test_ws_url(self, kucoin_api):
        """Test WebSocket URL."""
        assert "ws" in kucoin_api.ws_url

    def test_symbol_normalization(self, kucoin_api):
        """Test symbol normalization."""
        assert kucoin_api.normalize_symbol("BTCUSDT") == "XBTUSDTM"
        assert kucoin_api.normalize_symbol("ETHUSDT") == "ETHUSDTM"

    @pytest.mark.asyncio
    async def test_connect(self, kucoin_api):
        """Test connection initialization."""
        await kucoin_api.connect()
        assert kucoin_api._session is not None
        await kucoin_api.close()

    @pytest.mark.asyncio
    async def test_close(self, kucoin_api):
        """Test connection close."""
        await kucoin_api.connect()
        await kucoin_api.close()

    def test_parse_order(self, kucoin_api):
        """Test order parsing."""
        order_data = {
            "id": "test123",
            "clientOid": "client123",
            "symbol": "XBTUSDTM",
            "side": "buy",
            "type": "limit",
            "size": "1",
            "status": "open",
            "price": "50000",
            "dealSize": "0",
            "createdAt": 1700000000000,
        }
        order = kucoin_api._parse_order(order_data)

        assert order.order_id == "test123"
        assert order.client_order_id == "client123"
        assert order.symbol == "XBTUSDTM"
        assert order.side == Side.BUY


class TestKuCoinAPIError:
    """Tests for KuCoinAPIError."""

    def test_error_creation(self):
        """Test error creation."""
        error = KuCoinAPIError("400001", "Invalid parameter")
        assert error.code == "400001"
        assert error.message == "Invalid parameter"


# =============================================================================
# Exchange Factory Tests
# =============================================================================

class TestExchangeFactoryWithNewExchanges:
    """Tests for ExchangeFactory with new exchanges."""

    def test_kraken_in_registry(self):
        """Test Kraken is in exchange registry."""
        assert Exchange.KRAKEN in EXCHANGE_CLASSES
        assert EXCHANGE_CLASSES[Exchange.KRAKEN] == KrakenFuturesAPI

    def test_kucoin_in_registry(self):
        """Test KuCoin is in exchange registry."""
        assert Exchange.KUCOIN in EXCHANGE_CLASSES
        assert EXCHANGE_CLASSES[Exchange.KUCOIN] == KuCoinFuturesAPI

    def test_create_kraken(self, kraken_credentials):
        """Test creating Kraken API via factory."""
        factory = ExchangeFactory()
        api = factory.create(Exchange.KRAKEN, kraken_credentials)
        assert isinstance(api, KrakenFuturesAPI)

    def test_create_kucoin(self, kucoin_credentials):
        """Test creating KuCoin API via factory."""
        factory = ExchangeFactory()
        api = factory.create(Exchange.KUCOIN, kucoin_credentials)
        assert isinstance(api, KuCoinFuturesAPI)

    def test_create_exchange_function_kraken(self, kraken_credentials):
        """Test create_exchange convenience function for Kraken."""
        api = create_exchange("kraken", credentials=kraken_credentials)
        assert isinstance(api, KrakenFuturesAPI)

    def test_create_exchange_function_kucoin(self, kucoin_credentials):
        """Test create_exchange convenience function for KuCoin."""
        api = create_exchange("kucoin", credentials=kucoin_credentials)
        assert isinstance(api, KuCoinFuturesAPI)

    def test_create_exchange_error_message(self):
        """Test error message includes all supported exchanges."""
        with pytest.raises(ValueError) as exc_info:
            create_exchange("invalid_exchange")

        error_msg = str(exc_info.value)
        assert "kraken" in error_msg
        assert "kucoin" in error_msg
        assert "binance" in error_msg
        assert "bybit" in error_msg
        assert "okx" in error_msg


# =============================================================================
# Exchange Base Tests
# =============================================================================

class TestExchangeEnum:
    """Tests for Exchange enum."""

    def test_all_exchanges(self):
        """Test all exchanges are defined."""
        assert Exchange.BINANCE.value == "binance"
        assert Exchange.BYBIT.value == "bybit"
        assert Exchange.OKX.value == "okx"
        assert Exchange.KRAKEN.value == "kraken"
        assert Exchange.KUCOIN.value == "kucoin"

    def test_exchange_count(self):
        """Test total number of exchanges."""
        # BINANCE, BYBIT, OKX, KRAKEN, KUCOIN, GATEIO = 6
        assert len(Exchange) == 6


# =============================================================================
# Mock Request Tests
# =============================================================================

class TestKrakenMockedRequests:
    """Tests for Kraken API with mocked HTTP requests."""

    @pytest.mark.asyncio
    async def test_ping_success(self, kraken_api):
        """Test ping with mocked success response."""
        await kraken_api.connect()

        # Create proper async context manager mock
        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value={"result": "ok"})

        context_manager = MagicMock()
        context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        context_manager.__aexit__ = AsyncMock(return_value=None)

        with patch.object(
            kraken_api._session,
            "get",
            return_value=context_manager,
        ):
            result = await kraken_api.ping()
            assert result is True

        await kraken_api.close()

    @pytest.mark.asyncio
    async def test_get_ticker(self, kraken_api):
        """Test get_ticker with mocked response."""
        await kraken_api.connect()

        mock_ticker_data = {
            "result": {
                "tickers": [{
                    "symbol": "PI_XBTUSD",
                    "last": "50000",
                    "bid": "49990",
                    "ask": "50010",
                    "vol24h": "1000",
                    "change24h": "2.5",
                }]
            }
        }

        # Create proper async context manager mock
        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value=mock_ticker_data)

        context_manager = MagicMock()
        context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        context_manager.__aexit__ = AsyncMock(return_value=None)

        with patch.object(
            kraken_api._session,
            "get",
            return_value=context_manager,
        ):
            ticker = await kraken_api.get_ticker("PI_XBTUSD")
            assert ticker.symbol == "PI_XBTUSD"
            assert ticker.last_price == Decimal("50000")

        await kraken_api.close()


class TestKuCoinMockedRequests:
    """Tests for KuCoin API with mocked HTTP requests."""

    @pytest.mark.asyncio
    async def test_ping_success(self, kucoin_api):
        """Test ping with mocked success response."""
        await kucoin_api.connect()

        # Create proper async context manager mock
        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value={
            "code": "200000",
            "data": {"timestamp": 1700000000000}
        })

        context_manager = MagicMock()
        context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        context_manager.__aexit__ = AsyncMock(return_value=None)

        with patch.object(
            kucoin_api._session,
            "get",
            return_value=context_manager,
        ):
            result = await kucoin_api.ping()
            assert result is True

        await kucoin_api.close()

    @pytest.mark.asyncio
    async def test_get_ticker(self, kucoin_api):
        """Test get_ticker with mocked response."""
        await kucoin_api.connect()

        mock_ticker_data = {
            "code": "200000",
            "data": {
                "symbol": "XBTUSDTM",
                "price": "50000",
                "bestBidPrice": "49990",
                "bestAskPrice": "50010",
                "size": "1000",
                "priceChgPct": "0.025",
                "ts": 1700000000000000,
            }
        }

        # Create proper async context manager mock
        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value=mock_ticker_data)

        context_manager = MagicMock()
        context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        context_manager.__aexit__ = AsyncMock(return_value=None)

        with patch.object(
            kucoin_api._session,
            "get",
            return_value=context_manager,
        ):
            ticker = await kucoin_api.get_ticker("XBTUSDTM")
            assert ticker.symbol == "XBTUSDTM"
            assert ticker.last_price == Decimal("50000")

        await kucoin_api.close()
