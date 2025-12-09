"""
E2E Test Configuration and Fixtures
"""
import asyncio
import os
from typing import AsyncGenerator, Generator
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# Set test environment
os.environ["ENVIRONMENT"] = "testnet"
os.environ["ENABLE_PAPER_TRADING"] = "true"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_e2e.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for session scope."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def mock_exchange():
    """Mock exchange client for E2E tests."""
    exchange = AsyncMock()

    # Mock account info
    exchange.get_account_balance.return_value = {
        "total": Decimal("10000.0"),
        "available": Decimal("9500.0"),
        "unrealized_pnl": Decimal("50.0")
    }

    # Mock positions
    exchange.get_positions.return_value = []

    # Mock ticker
    exchange.get_ticker.return_value = {
        "symbol": "BTCUSDT",
        "last": Decimal("50000.0"),
        "bid": Decimal("49999.0"),
        "ask": Decimal("50001.0"),
        "volume": Decimal("1000.0"),
        "change_24h": Decimal("2.5")
    }

    # Mock orderbook
    exchange.get_orderbook.return_value = {
        "bids": [(Decimal("49999.0"), Decimal("1.0")), (Decimal("49998.0"), Decimal("2.0"))],
        "asks": [(Decimal("50001.0"), Decimal("1.0")), (Decimal("50002.0"), Decimal("2.0"))],
        "timestamp": 1700000000000
    }

    # Mock klines
    exchange.get_klines.return_value = [
        {
            "timestamp": 1700000000000 + i * 60000,
            "open": Decimal("50000.0") + i,
            "high": Decimal("50100.0") + i,
            "low": Decimal("49900.0") + i,
            "close": Decimal("50050.0") + i,
            "volume": Decimal("100.0")
        }
        for i in range(100)
    ]

    # Mock order placement
    order_counter = [0]
    async def place_order(symbol, side, quantity, order_type="MARKET", price=None, **kwargs):
        order_counter[0] += 1
        return {
            "order_id": f"TEST-{order_counter[0]}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": order_type,
            "price": price or Decimal("50000.0"),
            "status": "FILLED",
            "filled_qty": quantity,
            "avg_price": Decimal("50000.0"),
            "timestamp": 1700000000000
        }

    exchange.place_order = AsyncMock(side_effect=place_order)
    exchange.cancel_order = AsyncMock(return_value={"status": "CANCELLED"})
    exchange.get_open_orders.return_value = []

    # Mock leverage
    exchange.set_leverage.return_value = {"leverage": 10}
    exchange.get_leverage.return_value = 10

    # Connection mocks
    exchange.connect = AsyncMock()
    exchange.disconnect = AsyncMock()
    exchange.is_connected = True

    yield exchange


@pytest_asyncio.fixture(scope="function")
async def mock_database():
    """Mock database for E2E tests."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    # Create in-memory database
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False
    )

    # Import models and create tables
    try:
        from src.database.models import Base
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except ImportError:
        pass  # Models not available

    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    yield async_session

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def mock_redis():
    """Mock Redis client for E2E tests."""
    redis = AsyncMock()
    cache = {}

    async def mock_get(key):
        return cache.get(key)

    async def mock_set(key, value, ex=None):
        cache[key] = value
        return True

    async def mock_delete(*keys):
        for key in keys:
            cache.pop(key, None)
        return len(keys)

    async def mock_exists(*keys):
        return sum(1 for k in keys if k in cache)

    redis.get = AsyncMock(side_effect=mock_get)
    redis.set = AsyncMock(side_effect=mock_set)
    redis.delete = AsyncMock(side_effect=mock_delete)
    redis.exists = AsyncMock(side_effect=mock_exists)
    redis.keys = AsyncMock(return_value=list(cache.keys()))
    redis.ping = AsyncMock(return_value=True)
    redis.close = AsyncMock()

    yield redis


@pytest_asyncio.fixture(scope="function")
async def test_client(mock_exchange, mock_database, mock_redis):
    """Create test client for API E2E tests."""
    try:
        from src.web.app import create_app

        app = create_app()

        # Override dependencies
        app.state.exchange = mock_exchange
        app.state.db_session = mock_database
        app.state.redis = mock_redis

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
    except ImportError:
        # Dashboard not available, yield mock client
        yield AsyncMock()


@pytest_asyncio.fixture(scope="function")
async def trading_engine(mock_exchange, mock_database, mock_redis):
    """Create trading engine for E2E tests."""
    try:
        from src.core.engine import TradingEngine
        from src.core.config import Config

        config = Config(
            symbols=["BTCUSDT", "ETHUSDT"],
            paper_trading=True,
            max_position_size=Decimal("0.1"),
            max_daily_loss=Decimal("100.0")
        )

        engine = TradingEngine(
            config=config,
            exchange=mock_exchange,
            db_session=mock_database,
            redis=mock_redis
        )

        yield engine

        await engine.stop()
    except ImportError:
        yield AsyncMock()


class MockWebSocket:
    """Mock WebSocket for testing."""

    def __init__(self):
        self.messages = []
        self.closed = False
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def send_json(self, data):
        self.messages.append(data)

    async def send_text(self, data):
        self.messages.append(data)

    async def receive_json(self):
        if self.messages:
            return self.messages.pop(0)
        return {"type": "ping"}

    async def close(self):
        self.closed = True


@pytest.fixture
def mock_websocket():
    """Create mock WebSocket."""
    return MockWebSocket()


@pytest.fixture
def sample_trade_data():
    """Sample trade data for testing."""
    return {
        "id": "trade-001",
        "symbol": "BTCUSDT",
        "side": "BUY",
        "entry_price": Decimal("50000.0"),
        "exit_price": Decimal("50500.0"),
        "quantity": Decimal("0.1"),
        "pnl": Decimal("50.0"),
        "pnl_percent": Decimal("1.0"),
        "entry_time": "2024-01-01T10:00:00Z",
        "exit_time": "2024-01-01T11:00:00Z",
        "strategy": "momentum",
        "fees": Decimal("5.0")
    }


@pytest.fixture
def sample_signal():
    """Sample trading signal for testing."""
    return {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "strength": 0.8,
        "strategy": "momentum",
        "entry_price": Decimal("50000.0"),
        "stop_loss": Decimal("49500.0"),
        "take_profit": Decimal("51000.0"),
        "timestamp": 1700000000000
    }
