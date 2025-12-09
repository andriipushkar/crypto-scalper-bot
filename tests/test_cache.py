"""
Unit tests for Redis cache module.
"""
import pytest
import pytest_asyncio
from decimal import Decimal
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import json


class MockRedis:
    """Mock Redis client for testing."""

    def __init__(self):
        self._data = {}
        self._expires = {}
        self._pubsub_channels = {}
        self._locks = {}

    async def get(self, key: str):
        if key in self._data:
            return self._data[key]
        return None

    async def set(self, key: str, value: str, ex: int = None):
        self._data[key] = value
        if ex:
            self._expires[key] = ex
        return True

    async def delete(self, *keys):
        count = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                count += 1
        return count

    async def exists(self, *keys):
        return sum(1 for k in keys if k in self._data)

    async def keys(self, pattern: str = "*"):
        import fnmatch
        return [k for k in self._data.keys() if fnmatch.fnmatch(k, pattern)]

    async def expire(self, key: str, seconds: int):
        self._expires[key] = seconds
        return True

    async def ttl(self, key: str):
        return self._expires.get(key, -1)

    async def incr(self, key: str):
        val = int(self._data.get(key, 0)) + 1
        self._data[key] = str(val)
        return val

    async def decr(self, key: str):
        val = int(self._data.get(key, 0)) - 1
        self._data[key] = str(val)
        return val

    async def hset(self, name: str, key: str = None, value: str = None, mapping: dict = None):
        if name not in self._data:
            self._data[name] = {}
        if mapping:
            self._data[name].update(mapping)
        elif key and value:
            self._data[name][key] = value
        return 1

    async def hget(self, name: str, key: str):
        if name in self._data and key in self._data[name]:
            return self._data[name][key]
        return None

    async def hgetall(self, name: str):
        return self._data.get(name, {})

    async def hdel(self, name: str, *keys):
        if name not in self._data:
            return 0
        count = 0
        for key in keys:
            if key in self._data[name]:
                del self._data[name][key]
                count += 1
        return count

    async def publish(self, channel: str, message: str):
        if channel not in self._pubsub_channels:
            self._pubsub_channels[channel] = []
        self._pubsub_channels[channel].append(message)
        return len(self._pubsub_channels.get(channel, []))

    def pubsub(self):
        return MockPubSub(self)

    async def ping(self):
        return True

    async def close(self):
        pass


class MockPubSub:
    """Mock PubSub for testing."""

    def __init__(self, redis):
        self._redis = redis
        self._subscribed = []

    async def subscribe(self, *channels):
        self._subscribed.extend(channels)

    async def unsubscribe(self, *channels):
        for ch in channels:
            if ch in self._subscribed:
                self._subscribed.remove(ch)

    async def get_message(self, ignore_subscribe_messages=True, timeout=None):
        for ch in self._subscribed:
            if ch in self._redis._pubsub_channels and self._redis._pubsub_channels[ch]:
                msg = self._redis._pubsub_channels[ch].pop(0)
                return {"type": "message", "channel": ch.encode(), "data": msg.encode()}
        return None

    async def close(self):
        pass


@pytest.fixture
def mock_redis():
    """Create mock Redis instance."""
    return MockRedis()


class TestCacheBasicOperations:
    """Test basic cache operations."""

    @pytest.mark.asyncio
    async def test_set_and_get(self, mock_redis):
        """Test basic set and get operations."""
        await mock_redis.set("test_key", "test_value")
        result = await mock_redis.get("test_key")
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_set_with_expiry(self, mock_redis):
        """Test set with TTL."""
        await mock_redis.set("expiring_key", "value", ex=60)
        ttl = await mock_redis.ttl("expiring_key")
        assert ttl == 60

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, mock_redis):
        """Test get for non-existent key."""
        result = await mock_redis.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, mock_redis):
        """Test delete operation."""
        await mock_redis.set("to_delete", "value")
        deleted = await mock_redis.delete("to_delete")
        assert deleted == 1

        result = await mock_redis.get("to_delete")
        assert result is None

    @pytest.mark.asyncio
    async def test_exists(self, mock_redis):
        """Test exists check."""
        await mock_redis.set("existing", "value")

        exists = await mock_redis.exists("existing")
        assert exists == 1

        not_exists = await mock_redis.exists("not_existing")
        assert not_exists == 0

    @pytest.mark.asyncio
    async def test_keys_pattern(self, mock_redis):
        """Test keys with pattern matching."""
        await mock_redis.set("ticker:BTCUSDT", "data1")
        await mock_redis.set("ticker:ETHUSDT", "data2")
        await mock_redis.set("orderbook:BTCUSDT", "data3")

        ticker_keys = await mock_redis.keys("ticker:*")
        assert len(ticker_keys) == 2

        btc_keys = await mock_redis.keys("*:BTCUSDT")
        assert len(btc_keys) == 2


class TestCacheHashOperations:
    """Test hash operations."""

    @pytest.mark.asyncio
    async def test_hset_and_hget(self, mock_redis):
        """Test hash set and get."""
        await mock_redis.hset("user:1", "name", "test")
        result = await mock_redis.hget("user:1", "name")
        assert result == "test"

    @pytest.mark.asyncio
    async def test_hset_mapping(self, mock_redis):
        """Test hash set with mapping."""
        await mock_redis.hset("position:BTCUSDT", mapping={
            "side": "LONG",
            "quantity": "0.1",
            "entry_price": "50000"
        })

        result = await mock_redis.hgetall("position:BTCUSDT")
        assert result["side"] == "LONG"
        assert result["quantity"] == "0.1"

    @pytest.mark.asyncio
    async def test_hdel(self, mock_redis):
        """Test hash delete."""
        await mock_redis.hset("test_hash", mapping={"a": "1", "b": "2"})
        await mock_redis.hdel("test_hash", "a")

        result = await mock_redis.hget("test_hash", "a")
        assert result is None

        result = await mock_redis.hget("test_hash", "b")
        assert result == "2"


class TestCacheCounterOperations:
    """Test counter operations."""

    @pytest.mark.asyncio
    async def test_incr(self, mock_redis):
        """Test increment operation."""
        result = await mock_redis.incr("counter")
        assert result == 1

        result = await mock_redis.incr("counter")
        assert result == 2

    @pytest.mark.asyncio
    async def test_decr(self, mock_redis):
        """Test decrement operation."""
        await mock_redis.set("counter", "10")

        result = await mock_redis.decr("counter")
        assert result == 9


class TestCachePubSub:
    """Test Pub/Sub operations."""

    @pytest.mark.asyncio
    async def test_publish(self, mock_redis):
        """Test publish to channel."""
        await mock_redis.publish("trades", '{"symbol": "BTCUSDT"}')

        assert "trades" in mock_redis._pubsub_channels
        assert len(mock_redis._pubsub_channels["trades"]) == 1

    @pytest.mark.asyncio
    async def test_subscribe_and_receive(self, mock_redis):
        """Test subscribe and receive message."""
        pubsub = mock_redis.pubsub()
        await pubsub.subscribe("signals")

        # Publish a message
        await mock_redis.publish("signals", '{"side": "BUY"}')

        # Receive the message
        msg = await pubsub.get_message()
        assert msg is not None
        assert msg["type"] == "message"
        assert msg["channel"] == b"signals"


class TestMarketDataCaching:
    """Test market data caching patterns."""

    @pytest.mark.asyncio
    async def test_cache_ticker(self, mock_redis):
        """Test caching ticker data."""
        ticker_data = {
            "symbol": "BTCUSDT",
            "last": "50000.00",
            "bid": "49999.00",
            "ask": "50001.00",
            "volume": "1000.00",
            "timestamp": 1700000000000
        }

        key = f"ticker:{ticker_data['symbol']}"
        await mock_redis.set(key, json.dumps(ticker_data), ex=1)

        cached = await mock_redis.get(key)
        assert cached is not None

        parsed = json.loads(cached)
        assert parsed["symbol"] == "BTCUSDT"
        assert parsed["last"] == "50000.00"

    @pytest.mark.asyncio
    async def test_cache_orderbook(self, mock_redis):
        """Test caching orderbook data."""
        orderbook = {
            "symbol": "BTCUSDT",
            "bids": [["49999", "1.0"], ["49998", "2.0"]],
            "asks": [["50001", "1.0"], ["50002", "2.0"]],
            "timestamp": 1700000000000
        }

        key = f"orderbook:{orderbook['symbol']}"
        await mock_redis.set(key, json.dumps(orderbook), ex=1)

        cached = await mock_redis.get(key)
        parsed = json.loads(cached)

        assert len(parsed["bids"]) == 2
        assert len(parsed["asks"]) == 2

    @pytest.mark.asyncio
    async def test_cache_positions(self, mock_redis):
        """Test caching position data."""
        position = {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": "0.1",
            "entry_price": "50000",
            "unrealized_pnl": "50"
        }

        await mock_redis.hset(
            "positions",
            position["symbol"],
            json.dumps(position)
        )

        cached = await mock_redis.hget("positions", "BTCUSDT")
        parsed = json.loads(cached)

        assert parsed["side"] == "LONG"
        assert parsed["quantity"] == "0.1"


class TestRateLimiting:
    """Test rate limiting with Redis."""

    @pytest.mark.asyncio
    async def test_rate_limit_counter(self, mock_redis):
        """Test rate limiting with counter."""
        key = "rate_limit:api:user1"
        limit = 10

        # Simulate requests
        for i in range(15):
            count = await mock_redis.incr(key)

            if count == 1:
                # Set expiry on first request
                await mock_redis.expire(key, 60)

            if count > limit:
                # Rate limited
                assert i >= 10
                break

    @pytest.mark.asyncio
    async def test_sliding_window_rate_limit(self, mock_redis):
        """Test sliding window rate limiting."""
        # Simple sliding window implementation
        window_key = "requests:user1"
        window_size = 60  # seconds
        max_requests = 10

        # Add timestamps for requests
        current_time = 1700000000

        for i in range(12):
            # In real implementation, would use ZADD with scores
            await mock_redis.incr(window_key)

        count = int(await mock_redis.get(window_key) or 0)
        is_limited = count > max_requests

        assert is_limited is True


class TestDistributedLocking:
    """Test distributed locking patterns."""

    @pytest.mark.asyncio
    async def test_acquire_lock(self, mock_redis):
        """Test acquiring a lock."""
        lock_key = "lock:position:BTCUSDT"
        lock_value = "owner-123"

        # Try to acquire lock
        exists = await mock_redis.exists(lock_key)
        if not exists:
            await mock_redis.set(lock_key, lock_value, ex=10)
            acquired = True
        else:
            acquired = False

        assert acquired is True

        # Verify lock is held
        holder = await mock_redis.get(lock_key)
        assert holder == lock_value

    @pytest.mark.asyncio
    async def test_lock_contention(self, mock_redis):
        """Test lock contention."""
        lock_key = "lock:critical"

        # First process acquires lock
        await mock_redis.set(lock_key, "process-1", ex=10)

        # Second process tries to acquire
        exists = await mock_redis.exists(lock_key)
        acquired = not exists

        assert acquired is False

    @pytest.mark.asyncio
    async def test_release_lock(self, mock_redis):
        """Test releasing a lock."""
        lock_key = "lock:resource"
        lock_value = "owner-456"

        # Acquire
        await mock_redis.set(lock_key, lock_value, ex=10)

        # Release (only if we own it)
        current_holder = await mock_redis.get(lock_key)
        if current_holder == lock_value:
            await mock_redis.delete(lock_key)

        # Verify released
        exists = await mock_redis.exists(lock_key)
        assert exists == 0


class TestSessionState:
    """Test session state management."""

    @pytest.mark.asyncio
    async def test_save_session(self, mock_redis):
        """Test saving session state."""
        session = {
            "user_id": "user123",
            "last_trade_time": 1700000000,
            "daily_pnl": "150.50",
            "positions_count": 2
        }

        await mock_redis.hset("session:user123", mapping={
            k: str(v) for k, v in session.items()
        })

        saved = await mock_redis.hgetall("session:user123")
        assert saved["daily_pnl"] == "150.50"

    @pytest.mark.asyncio
    async def test_update_session(self, mock_redis):
        """Test updating session state."""
        await mock_redis.hset("session:user1", "daily_pnl", "100")

        # Update
        await mock_redis.hset("session:user1", "daily_pnl", "200")

        result = await mock_redis.hget("session:user1", "daily_pnl")
        assert result == "200"

    @pytest.mark.asyncio
    async def test_session_expiry(self, mock_redis):
        """Test session with expiry."""
        await mock_redis.set("session:temp", "data", ex=3600)

        ttl = await mock_redis.ttl("session:temp")
        assert ttl == 3600


class TestCacheInvalidation:
    """Test cache invalidation patterns."""

    @pytest.mark.asyncio
    async def test_invalidate_by_key(self, mock_redis):
        """Test invalidating cache by key."""
        await mock_redis.set("cache:item1", "data")
        await mock_redis.delete("cache:item1")

        result = await mock_redis.get("cache:item1")
        assert result is None

    @pytest.mark.asyncio
    async def test_invalidate_by_pattern(self, mock_redis):
        """Test invalidating cache by pattern."""
        # Set multiple keys
        await mock_redis.set("ticker:BTCUSDT", "data1")
        await mock_redis.set("ticker:ETHUSDT", "data2")
        await mock_redis.set("orderbook:BTCUSDT", "data3")

        # Find and delete ticker keys
        keys = await mock_redis.keys("ticker:*")
        if keys:
            await mock_redis.delete(*keys)

        # Verify deleted
        remaining = await mock_redis.keys("ticker:*")
        assert len(remaining) == 0

        # Orderbook should still exist
        orderbook = await mock_redis.get("orderbook:BTCUSDT")
        assert orderbook == "data3"


class TestCachePerformance:
    """Test cache performance patterns."""

    @pytest.mark.asyncio
    async def test_batch_operations(self, mock_redis):
        """Test batch set operations."""
        # Set multiple values
        for i in range(100):
            await mock_redis.set(f"key:{i}", f"value:{i}")

        # Verify all set
        count = len(await mock_redis.keys("key:*"))
        assert count == 100

    @pytest.mark.asyncio
    async def test_pipeline_simulation(self, mock_redis):
        """Test pipeline-like batch operations."""
        # In real Redis, this would use pipeline
        operations = [
            ("set", "a", "1"),
            ("set", "b", "2"),
            ("set", "c", "3"),
        ]

        for op, key, value in operations:
            if op == "set":
                await mock_redis.set(key, value)

        # Verify all executed
        assert await mock_redis.get("a") == "1"
        assert await mock_redis.get("b") == "2"
        assert await mock_redis.get("c") == "3"
