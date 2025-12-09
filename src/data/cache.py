"""
Redis caching layer for high-performance data access.

Features:
- Market data caching (orderbook, trades, tickers)
- Session state management
- Rate limit tracking
- Pub/Sub for real-time updates
"""

import asyncio
import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, TypeVar, Generic, Callable, Union

from loguru import logger

# Try to import redis
try:
    import redis.asyncio as redis
    from redis.asyncio import Redis, ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not installed. Caching disabled. Install with: pip install redis")


T = TypeVar('T')


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CacheConfig:
    """Redis cache configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    max_connections: int = 50

    # TTL defaults (seconds)
    ticker_ttl: int = 1
    orderbook_ttl: int = 1
    trade_ttl: int = 5
    position_ttl: int = 5
    balance_ttl: int = 10
    signal_ttl: int = 60
    session_ttl: int = 3600

    # Key prefixes
    prefix: str = "scalper:"

    @classmethod
    def from_env(cls) -> "CacheConfig":
        """Create config from environment variables."""
        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
        )


# =============================================================================
# JSON Encoder for Decimal
# =============================================================================

class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def decimal_decoder(obj: Dict) -> Dict:
    """Decode decimals from JSON."""
    for key, value in obj.items():
        if isinstance(value, str):
            try:
                # Try to parse as Decimal if it looks like a number
                if value.replace(".", "").replace("-", "").isdigit():
                    obj[key] = Decimal(value)
            except:
                pass
    return obj


# =============================================================================
# Cache Client
# =============================================================================

class CacheClient:
    """
    Redis cache client for trading bot.

    Usage:
        cache = CacheClient(CacheConfig.from_env())
        await cache.connect()

        # Cache ticker
        await cache.set_ticker("BTCUSDT", {"price": "50000", "volume": "1000"})
        ticker = await cache.get_ticker("BTCUSDT")

        # Cache orderbook
        await cache.set_orderbook("BTCUSDT", orderbook_data)

        # Pub/Sub
        await cache.publish("signals", signal_data)
        async for message in cache.subscribe("signals"):
            process(message)
    """

    def __init__(self, config: CacheConfig = None):
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis not installed. Install with: pip install redis")

        self.config = config or CacheConfig.from_env()
        self._pool: Optional[ConnectionPool] = None
        self._redis: Optional[Redis] = None
        self._pubsub = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _key(self, *parts: str) -> str:
        """Build cache key with prefix."""
        return f"{self.config.prefix}{':'.join(parts)}"

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self._pool = ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                max_connections=self.config.max_connections,
                decode_responses=True,
            )

            self._redis = Redis(connection_pool=self._pool)

            # Test connection
            await self._redis.ping()
            self._connected = True

            logger.info(f"Redis connected: {self.config.host}:{self.config.port}")

        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._pubsub:
            await self._pubsub.close()

        if self._redis:
            await self._redis.close()

        if self._pool:
            await self._pool.disconnect()

        self._connected = False
        logger.info("Redis disconnected")

    async def ping(self) -> bool:
        """Check connection health."""
        try:
            if self._redis:
                await self._redis.ping()
                return True
        except:
            pass
        return False

    # =========================================================================
    # Basic Operations
    # =========================================================================

    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        if not self._redis:
            return None
        return await self._redis.get(self._key(key))

    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value with optional TTL."""
        if not self._redis:
            return False
        await self._redis.set(self._key(key), value, ex=ttl)
        return True

    async def delete(self, key: str) -> bool:
        """Delete a key."""
        if not self._redis:
            return False
        await self._redis.delete(self._key(key))
        return True

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self._redis:
            return False
        return bool(await self._redis.exists(self._key(key)))

    async def get_json(self, key: str) -> Optional[Dict]:
        """Get JSON value."""
        value = await self.get(key)
        if value:
            return json.loads(value, object_hook=decimal_decoder)
        return None

    async def set_json(
        self,
        key: str,
        value: Dict,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set JSON value."""
        return await self.set(
            key,
            json.dumps(value, cls=DecimalEncoder),
            ttl,
        )

    # =========================================================================
    # Market Data Caching
    # =========================================================================

    async def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get cached ticker."""
        return await self.get_json(f"ticker:{symbol}")

    async def set_ticker(self, symbol: str, data: Dict) -> bool:
        """Cache ticker data."""
        return await self.set_json(
            f"ticker:{symbol}",
            data,
            self.config.ticker_ttl,
        )

    async def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """Get cached orderbook."""
        return await self.get_json(f"orderbook:{symbol}")

    async def set_orderbook(self, symbol: str, data: Dict) -> bool:
        """Cache orderbook data."""
        return await self.set_json(
            f"orderbook:{symbol}",
            data,
            self.config.orderbook_ttl,
        )

    async def get_trades(self, symbol: str) -> Optional[List[Dict]]:
        """Get cached recent trades."""
        data = await self.get_json(f"trades:{symbol}")
        return data.get("trades") if data else None

    async def set_trades(self, symbol: str, trades: List[Dict]) -> bool:
        """Cache recent trades."""
        return await self.set_json(
            f"trades:{symbol}",
            {"trades": trades, "timestamp": datetime.now().isoformat()},
            self.config.trade_ttl,
        )

    async def add_trade(self, symbol: str, trade: Dict, max_trades: int = 100) -> bool:
        """Add trade to cached list (FIFO)."""
        if not self._redis:
            return False

        key = self._key(f"trades:{symbol}:list")
        await self._redis.lpush(key, json.dumps(trade, cls=DecimalEncoder))
        await self._redis.ltrim(key, 0, max_trades - 1)
        await self._redis.expire(key, self.config.trade_ttl)
        return True

    # =========================================================================
    # Position & Balance Caching
    # =========================================================================

    async def get_positions(self) -> Optional[List[Dict]]:
        """Get cached positions."""
        data = await self.get_json("positions")
        return data.get("positions") if data else None

    async def set_positions(self, positions: List[Dict]) -> bool:
        """Cache positions."""
        return await self.set_json(
            "positions",
            {"positions": positions, "timestamp": datetime.now().isoformat()},
            self.config.position_ttl,
        )

    async def get_balance(self) -> Optional[Dict]:
        """Get cached balance."""
        return await self.get_json("balance")

    async def set_balance(self, balance: Dict) -> bool:
        """Cache balance."""
        return await self.set_json(
            "balance",
            {**balance, "cached_at": datetime.now().isoformat()},
            self.config.balance_ttl,
        )

    # =========================================================================
    # Signal Caching
    # =========================================================================

    async def get_signal(self, symbol: str) -> Optional[Dict]:
        """Get latest signal for symbol."""
        return await self.get_json(f"signal:{symbol}")

    async def set_signal(self, symbol: str, signal: Dict) -> bool:
        """Cache signal."""
        return await self.set_json(
            f"signal:{symbol}",
            signal,
            self.config.signal_ttl,
        )

    async def get_signal_history(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[Dict]:
        """Get signal history."""
        if not self._redis:
            return []

        key = self._key(f"signals:{symbol}:history")
        data = await self._redis.lrange(key, 0, limit - 1)
        return [json.loads(d, object_hook=decimal_decoder) for d in data]

    async def add_signal_to_history(self, symbol: str, signal: Dict) -> bool:
        """Add signal to history."""
        if not self._redis:
            return False

        key = self._key(f"signals:{symbol}:history")
        await self._redis.lpush(key, json.dumps(signal, cls=DecimalEncoder))
        await self._redis.ltrim(key, 0, 999)  # Keep last 1000
        return True

    # =========================================================================
    # Session State
    # =========================================================================

    async def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session state."""
        return await self.get_json(f"session:{session_id}")

    async def set_session(self, session_id: str, data: Dict) -> bool:
        """Set session state."""
        return await self.set_json(
            f"session:{session_id}",
            data,
            self.config.session_ttl,
        )

    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        return await self.delete(f"session:{session_id}")

    # =========================================================================
    # Rate Limiting
    # =========================================================================

    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int = 60,
    ) -> tuple[bool, int]:
        """
        Check rate limit using sliding window.

        Returns:
            (allowed, remaining_requests)
        """
        if not self._redis:
            return True, limit

        now = datetime.now().timestamp()
        window_start = now - window_seconds

        rate_key = self._key(f"ratelimit:{key}")

        # Remove old entries
        await self._redis.zremrangebyscore(rate_key, 0, window_start)

        # Count current requests
        count = await self._redis.zcard(rate_key)

        if count >= limit:
            return False, 0

        # Add current request
        await self._redis.zadd(rate_key, {str(now): now})
        await self._redis.expire(rate_key, window_seconds)

        return True, limit - count - 1

    async def get_rate_limit_remaining(
        self,
        key: str,
        limit: int,
        window_seconds: int = 60,
    ) -> int:
        """Get remaining rate limit."""
        if not self._redis:
            return limit

        now = datetime.now().timestamp()
        window_start = now - window_seconds

        rate_key = self._key(f"ratelimit:{key}")
        await self._redis.zremrangebyscore(rate_key, 0, window_start)
        count = await self._redis.zcard(rate_key)

        return max(0, limit - count)

    # =========================================================================
    # Pub/Sub
    # =========================================================================

    async def publish(self, channel: str, message: Dict) -> int:
        """Publish message to channel."""
        if not self._redis:
            return 0

        return await self._redis.publish(
            self._key(f"channel:{channel}"),
            json.dumps(message, cls=DecimalEncoder),
        )

    async def subscribe(self, *channels: str):
        """
        Subscribe to channels.

        Usage:
            async for message in cache.subscribe("signals", "orders"):
                print(message)
        """
        if not self._redis:
            return

        self._pubsub = self._redis.pubsub()
        await self._pubsub.subscribe(*[self._key(f"channel:{c}") for c in channels])

        async for message in self._pubsub.listen():
            if message["type"] == "message":
                yield json.loads(message["data"], object_hook=decimal_decoder)

    async def unsubscribe(self, *channels: str) -> None:
        """Unsubscribe from channels."""
        if self._pubsub:
            await self._pubsub.unsubscribe(
                *[self._key(f"channel:{c}") for c in channels]
            )

    # =========================================================================
    # Atomic Operations
    # =========================================================================

    async def increment(self, key: str, amount: int = 1) -> int:
        """Atomic increment."""
        if not self._redis:
            return 0
        return await self._redis.incrby(self._key(key), amount)

    async def decrement(self, key: str, amount: int = 1) -> int:
        """Atomic decrement."""
        if not self._redis:
            return 0
        return await self._redis.decrby(self._key(key), amount)

    async def acquire_lock(
        self,
        name: str,
        timeout: int = 10,
        blocking: bool = True,
        blocking_timeout: float = 5.0,
    ) -> Optional[str]:
        """
        Acquire a distributed lock.

        Returns lock token if acquired, None otherwise.
        """
        if not self._redis:
            return None

        import uuid
        token = str(uuid.uuid4())
        key = self._key(f"lock:{name}")

        if blocking:
            end_time = asyncio.get_event_loop().time() + blocking_timeout
            while asyncio.get_event_loop().time() < end_time:
                if await self._redis.set(key, token, nx=True, ex=timeout):
                    return token
                await asyncio.sleep(0.1)
            return None
        else:
            if await self._redis.set(key, token, nx=True, ex=timeout):
                return token
            return None

    async def release_lock(self, name: str, token: str) -> bool:
        """Release a distributed lock."""
        if not self._redis:
            return False

        key = self._key(f"lock:{name}")

        # Only release if we own the lock
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """

        result = await self._redis.eval(script, 1, key, token)
        return bool(result)

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._redis:
            return {}

        info = await self._redis.info()

        return {
            "connected": self._connected,
            "host": self.config.host,
            "port": self.config.port,
            "used_memory": info.get("used_memory_human", "N/A"),
            "connected_clients": info.get("connected_clients", 0),
            "total_commands": info.get("total_commands_processed", 0),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "hit_rate": (
                info.get("keyspace_hits", 0) /
                max(1, info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0))
            ),
        }


# =============================================================================
# Cache Decorator
# =============================================================================

def cached(
    ttl: int = 60,
    key_prefix: str = "func",
    cache_client: Optional[CacheClient] = None,
):
    """
    Decorator for caching function results.

    Usage:
        @cached(ttl=30, key_prefix="ticker")
        async def get_ticker(symbol: str):
            return await api.get_ticker(symbol)
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            # Build cache key
            cache_key = f"{key_prefix}:{func.__name__}:{hash((args, tuple(kwargs.items())))}"

            # Try to get from cache
            if cache_client and cache_client.is_connected:
                cached_value = await cache_client.get_json(cache_key)
                if cached_value is not None:
                    return cached_value

            # Call function
            result = await func(*args, **kwargs)

            # Cache result
            if cache_client and cache_client.is_connected and result is not None:
                await cache_client.set_json(cache_key, result, ttl)

            return result

        return wrapper
    return decorator


# =============================================================================
# Global Cache Instance
# =============================================================================

_cache: Optional[CacheClient] = None


async def get_cache() -> Optional[CacheClient]:
    """Get global cache instance."""
    global _cache

    if _cache is None:
        if REDIS_AVAILABLE:
            try:
                _cache = CacheClient(CacheConfig.from_env())
                await _cache.connect()
            except Exception as e:
                logger.warning(f"Failed to initialize cache: {e}")
                return None

    return _cache


async def close_cache() -> None:
    """Close global cache instance."""
    global _cache

    if _cache:
        await _cache.disconnect()
        _cache = None
