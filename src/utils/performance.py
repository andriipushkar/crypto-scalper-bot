"""
Performance optimization utilities.

Provides:
- uvloop event loop optimization
- Connection pooling
- Message batching
- Latency tracking
"""

import asyncio
import os
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Optional, Dict, Any, List, Callable, Deque
import statistics

from loguru import logger


# =============================================================================
# uvloop Integration
# =============================================================================

def setup_uvloop() -> bool:
    """
    Set up uvloop as the default event loop policy.

    uvloop is a fast, drop-in replacement for asyncio's event loop.
    It can provide 2-4x performance improvement.

    Returns:
        True if uvloop was set up successfully
    """
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        logger.info("uvloop event loop policy enabled")
        return True
    except ImportError:
        logger.warning(
            "uvloop not available, using default asyncio event loop. "
            "Install with: pip install uvloop"
        )
        return False


def get_optimal_event_loop():
    """Get the optimal event loop for the platform."""
    # Try uvloop first
    try:
        import uvloop
        return uvloop.new_event_loop()
    except ImportError:
        pass

    # Fall back to default
    return asyncio.new_event_loop()


# =============================================================================
# Latency Tracking
# =============================================================================

@dataclass
class LatencyStats:
    """Statistics for latency measurements."""
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    recent: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    @property
    def avg_ms(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_ms / self.count

    @property
    def p50_ms(self) -> float:
        if not self.recent:
            return 0.0
        sorted_values = sorted(self.recent)
        return sorted_values[len(sorted_values) // 2]

    @property
    def p95_ms(self) -> float:
        if not self.recent:
            return 0.0
        sorted_values = sorted(self.recent)
        idx = int(len(sorted_values) * 0.95)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    @property
    def p99_ms(self) -> float:
        if not self.recent:
            return 0.0
        sorted_values = sorted(self.recent)
        idx = int(len(sorted_values) * 0.99)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    @property
    def std_ms(self) -> float:
        if len(self.recent) < 2:
            return 0.0
        return statistics.stdev(self.recent)

    def add(self, latency_ms: float) -> None:
        """Add a latency measurement."""
        self.count += 1
        self.total_ms += latency_ms
        self.min_ms = min(self.min_ms, latency_ms)
        self.max_ms = max(self.max_ms, latency_ms)
        self.recent.append(latency_ms)

    def reset(self) -> None:
        """Reset statistics."""
        self.count = 0
        self.total_ms = 0.0
        self.min_ms = float('inf')
        self.max_ms = 0.0
        self.recent.clear()

    def summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            "count": self.count,
            "avg_ms": self.avg_ms,
            "min_ms": self.min_ms if self.min_ms != float('inf') else 0,
            "max_ms": self.max_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "std_ms": self.std_ms,
        }


class LatencyTracker:
    """
    Track latency across different operations.

    Usage:
        tracker = LatencyTracker()

        with tracker.measure("websocket_recv"):
            data = await ws.recv()

        # Or as decorator
        @tracker.track("order_submission")
        async def submit_order():
            ...
    """

    def __init__(self):
        self._stats: Dict[str, LatencyStats] = {}

    def get_stats(self, operation: str) -> LatencyStats:
        """Get or create stats for an operation."""
        if operation not in self._stats:
            self._stats[operation] = LatencyStats()
        return self._stats[operation]

    @contextmanager
    def measure(self, operation: str):
        """Context manager to measure latency."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.get_stats(operation).add(elapsed_ms)

    def track(self, operation: str):
        """Decorator to track async function latency."""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    self.get_stats(operation).add(elapsed_ms)
            return wrapper
        return decorator

    def record(self, operation: str, latency_ms: float) -> None:
        """Manually record a latency measurement."""
        self.get_stats(operation).add(latency_ms)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all operations."""
        return {op: stats.summary() for op, stats in self._stats.items()}

    def reset(self, operation: str = None) -> None:
        """Reset statistics for one or all operations."""
        if operation:
            if operation in self._stats:
                self._stats[operation].reset()
        else:
            for stats in self._stats.values():
                stats.reset()


# Global latency tracker
latency_tracker = LatencyTracker()


# =============================================================================
# Message Batching
# =============================================================================

class MessageBatcher:
    """
    Batch messages for more efficient processing.

    Collects messages until either:
    - Max batch size is reached
    - Max wait time is exceeded
    """

    def __init__(
        self,
        max_batch_size: int = 100,
        max_wait_ms: float = 10.0,
        callback: Callable[[List[Any]], Any] = None,
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.callback = callback

        self._batch: List[Any] = []
        self._last_flush = time.perf_counter()
        self._lock = asyncio.Lock()

    async def add(self, message: Any) -> Optional[List[Any]]:
        """
        Add a message to the batch.

        Returns:
            Flushed batch if batch was processed, None otherwise
        """
        async with self._lock:
            self._batch.append(message)

            # Check if we should flush
            should_flush = (
                len(self._batch) >= self.max_batch_size or
                (time.perf_counter() - self._last_flush) * 1000 >= self.max_wait_ms
            )

            if should_flush:
                return await self._flush()

        return None

    async def _flush(self) -> List[Any]:
        """Flush the current batch."""
        batch = self._batch
        self._batch = []
        self._last_flush = time.perf_counter()

        if self.callback and batch:
            await self.callback(batch)

        return batch

    async def flush(self) -> List[Any]:
        """Force flush the current batch."""
        async with self._lock:
            return await self._flush()

    @property
    def size(self) -> int:
        """Current batch size."""
        return len(self._batch)


# =============================================================================
# Connection Pool
# =============================================================================

class AsyncConnectionPool:
    """
    Generic async connection pool.

    Maintains a pool of reusable connections.
    """

    def __init__(
        self,
        factory: Callable,
        min_size: int = 1,
        max_size: int = 10,
    ):
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size

        self._pool: Deque = deque()
        self._size = 0
        self._lock = asyncio.Lock()
        self._available = asyncio.Event()

    async def acquire(self):
        """Acquire a connection from the pool."""
        async with self._lock:
            if self._pool:
                return self._pool.popleft()

            if self._size < self.max_size:
                self._size += 1
                return await self.factory()

        # Wait for a connection to be available
        await self._available.wait()
        return await self.acquire()

    async def release(self, conn):
        """Release a connection back to the pool."""
        async with self._lock:
            self._pool.append(conn)
            self._available.set()

    async def close(self):
        """Close all connections in the pool."""
        async with self._lock:
            while self._pool:
                conn = self._pool.popleft()
                if hasattr(conn, 'close'):
                    if asyncio.iscoroutinefunction(conn.close):
                        await conn.close()
                    else:
                        conn.close()
            self._size = 0

    @contextmanager
    async def connection(self):
        """Context manager for acquiring/releasing connections."""
        conn = await self.acquire()
        try:
            yield conn
        finally:
            await self.release(conn)


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter.

    Limits the rate of operations to prevent overwhelming services.
    """

    def __init__(
        self,
        rate: float,  # tokens per second
        burst: int = 1,  # max burst size
    ):
        self.rate = rate
        self.burst = burst

        self._tokens = burst
        self._last_update = time.perf_counter()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens, waiting if necessary.

        Returns:
            Wait time in seconds (0 if no wait)
        """
        async with self._lock:
            now = time.perf_counter()
            elapsed = now - self._last_update
            self._last_update = now

            # Add tokens based on elapsed time
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)

            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0

            # Calculate wait time
            needed = tokens - self._tokens
            wait_time = needed / self.rate

            return wait_time

    async def wait(self, tokens: int = 1) -> None:
        """Wait until tokens are available."""
        wait_time = await self.acquire(tokens)
        if wait_time > 0:
            await asyncio.sleep(wait_time)


# =============================================================================
# Timing Utilities
# =============================================================================

def time_ms() -> float:
    """Get current time in milliseconds."""
    return time.perf_counter() * 1000


def monotonic_ms() -> int:
    """Get monotonic time in milliseconds (integer)."""
    return int(time.monotonic() * 1000)


class Timer:
    """Simple timer for measuring elapsed time."""

    def __init__(self):
        self._start = None
        self._end = None

    def start(self) -> 'Timer':
        """Start the timer."""
        self._start = time.perf_counter()
        self._end = None
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed ms."""
        self._end = time.perf_counter()
        return self.elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self._start is None:
            return 0.0
        end = self._end or time.perf_counter()
        return (end - self._start) * 1000

    @property
    def elapsed_s(self) -> float:
        """Get elapsed time in seconds."""
        return self.elapsed_ms / 1000


# =============================================================================
# Process Optimization
# =============================================================================

def optimize_process() -> Dict[str, Any]:
    """
    Apply process-level optimizations.

    Returns:
        Dictionary of applied optimizations
    """
    optimizations = {}

    # Set process priority (Linux/Unix only)
    try:
        import resource
        # Try to set higher priority
        os.nice(-5)
        optimizations["nice"] = -5
    except (ImportError, PermissionError, OSError):
        pass

    # Disable garbage collection during critical sections
    import gc
    gc.disable()
    optimizations["gc_disabled"] = True

    # Set thread priority (if possible)
    try:
        import threading
        # Note: Thread priority setting is platform-specific
        optimizations["main_thread"] = threading.current_thread().name
    except Exception:
        pass

    return optimizations


def enable_gc() -> None:
    """Re-enable garbage collection."""
    import gc
    gc.enable()
    gc.collect()


# =============================================================================
# Async Utilities
# =============================================================================

async def gather_with_concurrency(
    n: int,
    *coros,
) -> List[Any]:
    """
    Run coroutines with limited concurrency.

    Args:
        n: Maximum concurrent tasks
        coros: Coroutines to run

    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(n)

    async def run_with_semaphore(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(run_with_semaphore(c) for c in coros))


async def timeout(coro, seconds: float, default=None):
    """
    Run coroutine with timeout.

    Args:
        coro: Coroutine to run
        seconds: Timeout in seconds
        default: Default value on timeout

    Returns:
        Result or default on timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        return default


# =============================================================================
# Memory Optimization
# =============================================================================

def get_object_size(obj) -> int:
    """Get approximate memory size of an object in bytes."""
    import sys
    seen = set()

    def sizeof(o):
        if id(o) in seen:
            return 0
        seen.add(id(o))
        size = sys.getsizeof(o)

        if isinstance(o, dict):
            size += sum(sizeof(k) + sizeof(v) for k, v in o.items())
        elif hasattr(o, '__dict__'):
            size += sizeof(o.__dict__)
        elif hasattr(o, '__iter__') and not isinstance(o, (str, bytes)):
            size += sum(sizeof(i) for i in o)

        return size

    return sizeof(obj)


def format_bytes(num_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"
