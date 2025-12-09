# Performance Tuning Guide

## Overview

This guide covers performance optimization strategies for the crypto trading bot.

## Python Performance

### Async Best Practices

```python
# Use connection pooling
import aiohttp

# Create session once, reuse for all requests
async def create_session():
    connector = aiohttp.TCPConnector(
        limit=100,  # Max concurrent connections
        limit_per_host=30,  # Max per host
        keepalive_timeout=30,
        enable_cleanup_closed=True,
    )
    return aiohttp.ClientSession(connector=connector)

# Use gather for concurrent operations
async def fetch_all_prices(symbols: List[str]):
    async with create_session() as session:
        tasks = [fetch_price(session, symbol) for symbol in symbols]
        return await asyncio.gather(*tasks, return_exceptions=True)
```

### Memory Optimization

```python
# Use __slots__ for frequently instantiated classes
class Trade:
    __slots__ = ['symbol', 'price', 'quantity', 'timestamp']

    def __init__(self, symbol, price, quantity, timestamp):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.timestamp = timestamp

# Use generators for large datasets
def process_trades(trades):
    for trade in trades:
        yield transform(trade)  # Lazy evaluation

# Use deque for fixed-size collections
from collections import deque
price_history = deque(maxlen=1000)  # Auto-removes old items
```

### CPU Optimization

```python
# Use numpy for numerical operations
import numpy as np

# Bad - Python loop
def calculate_sma_slow(prices: List[float], period: int) -> List[float]:
    result = []
    for i in range(period, len(prices)):
        result.append(sum(prices[i-period:i]) / period)
    return result

# Good - NumPy vectorized
def calculate_sma_fast(prices: np.ndarray, period: int) -> np.ndarray:
    weights = np.ones(period) / period
    return np.convolve(prices, weights, mode='valid')

# Use Cython for hot paths (compile with: cythonize -i indicators.pyx)
# indicators.pyx
cpdef double calculate_ema(double[:] prices, int period):
    cdef double alpha = 2.0 / (period + 1)
    cdef double ema = prices[0]
    cdef int i
    for i in range(1, len(prices)):
        ema = alpha * prices[i] + (1 - alpha) * ema
    return ema
```

## Database Performance

### PostgreSQL Tuning

```sql
-- Connection pooling settings (postgresql.conf)
max_connections = 200
shared_buffers = 2GB
effective_cache_size = 6GB
maintenance_work_mem = 512MB
work_mem = 64MB

-- Enable query optimization
random_page_cost = 1.1  -- For SSD
effective_io_concurrency = 200

-- Parallelism
max_worker_processes = 8
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
```

### Index Optimization

```sql
-- Composite index for common queries
CREATE INDEX idx_trades_symbol_time
ON trades (symbol, exit_time DESC);

-- Partial index for active positions
CREATE INDEX idx_positions_open
ON positions (symbol, entry_time)
WHERE status = 'open';

-- BRIN index for time-series data (huge tables)
CREATE INDEX idx_market_data_time
ON market_data USING BRIN (timestamp);

-- Analyze query performance
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT * FROM trades
WHERE symbol = 'BTCUSDT'
AND exit_time > NOW() - INTERVAL '1 day';
```

### Query Optimization

```python
# Use bulk inserts
async def insert_trades(trades: List[Trade]):
    async with pool.acquire() as conn:
        await conn.executemany(
            "INSERT INTO trades (symbol, price, quantity) VALUES ($1, $2, $3)",
            [(t.symbol, t.price, t.quantity) for t in trades]
        )

# Use prepared statements
async def get_trades(symbol: str, limit: int):
    async with pool.acquire() as conn:
        stmt = await conn.prepare(
            "SELECT * FROM trades WHERE symbol = $1 ORDER BY time DESC LIMIT $2"
        )
        return await stmt.fetch(symbol, limit)

# Connection pooling with asyncpg
import asyncpg

pool = await asyncpg.create_pool(
    DATABASE_URL,
    min_size=10,
    max_size=50,
    max_queries=50000,
    max_inactive_connection_lifetime=300,
)
```

## Redis Performance

### Configuration

```conf
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
tcp-keepalive 300

# Enable persistence (choose one)
save 900 1
appendonly yes
appendfsync everysec
```

### Usage Patterns

```python
import redis.asyncio as redis

# Connection pooling
pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    max_connections=50,
    decode_responses=True,
)
redis_client = redis.Redis(connection_pool=pool)

# Use pipelines for batch operations
async def cache_prices(prices: Dict[str, float]):
    async with redis_client.pipeline() as pipe:
        for symbol, price in prices.items():
            pipe.setex(f"price:{symbol}", 60, price)
        await pipe.execute()

# Use Lua scripts for atomic operations
UPDATE_POSITION_SCRIPT = """
local current = redis.call('GET', KEYS[1])
if current then
    local new_qty = tonumber(current) + tonumber(ARGV[1])
    redis.call('SET', KEYS[1], new_qty)
    return new_qty
end
return nil
"""

async def update_position(symbol: str, qty_change: float):
    return await redis_client.eval(UPDATE_POSITION_SCRIPT, 1, f"pos:{symbol}", qty_change)
```

## Network Optimization

### WebSocket Configuration

```python
import websockets

async def connect_ws():
    async with websockets.connect(
        WS_URL,
        ping_interval=20,
        ping_timeout=10,
        close_timeout=5,
        max_size=2**20,  # 1MB max message
        compression=None,  # Disable for latency
    ) as ws:
        async for message in ws:
            await process_message(message)
```

### HTTP Client Tuning

```python
import httpx

# Optimized client configuration
client = httpx.AsyncClient(
    http2=True,
    limits=httpx.Limits(
        max_keepalive_connections=20,
        max_connections=100,
        keepalive_expiry=30,
    ),
    timeout=httpx.Timeout(10.0, connect=5.0),
)
```

## Kubernetes Performance

### Resource Requests/Limits

```yaml
resources:
  requests:
    cpu: "500m"
    memory: "512Mi"
  limits:
    cpu: "2000m"
    memory: "2Gi"
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: trading-bot-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: trading-bot
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
```

### Pod Disruption Budget

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: trading-bot-pdb
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: trading-bot
```

## Profiling

### CPU Profiling

```python
import cProfile
import pstats

def profile_function(func):
    profiler = cProfile.Profile()
    profiler.enable()
    result = func()
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

    return result

# Async profiling with yappi
import yappi

yappi.set_clock_type("wall")
yappi.start()
# ... run async code ...
yappi.stop()
yappi.get_func_stats().print_all()
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def process_trades(trades):
    results = []
    for trade in trades:
        results.append(transform(trade))
    return results

# Track memory over time
import tracemalloc

tracemalloc.start()
# ... run code ...
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

## Benchmarks

### Latency Targets

| Operation | Target | Acceptable |
|-----------|--------|------------|
| Order placement | < 50ms | < 100ms |
| Price update processing | < 5ms | < 10ms |
| Signal generation | < 10ms | < 20ms |
| Database read | < 5ms | < 10ms |
| Database write | < 10ms | < 20ms |
| Cache read | < 1ms | < 2ms |

### Load Testing

```python
import asyncio
import time

async def benchmark_orders(n: int = 1000):
    start = time.perf_counter()

    tasks = [place_order(i) for i in range(n)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.perf_counter() - start
    success = sum(1 for r in results if not isinstance(r, Exception))

    print(f"Total: {elapsed:.2f}s")
    print(f"Orders/sec: {n/elapsed:.2f}")
    print(f"Success rate: {success/n*100:.1f}%")
    print(f"Avg latency: {elapsed/n*1000:.2f}ms")
```

## Monitoring

### Key Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Latency histogram
order_latency = Histogram(
    'order_latency_seconds',
    'Order placement latency',
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0]
)

# Throughput counter
orders_total = Counter(
    'orders_total',
    'Total orders placed',
    ['symbol', 'side', 'status']
)

# Current state gauge
active_positions = Gauge(
    'active_positions',
    'Number of active positions',
    ['symbol']
)

# Usage
with order_latency.time():
    await place_order(order)
orders_total.labels(symbol='BTCUSDT', side='buy', status='filled').inc()
active_positions.labels(symbol='BTCUSDT').set(5)
```

### Alerting Rules

```yaml
groups:
- name: performance
  rules:
  - alert: HighOrderLatency
    expr: histogram_quantile(0.95, order_latency_seconds_bucket) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High order latency detected"

  - alert: LowThroughput
    expr: rate(orders_total[5m]) < 1
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Low order throughput"
```
