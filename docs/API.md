# API Reference

Complete API documentation for the Crypto Scalper Bot.

## Table of Contents

1. [REST API](#rest-api)
   - [Authentication](#authentication)
   - [Account Endpoints](#account-endpoints)
   - [Trading Endpoints](#trading-endpoints)
   - [Bot Control](#bot-control)
   - [Strategy Endpoints](#strategy-endpoints)
   - [Backtesting Endpoints](#backtesting-endpoints)
2. [WebSocket API](#websocket-api)
3. [Python SDK](#python-sdk)
4. [Error Handling](#error-handling)
5. [Rate Limiting](#rate-limiting)

---

## REST API

Base URL: `http://localhost:8000/api/v1`

### Authentication

All API endpoints require authentication using JWT tokens or API keys.

#### JWT Authentication

```bash
# Login to get access token
curl -X POST /api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your_password"}'

# Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 900
}

# Use token in subsequent requests
curl -X GET /api/v1/account \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

#### API Key Authentication

```bash
curl -X GET /api/v1/account \
  -H "X-API-Key: your_api_key"
```

---

### Account Endpoints

#### Get Account Information

```http
GET /api/v1/account
```

**Response:**

```json
{
  "account_id": "acc_12345",
  "exchange": "binance",
  "account_type": "futures",
  "status": "active",
  "created_at": "2024-01-15T10:00:00Z"
}
```

---

#### Get Balance

```http
GET /api/v1/balance
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `asset` | string | No | Filter by specific asset (e.g., "USDT") |

**Response:**

```json
{
  "balances": [
    {
      "asset": "USDT",
      "free": 10000.50,
      "locked": 500.00,
      "total": 10500.50
    },
    {
      "asset": "BTC",
      "free": 0.5,
      "locked": 0.0,
      "total": 0.5
    }
  ],
  "total_equity_usd": 35000.00,
  "available_margin": 25000.00,
  "used_margin": 10000.00,
  "margin_level": 350.0
}
```

---

#### Get Positions

```http
GET /api/v1/positions
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbol` | string | No | Filter by trading pair |
| `side` | string | No | Filter by side ("long" or "short") |

**Response:**

```json
{
  "positions": [
    {
      "id": "pos_001",
      "symbol": "BTCUSDT",
      "side": "long",
      "quantity": 0.1,
      "entry_price": 50000.00,
      "current_price": 51000.00,
      "leverage": 10,
      "margin": 500.00,
      "unrealized_pnl": 100.00,
      "unrealized_pnl_pct": 2.0,
      "liquidation_price": 45000.00,
      "stop_loss": 49000.00,
      "take_profit": 55000.00,
      "opened_at": "2024-01-15T12:00:00Z"
    }
  ],
  "total_unrealized_pnl": 100.00,
  "total_margin_used": 500.00
}
```

---

#### Get Trade History

```http
GET /api/v1/trades
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbol` | string | No | Filter by trading pair |
| `start_time` | datetime | No | Start of time range (ISO 8601) |
| `end_time` | datetime | No | End of time range (ISO 8601) |
| `limit` | integer | No | Max results (default: 100, max: 1000) |
| `offset` | integer | No | Offset for pagination |

**Response:**

```json
{
  "trades": [
    {
      "id": "trade_001",
      "symbol": "BTCUSDT",
      "side": "long",
      "entry_price": 50000.00,
      "exit_price": 51000.00,
      "quantity": 0.1,
      "pnl": 100.00,
      "pnl_pct": 2.0,
      "commission": 5.00,
      "strategy": "momentum",
      "entry_time": "2024-01-15T10:00:00Z",
      "exit_time": "2024-01-15T12:00:00Z",
      "duration_seconds": 7200
    }
  ],
  "total_count": 156,
  "summary": {
    "total_pnl": 2450.00,
    "win_rate": 62.5,
    "profit_factor": 1.65
  }
}
```

---

### Trading Endpoints

#### Place Order

```http
POST /api/v1/orders
```

**Request Body:**

```json
{
  "symbol": "BTCUSDT",
  "side": "buy",
  "type": "market",
  "quantity": 0.1,
  "price": null,
  "stop_price": null,
  "stop_loss": 49000.00,
  "take_profit": 55000.00,
  "reduce_only": false,
  "time_in_force": "GTC"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `symbol` | string | Yes | Trading pair (e.g., "BTCUSDT") |
| `side` | string | Yes | Order side: "buy" or "sell" |
| `type` | string | Yes | Order type: "market", "limit", "stop_market", "stop_limit" |
| `quantity` | number | Yes | Order quantity |
| `price` | number | No | Limit price (required for limit orders) |
| `stop_price` | number | No | Stop trigger price (required for stop orders) |
| `stop_loss` | number | No | Stop-loss price for position |
| `take_profit` | number | No | Take-profit price for position |
| `reduce_only` | boolean | No | Only reduce existing position (default: false) |
| `time_in_force` | string | No | "GTC", "IOC", "FOK" (default: "GTC") |

**Response:**

```json
{
  "order_id": "ord_12345",
  "symbol": "BTCUSDT",
  "side": "buy",
  "type": "market",
  "quantity": 0.1,
  "filled_quantity": 0.1,
  "avg_fill_price": 50025.00,
  "status": "filled",
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:00:01Z"
}
```

---

#### Cancel Order

```http
DELETE /api/v1/orders/{order_id}
```

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `order_id` | string | Yes | Order ID to cancel |

**Response:**

```json
{
  "order_id": "ord_12345",
  "status": "cancelled",
  "cancelled_at": "2024-01-15T10:05:00Z"
}
```

---

#### Close Position

```http
POST /api/v1/positions/{symbol}/close
```

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbol` | string | Yes | Symbol of position to close |

**Request Body (optional):**

```json
{
  "quantity": 0.05,
  "type": "market"
}
```

**Response:**

```json
{
  "order_id": "ord_12346",
  "symbol": "BTCUSDT",
  "side": "sell",
  "quantity": 0.1,
  "filled_quantity": 0.1,
  "avg_fill_price": 51000.00,
  "realized_pnl": 100.00,
  "status": "filled"
}
```

---

#### Close All Positions

```http
POST /api/v1/positions/close-all
```

**Response:**

```json
{
  "closed_positions": 3,
  "total_realized_pnl": 250.00,
  "orders": [
    {"symbol": "BTCUSDT", "order_id": "ord_001", "pnl": 100.00},
    {"symbol": "ETHUSDT", "order_id": "ord_002", "pnl": 80.00},
    {"symbol": "BNBUSDT", "order_id": "ord_003", "pnl": 70.00}
  ]
}
```

---

#### Set Leverage

```http
PUT /api/v1/leverage/{symbol}
```

**Request Body:**

```json
{
  "leverage": 10
}
```

**Response:**

```json
{
  "symbol": "BTCUSDT",
  "leverage": 10,
  "max_leverage": 125
}
```

---

### Bot Control

#### Get Bot Status

```http
GET /api/v1/bot/status
```

**Response:**

```json
{
  "status": "running",
  "trading_enabled": true,
  "paper_trading": false,
  "uptime_seconds": 86400,
  "started_at": "2024-01-14T10:00:00Z",
  "strategies": [
    {"name": "momentum", "enabled": true, "signals_today": 15},
    {"name": "orderbook_imbalance", "enabled": true, "signals_today": 8},
    {"name": "ml_signals", "enabled": false, "signals_today": 0}
  ],
  "performance": {
    "daily_pnl": 150.00,
    "daily_trades": 12,
    "win_rate": 66.7
  }
}
```

---

#### Start Bot

```http
POST /api/v1/bot/start
```

**Request Body (optional):**

```json
{
  "paper_trading": true,
  "strategies": ["momentum", "orderbook_imbalance"]
}
```

**Response:**

```json
{
  "status": "started",
  "message": "Trading bot started successfully",
  "paper_trading": true,
  "active_strategies": ["momentum", "orderbook_imbalance"]
}
```

---

#### Stop Bot

```http
POST /api/v1/bot/stop
```

**Request Body (optional):**

```json
{
  "close_positions": false
}
```

**Response:**

```json
{
  "status": "stopped",
  "message": "Trading bot stopped",
  "open_positions": 2
}
```

---

#### Get P&L Summary

```http
GET /api/v1/pnl
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `period` | string | No | "daily", "weekly", "monthly", "all" (default: "daily") |

**Response:**

```json
{
  "period": "daily",
  "date": "2024-01-15",
  "realized_pnl": 250.00,
  "unrealized_pnl": 100.00,
  "total_pnl": 350.00,
  "trades": 15,
  "winning_trades": 10,
  "losing_trades": 5,
  "win_rate": 66.67,
  "profit_factor": 1.85,
  "largest_win": 150.00,
  "largest_loss": -50.00,
  "avg_win": 40.00,
  "avg_loss": -25.00,
  "by_strategy": [
    {"strategy": "momentum", "pnl": 180.00, "trades": 8},
    {"strategy": "orderbook_imbalance", "pnl": 70.00, "trades": 7}
  ],
  "by_symbol": [
    {"symbol": "BTCUSDT", "pnl": 200.00, "trades": 10},
    {"symbol": "ETHUSDT", "pnl": 50.00, "trades": 5}
  ]
}
```

---

### Strategy Endpoints

#### List Strategies

```http
GET /api/v1/strategies
```

**Response:**

```json
{
  "strategies": [
    {
      "name": "momentum",
      "type": "technical",
      "enabled": true,
      "parameters": {
        "rsi_period": 14,
        "overbought": 70,
        "oversold": 30
      },
      "performance": {
        "total_signals": 156,
        "accuracy": 62.5,
        "avg_return": 0.8
      }
    },
    {
      "name": "ml_signals",
      "type": "machine_learning",
      "enabled": true,
      "parameters": {
        "model_type": "random_forest",
        "confidence_threshold": 0.7
      },
      "performance": {
        "total_signals": 89,
        "accuracy": 68.5,
        "avg_return": 1.2
      }
    }
  ]
}
```

---

#### Update Strategy

```http
PUT /api/v1/strategies/{strategy_name}
```

**Request Body:**

```json
{
  "enabled": true,
  "parameters": {
    "rsi_period": 21,
    "overbought": 75,
    "oversold": 25
  }
}
```

**Response:**

```json
{
  "name": "momentum",
  "enabled": true,
  "parameters": {
    "rsi_period": 21,
    "overbought": 75,
    "oversold": 25
  },
  "updated_at": "2024-01-15T10:00:00Z"
}
```

---

### Backtesting Endpoints

#### Run Backtest

```http
POST /api/v1/backtest
```

**Request Body:**

```json
{
  "strategy": "momentum",
  "symbol": "BTCUSDT",
  "start_date": "2024-01-01",
  "end_date": "2024-06-01",
  "initial_balance": 10000,
  "leverage": 10,
  "commission_rate": 0.001,
  "parameters": {
    "rsi_period": 14,
    "overbought": 70,
    "oversold": 30
  }
}
```

**Response:**

```json
{
  "backtest_id": "bt_12345",
  "status": "running",
  "message": "Backtest started, check status endpoint for results"
}
```

---

#### Get Backtest Status

```http
GET /api/v1/backtest/{backtest_id}
```

**Response:**

```json
{
  "backtest_id": "bt_12345",
  "status": "completed",
  "progress": 100,
  "result": {
    "initial_balance": 10000,
    "final_balance": 12450,
    "total_return": 24.5,
    "total_trades": 156,
    "winning_trades": 98,
    "losing_trades": 58,
    "win_rate": 62.82,
    "profit_factor": 1.65,
    "max_drawdown": 8.5,
    "sharpe_ratio": 1.85,
    "sortino_ratio": 2.10,
    "calmar_ratio": 2.88,
    "avg_trade_duration": "2h 15m"
  }
}
```

---

## WebSocket API

Connect to: `ws://localhost:8000/ws`

### Connection

```javascript
const ws = new WebSocket("ws://localhost:8000/ws");

ws.onopen = () => {
  // Authenticate
  ws.send(JSON.stringify({
    type: "auth",
    token: "your_jwt_token"
  }));
};
```

### Subscribe to Channels

```javascript
// Subscribe to market data
ws.send(JSON.stringify({
  type: "subscribe",
  channels: ["market:BTCUSDT", "market:ETHUSDT"]
}));

// Subscribe to account updates
ws.send(JSON.stringify({
  type: "subscribe",
  channels: ["account", "orders", "positions"]
}));

// Subscribe to signals
ws.send(JSON.stringify({
  type: "subscribe",
  channels: ["signals"]
}));
```

### Message Types

#### Market Update

```json
{
  "type": "market",
  "symbol": "BTCUSDT",
  "data": {
    "price": 50000.00,
    "bid": 49999.50,
    "ask": 50000.50,
    "volume_24h": 15000000000,
    "change_24h": 2.5,
    "timestamp": "2024-01-15T10:00:00.123Z"
  }
}
```

#### Order Update

```json
{
  "type": "order",
  "data": {
    "order_id": "ord_12345",
    "symbol": "BTCUSDT",
    "side": "buy",
    "status": "filled",
    "filled_quantity": 0.1,
    "avg_fill_price": 50000.00,
    "timestamp": "2024-01-15T10:00:00.123Z"
  }
}
```

#### Position Update

```json
{
  "type": "position",
  "data": {
    "symbol": "BTCUSDT",
    "side": "long",
    "quantity": 0.1,
    "entry_price": 50000.00,
    "current_price": 50100.00,
    "unrealized_pnl": 10.00,
    "timestamp": "2024-01-15T10:00:00.123Z"
  }
}
```

#### Signal Generated

```json
{
  "type": "signal",
  "data": {
    "strategy": "momentum",
    "symbol": "BTCUSDT",
    "action": "buy",
    "strength": 0.85,
    "entry_price": 50000.00,
    "stop_loss": 49000.00,
    "take_profit": 52000.00,
    "timestamp": "2024-01-15T10:00:00.123Z"
  }
}
```

---

## Python SDK

### Installation

```bash
pip install crypto-scalper-client
```

### Usage

```python
from crypto_scalper import TradingClient

# Initialize client
client = TradingClient(
    api_url="http://localhost:8000",
    api_key="your_api_key"
)

# Get balance
balance = await client.get_balance()
print(f"Available: ${balance.available_margin}")

# Get positions
positions = await client.get_positions()
for pos in positions:
    print(f"{pos.symbol}: {pos.unrealized_pnl}")

# Place order
order = await client.place_order(
    symbol="BTCUSDT",
    side="buy",
    quantity=0.1,
    order_type="market",
    stop_loss=49000,
    take_profit=55000
)
print(f"Order filled at {order.avg_fill_price}")

# Close position
result = await client.close_position("BTCUSDT")
print(f"P&L: ${result.realized_pnl}")

# Get P&L summary
pnl = await client.get_pnl(period="daily")
print(f"Daily P&L: ${pnl.total_pnl}")

# Bot control
await client.start_bot(paper_trading=True)
await client.stop_bot()
```

### Async Context Manager

```python
async with TradingClient(api_url="http://localhost:8000", api_key="key") as client:
    balance = await client.get_balance()
    # ... operations
```

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "INSUFFICIENT_BALANCE",
    "message": "Insufficient balance to place order",
    "details": {
      "required": 1000.00,
      "available": 500.00
    }
  },
  "request_id": "req_12345"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Invalid or expired token |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `INSUFFICIENT_BALANCE` | 400 | Not enough balance for operation |
| `POSITION_NOT_FOUND` | 404 | No position for specified symbol |
| `ORDER_REJECTED` | 400 | Order rejected by exchange |
| `RATE_LIMITED` | 429 | Too many requests |
| `EXCHANGE_ERROR` | 502 | Exchange API error |
| `INTERNAL_ERROR` | 500 | Internal server error |

---

## Rate Limiting

Rate limits are applied per API key:

| Endpoint Category | Rate Limit |
|-------------------|------------|
| Read operations | 100 requests/minute |
| Write operations | 30 requests/minute |
| Backtest | 5 requests/minute |
| WebSocket | 100 messages/minute |

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705312860
```

When rate limited:

```json
{
  "error": {
    "code": "RATE_LIMITED",
    "message": "Rate limit exceeded. Retry after 30 seconds.",
    "retry_after": 30
  }
}
```
