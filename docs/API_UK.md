# API Документація (Українська)

Повна документація REST API та WebSocket для Crypto Scalper Bot.

## Зміст

1. [REST API](#rest-api)
   - [Аутентифікація](#аутентифікація)
   - [Статус та контроль](#статус-та-контроль)
   - [Торгівля](#торгівля)
   - [Позиції](#позиції)
   - [Стратегії](#стратегії)
   - [Ризик-менеджмент](#ризик-менеджмент)
   - [Біржа](#біржа)
   - [Логи та алерти](#логи-та-алерти)
2. [WebSocket API](#websocket-api)
3. [Обробка помилок](#обробка-помилок)
4. [Rate Limiting](#rate-limiting)

---

## REST API

### Базовий URL

```
http://localhost:8000
```

### Аутентифікація

#### JWT Аутентифікація

```bash
# Отримання токену
POST /api/auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "your_password"
}

# Відповідь
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

#### Використання токену

```bash
# У заголовках запитів
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### Оновлення токену

```bash
POST /api/auth/refresh
Authorization: Bearer <refresh_token>

# Відповідь
{
  "access_token": "новий_токен...",
  "refresh_token": "новий_refresh_токен...",
  "expires_in": 3600
}
```

#### Інформація про користувача

```bash
GET /api/auth/me
Authorization: Bearer <token>

# Відповідь
{
  "type": "jwt",
  "username": "admin",
  "permissions": ["read", "write", "admin"]
}
```

---

### Статус та контроль

#### Health Check

```http
GET /health
```

**Відповідь:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:00:00Z"
}
```

#### Статус бота

```http
GET /api/status
```

**Відповідь:**
```json
{
  "status": "running",
  "uptime_seconds": 3600.5,
  "mode": "paper",
  "connected_exchanges": ["binance_futures"],
  "active_symbols": ["BTCUSDT", "ETHUSDT"]
}
```

#### Команди бота

```http
POST /api/command
Content-Type: application/json

{
  "command": "start"  // start, stop, pause, resume
}
```

**Відповідь:**
```json
{
  "success": true,
  "message": "Bot started"
}
```

---

### Торгівля

#### Розміщення ордеру

```http
POST /api/order
Content-Type: application/json

{
  "symbol": "BTCUSDT",
  "side": "BUY",
  "quantity": 0.001,
  "order_type": "MARKET",
  "price": null,
  "stop_loss": 49000.0,
  "take_profit": 52000.0,
  "leverage": 10
}
```

**Параметри:**

| Поле | Тип | Обов'язково | Опис |
|------|-----|-------------|------|
| `symbol` | string | Так | Торгова пара (BTCUSDT, ETHUSDT) |
| `side` | string | Так | BUY або SELL |
| `quantity` | number | Так | Обсяг ордеру |
| `order_type` | string | Ні | MARKET (за замовч.) або LIMIT |
| `price` | number | Для LIMIT | Ціна лімітного ордеру |
| `stop_loss` | number | Ні | Ціна стоп-лосс |
| `take_profit` | number | Ні | Ціна тейк-профіт |
| `leverage` | number | Ні | Плече (1-125) |

**Відповідь:**
```json
{
  "order_id": "abc12345",
  "symbol": "BTCUSDT",
  "side": "BUY",
  "quantity": 0.001,
  "price": 50000.0,
  "status": "FILLED",
  "created_at": "2024-01-15T10:00:00Z"
}
```

#### Скасування ордеру

```http
DELETE /api/order/{order_id}
```

**Відповідь:**
```json
{
  "success": true,
  "message": "Order abc12345 cancelled",
  "order": {
    "order_id": "abc12345",
    "status": "CANCELLED"
  }
}
```

#### Встановлення плеча

```http
PUT /api/leverage
Content-Type: application/json

{
  "symbol": "BTCUSDT",
  "leverage": 20
}
```

**Відповідь:**
```json
{
  "success": true,
  "message": "Leverage for BTCUSDT set to 20x"
}
```

#### Встановлення SL/TP

```http
PUT /api/sl-tp
Content-Type: application/json

{
  "symbol": "BTCUSDT",
  "stop_loss": 49000.0,
  "take_profit": 52000.0
}
```

**Відповідь:**
```json
{
  "success": true,
  "message": "SL/TP updated for BTCUSDT",
  "sl_tp": {
    "stop_loss": 49000.0,
    "take_profit": 52000.0
  }
}
```

---

### Позиції

#### Отримання позицій

```http
GET /api/positions
```

**Відповідь:**
```json
[
  {
    "symbol": "BTCUSDT",
    "side": "BUY",
    "size": 0.001,
    "entry_price": 50000.0,
    "mark_price": 50500.0,
    "unrealized_pnl": 0.50,
    "unrealized_pnl_pct": 1.0,
    "leverage": 10
  }
]
```

#### Закриття позиції

```http
POST /api/position/close
Content-Type: application/json

{
  "symbol": "BTCUSDT",
  "quantity": null  // null = закрити все
}
```

**Відповідь:**
```json
{
  "success": true,
  "message": "Position closed: BTCUSDT",
  "pnl": 0.50,
  "trade": {
    "id": "trade123",
    "symbol": "BTCUSDT",
    "side": "BUY",
    "entry_price": 50000.0,
    "exit_price": 50500.0,
    "quantity": 0.001,
    "pnl": 0.50,
    "pnl_pct": 1.0,
    "strategy": "manual"
  }
}
```

---

### Стратегії

#### Отримання стратегій

```http
GET /api/strategies
```

**Відповідь:**
```json
{
  "orderbook_imbalance": {
    "enabled": true,
    "imbalance_threshold": 1.5,
    "max_spread": 0.0005,
    "signal_cooldown": 10,
    "levels": 5,
    "min_volume_usd": 5000
  },
  "volume_spike": {
    "enabled": true,
    "volume_multiplier": 3.0,
    "lookback_seconds": 60,
    "min_volume_usd": 10000,
    "signal_cooldown": 15
  },
  "mean_reversion": {
    "enabled": false,
    "lookback_period": 20,
    "std_dev_multiplier": 2.0,
    "entry_z_score": 2.0
  },
  "grid_trading": {
    "enabled": false,
    "grid_levels": 10,
    "range_percent": 0.05
  },
  "dca": {
    "enabled": false,
    "mode": "hybrid",
    "interval_minutes": 60
  }
}
```

#### Оновлення стратегії

```http
PUT /api/strategies
Content-Type: application/json

{
  "strategy_name": "orderbook_imbalance",
  "enabled": true,
  "params": {
    "imbalance_threshold": 2.0,
    "levels": 10
  }
}
```

**Відповідь:**
```json
{
  "success": true,
  "message": "Strategy orderbook_imbalance updated",
  "config": {
    "enabled": true,
    "imbalance_threshold": 2.0,
    "levels": 10,
    "max_spread": 0.0005
  }
}
```

#### Перемикання стратегії

```http
POST /api/strategies/{strategy_name}/toggle
```

**Відповідь:**
```json
{
  "success": true,
  "message": "Strategy orderbook_imbalance disabled",
  "enabled": false
}
```

---

### Ризик-менеджмент

#### Отримання конфігурації ризику

```http
GET /api/risk
```

**Відповідь:**
```json
{
  "max_position_usd": 30.0,
  "max_daily_loss_usd": 5.0,
  "max_trades_per_day": 20,
  "max_open_positions": 1,
  "default_stop_loss_pct": 0.5,
  "default_take_profit_pct": 0.3,
  "risk_per_trade_pct": 1.0
}
```

#### Оновлення конфігурації ризику

```http
PUT /api/risk
Content-Type: application/json

{
  "max_position_usd": 50.0,
  "max_daily_loss_usd": 10.0,
  "max_trades_per_day": 30
}
```

**Параметри (всі опціональні):**

| Поле | Тип | Опис |
|------|-----|------|
| `max_position_usd` | number | Макс. розмір позиції (USD) |
| `max_daily_loss_usd` | number | Денний ліміт втрат (USD) |
| `max_trades_per_day` | integer | Макс. угод на день |
| `max_open_positions` | integer | Макс. відкритих позицій |
| `default_stop_loss_pct` | number | Стоп-лосс за замовч. (%) |
| `default_take_profit_pct` | number | Тейк-профіт за замовч. (%) |
| `risk_per_trade_pct` | number | Ризик на угоду (%) |

**Відповідь:**
```json
{
  "success": true,
  "message": "Risk configuration updated",
  "config": {
    "max_position_usd": 50.0,
    "max_daily_loss_usd": 10.0,
    "max_trades_per_day": 30,
    "max_open_positions": 1,
    "default_stop_loss_pct": 0.5,
    "default_take_profit_pct": 0.3,
    "risk_per_trade_pct": 1.0
  }
}
```

---

### Біржа

#### Отримання конфігурації біржі

```http
GET /api/exchange
```

**Відповідь:**
```json
{
  "current": {
    "exchange": "binance",
    "testnet": true
  },
  "available": ["binance", "bybit", "okx", "kraken", "kucoin", "gateio"]
}
```

#### Зміна біржі

```http
PUT /api/exchange
Content-Type: application/json

{
  "exchange": "bybit",
  "testnet": true
}
```

**Відповідь:**
```json
{
  "success": true,
  "message": "Exchange set to bybit",
  "config": {
    "exchange": "bybit",
    "testnet": true
  }
}
```

---

### Дані та метрики

#### Історія угод

```http
GET /api/trades?limit=100&offset=0&symbol=BTCUSDT
```

**Query параметри:**

| Параметр | Тип | Опис |
|----------|-----|------|
| `limit` | integer | Кількість записів (макс. 1000) |
| `offset` | integer | Зсув для пагінації |
| `symbol` | string | Фільтр за символом |

**Відповідь:**
```json
[
  {
    "id": "trade123",
    "symbol": "BTCUSDT",
    "side": "BUY",
    "entry_time": "2024-01-15T10:00:00Z",
    "exit_time": "2024-01-15T11:00:00Z",
    "entry_price": 50000.0,
    "exit_price": 50500.0,
    "quantity": 0.001,
    "pnl": 0.50,
    "pnl_pct": 1.0,
    "strategy": "orderbook_imbalance"
  }
]
```

#### Метрики

```http
GET /api/metrics
```

**Відповідь:**
```json
{
  "total_trades": 100,
  "winning_trades": 65,
  "losing_trades": 35,
  "win_rate": 65.0,
  "total_pnl": 250.50,
  "gross_profit": 400.0,
  "gross_loss": 149.50,
  "profit_factor": 2.67,
  "sharpe_ratio": 1.85,
  "sortino_ratio": 2.10,
  "max_drawdown": 50.0,
  "max_drawdown_pct": 5.0
}
```

#### Сигнали

```http
GET /api/signals?limit=50
```

**Відповідь:**
```json
[
  {
    "timestamp": "2024-01-15T10:00:00Z",
    "symbol": "BTCUSDT",
    "type": "LONG",
    "strength": 0.85,
    "price": 50000.0,
    "strategy": "orderbook_imbalance"
  }
]
```

#### Крива еквіті

```http
GET /api/equity?start=2024-01-01&end=2024-01-31
```

**Відповідь:**
```json
{
  "data": [
    {"timestamp": "2024-01-01T00:00:00Z", "equity": 1000.0},
    {"timestamp": "2024-01-02T00:00:00Z", "equity": 1025.0},
    {"timestamp": "2024-01-03T00:00:00Z", "equity": 1050.5}
  ]
}
```

#### Ринкові дані

```http
GET /api/market/BTCUSDT
```

**Відповідь:**
```json
{
  "symbol": "BTCUSDT",
  "price": 50000.0,
  "orderbook": {
    "bid_volume": 10.5,
    "ask_volume": 8.2,
    "imbalance": 1.28
  }
}
```

---

### Логи та алерти

#### Отримання логів

```http
GET /api/logs?limit=100&level=ERROR&module=orders
```

**Query параметри:**

| Параметр | Тип | Опис |
|----------|-----|------|
| `limit` | integer | Кількість записів (макс. 1000) |
| `level` | string | Фільтр за рівнем (INFO, WARNING, ERROR) |
| `module` | string | Фільтр за модулем |

**Відповідь:**
```json
{
  "total": 1000,
  "filtered": 50,
  "logs": [
    {
      "timestamp": "2024-01-15T10:00:00Z",
      "level": "ERROR",
      "message": "Failed to place order",
      "module": "orders"
    }
  ]
}
```

#### Очищення логів

```http
DELETE /api/logs
```

**Відповідь:**
```json
{
  "success": true,
  "message": "Logs cleared"
}
```

#### Конфігурація алертів

```http
GET /api/alerts
```

**Відповідь:**
```json
{
  "enabled": true
}
```

#### Увімкнення/вимкнення алертів

```http
PUT /api/alerts?enabled=true
```

**Відповідь:**
```json
{
  "success": true,
  "message": "Alerts enabled"
}
```

---

### Конфігурація

#### Отримання загальної конфігурації

```http
GET /api/config
```

**Відповідь:**
```json
{
  "mode": "paper",
  "symbols": ["BTCUSDT", "ETHUSDT"],
  "leverage": {
    "BTCUSDT": 10,
    "ETHUSDT": 10
  },
  "exchange": {
    "exchange": "binance",
    "testnet": true
  },
  "risk": {
    "max_position_usd": 30.0,
    "max_daily_loss_usd": 5.0
  },
  "strategies": {
    "orderbook_imbalance": {"enabled": true},
    "volume_spike": {"enabled": true}
  }
}
```

#### Оновлення конфігурації

```http
POST /api/config
Content-Type: application/json

{
  "key": "mode",
  "value": "live"
}
```

**Доступні ключі:**
- `mode` - режим торгівлі ("paper" або "live")
- `symbols` - список символів

---

## WebSocket API

### Підключення

```
ws://localhost:8000/ws
```

### Початковий стан

Після підключення сервер автоматично надсилає початковий стан:

```json
{
  "type": "initial_state",
  "data": {
    "status": "running",
    "mode": "paper",
    "symbols": ["BTCUSDT"],
    "positions": [],
    "metrics": {
      "total_trades": 0,
      "win_rate": 0,
      "total_pnl": 0
    }
  }
}
```

### Підписка на канали

```json
{
  "type": "subscribe",
  "channels": ["all"]
}
```

### Типи повідомлень

#### Зміна статусу

```json
{
  "type": "status_change",
  "status": "running"
}
```

#### Розміщення ордеру

```json
{
  "type": "order_placed",
  "data": {
    "order_id": "abc123",
    "symbol": "BTCUSDT",
    "side": "BUY",
    "quantity": 0.001,
    "price": 50000.0,
    "status": "FILLED"
  }
}
```

#### Оновлення позицій

```json
{
  "type": "position_update",
  "data": [
    {
      "symbol": "BTCUSDT",
      "side": "BUY",
      "size": 0.001,
      "entry_price": 50000.0,
      "unrealized_pnl": 0.50
    }
  ]
}
```

#### Закриття позиції

```json
{
  "type": "position_closed",
  "data": {
    "symbol": "BTCUSDT",
    "pnl": 0.50,
    "pnl_pct": 1.0
  }
}
```

#### Новий сигнал

```json
{
  "type": "signal",
  "data": {
    "timestamp": "2024-01-15T10:00:00Z",
    "symbol": "BTCUSDT",
    "type": "LONG",
    "strength": 0.85,
    "price": 50000.0,
    "strategy": "orderbook_imbalance"
  }
}
```

#### Нова угода

```json
{
  "type": "trade",
  "data": {
    "symbol": "BTCUSDT",
    "side": "BUY",
    "pnl": 0.50,
    "strategy": "orderbook_imbalance"
  }
}
```

#### Оновлення метрик

```json
{
  "type": "metrics_update",
  "data": {
    "total_trades": 100,
    "win_rate": 65.0,
    "total_pnl": 250.50
  }
}
```

#### Оновлення еквіті

```json
{
  "type": "equity_update",
  "data": {
    "timestamp": "2024-01-15T10:00:00Z",
    "equity": 1050.50
  }
}
```

### Ping/Pong

```json
// Запит
{"type": "ping"}

// Відповідь
{"type": "pong"}
```

---

## Обробка помилок

### Формат помилок

```json
{
  "detail": "Опис помилки"
}
```

### HTTP коди відповідей

| Код | Опис |
|-----|------|
| 200 | Успіх |
| 201 | Створено |
| 400 | Некоректний запит |
| 401 | Не авторизовано |
| 403 | Доступ заборонено |
| 404 | Не знайдено |
| 422 | Помилка валідації |
| 429 | Перевищено ліміт запитів |
| 500 | Внутрішня помилка сервера |
| 503 | Сервіс недоступний |

### Приклади помилок

**Некоректний ордер:**
```json
{
  "detail": "Side must be BUY or SELL"
}
```

**Позиція не знайдена:**
```json
{
  "detail": "No position found for BTCUSDT"
}
```

**Невідома стратегія:**
```json
{
  "detail": "Unknown strategy: invalid_strategy. Available: orderbook_imbalance, volume_spike, mean_reversion, grid_trading, dca"
}
```

---

## Rate Limiting

### Ліміти

| Тип операції | Ліміт |
|--------------|-------|
| Читання | 120 запитів/хвилину |
| Запис | 60 запитів/хвилину |
| WebSocket | 100 повідомлень/хвилину |

### Заголовки rate limit

```http
X-RateLimit-Limit: 120
X-RateLimit-Remaining: 115
X-RateLimit-Reset: 1705312860
```

### Обробка rate limit

```json
{
  "detail": "Rate limit exceeded. Retry after 30 seconds."
}
```

---

## Приклади використання

### cURL

```bash
# Отримати статус
curl http://localhost:8000/api/status

# Розмістити ордер
curl -X POST http://localhost:8000/api/order \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTCUSDT","side":"BUY","quantity":0.001}'

# Закрити позицію
curl -X POST http://localhost:8000/api/position/close \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTCUSDT"}'

# Оновити ризик
curl -X PUT http://localhost:8000/api/risk \
  -H "Content-Type: application/json" \
  -d '{"max_position_usd":50,"max_daily_loss_usd":10}'
```

### Python

```python
import httpx
import asyncio

async def main():
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        # Отримати статус
        status = await client.get("/api/status")
        print(status.json())

        # Розмістити ордер
        order = await client.post("/api/order", json={
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.001
        })
        print(order.json())

        # Закрити позицію
        close = await client.post("/api/position/close", json={
            "symbol": "BTCUSDT"
        })
        print(close.json())

asyncio.run(main())
```

### JavaScript

```javascript
// Отримати статус
const status = await fetch('/api/status');
console.log(await status.json());

// Розмістити ордер
const order = await fetch('/api/order', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        symbol: 'BTCUSDT',
        side: 'BUY',
        quantity: 0.001
    })
});
console.log(await order.json());

// WebSocket
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data);
};
```

---

*Останнє оновлення: Грудень 2025*
