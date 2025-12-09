# План розробки (Roadmap)

## Загальний таймлайн

```
Тиждень 1: Інфраструктура та збір даних
Тиждень 2: Аналіз даних та пошук edge
Тиждень 3: Розробка стратегій та бота
Тиждень 4: Paper trading та оптимізація
Тиждень 5+: Live trading (якщо результати позитивні)
```

---

## Тиждень 1: Інфраструктура

### День 1-2: Базовий setup

**Завдання:**
- [ ] Створити структуру проєкту (всі папки та файли)
- [ ] Налаштувати `pyproject.toml` та `requirements.txt`
- [ ] Створити `.env.example` та `.gitignore`
- [ ] Зареєструвати Binance Testnet акаунт
- [ ] Отримати API ключі Testnet

**Файли для створення:**
```
requirements.txt
pyproject.toml
.env.example
.gitignore
src/__init__.py
src/utils/__init__.py
src/utils/logger.py
config/settings.yaml
```

**Критерій успіху:** Проєкт запускається без помилок

### День 3-4: WebSocket підключення

**Завдання:**
- [ ] Реалізувати `src/data/websocket.py`
- [ ] Підключитися до Binance Futures WebSocket (Testnet)
- [ ] Обробляти потоки: depth, aggTrade, markPrice
- [ ] Логувати отримані дані

**Файли для створення:**
```
src/data/__init__.py
src/data/websocket.py
src/data/models.py
```

**Тест:** Запустити і бачити live дані в консолі

### День 5-7: Order Book та зберігання

**Завдання:**
- [ ] Реалізувати `src/data/orderbook.py`
- [ ] Підтримувати локальну копію order book
- [ ] Реалізувати `src/data/storage.py` (SQLite)
- [ ] Зберігати всі дані для аналізу

**Файли для створення:**
```
src/data/orderbook.py
src/data/storage.py
scripts/collect_data.py
```

**Критерій успіху:** 
- Order book синхронізований з біржею
- Дані зберігаються в SQLite
- Скрипт працює 24+ години без падінь

---

## Тиждень 2: Аналіз даних

### День 1-2: Підготовка даних

**Завдання:**
- [ ] Зібрати мінімум 5-7 днів даних
- [ ] Створити Jupyter notebook для аналізу
- [ ] Очистити та структурувати дані
- [ ] Візуалізувати order book dynamics

**Файли для створення:**
```
notebooks/01_data_exploration.ipynb
src/utils/metrics.py
```

### День 3-4: Пошук патернів

**Завдання:**
- [ ] Аналіз order book imbalance vs price movement
- [ ] Аналіз volume spikes vs price movement
- [ ] Аналіз funding rate correlation
- [ ] Документувати знайдені патерни

**Аналізи для проведення:**
```python
# Питання на які шукаємо відповіді:
1. При якому imbalance ціна рухається в бік більшого об'єму?
2. Як швидко ціна реагує на volume spike?
3. Чи є кореляція між funding rate та короткостроковим рухом?
4. Який оптимальний час утримання позиції?
5. Які години доби найприбутковіші?
```

### День 5-7: Формалізація стратегій

**Завдання:**
- [ ] Визначити точні правила входу/виходу
- [ ] Визначити параметри для кожної стратегії
- [ ] Створити простий бектест
- [ ] Оцінити потенційну прибутковість

**Файли для створення:**
```
notebooks/02_strategy_analysis.ipynb
notebooks/03_backtest.ipynb
```

**Критерій успіху:**
- Мінімум 1 стратегія з позитивним expected value
- Win rate > 55% на історичних даних
- Profit factor > 1.2

---

## Тиждень 3: Розробка бота

### День 1-2: Core система

**Завдання:**
- [ ] Реалізувати Event Bus (`src/core/events.py`)
- [ ] Реалізувати головний Engine (`src/core/engine.py`)
- [ ] Інтегрувати WebSocket з Event Bus
- [ ] Написати unit тести

**Файли для створення:**
```
src/core/__init__.py
src/core/events.py
src/core/engine.py
tests/test_events.py
```

### День 3-4: Стратегії

**Завдання:**
- [ ] Реалізувати `src/strategy/base.py`
- [ ] Реалізувати `src/strategy/orderbook_imbalance.py`
- [ ] Реалізувати `src/strategy/volume_spike.py`
- [ ] Написати unit тести

**Файли для створення:**
```
src/strategy/__init__.py
src/strategy/base.py
src/strategy/orderbook_imbalance.py
src/strategy/volume_spike.py
tests/test_strategies.py
```

### День 5-6: Risk Management

**Завдання:**
- [ ] Реалізувати `src/risk/manager.py`
- [ ] Реалізувати `src/risk/position.py`
- [ ] Інтегрувати з Engine
- [ ] Написати unit тести

**Файли для створення:**
```
src/risk/__init__.py
src/risk/manager.py
src/risk/position.py
config/risk.yaml
tests/test_risk.py
```

### День 7: Execution

**Завдання:**
- [ ] Реалізувати `src/execution/binance_api.py`
- [ ] Реалізувати `src/execution/orders.py`
- [ ] Інтегрувати з Engine
- [ ] Тестування на Testnet

**Файли для створення:**
```
src/execution/__init__.py
src/execution/binance_api.py
src/execution/orders.py
tests/test_execution.py
```

**Критерій успіху:**
- Бот успішно розміщує ордери на Testnet
- Всі компоненти інтегровані
- Unit тести проходять

---

## Тиждень 4: Paper Trading

### День 1-2: Інтеграція

**Завдання:**
- [ ] Створити `src/main.py` — точку входу
- [ ] Створити `scripts/paper_trade.py`
- [ ] End-to-end тестування
- [ ] Фікс багів

**Файли для створення:**
```
src/main.py
scripts/paper_trade.py
```

### День 3-5: Paper Trading

**Завдання:**
- [ ] Запустити бота на Testnet
- [ ] Моніторити 24/7
- [ ] Збирати статистику
- [ ] Логувати всі трейди

**Метрики для відстеження:**
```
- Total trades
- Win rate
- Average profit per trade
- Average loss per trade
- Profit factor
- Max drawdown
- Sharpe ratio
- Average holding time
- Trades per hour distribution
```

### День 6-7: Аналіз та оптимізація

**Завдання:**
- [ ] Аналіз результатів paper trading
- [ ] Порівняння з бектестом
- [ ] Tune параметрів
- [ ] Документація результатів

**Файли для створення:**
```
notebooks/04_paper_trading_analysis.ipynb
docs/RESULTS.md
```

**Критерій успіху:**
- Реальні результати близькі до бектесту (±20%)
- Win rate > 50%
- Позитивний P&L за тиждень
- Система стабільна (без падінь)

---

## Тиждень 5+: Live Trading

### Передумови для переходу на live:
- [ ] Мінімум 7 днів успішного paper trading
- [ ] Win rate > 55%
- [ ] Profit factor > 1.2
- [ ] Max drawdown < 15%
- [ ] Система працює стабільно 24/7

### План live trading:

**Фаза A (тиждень 5-6): Мікро-капітал**
- Депозит: $100
- Position size: $10 max
- Мета: Валідація системи на реальних грошах

**Фаза B (тиждень 7-8): Масштабування**
- Якщо Фаза A прибуткова
- Position size: $20-30
- Мета: Збільшення прибутку

**Фаза C (місяць 3+): Розширення**
- Додати більше symbols
- Додати нові стратегії
- Оптимізувати latency

---

## Чеклист перед кожним етапом

### Перед Paper Trading:
- [ ] Всі unit тести проходять
- [ ] Бектест показує позитивні результати
- [ ] Risk management правильно налаштований
- [ ] Logging працює коректно
- [ ] Testnet API ключі налаштовані

### Перед Live Trading:
- [ ] 7+ днів успішного paper trading
- [ ] Production API ключі (без withdraw)
- [ ] IP whitelist налаштований
- [ ] Моніторинг та алерти налаштовані
- [ ] План на випадок збоїв
- [ ] Психологічна готовність втратити весь депозит

---

## Ризики та мітигація

| Ризик | Ймовірність | Вплив | Мітигація |
|-------|-------------|-------|-----------|
| Втрата всього депозиту | Висока | $100 | Прийняти як навчання |
| API disconnection | Середня | Пропущені сигнали | Auto-reconnect, timeout close |
| Slippage | Висока | Менший прибуток | Limit orders, враховувати в бектесті |
| Exchange downtime | Низька | Заморожена позиція | Position timeout, manual override |
| Bug у коді | Середня | Неправильні ордери | Unit тести, paper trading |
