# Стратегії з книги "Скальпинг: практическое руководство трейдера"

Цей документ описує стратегії та концепції, реалізовані на основі книги "Скальпинг: практическое руководство трейдера" від United Traders (автори: Артем Євсєєв, Антон Клєвцов).

## Зміст

1. [Огляд](#огляд)
2. [Range Trading Strategy](#range-trading-strategy)
3. [Session Trading Strategy](#session-trading-strategy)
4. [Trendline Breakout Strategy](#trendline-breakout-strategy)
5. [Size Bounce/Breakout](#size-bouncebreakout)
6. [Bid/Ask Flip Detection](#bidask-flip-detection)
7. [Order Flow Velocity](#order-flow-velocity)
8. [Fee Optimizer](#fee-optimizer)
9. [Конфігурація](#конфігурація)

---

## Огляд

Книга описує класичний скальпінг на американському фондовому ринку. Ми адаптували ці концепції для криптовалютного ринку:

| Концепція з книги | Адаптація для крипто |
|-------------------|---------------------|
| ECN rebates | Maker/Taker fees |
| Morning return | Session-based trading (Азія/Європа/США) |
| Level II | Order Book analysis |
| Time & Sales | Print Tape analyzer |
| Leaders (SPX, DXY) | BTC, ETH, Total Market Cap |

---

## Range Trading Strategy

**Файл:** `src/strategy/range_trading.py`

### Концепція

Стратегії #1 та #2 з книги:
- **Повернення в рейндж після хибного пробою**
- **Торгівля від меж рейнджу**

> "Ціна більшу частину часу рухається в рейнджі. Хибні пробої рейнджу часто призводять до швидкого повернення ціни назад."

### Логіка

1. **Визначення рейнджу:**
   - Аналіз N останніх барів (default: 50)
   - Знаходження high/low
   - Підрахунок дотиків до меж

2. **Класифікація пробою:**
   - **True Breakout:** пробій + обсяг + час підтвердження
   - **False Breakout:** пробій без підтвердження → повернення

3. **Сигнали:**
   - На хибному пробої вгору → SHORT
   - На хибному пробої вниз → LONG
   - Біля верхньої межі + продавці → SHORT
   - Біля нижньої межі + покупці → LONG

### Параметри

```yaml
range_trading:
  enabled: true
  range_lookback_periods: 50      # Барів для визначення рейнджу
  min_range_width_pct: 0.2        # Мін. ширина рейнджу (%)
  max_range_width_pct: 2.0        # Макс. ширина
  min_touches: 2                   # Мін. дотиків до межі
  breakout_threshold_pct: 0.1      # Поріг пробою (%)
  volume_confirmation_multiplier: 1.5
```

---

## Session Trading Strategy

**Файл:** `src/strategy/session_trading.py`

### Концепція

Стратегія #4 з книги "Morning Return", адаптована для 24/7 крипто-ринку:

> "На початку торгової сесії ціна часто повертається до закриття попередньої сесії"

### Крипто-сесії

| Сесія | Час (UTC) | Характеристика |
|-------|-----------|----------------|
| Азіатська | 00:00-08:00 | Низька волатильність |
| Європейська | 07:00-16:00 | Середня волатильність |
| Американська | 13:00-22:00 | Висока волатильність |

### Логіка

1. **Трекінг рівнів сесії:**
   - Open, High, Low, Close, VWAP

2. **Gap Detection:**
   - Визначення gap між сесіями
   - Gap fill probability ≈ 70%

3. **Сигнали:**
   - Gap fill (торгуємо повернення)
   - Рівні попередньої сесії як S/R
   - Overlap momentum

### Параметри

```yaml
session_trading:
  enabled: true
  min_gap_pct: 0.3               # Мін. gap для сигналу
  max_gap_pct: 3.0               # Макс. gap (більше = тренд)
  use_prev_session_levels: true
  trade_overlaps: true           # Торгувати під час overlap
```

---

## Trendline Breakout Strategy

**Файл:** `src/strategy/trendline_breakout.py`

### Концепція

Стратегія #3 з книги:

> "Будуємо трендову лінію по локальних мінімумах (для висхідного тренду) або максимумах (для низхідного тренду). Пробій лінії з обсягом - сигнал для входу."

### Логіка

1. **Pivot Detection:**
   - Знаходження локальних екстремумів
   - Мін. N барів зліва/справа

2. **Trendline Creation:**
   - Побудова лінії по 2+ pivots
   - Перевірка кута нахилу

3. **Breakout Detection:**
   - Пробій лінії > threshold
   - Підтвердження обсягом

4. **Сигнали:**
   - Пробій лінії підтримки → SHORT
   - Пробій лінії опору → LONG

### Параметри

```yaml
trendline_breakout:
  enabled: true
  pivot_lookback: 5              # Барів для pivot
  min_trendline_touches: 2       # Мін. дотиків до лінії
  breakout_threshold_pct: 0.15   # Поріг пробою
  volume_confirmation_multiplier: 1.3
```

---

## Size Bounce/Breakout

**Файл:** `src/strategy/advanced_orderbook.py`

### Концепція

Стратегії #5 та #6 з книги про торгівлю великих ордерів:

> "Size Breakout: пробій великого ордера → торгуємо в напрямку пробою"
> "Size Bounce: відскок від великого ордера → торгуємо відскок"

### Логіка

1. **Wall Detection:**
   - Виявлення великих ордерів (> threshold USD)
   - Трекінг стабільності стіни

2. **Reaction Classification:**
   - **BOUNCE:** ціна відскочила, стіна залишилась
   - **BREAKOUT:** стіна поглинена, ціна пішла далі
   - **ABSORPTION:** стіна поступово зменшується

3. **Сигнали:**
   - Bounce від bid wall → LONG
   - Bounce від ask wall → SHORT
   - Breakout bid wall → SHORT
   - Breakout ask wall → LONG

### Параметри

```yaml
advanced_orderbook:
  size_reaction_enabled: true
  wall_threshold_usd: 100000     # Мін. розмір стіни
  bounce_confirmation_pct: 0.1   # Поріг відскоку
  breakout_confirmation_pct: 0.15 # Поріг пробою
  absorption_volume_threshold: 0.5 # Поглинання якщо < 50%
```

---

## Bid/Ask Flip Detection

**Файл:** `src/strategy/advanced_orderbook.py`

### Концепція

Концепція з книги:

> "Коли великий bid раптово зникає і з'являється великий ask на тій же ціні або вище - це сигнал що великий гравець змінив позицію"

### Логіка

1. **Tracking:**
   - Запам'ятовування великих ордерів
   - Вікно спостереження (default: 5 сек)

2. **Flip Detection:**
   - BID → ASK: ведмежий сигнал
   - ASK → BID: бичачий сигнал

### Параметри

```yaml
advanced_orderbook:
  flip_detection_enabled: true
  flip_min_volume_usd: 50000     # Мін. розмір для flip
  flip_price_tolerance_pct: 0.1  # Толерантність ціни
  flip_time_window_sec: 5        # Вікно часу
```

---

## Order Flow Velocity

**Файл:** `src/analytics/print_tape.py`

### Концепція

Книга наголошує на швидкості зміни потоку ордерів:

> "Швидкість потоку важливіша за сам потік. Прискорення покупок/продажів - сигнал momentum."

### Метрики

- **trades_per_second:** швидкість потоку
- **trade_acceleration:** зміна швидкості
- **delta_acceleration:** зміна напрямку тиску
- **buy_pressure / sell_pressure:** частка покупок/продажів
- **pressure_change:** зміна тиску

### Сигнали

1. **Acceleration Signal:**
   - Прискорення бичачого потоку → LONG
   - Прискорення ведмежого потоку → SHORT

2. **Pressure Reversal:**
   - Різка зміна buy/sell pressure → сигнал розвороту

### Параметри

```yaml
print_tape:
  short_window_seconds: 10
  medium_window_seconds: 60
  delta_threshold_percent: 20.0
```

---

## Fee Optimizer

**Файл:** `src/execution/fee_optimizer.py`

### Концепція

Адаптація концепції ECN rebates з книги:

> "В класичному скальпінгу важливо розуміти структуру комісій:
> - Add liquidity (maker) = отримуємо rebate
> - Remove liquidity (taker) = платимо комісію"

### Функціонал

1. **Execution Plan:**
   - Вибір maker vs taker
   - Post-only ордери

2. **Trade Analysis:**
   - Break-even з урахуванням комісій
   - Net R:R ratio

3. **Fee Comparison:**
   - Економія maker vs taker
   - Річна економія

### Приклад

```python
from src.execution.fee_optimizer import FeeOptimizer

optimizer = FeeOptimizer("binance")

# План виконання
plan = optimizer.get_execution_plan(
    side="buy",
    best_bid=Decimal("50000"),
    best_ask=Decimal("50010"),
    urgency="normal"
)
print(f"Order type: {plan.order_type}")
print(f"Fee: {plan.estimated_fee_bps} bps")

# Аналіз трейду
analysis = optimizer.analyze_trade(
    entry_price=Decimal("50000"),
    target_price=Decimal("50100"),
    stop_price=Decimal("49950"),
    position_size=Decimal("1"),
    side="long"
)
print(f"Break-even: {analysis.break_even_price}")
print(f"Net R:R: {analysis.risk_reward_net:.2f}")
print(f"Recommendation: {analysis.recommendation}")
```

### Структура комісій

| Біржа | Maker (bps) | Taker (bps) | Економія |
|-------|-------------|-------------|----------|
| Binance | 1.0 | 5.0 | 4.0 bps |
| Bybit | 1.0 | 6.0 | 5.0 bps |
| OKX | 2.0 | 5.0 | 3.0 bps |

---

## Конфігурація

### Повний приклад config/settings.yaml

```yaml
# Book-based strategies
strategies:
  range_trading:
    enabled: true
    signal_cooldown: 10.0
    range_lookback_periods: 50
    min_range_width_pct: 0.2
    max_range_width_pct: 2.0
    min_touches: 2
    breakout_threshold_pct: 0.1

  session_trading:
    enabled: true
    signal_cooldown: 30.0
    min_gap_pct: 0.3
    max_gap_pct: 3.0
    use_prev_session_levels: true
    trade_overlaps: true

  trendline_breakout:
    enabled: true
    signal_cooldown: 15.0
    pivot_lookback: 5
    min_trendline_touches: 2
    breakout_threshold_pct: 0.15

  advanced_orderbook:
    enabled: true
    wall_threshold_usd: 100000
    size_reaction_enabled: true
    flip_detection_enabled: true
    bounce_confirmation_pct: 0.1
    breakout_confirmation_pct: 0.15

# Analytics
analytics:
  print_tape:
    whale_threshold: 100000
    delta_threshold_percent: 20.0
    short_window_seconds: 10

# Execution
execution:
  fee_optimizer:
    prefer_maker: true
    use_post_only: true
    max_wait_for_fill_ms: 5000
    min_profit_after_fees_bps: 5.0
```

---

## Тестування

```bash
# Запуск тестів для book-based стратегій
pytest tests/test_book_strategies.py -v

# Запуск конкретного тесту
pytest tests/test_book_strategies.py::TestRangeTradingStrategy -v
```

---

## Посилання

- [Книга "Скальпинг: практическое руководство трейдера"](https://download.unitedtraders.com/education/books/Scalping-prakticheskoe_rukovodstvo_treidera.pdf)
- United Traders
