# Детальна конфігурація

Повний посібник з налаштування всіх параметрів криптовалютного скальпінг-бота.

## Зміст

1. [Структура конфігурації](#структура-конфігурації)
2. [Налаштування біржі](#налаштування-біржі)
3. [Торгові параметри](#торгові-параметри)
4. [Ризик-менеджмент](#ризик-менеджмент)
5. [Стратегії](#стратегії)
6. [Signal Provider](#signal-provider)
7. [Smart DCA з AI/ML](#smart-dca-з-aiml)
8. [Liquidation Heatmap](#liquidation-heatmap)
9. [News/Event Trading](#newsevent-trading)
10. [Advanced Alerts](#advanced-alerts)
11. [Telegram Bot](#telegram-bot)
12. [Web Dashboard](#web-dashboard)
13. [Логування](#логування)
14. [Бектестинг](#бектестинг)
15. [Приклади конфігурацій](#приклади-конфігурацій)

---

## Структура конфігурації

### Файли конфігурації

```
crypto-scalper-bot/
├── .env                      # Секретні ключі та токени
├── config/
│   ├── config.yaml           # Основна конфігурація
│   ├── strategies.yaml       # Налаштування стратегій
│   └── logging.yaml          # Налаштування логування
```

### Пріоритет конфігурації

1. Змінні середовища (`.env`) - найвищий пріоритет
2. Аргументи командного рядка
3. YAML конфігурація
4. Значення за замовчуванням

---

## Налаштування біржі

### Binance Futures

```yaml
exchange:
  name: binance
  testnet: true                    # true для тестової мережі

  # API налаштування
  api:
    recv_window: 5000              # Вікно прийому (мс)
    request_timeout: 10            # Таймаут запиту (сек)
    retry_count: 3                 # Кількість повторів
    retry_delay: 1                 # Затримка між повторами (сек)

  # WebSocket
  websocket:
    ping_interval: 30              # Інтервал пінгу (сек)
    ping_timeout: 10               # Таймаут пінгу (сек)
    reconnect_delay: 5             # Затримка реконекту (сек)
    max_reconnects: 10             # Макс. кількість реконектів
```

### Bybit

```yaml
exchange:
  name: bybit
  testnet: true

  api:
    recv_window: 5000
    rate_limit: 120                # Запитів на хвилину
```

### OKX

```yaml
exchange:
  name: okx
  testnet: true

  # OKX потребує passphrase
  passphrase_env: OKX_PASSPHRASE   # Змінна середовища
```

### Kraken

```yaml
exchange:
  name: kraken
  testnet: false                   # Kraken не має testnet

  api:
    rate_limit: 15                 # Запитів на секунду
```

### KuCoin

```yaml
exchange:
  name: kucoin
  testnet: true

  passphrase_env: KUCOIN_PASSPHRASE
```

### Gate.io

```yaml
exchange:
  name: gateio
  testnet: true
```

---

## Торгові параметри

### Базові налаштування

```yaml
trading:
  # Торгові пари
  symbols:
    - BTCUSDT
    - ETHUSDT
    - SOLUSDT

  # Кредитне плече
  leverage: 10                     # 1-125 для Binance

  # Тип маржі
  margin_type: cross               # cross або isolated

  # Тип ордерів
  order_type: market               # market або limit

  # Для limit ордерів
  limit_order:
    price_offset_pct: 0.01         # Відступ від ринкової ціни (%)
    time_in_force: GTC             # GTC, IOC, FOK
    post_only: true                # Тільки maker ордери
```

### Фільтри символів

```yaml
trading:
  symbol_filters:
    min_volume_24h: 100000000      # Мін. обсяг за 24г ($)
    min_trades_24h: 100000         # Мін. кількість угод
    max_spread_pct: 0.05           # Макс. спред (%)
    exclude:                       # Виключити символи
      - LUNAPERP
      - USTPERP
```

### Торгові години

```yaml
trading:
  trading_hours:
    enabled: true
    timezone: UTC

    # Торгувати тільки в ці години
    schedule:
      monday:
        - start: "00:00"
          end: "23:59"
      tuesday:
        - start: "00:00"
          end: "23:59"
      # ... інші дні

    # Не торгувати під час новин
    blackout_periods:
      - "2024-01-15T14:30:00Z"     # Приклад: FOMC
      - "2024-02-02T13:30:00Z"     # Приклад: NFP
```

---

## Ризик-менеджмент

### Капітал та позиції

```yaml
risk:
  # Капітал
  total_capital: 1000              # Загальний капітал (USDT)
  max_position_pct: 0.1            # Макс. % на одну позицію

  # Ризик на угоду
  risk_per_trade_pct: 0.01         # Ризикуємо 1% на угоду
  max_loss_per_trade: 10           # Макс. збиток на угоду ($)

  # Ліміти позицій
  max_positions: 3                 # Макс. одночасних позицій
  max_positions_per_symbol: 1      # Макс. позицій на символ
```

### Денні ліміти

```yaml
risk:
  # Денні ліміти
  max_daily_trades: 50             # Макс. угод на день
  max_daily_loss: 30               # Макс. денний збиток ($)
  max_daily_loss_pct: 0.03         # Макс. денний збиток (%)

  # Реакція на ліміти
  on_daily_limit:
    action: pause                  # pause, alert, stop
    resume_next_day: true
```

### Stop Loss та Take Profit

```yaml
risk:
  # Stop Loss
  stop_loss:
    enabled: true
    default_pct: 0.002             # 0.2%
    trailing:
      enabled: true
      activation_pct: 0.001        # Активація при +0.1%
      callback_pct: 0.0005         # Відступ 0.05%

  # Take Profit
  take_profit:
    enabled: true
    default_pct: 0.003             # 0.3%
    partial:
      enabled: true
      levels:
        - pct: 0.002               # На +0.2%
          close_pct: 0.5           # Закрити 50%
        - pct: 0.003               # На +0.3%
          close_pct: 1.0           # Закрити решту
```

### Cooldown періоди

```yaml
risk:
  cooldowns:
    after_loss: 60                 # Пауза після збиткової угоди (сек)
    after_max_loss: 3600           # Пауза після макс. збитку (сек)
    after_win_streak: 0            # Пауза після серії перемог
    after_loss_streak: 300         # Пауза після серії збитків
```

### Аварійні контролі

```yaml
risk:
  emergency:
    # Автоматичний стоп
    auto_stop_loss_pct: 0.05       # Стоп при -5% портфелю

    # Максимальна просадка
    max_drawdown_pct: 0.10         # Стоп при -10% drawdown

    # Аномальна волатильність
    volatility_pause:
      enabled: true
      threshold: 0.05              # При 5% руху за 5 хв
```

---

## Стратегії

### Orderbook Imbalance

```yaml
strategies:
  orderbook_imbalance:
    enabled: true

    # Параметри дисбалансу
    imbalance_threshold: 2.0       # Bid/Ask ratio >= 2.0
    levels_to_analyze: 10          # Рівні книги ордерів

    # Фільтри обсягу
    min_total_volume: 50           # Мін. загальний обсяг
    volume_weight_decay: 0.9       # Зважування за рівнями

    # Сигнали
    signal_strength_threshold: 0.7
    signal_cooldown: 5             # Секунд між сигналами
```

### Volume Spike

```yaml
strategies:
  volume_spike:
    enabled: true

    # Детекція спайків
    spike_multiplier: 3.0          # Обсяг > 3x середнього
    lookback_trades: 100           # Останні N угод
    lookback_seconds: 60           # Або останні N секунд

    # Фільтри
    min_spike_volume: 10           # Мін. обсяг спайку
    min_price_move_pct: 0.001      # Мін. рух ціни

    # Підтвердження
    require_orderbook_confirmation: true
    confirmation_imbalance: 1.5
```

### DCA Strategy

```yaml
strategies:
  dca:
    enabled: true

    # Режим DCA
    mode: hybrid                   # time_based, price_based, hybrid
    direction: accumulate          # accumulate, distribute, bidirectional

    # Часові параметри
    interval_minutes: 60           # Інтервал входу (для time_based/hybrid)
    lookback_minutes: 60           # Період аналізу цін

    # Цінові параметри
    dip_threshold: 0.02            # Поріг падіння (2%)
    pump_threshold: 0.02           # Поріг росту (2%)

    # Розмір позиції
    quantity_per_entry: 0.001      # Кількість на вхід
    max_entries: 100               # Макс. кількість входів
    total_budget: 1000             # Загальний бюджет

    # Масштабування на просадці
    scale_on_dip: true             # Збільшувати розмір при більшій просадці
    scale_factor: 1.5              # Макс. множник

    # === Safety Orders ===
    safety_orders_enabled: true    # Увімкнути safety orders
    safety_orders_count: 5         # Кількість SO
    safety_order_step_pct: 0.01    # Крок ціни (1%)
    safety_order_step_scale: 1.5   # Множник кроку (exp. growth)
    safety_order_size_scale: 1.5   # Множник розміру

    # === Take Profit / Stop Loss ===
    take_profit_pct: 0.10          # Take profit 10%
    stop_loss_pct: 0.05            # Stop loss 5%

    # === Trailing Take Profit ===
    trailing_tp_enabled: true      # Увімкнути trailing TP
    trailing_tp_activation_pct: 0.05  # Активація при +5%
    trailing_tp_deviation_pct: 0.02   # Тригер при -2% від хаю
```

### Grid Trading

```yaml
strategies:
  grid:
    enabled: true

    # Напрямок
    direction: neutral             # neutral, long, short

    # Діапазон сітки
    range:
      auto: true                   # Автоматичне визначення
      auto_range_percentile: 0.9   # 90% діапазону за 24г
      # Або фіксований:
      # lower_price: 48000
      # upper_price: 52000

    # Кількість рівнів
    grid_levels: 10

    # Розмір ордеру
    order_size: 0.001              # Розмір на рівень (BTC)

    # Trailing
    trailing:
      enabled: true
      trigger_pct: 0.02            # Переміщення сітки при +2%
```

### Mean Reversion

```yaml
strategies:
  mean_reversion:
    enabled: true

    # Індикатори
    bollinger_bands:
      period: 20
      std_dev: 2.0

    z_score:
      lookback: 50
      entry_threshold: 2.0
      exit_threshold: 0.5

    rsi:
      enabled: true
      period: 14
      oversold: 30
      overbought: 70

    # Exit
    exit:
      on_mean_reversion: true
      max_holding_time: 300        # Макс. час позиції (сек)
```

### Composite Strategy

```yaml
strategies:
  composite:
    enabled: true

    # Стратегії та ваги
    strategies:
      - name: orderbook_imbalance
        weight: 0.4
      - name: volume_spike
        weight: 0.3
      - name: mean_reversion
        weight: 0.3

    # Порогове значення
    threshold: 0.6                 # Сумарний сигнал >= 0.6

    # Агрегація
    aggregation: weighted_average  # weighted_average, majority, unanimous
```

---

## Signal Provider

Система розсилки торгових сигналів підписникам з різними рівнями доступу.

### Базові налаштування

```yaml
signal_provider:
  enabled: true

  # Ідентифікація провайдера
  provider_id: "my_signals"
  provider_name: "Pro Trading Signals"

  # Рівні підписки
  tiers:
    FREE:
      delay_seconds: 300              # Затримка 5 хв для безкоштовних
      channels: ["telegram"]          # Тільки Telegram
    BASIC:
      delay_seconds: 60               # Затримка 1 хв
      channels: ["telegram", "email"]
    PREMIUM:
      delay_seconds: 10               # Затримка 10 сек
      channels: ["telegram", "email", "webhook"]
    VIP:
      delay_seconds: 0                # Без затримки
      channels: ["telegram", "email", "webhook", "api"]
```

### Налаштування сигналів

```yaml
signal_provider:
  signals:
    # Типи сигналів
    types:
      - LONG
      - SHORT
      - CLOSE_LONG
      - CLOSE_SHORT
      - SCALE_IN
      - SCALE_OUT

    # Фільтри
    min_confidence: 0.7               # Мін. впевненість для публікації
    min_risk_reward: 1.5              # Мін. співвідношення ризик/прибуток

    # Форматування
    include_chart: true               # Включати графік
    include_analysis: true            # Включати аналіз
```

### Канали доставки

```yaml
signal_provider:
  delivery:
    telegram:
      enabled: true
      parse_mode: "HTML"
      include_buttons: true           # Кнопки "Copy Trade"

    email:
      enabled: true
      smtp_host: smtp.gmail.com
      smtp_port: 587
      sender_email_env: SIGNAL_EMAIL
      sender_password_env: SIGNAL_EMAIL_PASS

    webhook:
      enabled: true
      timeout: 10
      retry_count: 3

    api:
      enabled: true
      rate_limit: 100                 # Запитів на хвилину
```

### Статистика та аналітика

```yaml
signal_provider:
  analytics:
    track_performance: true           # Відстежувати результати
    retention_days: 90                # Зберігати історію 90 днів

    # Метрики
    metrics:
      - win_rate
      - profit_factor
      - average_pnl
      - sharpe_ratio
```

---

## Smart DCA з AI/ML

Інтелектуальна стратегія усереднення з машинним навчанням.

### Базові налаштування

```yaml
smart_dca:
  enabled: true

  # Режим оцінки
  mode: hybrid                        # rule_based, ml_based, hybrid

  # Капітал
  total_capital: 1000                 # Загальний капітал для DCA
  base_order_pct: 0.1                 # Базовий ордер 10%
```

### Налаштування рівнів входу

```yaml
smart_dca:
  entry_levels:
    # Автоматичні рівні
    auto:
      enabled: true
      levels_count: 5                 # Кількість рівнів
      price_step_pct: 0.02            # Крок 2%
      multiplier: 1.5                 # Множник розміру

    # Або ручні рівні
    manual:
      - price_drop_pct: 0.02
        size_multiplier: 1.0
      - price_drop_pct: 0.04
        size_multiplier: 1.5
      - price_drop_pct: 0.06
        size_multiplier: 2.0
      - price_drop_pct: 0.10
        size_multiplier: 3.0
```

### Налаштування ML моделі

```yaml
smart_dca:
  ml:
    enabled: true

    # Тип моделі
    model_type: random_forest         # random_forest, gradient_boosting

    # Параметри моделі
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5

    # Навчання
    training:
      min_samples: 1000               # Мін. зразків для навчання
      retrain_interval: 86400         # Перенавчання щодня
      validation_split: 0.2           # 20% для валідації

    # Фічі для моделі
    features:
      - rsi_14
      - macd_histogram
      - bollinger_position
      - volume_sma_ratio
      - fear_greed_index
      - funding_rate
```

### Fear & Greed інтеграція

```yaml
smart_dca:
  fear_greed:
    enabled: true
    api_url: "https://api.alternative.me/fng/"

    # Інтерпретація
    thresholds:
      extreme_fear: 20                # Агресивніші входи
      fear: 40
      neutral: 60
      greed: 80                       # Обережніші входи
      extreme_greed: 100

    # Множники
    multipliers:
      extreme_fear: 1.5               # Збільшити розмір на 50%
      fear: 1.2
      neutral: 1.0
      greed: 0.8
      extreme_greed: 0.5              # Зменшити розмір на 50%
```

### Ризик-менеджмент DCA

```yaml
smart_dca:
  risk:
    max_total_investment: 5000        # Макс. інвестиція
    max_positions: 3                  # Макс. позицій DCA
    stop_loss_pct: 0.15               # Стоп-лосс 15% від середньої ціни
    take_profit_pct: 0.05             # Тейк-профіт 5%

    # Умови виходу
    exit_conditions:
      max_holding_days: 30            # Макс. час утримання
      force_exit_loss_pct: 0.20       # Примусовий вихід при -20%
```

---

## Liquidation Heatmap

Аналіз рівнів ліквідації та Open Interest.

### Базові налаштування

```yaml
liquidation_heatmap:
  enabled: true

  # Джерела даних
  data_sources:
    - coinglass
    - binance_futures
    - bybit_futures

  # Символи для моніторингу
  symbols:
    - BTCUSDT
    - ETHUSDT
```

### Налаштування аналізу

```yaml
liquidation_heatmap:
  analysis:
    # Кластеризація рівнів
    clustering:
      enabled: true
      method: density                 # density, fixed_levels
      min_cluster_size: 1000000       # Мін. $1M в кластері
      price_tolerance_pct: 0.5        # Об'єднувати рівні в межах 0.5%

    # Кредитні плечі для розрахунку
    leverage_levels:
      - 5
      - 10
      - 25
      - 50
      - 100

    # Open Interest
    open_interest:
      enabled: true
      track_changes: true
      alert_change_pct: 5             # Алерт при зміні OI на 5%
```

### Магніти ліквідацій

```yaml
liquidation_heatmap:
  magnets:
    enabled: true

    # Поріг для визначення магніту
    min_volume_usd: 50000000          # Мін. $50M обсягу

    # Оцінка сили притягування
    gravity_decay: 0.95               # Затухання з відстанню

    # Торгові сигнали
    generate_signals: true
    min_distance_pct: 1.0             # Мін. відстань до магніту
    max_distance_pct: 5.0             # Макс. відстань
```

### Каскадні ліквідації

```yaml
liquidation_heatmap:
  cascade:
    enabled: true

    # Виявлення каскадів
    detection:
      min_levels: 3                   # Мін. послідовних рівнів
      max_gap_pct: 0.5                # Макс. проміжок між рівнями

    # Оцінка ризику
    risk_levels:
      low: 10000000                   # < $10M
      medium: 50000000                # $10M - $50M
      high: 100000000                 # $50M - $100M
      extreme: 500000000              # > $500M
```

### Візуалізація

```yaml
liquidation_heatmap:
  visualization:
    enabled: true

    # Формат виводу
    output:
      - console
      - chart
      - json

    # Оновлення
    refresh_interval: 60              # Оновлювати щохвилини

    # Кольорова схема
    colors:
      long_liquidations: "#FF6B6B"    # Червоний
      short_liquidations: "#4ECDC4"   # Зелений
```

---

## News/Event Trading

Торгівля на основі новин та економічних подій.

### Базові налаштування

```yaml
news_event_trading:
  enabled: true

  # Джерела
  sources:
    economic_calendar: true
    token_unlocks: true
    news_sentiment: true
```

### Економічний календар

```yaml
news_event_trading:
  economic_calendar:
    enabled: true

    # Типи подій
    events:
      - type: CPI
        impact: high
        pre_event_action: reduce_exposure
        post_event_wait: 300          # Чекати 5 хв після

      - type: FOMC
        impact: high
        pre_event_action: close_all
        post_event_wait: 900          # Чекати 15 хв

      - type: NFP
        impact: high
        pre_event_action: reduce_exposure
        post_event_wait: 600

      - type: GDP
        impact: medium
        pre_event_action: none
        post_event_wait: 300

    # Попередження
    alerts:
      before_hours: [24, 1, 0.25]     # За 24г, 1г, 15хв
```

### Token Unlocks

```yaml
news_event_trading:
  token_unlocks:
    enabled: true

    # Джерела даних
    data_sources:
      - tokenunlocks_io
      - defillama

    # Фільтри
    min_unlock_value_usd: 10000000    # Мін. $10M
    min_supply_pct: 1.0               # Мін. 1% від supply

    # Торгова стратегія
    strategy:
      # До анлоку
      pre_unlock:
        days_before: 7
        action: SHORT
        size_pct: 0.5

      # Під час анлоку
      during_unlock:
        action: CLOSE

      # Після анлоку
      post_unlock:
        wait_hours: 24
        action: EVALUATE              # Оцінити ситуацію
```

### Аналіз новин

```yaml
news_event_trading:
  news_sentiment:
    enabled: true

    # Джерела новин
    sources:
      - cryptopanic
      - twitter
      - reddit

    # API ключі
    api_keys:
      cryptopanic_env: CRYPTOPANIC_API
      twitter_bearer_env: TWITTER_BEARER

    # Аналіз настрою
    sentiment:
      model: vader                    # vader, transformers
      min_score: 0.6                  # Мін. впевненість

      # Пороги
      thresholds:
        very_bullish: 0.8
        bullish: 0.6
        neutral_low: 0.4
        neutral_high: 0.6
        bearish: 0.4
        very_bearish: 0.2

    # Торгові сигнали
    signals:
      generate: true
      min_mentions: 10                # Мін. згадок для сигналу
      cooldown_minutes: 30
```

### Кореляція з ринком

```yaml
news_event_trading:
  market_correlation:
    enabled: true

    # Традиційні ринки
    correlations:
      - asset: SPY
        correlation_threshold: 0.7
        impact_on: [BTCUSDT, ETHUSDT]

      - asset: DXY
        correlation_threshold: -0.6
        impact_on: [BTCUSDT]

      - asset: GOLD
        correlation_threshold: 0.5
        impact_on: [BTCUSDT]
```

---

## Advanced Alerts

Розширена система алертів та сповіщень.

### Базові налаштування

```yaml
alerts:
  enabled: true

  # Максимум активних алертів
  max_active_alerts: 100

  # Збереження історії
  history_retention_days: 30
```

### Цінові алерти

```yaml
alerts:
  price:
    enabled: true

    # Типи
    types:
      - PRICE_ABOVE                   # Ціна вище рівня
      - PRICE_BELOW                   # Ціна нижче рівня
      - PRICE_CROSS                   # Перетин рівня
      - PRICE_CHANGE_PCT              # Зміна на %

    # Приклади
    examples:
      - symbol: BTCUSDT
        type: PRICE_ABOVE
        value: 100000
        message: "BTC пробив $100k!"

      - symbol: ETHUSDT
        type: PRICE_CHANGE_PCT
        value: 5
        timeframe: 1h
        message: "ETH змінився на 5% за годину"
```

### Алерти обсягу

```yaml
alerts:
  volume:
    enabled: true

    types:
      - VOLUME_SPIKE                  # Сплеск обсягу
      - OI_CHANGE                     # Зміна Open Interest

    # Налаштування
    volume_spike:
      multiplier: 3.0                 # 3x від середнього
      lookback_minutes: 60

    oi_change:
      threshold_pct: 5                # Зміна OI на 5%
      lookback_minutes: 15
```

### Алерти китів

```yaml
alerts:
  whale:
    enabled: true

    # Відстеження транзакцій
    transactions:
      min_value_usd: 1000000          # Мін. $1M
      chains:
        - ethereum
        - bitcoin
        - binance_smart_chain

    # Біржові перекази
    exchange_flows:
      track_inflows: true
      track_outflows: true
      min_value_usd: 500000

    # Джерела даних
    data_sources:
      - whale_alert
      - etherscan
```

### On-chain метрики

```yaml
alerts:
  onchain:
    enabled: true

    # Метрики
    metrics:
      nvt_ratio:
        enabled: true
        overbought: 95                # NVT > 95 = перекуплено
        oversold: 45                  # NVT < 45 = перепродано

      mvrv_zscore:
        enabled: true
        overbought: 7
        oversold: -0.5

      funding_rate:
        enabled: true
        extreme_positive: 0.01        # > 1% = перегрів лонгів
        extreme_negative: -0.01       # < -1% = перегрів шортів

      exchange_netflow:
        enabled: true
        inflow_alert: 10000           # > 10k BTC на біржі
        outflow_alert: -10000
```

### Технічні індикатори

```yaml
alerts:
  technical:
    enabled: true

    indicators:
      rsi:
        overbought: 70
        oversold: 30
        timeframes: [1h, 4h, 1d]

      macd:
        enabled: true
        signal_cross: true
        zero_cross: true

      bollinger:
        enabled: true
        alert_on_touch: true
        alert_on_breakout: true
```

### Канали сповіщень

```yaml
alerts:
  notifications:
    # Telegram
    telegram:
      enabled: true
      chat_id_env: TELEGRAM_CHAT_ID
      parse_mode: HTML

      # Фільтри за пріоритетом
      priorities:
        - critical                    # Завжди
        - high                        # Завжди
        - medium                      # 08:00 - 22:00
        - low                         # Тільки в дашборді

    # Discord
    discord:
      enabled: true
      webhook_url_env: DISCORD_WEBHOOK
      mention_roles:
        critical: "@everyone"
        high: "@traders"

    # Webhook
    webhook:
      enabled: true
      url_env: ALERT_WEBHOOK_URL
      method: POST
      headers:
        Content-Type: application/json

    # Email
    email:
      enabled: false
      smtp_host: smtp.gmail.com
      smtp_port: 587
      only_critical: true
```

### Розклад та cooldown

```yaml
alerts:
  scheduling:
    # Активні години
    active_hours:
      enabled: true
      timezone: UTC
      start: "00:00"
      end: "23:59"

    # Cooldown між алертами
    cooldown:
      same_alert: 300                 # 5 хв між однаковими
      same_symbol: 60                 # 1 хв для того ж символу
      global: 10                      # 10 сек глобально

    # Максимум спрацювань
    max_triggers:
      per_alert: 10                   # Макс. 10 разів на алерт
      reset_after_hours: 24           # Скидати через 24 год
```

---

## Telegram Bot

### Базові налаштування

```yaml
telegram:
  enabled: true

  # Авторизація
  admin_ids:                       # Дозволені ID
    - 123456789
    - 987654321

  # Команди
  commands:
    trading:
      enabled: true                # /buy, /sell, /close
    admin:
      enabled: true                # /stop, /start, /panic
    info:
      enabled: true                # /status, /balance, /positions
```

### Сповіщення

```yaml
telegram:
  notifications:
    # Позиції
    position_opened: true
    position_closed: true
    stop_loss_hit: true
    take_profit_hit: true

    # Сигнали
    signal_generated: false        # Занадто багато

    # Помилки
    errors: true
    warnings: true

    # Звіти
    daily_report: true
    daily_report_time: "00:00"     # UTC

    # Форматування
    include_charts: true
    include_pnl: true
```

### Rate Limits

```yaml
telegram:
  rate_limits:
    messages_per_minute: 20
    batch_delay: 1                 # Затримка між повідомленнями
```

---

## Web Dashboard

### Сервер

```yaml
web:
  enabled: true

  # Сервер
  host: 0.0.0.0
  port: 8080

  # SSL (рекомендовано для production)
  ssl:
    enabled: false
    cert_file: /path/to/cert.pem
    key_file: /path/to/key.pem

  # CORS
  cors:
    enabled: true
    origins:
      - "*"                        # Або конкретні домени
```

### Автентифікація

```yaml
web:
  auth:
    enabled: true
    method: jwt                    # jwt або basic

    # JWT налаштування
    jwt:
      secret_env: JWT_SECRET       # Змінна середовища
      algorithm: HS256
      expiration: 3600             # Час життя токену (сек)

    # Basic auth
    basic:
      username: admin
      password_env: WEB_PASSWORD
```

### WebSocket

```yaml
web:
  websocket:
    enabled: true

    # Оновлення в реальному часі
    updates:
      orderbook: true
      positions: true
      trades: true
      stats: true

    # Інтервали
    orderbook_interval: 100        # мс
    positions_interval: 1000       # мс
```

---

## Логування

### Загальні налаштування

```yaml
logging:
  # Рівень
  level: INFO                      # DEBUG, INFO, WARNING, ERROR

  # Формат
  format: "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

  # Файли
  files:
    main:
      path: logs/trading.log
      rotation: 10 MB
      retention: 30 days

    errors:
      path: logs/errors.log
      level: ERROR
      rotation: 10 MB

    trades:
      path: logs/trades.log
      rotation: 50 MB
```

### Структуроване логування

```yaml
logging:
  json:
    enabled: true
    path: logs/structured.json

  # Поля для логування
  extra_fields:
    - symbol
    - strategy
    - pnl
```

### Моніторинг

```yaml
logging:
  # Prometheus метрики
  prometheus:
    enabled: true
    port: 9090

  # Алерти
  alerts:
    email:
      enabled: false
      smtp_server: smtp.gmail.com
      recipients:
        - admin@example.com

    slack:
      enabled: false
      webhook_url_env: SLACK_WEBHOOK
```

---

## Бектестинг

### Налаштування

```yaml
backtest:
  # Капітал
  initial_capital: 1000
  leverage: 10

  # Симуляція
  slippage_bps: 1.0                # Базисні пункти
  commission_rate: 0.0004          # 0.04%
  latency_ms: 50

  # Дані
  data:
    source: binance                # binance, csv, parquet
    interval: 1m
    warmup_bars: 100

  # Walk-forward
  walk_forward:
    train_period_days: 30
    test_period_days: 7
```

### Monte Carlo

```yaml
backtest:
  monte_carlo:
    enabled: true
    simulations: 1000
    confidence_level: 0.95
```

---

## Приклади конфігурацій

### Консервативний (низький ризик)

```yaml
risk:
  total_capital: 1000
  max_position_pct: 0.05           # 5% на позицію
  risk_per_trade_pct: 0.005        # 0.5% ризику
  max_positions: 1
  max_daily_loss: 10               # 1% від капіталу

strategies:
  orderbook_imbalance:
    enabled: true
    imbalance_threshold: 3.0       # Високий поріг
    signal_strength_threshold: 0.8
```

### Агресивний (високий ризик)

```yaml
risk:
  total_capital: 1000
  max_position_pct: 0.2            # 20% на позицію
  risk_per_trade_pct: 0.02         # 2% ризику
  max_positions: 3
  max_daily_loss: 50               # 5% від капіталу

strategies:
  orderbook_imbalance:
    enabled: true
    imbalance_threshold: 1.5       # Низький поріг
    signal_strength_threshold: 0.6

  volume_spike:
    enabled: true
    spike_multiplier: 2.0
```

### Тестування (Paper Trading)

```yaml
exchange:
  name: binance
  testnet: true

trading:
  symbols:
    - BTCUSDT
  leverage: 5

risk:
  total_capital: 10000             # Віртуальні кошти
  max_daily_loss: 1000

# Включити всі логи
logging:
  level: DEBUG
```

---

## Змінні середовища (.env)

```env
# =============================================================================
# БІРЖІ
# =============================================================================
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
BINANCE_TESTNET=true

BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret
BYBIT_TESTNET=true

OKX_API_KEY=your_key
OKX_API_SECRET=your_secret
OKX_PASSPHRASE=your_passphrase
OKX_TESTNET=true

KRAKEN_API_KEY=your_key
KRAKEN_API_SECRET=your_secret

KUCOIN_API_KEY=your_key
KUCOIN_API_SECRET=your_secret
KUCOIN_PASSPHRASE=your_passphrase
KUCOIN_TESTNET=true

GATEIO_API_KEY=your_key
GATEIO_API_SECRET=your_secret
GATEIO_TESTNET=true

# =============================================================================
# TELEGRAM
# =============================================================================
TELEGRAM_BOT_TOKEN=123456789:ABC...
TELEGRAM_CHAT_ID=-1001234567890
TELEGRAM_ADMIN_IDS=123456789,987654321

# =============================================================================
# WEB DASHBOARD
# =============================================================================
WEB_HOST=0.0.0.0
WEB_PORT=8080
JWT_SECRET=your_very_secret_key_at_least_32_characters
WEB_PASSWORD=admin_password

# =============================================================================
# БАЗА ДАНИХ
# =============================================================================
DATABASE_URL=sqlite:///data/trading.db
# Або PostgreSQL:
# DATABASE_URL=postgresql://user:pass@localhost:5432/trading

# =============================================================================
# МОНІТОРИНГ
# =============================================================================
SLACK_WEBHOOK=https://hooks.slack.com/services/...
PROMETHEUS_PORT=9090

# =============================================================================
# ІНШЕ
# =============================================================================
LOG_LEVEL=INFO
ENVIRONMENT=development
```

---

## Валідація конфігурації

Перевірте конфігурацію перед запуском:

```bash
python -m src.utils.config --validate
```

Тест з'єднання з біржею:

```bash
python -m src.utils.config --test-connection
```

---

*Детальніше про кожен параметр - у [USER_GUIDE_UK.md](./USER_GUIDE_UK.md)*
