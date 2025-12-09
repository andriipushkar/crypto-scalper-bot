# Швидкий старт

Цей документ містить покрокову інструкцію для швидкого налаштування та запуску криптовалютного скальпінг-бота.

## Зміст

1. [Вимоги](#вимоги)
2. [Встановлення](#встановлення)
3. [Конфігурація](#конфігурація)
4. [Запуск](#запуск)
5. [Перші кроки](#перші-кроки)

---

## Вимоги

### Системні вимоги

- **Python**: 3.10 або вище
- **RAM**: мінімум 4 GB (рекомендовано 8 GB)
- **CPU**: 2+ ядра
- **Інтернет**: стабільне з'єднання (рекомендовано < 50ms latency до біржі)

### Операційні системи

- Linux (рекомендовано Ubuntu 22.04+)
- macOS 12+
- Windows 10/11 (з WSL2)

---

## Встановлення

### Крок 1: Клонування репозиторію

```bash
git clone https://github.com/your-repo/crypto-scalper-bot.git
cd crypto-scalper-bot
```

### Крок 2: Створення віртуального середовища

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Крок 3: Встановлення залежностей

```bash
# Основні залежності
pip install -r requirements.txt

# Додаткові залежності для розробки
pip install -r requirements-dev.txt
```

### Крок 4: Перевірка встановлення

```bash
python -c "import src; print('Встановлення успішне!')"
```

---

## Конфігурація

### Крок 1: Створення файлу конфігурації

```bash
cp config/config.example.yaml config/config.yaml
```

### Крок 2: Створення файлу змінних середовища

```bash
cp .env.example .env
```

### Крок 3: Налаштування API ключів біржі

Відредагуйте файл `.env`:

```env
# =============================================================================
# Налаштування біржі
# =============================================================================

# Binance (основна біржа)
BINANCE_API_KEY=ваш_api_key
BINANCE_API_SECRET=ваш_api_secret
BINANCE_TESTNET=true

# Bybit (опціонально)
BYBIT_API_KEY=ваш_api_key
BYBIT_API_SECRET=ваш_api_secret
BYBIT_TESTNET=true

# OKX (опціонально)
OKX_API_KEY=ваш_api_key
OKX_API_SECRET=ваш_api_secret
OKX_PASSPHRASE=ваш_passphrase
OKX_TESTNET=true

# =============================================================================
# Telegram Bot (опціонально)
# =============================================================================
TELEGRAM_BOT_TOKEN=ваш_bot_token
TELEGRAM_CHAT_ID=ваш_chat_id
TELEGRAM_ADMIN_IDS=123456789,987654321

# =============================================================================
# Web Dashboard
# =============================================================================
WEB_HOST=0.0.0.0
WEB_PORT=8080
JWT_SECRET=ваш_секретний_ключ_для_jwt
```

### Крок 4: Налаштування торгових параметрів

Відредагуйте `config/config.yaml`:

```yaml
# =============================================================================
# Налаштування біржі
# =============================================================================
exchange:
  name: binance                    # binance, bybit, okx, kraken, kucoin, gateio
  testnet: true                    # ВАЖЛИВО: почніть з testnet!

# =============================================================================
# Торгові параметри
# =============================================================================
trading:
  symbols:
    - BTCUSDT
    - ETHUSDT
  leverage: 10                     # Кредитне плече
  margin_type: cross               # cross або isolated

# =============================================================================
# Ризик-менеджмент
# =============================================================================
risk:
  total_capital: 100               # Загальний капітал (USDT)
  max_position_pct: 0.1            # Макс. 10% на позицію
  risk_per_trade_pct: 0.01         # Ризик 1% на угоду
  max_positions: 1                 # Макс. одночасних позицій
  max_daily_trades: 50             # Макс. угод на день
  max_daily_loss: 10               # Макс. денний збиток ($)
  default_stop_loss_pct: 0.002     # Stop Loss 0.2%
  default_take_profit_pct: 0.003   # Take Profit 0.3%

# =============================================================================
# Стратегії
# =============================================================================
strategies:
  orderbook_imbalance:
    enabled: true
    imbalance_threshold: 2.0       # Поріг дисбалансу
    volume_threshold: 50           # Мін. обсяг (BTC)
    signal_strength_threshold: 0.7

  volume_spike:
    enabled: true
    spike_multiplier: 3.0          # Множник обсягу
    lookback_trades: 100
    min_spike_volume: 10
```

---

## Запуск

### Режим 1: Paper Trading (Симуляція)

Рекомендовано для тестування:

```bash
python -m src.main --mode paper
```

### Режим 2: Testnet

Торгівля на тестовій мережі біржі:

```bash
python -m src.main --mode testnet
```

### Режим 3: Live Trading

**УВАГА**: Лише після успішного тестування!

```bash
# Переконайтеся, що BINANCE_TESTNET=false в .env
python -m src.main --mode live
```

### Запуск з Telegram ботом

```bash
python -m src.main --mode testnet --telegram
```

### Запуск з Web Dashboard

```bash
python -m src.main --mode testnet --web
```

### Запуск всіх компонентів

```bash
python -m src.main --mode testnet --telegram --web
```

---

## Перші кроки

### 1. Перевірка з'єднання

Переконайтеся, що бот підключився до біржі:

```bash
# У логах ви повинні побачити:
# INFO - Connected to Binance Futures Testnet
# INFO - WebSocket connected
# INFO - Order book synced
```

### 2. Моніторинг через Telegram

Якщо налаштований Telegram бот:

```
/start        - Активація бота
/status       - Поточний статус
/balance      - Баланс акаунту
/positions    - Відкриті позиції
/help         - Список команд
```

### 3. Моніторинг через Web Dashboard

Відкрийте у браузері: `http://localhost:8080`

- **Dashboard**: Огляд портфелю
- **Positions**: Відкриті позиції
- **Trades**: Історія угод
- **Settings**: Налаштування

### 4. Перегляд логів

```bash
# Основні логи
tail -f logs/trading.log

# Тільки помилки
tail -f logs/errors.log

# У реальному часі
python -m src.main --mode testnet --log-level DEBUG
```

---

## Часті проблеми та рішення

### Помилка: "Invalid API key"

```
Рішення:
1. Перевірте правильність API ключів у .env
2. Переконайтеся, що включені Futures торги
3. Перевірте IP whitelist на біржі
```

### Помилка: "Connection refused"

```
Рішення:
1. Перевірте інтернет-з'єднання
2. Перевірте firewall налаштування
3. Спробуйте VPN якщо біржа заблокована
```

### Помилка: "Insufficient margin"

```
Рішення:
1. Поповніть баланс на біржі
2. Зменшіть розмір позиції в config.yaml
3. Зменшіть кредитне плече
```

### Бот не відкриває позиції

```
Рішення:
1. Перевірте чи є сигнали в логах
2. Збільшіть signal_strength_threshold
3. Перевірте ризик-менеджмент параметри
4. Переконайтеся, що не досягнуто денний ліміт
```

---

## Наступні кроки

1. **Вивчіть документацію**:
   - [USER_GUIDE_UK.md](./USER_GUIDE_UK.md) - Повний посібник користувача
   - [API_UK.md](./API_UK.md) - API документація
   - [CONFIGURATION_UK.md](./CONFIGURATION_UK.md) - Детальні налаштування

2. **Налаштуйте стратегії**:
   - Протестуйте різні стратегії на paper trading
   - Оптимізуйте параметри через backtesting
   - Налаштуйте алерти

3. **Розширені модулі**:
   - **Signal Provider** - Публікуйте сигнали для підписників
   - **Smart DCA з AI/ML** - Інтелектуальне усереднення позицій
   - **Liquidation Heatmap** - Аналіз рівнів ліквідацій
   - **News/Event Trading** - Торгівля на новинах та подіях
   - **Advanced Alerts** - Розширені алерти (ціна, обсяг, кити, on-chain)

4. **Безпека**:
   - Налаштуйте IP whitelist на біржі
   - Використовуйте окремі API ключі для бота
   - Обмежте дозволи (лише торгівля, без виведення)

5. **Моніторинг**:
   - Налаштуйте алерти в Telegram
   - Моніторте логи регулярно
   - Слідкуйте за денним P&L

---

## Контакти та підтримка

- **GitHub Issues**: Повідомлення про баги
- **Telegram**: @your_support_bot
- **Email**: support@example.com

---

*Успіхів у торгівлі!*
