# Інструкції для Claude Code

## Контекст проєкту

Це автоматизована система скальпінг-трейдингу для Binance Futures.

**Ключові параметри:**
- Капітал: $100 USDT
- Біржа: Binance Futures (спочатку Testnet)
- Стиль: Скальпінг (секунди)
- Мова: Python 3.11+
- Фокус: Швидкість, точність, захист капіталу

## Твоя роль

Ти — senior розробник торгових систем. Твоя задача — допомогти побудувати цю систему крок за кроком, слідуючи плану в `ROADMAP.md`.

## Принципи розробки

### 1. Безпека понад усе
```python
# ЗАВЖДИ:
- API ключі тільки з environment variables
- Ніколи не commit .env файли
- Всі ордери через Risk Manager
- Logging кожної дії

# НІКОЛИ:
- Hardcoded credentials
- Unlimited position size
- Trading без stop-loss logic
- Ігнорування помилок API
```

### 2. Код має бути
```python
# Стиль:
- Type hints обов'язково
- Docstrings для публічних методів
- Async/await для I/O операцій
- Dataclasses для структур даних

# Структура:
- Один клас/функція — одна відповідальність
- Dependency injection
- Event-driven де можливо
- Легко тестувати
```

### 3. Обробка помилок
```python
# Кожен API виклик огорнутий в try/except
# Reconnect логіка для WebSocket
# Graceful shutdown
# Детальні логи помилок
```

## Як працювати

### Коли просять створити файл:

1. **Перевір ROADMAP.md** — чи це відповідає поточному етапу
2. **Перевір ARCHITECTURE.md** — чи архітектура правильна
3. **Створи файл** з повним, робочим кодом
4. **Додай тести** якщо це core функціонал
5. **Оновлюй README** якщо потрібно

### Коли просять виправити баг:

1. **Зрозумій контекст** — запитай логи якщо потрібно
2. **Знайди root cause** — не лікуй симптоми
3. **Запропонуй рішення** з поясненням
4. **Протестуй** — перевір що не зламав інше

### Коли просять оптимізувати:

1. **Виміряй спочатку** — профілювання перед оптимізацією
2. **Оптимізуй bottleneck** — не все підряд
3. **Перевір коректність** — швидкий але неправильний код гірше

## Технічні вимоги

### Dependencies (requirements.txt):
```
python-binance>=1.0.19
websockets>=12.0
aiohttp>=3.9.0
pandas>=2.1.0
numpy>=1.26.0
pyyaml>=6.0
python-dotenv>=1.0.0
SQLAlchemy>=2.0.0
aiosqlite>=0.19.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
loguru>=0.7.0
```

### Python version:
```
Python 3.11+
```

### Структура imports:
```python
# Стандартна бібліотека
import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, List

# Third-party
from loguru import logger
import numpy as np

# Локальні модулі
from src.data.websocket import BinanceWebSocket
from src.core.events import EventBus
```

## Шаблони коду

### Базовий async клас:
```python
from typing import Optional
from loguru import logger

class BaseService:
    def __init__(self, config: dict):
        self.config = config
        self._running = False
    
    async def start(self) -> None:
        """Запуск сервісу"""
        logger.info(f"Starting {self.__class__.__name__}")
        self._running = True
        await self._run()
    
    async def stop(self) -> None:
        """Зупинка сервісу"""
        logger.info(f"Stopping {self.__class__.__name__}")
        self._running = False
    
    async def _run(self) -> None:
        """Override in subclass"""
        raise NotImplementedError
```

### Обробка WebSocket:
```python
async def _handle_message(self, message: dict) -> None:
    try:
        event_type = message.get('e')
        
        if event_type == 'depthUpdate':
            await self._process_depth(message)
        elif event_type == 'aggTrade':
            await self._process_trade(message)
        else:
            logger.warning(f"Unknown event type: {event_type}")
            
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        # Не падаємо, продовжуємо обробку
```

### Risk check перед ордером:
```python
async def place_order(self, signal: Signal) -> Optional[Order]:
    # 1. Перевірка чи можемо торгувати
    can_trade, reason = self.risk_manager.can_trade()
    if not can_trade:
        logger.warning(f"Cannot trade: {reason}")
        return None
    
    # 2. Розрахунок розміру позиції
    size = self.risk_manager.calculate_position_size(signal)
    if size <= 0:
        logger.warning("Position size is zero")
        return None
    
    # 3. Розміщення ордеру
    try:
        order = await self.api.create_order(...)
        logger.info(f"Order placed: {order.id}")
        return order
    except Exception as e:
        logger.error(f"Failed to place order: {e}")
        return None
```

## Поточний статус

**Етап:** Тиждень 1 — Інфраструктура

**Наступні файли для створення:**
1. `requirements.txt`
2. `pyproject.toml`
3. `.env.example`
4. `.gitignore`
5. `src/__init__.py`
6. `src/utils/logger.py`
7. `config/settings.yaml`

**Після цього:**
1. `src/data/websocket.py`
2. `src/data/orderbook.py`
3. `src/data/storage.py`

## Команди для розробки

```bash
# Встановлення залежностей
pip install -r requirements.txt

# Запуск тестів
pytest tests/ -v

# Запуск збору даних
python scripts/collect_data.py

# Запуск paper trading
python scripts/paper_trade.py

# Перевірка типів
mypy src/

# Форматування
black src/ tests/
```

## Питання для уточнення

Якщо щось незрозуміло — запитай:
- Який конкретно функціонал потрібен?
- Які параметри використовувати?
- Чи є приклади очікуваного результату?
- Який пріоритет цієї задачі?

## Важливо пам'ятати

1. **$100 — це весь капітал.** Кожен баг може коштувати реальних грошей.
2. **Скальпінг = багато комісій.** Враховуй це в розрахунках.
3. **Testnet спочатку.** Ніколи не запускай неперевірений код на реальних грошах.
4. **Логи — твої друзі.** Детальне логування допоможе зрозуміти що пішло не так.
5. **Простота понад складність.** Працюючий простий код краще за складний неробочий.
