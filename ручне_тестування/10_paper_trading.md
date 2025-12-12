# Тести Paper Trading

## 1. Режим Paper Trading

### TC-264: Активація Paper mode
**Кроки:**
1. Вибрати режим "Paper" в header

**Очікуваний результат:**
- [ ] Indicator "PAPER" видимий
- [ ] Синій колір mode pill
- [ ] Реальні API не викликаються

### TC-265: Paper balance
**Кроки:**
1. Перевірити initial paper balance

**Очікуваний результат:**
- [ ] Default balance (напр. $10,000)
- [ ] Можна змінити в settings
- [ ] Reset balance option

### TC-266: Reset paper account
**Кроки:**
1. Натиснути "Reset Paper Account"

**Очікуваний результат:**
- [ ] Balance скинуто до initial
- [ ] Всі позиції закриті
- [ ] Історія очищена (опціонально)

---

## 2. Paper Trading Operations

### TC-267: Paper order execution
**Кроки:**
1. Відкрити позицію в paper mode

**Очікуваний результат:**
- [ ] Order "виконано" миттєво (market)
- [ ] Entry price = current market price
- [ ] Position створено

### TC-268: Paper limit order
**Кроки:**
1. Створити limit order в paper mode
2. Ціна досягла limit

**Очікуваний результат:**
- [ ] Order pending до досягнення ціни
- [ ] Виконується при досягненні
- [ ] Slippage simulation (optional)

### TC-269: Paper position P&L
**Кроки:**
1. Мати paper позицію
2. Ціна рухається

**Очікуваний результат:**
- [ ] Unrealized P&L оновлюється
- [ ] Real-time на основі market price
- [ ] Leverage враховано

### TC-270: Paper position close
**Кроки:**
1. Закрити paper позицію

**Очікуваний результат:**
- [ ] P&L зафіксовано
- [ ] Balance оновлено
- [ ] Trade в історії

### TC-271: Paper SL/TP execution
**Кроки:**
1. Встановити SL/TP на paper позицію
2. Ціна досягла SL або TP

**Очікуваний результат:**
- [ ] Позиція закрита автоматично
- [ ] P&L коректний
- [ ] Alert про спрацювання

---

## 3. Paper vs Live comparison

### TC-272: Paper mode indicator
**Кроки:**
1. Перевірити візуальні індикатори

**Очікуваний результат:**
- [ ] Чіткий "PAPER TRADING" badge
- [ ] Різний колір від Live
- [ ] Неможливо сплутати

### TC-273: Paper data isolation
**Кроки:**
1. Зробити trades в paper
2. Перемикнути на live

**Очікуваний результат:**
- [ ] Paper trades не в live history
- [ ] Balances незалежні
- [ ] Metrics окремі

### TC-274: Paper to live transition
**Кроки:**
1. Спробувати switch paper -> live

**Очікуваний результат:**
- [ ] Confirmation dialog
- [ ] Warning про реальні гроші
- [ ] Clear mode switch

---

## 4. Paper Trading Simulation

### TC-275: Simulated fees
**Кроки:**
1. Виконати trade в paper
2. Перевірити fee deduction

**Очікуваний результат:**
- [ ] Fees враховані (0.04% maker, 0.1% taker наприклад)
- [ ] Net P&L = Gross P&L - Fees

### TC-276: Simulated slippage
**Кроки:**
1. Виконати market order
2. Перевірити execution price

**Очікуваний результат:**
- [ ] Slippage simulation (optional)
- [ ] Realistic fills

### TC-277: Simulated funding
**Кроки:**
1. Тримати позицію через funding time (00:00, 08:00, 16:00 UTC)

**Очікуваний результат:**
- [ ] Funding fee applied
- [ ] Long/Short різниця

### TC-278: Simulated liquidation
**Кроки:**
1. Позиція досягла liquidation price

**Очікуваний результат:**
- [ ] Position liquidated
- [ ] Loss = margin (isolated)
- [ ] или частковий loss (cross)

---

## 5. Paper Trading Strategies

### TC-279: Strategy testing in paper
**Кроки:**
1. Увімкнути стратегію в paper mode
2. Дочекатись сигналів

**Очікуваний результат:**
- [ ] Сигнали генеруються
- [ ] Orders виконуються в paper
- [ ] Реальних trades немає

### TC-280: Strategy performance tracking
**Кроки:**
1. Запустити стратегію на декілька годин/днів
2. Перевірити metrics

**Очікуваний результат:**
- [ ] Paper performance tracked
- [ ] Win rate, P&L, etc.
- [ ] Basis for live decision

---

## 6. Paper Account Management

### TC-281: Multiple paper accounts
**Кроки:**
1. Створити декілька paper accounts (якщо підтримується)

**Очікуваний результат:**
- [ ] Different strategies per account
- [ ] Independent balances
- [ ] Switch between accounts

### TC-282: Paper account history
**Кроки:**
1. Переглянути повну історію paper account

**Очікуваний результат:**
- [ ] All trades
- [ ] Equity curve
- [ ] Performance metrics

### TC-283: Paper trading leaderboard
**Кроки:**
1. Якщо є competition mode

**Очікуваний результат:**
- [ ] Ranking by P&L
- [ ] Period-based competitions
