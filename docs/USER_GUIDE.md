# User Guide

Complete guide for using the Crypto Scalper Bot.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Configuration](#configuration)
3. [Trading Modes](#trading-modes)
4. [Strategies](#strategies)
5. [Risk Management](#risk-management)
6. [Telegram Bot](#telegram-bot)
7. [Web Dashboard](#web-dashboard)
8. [Backtesting](#backtesting)
9. [Paper Trading](#paper-trading)
10. [Multi-Account Trading](#multi-account-trading)
11. [Alerts & Notifications](#alerts--notifications)
12. [Best Practices](#best-practices)
13. [FAQ](#faq)

---

## Getting Started

### Quick Start (5 Minutes)

1. **Install the bot:**
   ```bash
   git clone <repo-url>
   cd crypto-scalper-bot
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure API keys:**
   ```bash
   cp .env.example .env
   # Edit .env with your exchange API keys
   ```

3. **Start in paper trading mode:**
   ```bash
   python -m src.main --paper
   ```

4. **Open the dashboard:**
   ```bash
   python -m src.dashboard.app
   # Visit http://localhost:8000
   ```

### First Trading Session

Before going live, we recommend:

1. **Paper trade for at least 1 week** to understand bot behavior
2. **Review all strategies** and their parameters
3. **Set conservative risk limits** (max 1% per trade)
4. **Start with one symbol** (e.g., BTCUSDT)
5. **Monitor closely** for the first few days

---

## Configuration

### Environment Variables

Create a `.env` file with your settings:

```bash
# ═══════════════════════════════════════════════════════════════════
# EXCHANGE CREDENTIALS
# ═══════════════════════════════════════════════════════════════════
# At least one exchange is required
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
BINANCE_TESTNET=false                  # Use true for testing

BYBIT_API_KEY=your_bybit_api_key
BYBIT_API_SECRET=your_bybit_api_secret

OKX_API_KEY=your_okx_api_key
OKX_API_SECRET=your_okx_api_secret
OKX_PASSPHRASE=your_okx_passphrase

# ═══════════════════════════════════════════════════════════════════
# TELEGRAM (Optional but recommended)
# ═══════════════════════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_ALLOWED_USERS=123456789,987654321   # Your Telegram user IDs
TELEGRAM_ADMIN_USERS=123456789               # Admin user IDs

# ═══════════════════════════════════════════════════════════════════
# SLACK (Optional)
# ═══════════════════════════════════════════════════════════════════
SLACK_BOT_TOKEN=xoxb-your-slack-token
SLACK_DEFAULT_CHANNEL=#trading-alerts

# ═══════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/trading
# For SQLite (development only):
# DATABASE_URL=sqlite:///./trading.db

# ═══════════════════════════════════════════════════════════════════
# REDIS (Optional, improves performance)
# ═══════════════════════════════════════════════════════════════════
REDIS_URL=redis://localhost:6379/0
```

### Trading Configuration (config/settings.yaml)

```yaml
# ═══════════════════════════════════════════════════════════════════
# GENERAL SETTINGS
# ═══════════════════════════════════════════════════════════════════
trading:
  # Symbols to trade (start with 1-2 for testing)
  symbols:
    - BTCUSDT
    - ETHUSDT

  # Default leverage (1-125, recommend 5-10 for safety)
  leverage: 10

  # Paper trading mode (always start with true)
  paper_trading: true

  # Default exchange
  default_exchange: binance

# ═══════════════════════════════════════════════════════════════════
# RISK MANAGEMENT (CRITICAL - Set these carefully!)
# ═══════════════════════════════════════════════════════════════════
risk:
  # Maximum position size as percentage of balance (0.1 = 10%)
  max_position_size: 0.1

  # Stop trading after this daily loss (in USD)
  max_daily_loss: 100

  # Maximum drawdown before stopping (0.1 = 10%)
  max_drawdown: 0.1

  # Risk per trade (0.01 = 1% of balance)
  risk_per_trade: 0.01

  # Maximum concurrent positions
  max_positions: 3

  # Close positions after this many seconds (0 = disabled)
  position_timeout: 0

# ═══════════════════════════════════════════════════════════════════
# STRATEGY SETTINGS
# ═══════════════════════════════════════════════════════════════════
strategies:
  orderbook_imbalance:
    enabled: true
    threshold: 1.5           # Bid/ask ratio to trigger
    levels: 5                # Order book depth levels

  volume_spike:
    enabled: true
    multiplier: 3.0          # Volume spike detection multiplier
    lookback_seconds: 60     # Lookback period

  momentum:
    enabled: true
    rsi_period: 14
    overbought: 70           # RSI level for overbought
    oversold: 30             # RSI level for oversold

  ml_signals:
    enabled: false           # Disable until you have trained model
    model_path: models/rf_model.pkl
    confidence_threshold: 0.7

  sentiment:
    enabled: false           # Requires API keys
    sources:
      - twitter
      - reddit
    weight: 0.3

  arbitrage:
    enabled: false           # Requires multiple exchanges
    min_spread: 0.001        # Minimum 0.1% spread
```

---

## Trading Modes

### Paper Trading Mode

**Recommended for beginners and testing strategies.**

```bash
# Start in paper mode
python -m src.main --paper

# Or set in config
trading:
  paper_trading: true
```

Features:
- Uses real market data
- Simulates order execution with slippage
- Tracks P&L without real money
- Perfect for strategy testing

### Live Trading Mode

**Use with caution! Real money at risk.**

```bash
# Start in live mode
python -m src.main --live

# Or set in config
trading:
  paper_trading: false
```

Pre-flight checklist:
- [ ] Paper traded for at least 1 week
- [ ] Verified API keys work correctly
- [ ] Set conservative risk limits
- [ ] Enabled Telegram notifications
- [ ] Disabled withdrawal on API keys
- [ ] Set IP whitelist on exchange

### Testnet Mode

**Use exchange testnets for risk-free testing with real order execution.**

```bash
# Enable testnet
BINANCE_TESTNET=true
```

Get testnet API keys:
- Binance: https://testnet.binancefuture.com
- Bybit: https://testnet.bybit.com

---

## Strategies

### Orderbook Imbalance

Detects when buy or sell pressure dominates the order book.

**How it works:**
1. Calculates total bid volume vs ask volume
2. Generates signal when ratio exceeds threshold
3. Long when bids dominate, short when asks dominate

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| threshold | 1.5 | Ratio needed to trigger |
| levels | 5 | Order book depth |

**Best for:** High-volume pairs, volatile markets

### Volume Spike

Detects abnormal trading volume that may precede price moves.

**How it works:**
1. Calculates average volume over lookback period
2. Triggers when current volume exceeds average by multiplier
3. Direction based on price movement during spike

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| multiplier | 3.0 | Volume spike multiplier |
| lookback_seconds | 60 | Period for average calculation |

**Best for:** News-driven moves, breakouts

### Momentum

Uses RSI and price momentum for trend following.

**How it works:**
1. Calculates RSI over specified period
2. Long when RSI crosses above oversold
3. Short when RSI crosses below overbought

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| rsi_period | 14 | RSI calculation period |
| overbought | 70 | Overbought threshold |
| oversold | 30 | Oversold threshold |

**Best for:** Trending markets, mean reversion

### ML Signals

Machine learning based predictions.

**Setup:**
```bash
# Train model first
python scripts/train_model.py --symbol BTCUSDT --start 2023-01-01 --end 2024-01-01
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| model_type | random_forest | Model algorithm |
| confidence_threshold | 0.7 | Minimum confidence |

### Sentiment Analysis

Aggregates sentiment from social media.

**Setup:**
```bash
# Add API keys
TWITTER_BEARER_TOKEN=your_token
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| sources | [twitter, reddit] | Data sources |
| weight | 0.3 | Weight in signal aggregation |

### Arbitrage

Finds price discrepancies across exchanges.

**Requirements:**
- API keys for multiple exchanges
- Fast network connection
- Sufficient balance on all exchanges

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| min_spread | 0.001 | Minimum spread (0.1%) |
| max_execution_time | 2.0 | Max seconds to execute |

---

## Risk Management

### Position Sizing

The bot uses risk-based position sizing:

```
Position Size = (Balance × Risk Per Trade) / Stop Distance
```

Example:
- Balance: $10,000
- Risk per trade: 1%
- Stop distance: 2%
- Position size: ($10,000 × 0.01) / 0.02 = $5,000

### Stop-Loss Orders

Always use stop-losses:

```yaml
risk:
  default_stop_loss_pct: 2.0     # 2% stop loss
  trailing_stop_enabled: true
  trailing_stop_pct: 1.0         # 1% trailing
```

### Daily Limits

Protect against catastrophic losses:

```yaml
risk:
  max_daily_loss: 100            # Stop after $100 loss
  max_daily_trades: 50           # Max trades per day
```

### Drawdown Protection

Automatically stops trading on drawdown:

```yaml
risk:
  max_drawdown: 0.1              # Stop at 10% drawdown
  drawdown_reset: daily          # Reset counter daily
```

---

## Telegram Bot

### Setup

1. **Create bot with BotFather:**
   - Message @BotFather on Telegram
   - Send `/newbot` and follow instructions
   - Save the token

2. **Get your user ID:**
   - Message @userinfobot
   - Copy your user ID

3. **Configure bot:**
   ```bash
   TELEGRAM_BOT_TOKEN=your_token
   TELEGRAM_ALLOWED_USERS=your_user_id
   ```

### Commands

| Command | Description |
|---------|-------------|
| `/start` | Initialize bot |
| `/status` | Bot status and P&L |
| `/balance` | Account balance |
| `/positions` | Open positions |
| `/trades` | Recent trades |
| `/pnl` | Daily P&L summary |
| `/buy BTCUSDT 0.1` | Buy 0.1 BTC |
| `/sell ETHUSDT 1` | Sell 1 ETH |
| `/close BTCUSDT` | Close BTC position |
| `/closeall` | Close all positions |
| `/stop` | Stop trading |
| `/resume` | Resume trading |

### Notifications

You'll receive alerts for:
- Trade executions
- Significant P&L changes
- Risk limit warnings
- System errors
- Daily summary

---

## Web Dashboard

### Accessing

```bash
# Start dashboard
python -m src.dashboard.app

# Open in browser
http://localhost:8000
```

### Features

1. **Overview Tab**
   - Real-time P&L
   - Equity curve
   - Open positions
   - Recent trades

2. **Positions Tab**
   - Position details
   - Unrealized P&L
   - One-click close

3. **Trades Tab**
   - Trade history
   - Filter by symbol/date
   - Performance stats

4. **Strategies Tab**
   - Strategy status
   - Enable/disable
   - Parameter tuning

5. **Settings Tab**
   - Risk limits
   - Notification preferences
   - API configuration

---

## Backtesting

### Running a Backtest

```bash
# Basic backtest
python scripts/backtest.py \
  --symbol BTCUSDT \
  --start 2024-01-01 \
  --end 2024-06-01 \
  --strategy momentum

# With custom parameters
python scripts/backtest.py \
  --symbol BTCUSDT \
  --start 2024-01-01 \
  --end 2024-06-01 \
  --strategy momentum \
  --param rsi_period=21 \
  --param overbought=80
```

### Interpreting Results

```
═══════════════════════════════════════════════════════════════════
                         BACKTEST REPORT
═══════════════════════════════════════════════════════════════════

RETURNS
  Initial Balance: $10,000.00
  Final Balance:   $12,450.00
  Total Return:    24.50%

TRADES
  Total Trades:    156
  Win Rate:        62.82%
  Profit Factor:   1.65

RISK METRICS
  Max Drawdown:    8.5%
  Sharpe Ratio:    1.85    ← Above 1.0 is good
  Sortino Ratio:   2.10    ← Above 1.5 is good
═══════════════════════════════════════════════════════════════════
```

**Key Metrics to Look For:**

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Win Rate | >50% | >55% | >60% |
| Profit Factor | >1.0 | >1.3 | >1.5 |
| Sharpe Ratio | >0.5 | >1.0 | >1.5 |
| Max Drawdown | <30% | <15% | <10% |

### Strategy Optimization

```bash
# Optimize strategy parameters
python scripts/optimize.py \
  --symbol BTCUSDT \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --strategy momentum \
  --trials 100
```

This will find optimal parameters using walk-forward optimization.

---

## Paper Trading

### Starting Paper Trading

```bash
# Via command line
python -m src.main --paper

# Or in config
trading:
  paper_trading: true
```

### Paper Trading Features

- Real-time market data
- Simulated order execution
- Realistic slippage (0.05%)
- Commission simulation (0.1%)
- Leverage support
- Liquidation simulation

### Monitoring Paper Trades

```bash
# Check status via Telegram
/status
/positions
/pnl

# Or via dashboard
http://localhost:8000
```

### Transitioning to Live

After successful paper trading:

1. Verify consistent positive results over 1+ weeks
2. Ensure drawdowns are within acceptable limits
3. Confirm all notifications work
4. Set up live API keys
5. Start with reduced position sizes
6. Monitor closely for first few days

---

## Multi-Account Trading

### Setting Up Multiple Accounts

```python
from src.trading.multi_account import MultiAccountManager, AccountType

manager = MultiAccountManager()

# Add accounts
acc1 = manager.add_account(
    name="Main Account",
    exchange="binance",
    api_key="key1",
    api_secret="secret1",
    account_type=AccountType.FUTURES
)

acc2 = manager.add_account(
    name="Secondary Account",
    exchange="bybit",
    api_key="key2",
    api_secret="secret2",
    account_type=AccountType.FUTURES
)
```

### Creating Account Groups

```python
# Create group with allocation
group = manager.create_group(
    name="Main Portfolio",
    account_ids=[acc1.id, acc2.id],
    allocation={
        acc1.id: Decimal("60"),  # 60%
        acc2.id: Decimal("40")   # 40%
    }
)
```

### Trading Across Accounts

```python
# Place order across all accounts in group
results = await manager.place_group_order(
    group_id=group.id,
    symbol="BTCUSDT",
    side="buy",
    total_quantity=Decimal("1.0")
)
# Automatically places 0.6 on acc1, 0.4 on acc2
```

---

## Alerts & Notifications

### Telegram Alerts

Configure alert levels:

```yaml
notifications:
  telegram:
    trade_alerts: true           # Every trade
    signal_alerts: false         # Every signal (noisy)
    daily_summary: true          # Daily report at 00:00 UTC
    risk_warnings: true          # Risk limit warnings
    error_alerts: true           # System errors
```

### Slack Alerts

```yaml
notifications:
  slack:
    channel: "#trading-alerts"
    trade_alerts: true
    daily_summary: true
```

### Email Alerts (Coming Soon)

```yaml
notifications:
  email:
    recipients: ["your@email.com"]
    daily_digest: true
```

---

## Best Practices

### 1. Start Small

- Begin with paper trading
- Use minimum position sizes initially
- Trade only 1-2 symbols first

### 2. Risk Management First

- Never risk more than 1-2% per trade
- Set daily loss limits
- Use stop-losses on every trade
- Disable leverage until experienced

### 3. Monitor Regularly

- Check positions at least twice daily
- Review daily P&L summaries
- Watch for unusual patterns

### 4. Keep Learning

- Backtest new strategies before live trading
- Analyze losing trades for improvements
- Stay updated on market conditions

### 5. Security

- Use IP whitelist on exchange
- Disable withdrawal on API keys
- Use separate API keys for testing
- Enable 2FA on all accounts

---

## FAQ

### General

**Q: How much money do I need to start?**
A: We recommend starting with at least $1,000 to have meaningful position sizes while maintaining proper risk management.

**Q: Which exchange is best?**
A: Binance Futures has the highest liquidity and lowest fees. Start there unless you have specific requirements.

**Q: Can I run multiple strategies at once?**
A: Yes, the bot aggregates signals from all enabled strategies and uses configurable weights.

### Technical

**Q: Why is my bot not opening positions?**
A: Check:
- Risk limits (daily loss, max positions)
- Strategy parameters (thresholds may be too strict)
- Market conditions (low volatility)
- Logs for errors

**Q: How do I reset paper trading?**
A: Delete the database or use the reset command:
```bash
python -m src.main --paper --reset
```

**Q: Can I run the bot 24/7?**
A: Yes, it's designed for continuous operation. Use Docker or Kubernetes for production deployments.

### Strategy

**Q: Which strategy is most profitable?**
A: Depends on market conditions. Backtest all strategies on recent data to find what works best.

**Q: How do I create a custom strategy?**
A: Extend the `BaseStrategy` class:
```python
class MyStrategy(BaseStrategy):
    def on_candle(self, candle, history):
        # Your logic here
        return Signal(...)
```

### Troubleshooting

**Q: Bot keeps disconnecting from exchange**
A: Check:
- Network stability
- API rate limits
- Exchange status page

**Q: Orders not being filled**
A: Check:
- Sufficient balance
- Correct symbol format
- Exchange trading enabled

**Q: High slippage on trades**
A: Consider:
- Trading larger/more liquid pairs
- Using limit orders instead of market
- Reducing position sizes
