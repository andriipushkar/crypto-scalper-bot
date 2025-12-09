# Crypto Scalper Bot

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](tests/)

Production-ready automated scalping system for cryptocurrency futures trading with advanced features including ML-powered signals, sentiment analysis, multi-exchange arbitrage, and comprehensive backtesting.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Trading Strategies](#trading-strategies)
- [Integrations](#integrations)
- [Backtesting](#backtesting)
- [Paper Trading](#paper-trading)
- [Multi-Account Management](#multi-account-management)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [API Reference](#api-reference)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Development](#development)
- [Performance](#performance)
- [Security](#security)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Core Trading
- **Multi-Exchange Support**: Binance Futures, Bybit, OKX with unified interface
- **Low Latency Execution**: < 50ms signal-to-order pipeline
- **Smart Order Routing**: Automatic best-price routing across exchanges
- **Position Management**: Stop-loss, take-profit, trailing stops

### Advanced Strategies
- **Orderbook Imbalance**: Detects bid/ask imbalances for momentum trades
- **Volume Spike Detection**: Identifies abnormal volume patterns
- **Momentum Strategy**: RSI and price momentum-based signals
- **ML-Powered Signals**: Random Forest and LSTM neural networks
- **Sentiment Analysis**: Twitter, Reddit, Fear & Greed Index integration
- **Cross-Exchange Arbitrage**: Price discrepancy detection and execution
- **Triangular Arbitrage**: Intra-exchange path-finding algorithm
- **Smart DCA with AI/ML**: Intelligent dollar-cost averaging with machine learning
- **News/Event Trading**: Economic calendar, token unlocks, sentiment analysis

### Book-Based Strategies (from "Скальпинг: практическое руководство трейдера")
- **Range Trading**: False breakout detection, boundary trading
- **Session Trading**: Asia/Europe/US session-based trading, gap fill
- **Trendline Breakout**: Automatic pivot detection, trendline construction
- **Size Bounce/Breakout**: Wall reaction detection (bounce vs breakout)
- **Bid/Ask Flip Detection**: Large order manipulation detection
- **Impulse Scalping**: BTC/ETH correlation for altcoin trading
- **Order Flow Velocity**: Trade acceleration and momentum analysis
- **Fee Optimizer**: Maker/Taker fee optimization for scalping

### Signal Provider
- **Multi-tier Subscriptions**: FREE, BASIC, PREMIUM, VIP levels
- **Delayed Delivery**: Configurable signal delays per tier
- **Multi-channel Distribution**: Telegram, Email, Webhook, API
- **Performance Tracking**: Win rate, profit factor, Sharpe ratio

### Analytics & Alerts
- **Liquidation Heatmap**: Liquidation level analysis and clustering
- **Cascade Detection**: Multi-level liquidation cascade risk assessment
- **Advanced Alerts**: Price, volume, whale, on-chain, technical indicators
- **Multi-channel Notifications**: Telegram, Discord, Webhook, Email

### Risk Management
- **Position Sizing**: Kelly criterion and fixed-risk models
- **Daily Loss Limits**: Automatic trading halt on losses
- **Maximum Drawdown Protection**: Equity curve monitoring
- **VaR/CVaR Calculations**: Statistical risk metrics
- **Correlated Position Limits**: Multi-asset exposure control

### Backtesting & Simulation
- **Historical Backtesting**: Walk-forward optimization with Optuna
- **Paper Trading Mode**: Real-time simulation without capital risk
- **Performance Analytics**: Sharpe ratio, sortino, calmar, profit factor
- **Equity Curve Analysis**: Drawdown periods, recovery analysis

### Integrations
- **Telegram Bot**: Full trading control via mobile
- **Slack Notifications**: Rich message formatting with Block Kit
- **TradingView Webhooks**: Alert-based trade execution
- **Web Dashboard**: Real-time monitoring with WebSocket updates

### Multi-Account
- **Account Groups**: Manage multiple accounts as portfolios
- **Allocation Strategies**: Percentage-based fund distribution
- **Synchronized Execution**: Parallel order placement

### Infrastructure
- **Kubernetes Ready**: Helm charts, ArgoCD, autoscaling
- **Message Queue**: Kafka integration for event streaming
- **Terraform IaC**: AWS EKS, RDS, ElastiCache provisioning
- **Disaster Recovery**: Automated backups, multi-region failover

---

## Quick Start

```bash
# Clone repository
git clone <repo-url>
cd crypto-scalper-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run in paper trading mode (safe testing)
python -m src.main --paper

# Run backtesting
python scripts/backtest.py --symbol BTCUSDT --start 2024-01-01 --end 2024-06-01

# Start web dashboard
python -m src.dashboard.app
```

---

## Installation

### Requirements

- Python 3.10+
- Redis 6+
- PostgreSQL 14+
- Docker (optional)
- Kubernetes (production)

### Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Optional: ML features
pip install -r requirements-ml.txt

# Optional: Development tools
pip install -r requirements-dev.txt
```

### Docker Installation

```bash
# Build image
docker build -t crypto-scalper-bot .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f trading-bot
```

---

## Configuration

### Environment Variables

```bash
# ═══════════════════════════════════════════════════════════════════
# EXCHANGE API CREDENTIALS
# ═══════════════════════════════════════════════════════════════════
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
BINANCE_TESTNET=true                    # Use testnet for development

BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret

OKX_API_KEY=your_api_key
OKX_API_SECRET=your_api_secret
OKX_PASSPHRASE=your_passphrase

# ═══════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/trading

# ═══════════════════════════════════════════════════════════════════
# REDIS
# ═══════════════════════════════════════════════════════════════════
REDIS_URL=redis://localhost:6379/0

# ═══════════════════════════════════════════════════════════════════
# TELEGRAM BOT
# ═══════════════════════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_ALLOWED_USERS=123456,789012    # Comma-separated user IDs
TELEGRAM_ADMIN_USERS=123456             # Admin user IDs

# ═══════════════════════════════════════════════════════════════════
# SLACK
# ═══════════════════════════════════════════════════════════════════
SLACK_BOT_TOKEN=xoxb-your-token
SLACK_DEFAULT_CHANNEL=#trading-alerts

# ═══════════════════════════════════════════════════════════════════
# TRADINGVIEW
# ═══════════════════════════════════════════════════════════════════
TRADINGVIEW_SECRET_KEY=your_webhook_secret
TRADINGVIEW_ALLOWED_IPS=52.89.214.238,34.212.75.30

# ═══════════════════════════════════════════════════════════════════
# SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════
TWITTER_BEARER_TOKEN=your_twitter_token
REDDIT_CLIENT_ID=your_reddit_id
REDDIT_CLIENT_SECRET=your_reddit_secret

# ═══════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════
DASHBOARD_SECRET_KEY=your_secret_key
DASHBOARD_PORT=8000
JWT_SECRET_KEY=your_jwt_secret

# ═══════════════════════════════════════════════════════════════════
# MESSAGE QUEUE
# ═══════════════════════════════════════════════════════════════════
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

### Example Configurations

Ready-to-use configurations for different trading styles:

| Config | Description | Risk Level |
|--------|-------------|------------|
| `config-conservative.yaml` | Low risk, 3x leverage, single position | Low |
| `config-aggressive.yaml` | High risk, 20x leverage, all strategies | High |
| `config-multi-exchange.yaml` | Multi-exchange arbitrage setup | Medium |
| `config-ml-focused.yaml` | ML/AI-driven trading with sentiment | Medium |

```bash
# Use example configuration
cp examples/config-conservative.yaml config/settings.yaml

# Or specify directly
python -m src.main --config examples/config-aggressive.yaml --paper
```

### Trading Settings (config/settings.yaml)

```yaml
# ═══════════════════════════════════════════════════════════════════
# TRADING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
trading:
  symbols:
    - BTCUSDT
    - ETHUSDT
    - BNBUSDT
  leverage: 10
  paper_trading: true
  default_exchange: binance

# ═══════════════════════════════════════════════════════════════════
# RISK MANAGEMENT
# ═══════════════════════════════════════════════════════════════════
risk:
  max_position_size: 0.1          # Max 10% of balance per trade
  max_daily_loss: 100             # Stop trading after $100 loss
  max_drawdown: 0.1               # Max 10% drawdown
  risk_per_trade: 0.01            # Risk 1% per trade
  max_positions: 3                # Max concurrent positions
  position_timeout: 300           # Close positions after 5 min

# ═══════════════════════════════════════════════════════════════════
# STRATEGY SETTINGS
# ═══════════════════════════════════════════════════════════════════
strategies:
  orderbook_imbalance:
    enabled: true
    threshold: 1.5
    levels: 5

  volume_spike:
    enabled: true
    multiplier: 3.0
    lookback_seconds: 60

  momentum:
    enabled: true
    rsi_period: 14
    overbought: 70
    oversold: 30

  ml_signals:
    enabled: true
    model_path: models/rf_model.pkl
    confidence_threshold: 0.7

  sentiment:
    enabled: true
    sources:
      - twitter
      - reddit
      - fear_greed
    weight: 0.3

  arbitrage:
    enabled: true
    min_spread: 0.001            # Minimum 0.1% spread
    max_execution_time: 2.0      # Max 2 seconds
    exchanges:
      - binance
      - bybit
      - okx
```

---

## Usage

### Command Line Interface

```bash
# Start trading bot (paper mode)
python -m src.main --paper

# Start trading bot (live mode - use with caution!)
python -m src.main --live

# Specify exchange
python -m src.main --exchange binance --paper

# Custom config file
python -m src.main --config my_config.yaml --paper

# Run specific strategy only
python -m src.main --strategy momentum --paper

# Debug mode with verbose logging
python -m src.main --paper --debug
```

### Python API

```python
from src.core.engine import TradingEngine
from src.core.config import Config

# Initialize
config = Config.from_yaml("config/settings.yaml")
engine = TradingEngine(config)

# Start trading
await engine.start()

# Place manual order
order = await engine.place_order(
    symbol="BTCUSDT",
    side="buy",
    quantity=0.01,
    order_type="market"
)

# Get positions
positions = await engine.get_positions()

# Stop trading
await engine.stop()
```

---

## Trading Strategies

### 1. Orderbook Imbalance

Detects significant differences between bid and ask volumes to predict short-term price movements.

```python
from src.strategy.orderbook_imbalance import OrderBookImbalanceStrategy

strategy = OrderBookImbalanceStrategy(
    threshold=1.5,      # Imbalance ratio threshold
    levels=5,           # Order book depth levels
    min_volume=1000     # Minimum volume filter
)
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| threshold | 1.5 | Bid/Ask volume ratio to trigger signal |
| levels | 5 | Number of price levels to analyze |
| min_volume | 1000 | Minimum total volume to consider |

### 2. Volume Spike Detection

Identifies abnormal trading volume that may precede significant price movements.

```python
from src.strategy.volume_spike import VolumeSpikeStrategy

strategy = VolumeSpikeStrategy(
    multiplier=3.0,     # Volume spike multiplier
    lookback=60,        # Seconds to look back
    min_spike_size=100  # Minimum spike in USD
)
```

### 3. ML-Powered Signals

Machine learning models trained on historical price data and technical indicators.

```python
from src.strategies.ml_signals import MLSignalGenerator, ModelType

generator = MLSignalGenerator(
    model_type=ModelType.RANDOM_FOREST,
    lookback_periods=20,
    confidence_threshold=0.7
)

# Train model
await generator.train(historical_data)

# Generate signals
signal = await generator.generate_signal(current_data)
```

**Features Used:**
- RSI (14 periods)
- MACD (12, 26, 9)
- Bollinger Bands (20, 2)
- ATR (14 periods)
- Volume moving averages
- Price momentum

### 4. Sentiment Analysis

Aggregates sentiment from multiple sources to inform trading decisions.

```python
from src.strategies.sentiment import SentimentAggregator

aggregator = SentimentAggregator()
aggregator.add_source(TwitterSentimentFetcher(bearer_token))
aggregator.add_source(RedditSentimentFetcher(client_id, secret))
aggregator.add_source(FearGreedFetcher())

# Get aggregated sentiment
sentiment = await aggregator.get_aggregated_sentiment("BTCUSDT")
# Returns: {"score": 0.65, "label": "bullish", "confidence": 0.8}
```

### 5. Arbitrage Detection

Finds and executes arbitrage opportunities across exchanges.

```python
from src.strategies.arbitrage import ArbitrageDetector

detector = ArbitrageDetector(
    exchanges=["binance", "bybit", "okx"],
    min_spread=0.001,           # Minimum 0.1% spread
    max_execution_time=2.0      # Maximum 2 seconds
)

# Scan for opportunities
opportunities = await detector.scan_opportunities()

# Execute arbitrage
result = await detector.execute_arbitrage(opportunity)
```

**Arbitrage Types:**
- **Cross-Exchange**: Buy on exchange A, sell on exchange B
- **Triangular**: BTC → ETH → USDT → BTC cycle

---

## Integrations

### Telegram Bot

Full trading control from your mobile device.

**Commands:**

| Command | Description |
|---------|-------------|
| `/start` | Initialize bot and show menu |
| `/status` | Bot status and current P&L |
| `/balance` | Account balance across exchanges |
| `/positions` | Open positions with P&L |
| `/trades` | Recent trade history |
| `/buy <symbol> <quantity>` | Place buy order |
| `/sell <symbol> <quantity>` | Place sell order |
| `/close <symbol>` | Close specific position |
| `/closeall` | Close all positions |
| `/pnl` | Daily P&L summary |
| `/set_leverage <value>` | Set trading leverage |
| `/stop` | Stop trading bot |
| `/resume` | Resume trading |

**Setup:**

```python
from src.integrations.telegram_bot import TelegramTradingBot

bot = TelegramTradingBot(
    token=os.environ["TELEGRAM_BOT_TOKEN"],
    allowed_users=[123456, 789012],
    admin_users=[123456]
)

# Set trading callbacks
bot.set_callback("get_balance", trading_engine.get_balance)
bot.set_callback("get_positions", trading_engine.get_positions)
bot.set_callback("place_order", trading_engine.place_order)

# Start bot
await bot.start()
```

### Slack Integration

Rich notifications using Slack Block Kit.

**Notification Types:**
- Trade executed alerts
- Signal notifications with action buttons
- Daily/weekly reports
- Risk alerts (drawdown, exposure)
- System status updates

```python
from src.integrations.slack import SlackNotifier, SlackClient

client = SlackClient(
    bot_token=os.environ["SLACK_BOT_TOKEN"],
    default_channel="#trading-alerts"
)
notifier = SlackNotifier(client)

# Send trade alert
await notifier.send_trade_alert({
    "symbol": "BTCUSDT",
    "side": "long",
    "entry_price": 50000,
    "exit_price": 51000,
    "pnl": 100,
    "strategy": "momentum"
})

# Send daily report
await notifier.send_daily_report({
    "date": "2024-01-15",
    "pnl": 1500.50,
    "trades": 25,
    "win_rate": 68.0,
    "balance": 15000.0
})
```

### TradingView Webhooks

Execute trades based on TradingView alerts.

**Alert Payload Format:**
```json
{
    "ticker": "BTCUSDT",
    "exchange": "BINANCE",
    "action": "buy",
    "price": 50000,
    "quantity": 0.1,
    "strategy": "MACD Crossover"
}
```

```python
from src.integrations.tradingview import TradingViewWebhook

webhook = TradingViewWebhook(
    secret_key=os.environ["TRADINGVIEW_SECRET_KEY"],
    allowed_ips=["52.89.214.238", "34.212.75.30"]
)

@webhook.on_alert
def handle_alert(alert):
    if alert.action == AlertAction.BUY:
        trading_engine.place_order(
            symbol=alert.ticker,
            side="buy",
            quantity=alert.quantity
        )
```

---

## Backtesting

Comprehensive historical testing with detailed analytics.

### Basic Backtest

```python
from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.backtesting.data import HistoricalDataLoader
from src.backtesting.analysis import BacktestAnalyzer

# Load historical data
loader = HistoricalDataLoader(data_dir="data/historical")
data = await loader.fetch_data(
    symbol="BTCUSDT",
    interval="1h",
    start_date="2024-01-01",
    end_date="2024-06-01"
)

# Configure backtest
config = BacktestConfig(
    initial_balance=10000.0,
    commission_rate=0.001,      # 0.1% commission
    slippage=0.0005,            # 0.05% slippage
    leverage=10
)

# Create engine
engine = BacktestEngine(config)

# Define strategy
def my_strategy(engine, symbol, candle, history):
    if len(history) < 20:
        return

    # Simple MA crossover
    ma_fast = sum(c.close for c in history[-5:]) / 5
    ma_slow = sum(c.close for c in history[-20:]) / 20

    if symbol not in engine.positions:
        if ma_fast > ma_slow:
            engine.place_order(symbol, OrderSide.BUY, 0.1, OrderType.MARKET)
    else:
        if ma_fast < ma_slow:
            engine.place_order(symbol, OrderSide.SELL, 0.1, OrderType.MARKET)

# Run backtest
result = engine.run(my_strategy, {"BTCUSDT": data}, warmup_period=50)

# Analyze results
analyzer = BacktestAnalyzer(result)
print(analyzer.generate_report())
```

### Backtest Results

```
═══════════════════════════════════════════════════════════════════
                         BACKTEST REPORT
═══════════════════════════════════════════════════════════════════

PERIOD
  Start:           2024-01-01 00:00:00
  End:             2024-06-01 00:00:00
  Duration:        152 days

RETURNS
  Initial Balance: $10,000.00
  Final Balance:   $12,450.00
  Total Return:    $2,450.00 (24.50%)

TRADES
  Total Trades:    156
  Winning:         98 (62.82%)
  Losing:          58 (37.18%)
  Avg Win:         $45.20
  Avg Loss:        -$28.50

RISK METRICS
  Max Drawdown:    8.5%
  Sharpe Ratio:    1.85
  Sortino Ratio:   2.10
  Profit Factor:   1.65
  Calmar Ratio:    2.88
═══════════════════════════════════════════════════════════════════
```

### Walk-Forward Optimization

```bash
python scripts/optimize.py \
    --symbol BTCUSDT \
    --start 2023-01-01 \
    --end 2024-01-01 \
    --strategy momentum \
    --trials 100
```

---

## Paper Trading

Safe real-time simulation without risking capital.

### Features
- Real-time price feeds from exchanges
- Simulated order execution with slippage
- Commission calculation
- Margin and leverage support
- Liquidation simulation
- Full position management

### Usage

```python
from src.trading.paper_trading import PaperTradingEngine, OrderType

engine = PaperTradingEngine(
    initial_balance=Decimal("10000"),
    commission_rate=Decimal("0.001"),   # 0.1%
    slippage_rate=Decimal("0.0005"),    # 0.05%
    leverage=10
)

# Update prices (from real exchange feed)
engine.update_price("BTCUSDT", Decimal("50000"))

# Place orders
order = await engine.place_order(
    symbol="BTCUSDT",
    side="buy",
    quantity=Decimal("0.1"),
    order_type=OrderType.MARKET
)

# Check positions
positions = engine.positions
print(f"BTCUSDT position: {positions.get('BTCUSDT')}")

# Get account stats
stats = engine.get_stats()
print(f"Total P&L: {stats['total_pnl']}")
print(f"Win Rate: {stats['win_rate']}%")
```

---

## Multi-Account Management

Manage multiple trading accounts with coordinated execution.

### Account Groups

```python
from src.trading.multi_account import MultiAccountManager, AccountType

manager = MultiAccountManager()

# Add accounts
acc1 = manager.add_account(
    name="Main Binance",
    exchange="binance",
    api_key="key1",
    api_secret="secret1",
    account_type=AccountType.FUTURES
)

acc2 = manager.add_account(
    name="Secondary Bybit",
    exchange="bybit",
    api_key="key2",
    api_secret="secret2",
    account_type=AccountType.FUTURES
)

# Create account group with allocation
group = manager.create_group(
    name="Primary Portfolio",
    account_ids=[acc1.id, acc2.id],
    allocation={
        acc1.id: Decimal("60"),  # 60% allocation
        acc2.id: Decimal("40")   # 40% allocation
    }
)

# Connect all accounts
await manager.connect_all()

# Place order across group (automatically splits by allocation)
results = await manager.place_group_order(
    group_id=group.id,
    symbol="BTCUSDT",
    side="buy",
    total_quantity=Decimal("1.0")  # Will place 0.6 on acc1, 0.4 on acc2
)

# Get total balance across all accounts
total_usdt = manager.get_total_balance("USDT")

# Get total exposure per symbol
exposure = manager.get_total_exposure()
```

---

## Deployment

### Docker Compose (Development)

```bash
docker-compose up -d
```

### Kubernetes (Production)

```bash
# Using Kustomize
kubectl apply -k k8s/overlays/prod

# Using Helm
helm install trading-bot ./helm/trading-bot \
    --namespace trading \
    --values values-prod.yaml

# Using ArgoCD
kubectl apply -f argocd/application.yaml
```

### Terraform (AWS Infrastructure)

```bash
cd terraform/environments/prod
terraform init
terraform plan
terraform apply
```

**Provisioned Resources:**
- EKS Kubernetes cluster
- RDS PostgreSQL database
- ElastiCache Redis cluster
- VPC with private subnets
- Application Load Balancer

---

## Monitoring

### Prometheus Metrics

```
# Order metrics
trading_orders_total{exchange, symbol, side, status}
trading_order_latency_seconds{exchange}

# P&L metrics
trading_pnl_total{strategy}
trading_daily_pnl

# Position metrics
trading_positions_current{symbol}
trading_position_value{symbol}

# System metrics
trading_websocket_reconnects_total
trading_cache_hits_total
trading_cache_misses_total
trading_api_requests_total{exchange, endpoint}
```

### Grafana Dashboards

Pre-built dashboards are available in `grafana/dashboards/`:

| Dashboard | File | Panels | Description |
|-----------|------|--------|-------------|
| **Trading Overview** | `trading-overview.json` | 15+ | PnL, win rate, trades, signals |
| **Risk Management** | `risk-management.json` | 12+ | Drawdown, exposure, positions |
| **Exchange Health** | `exchange-health.json` | 14+ | Latency, errors, rate limits |
| **Strategy Performance** | `strategy-performance.json` | 16+ | Per-strategy metrics, ML, sentiment |

**Setup:**

```bash
# Copy provisioning configs
cp grafana/provisioning/*.yml /etc/grafana/provisioning/
cp grafana/dashboards/*.json /var/lib/grafana/dashboards/

# Or use Docker
docker-compose up -d grafana
```

**Features:**
- Auto-refresh (10-30 seconds)
- Template variables for filtering (symbol, exchange, strategy)
- Threshold alerts for drawdown, latency, errors
- Dashboard linking for navigation
- Prometheus datasource integration

### Dashboard Web UI

Access at `http://localhost:8000`:
- Real-time P&L and equity curve
- Open positions and orders
- Strategy performance metrics
- Risk indicators
- Trade journal

---

## API Reference

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/account` | GET | Account information |
| `/api/v1/balance` | GET | Balance details |
| `/api/v1/positions` | GET | Open positions |
| `/api/v1/orders` | GET | Order history |
| `/api/v1/orders` | POST | Place new order |
| `/api/v1/orders/{id}` | DELETE | Cancel order |
| `/api/v1/trades` | GET | Trade history |
| `/api/v1/bot/start` | POST | Start trading |
| `/api/v1/bot/stop` | POST | Stop trading |
| `/api/v1/bot/status` | GET | Bot status |

### WebSocket API

```javascript
// Connect to WebSocket
const ws = new WebSocket("ws://localhost:8000/ws/market/BTCUSDT");

// Receive real-time updates
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log("Price update:", data.price);
};
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_strategies.py -v

# Run async tests
pytest tests/ -v --asyncio-mode=auto

# Run E2E tests
./scripts/run_e2e_tests.sh

# Type checking
mypy src/

# Linting
ruff check src/
ruff format src/ --check
```

---

## CI/CD

### GitHub Actions Workflows

| Workflow | File | Trigger | Description |
|----------|------|---------|-------------|
| **CI** | `ci.yml` | Push, PR | Tests, linting, type checking |
| **CD** | `cd.yml` | Tag push | Build, deploy to staging/prod |
| **Security** | `security.yml` | Daily, PR | Vulnerability scanning |

### CI Pipeline

```yaml
# Runs on every push and PR
- Lint (Ruff)
- Type check (MyPy)
- Unit tests (pytest)
- Integration tests
- Coverage report
- Build Docker image
```

### CD Pipeline

```yaml
# Runs on version tags (v*)
- Build multi-arch Docker image (amd64, arm64)
- Push to container registry
- Deploy to staging (auto)
- Deploy to production (manual approval)
- Create GitHub Release
- Slack notifications
```

### Security Scanning

```yaml
# Daily + on PRs
- Dependency vulnerabilities (Safety, pip-audit)
- Code security (Bandit, CodeQL)
- Container scanning (Trivy)
- Secret detection (Gitleaks, TruffleHog)
```

---

## Development

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

**Configured Hooks:**

| Hook | Description |
|------|-------------|
| `ruff` | Fast Python linter |
| `ruff-format` | Code formatting |
| `mypy` | Type checking |
| `bandit` | Security linter |
| `check-yaml` | YAML syntax validation |
| `detect-secrets` | Prevent secret commits |
| `trailing-whitespace` | Clean whitespace |

### Code Quality

```bash
# Format code
ruff format src/

# Lint and auto-fix
ruff check src/ --fix

# Type check
mypy src/

# Security scan
bandit -r src/

# All checks
pre-commit run --all-files
```

### Project Structure

```
crypto-scalper-bot/
├── src/                    # Source code
│   ├── core/              # Core trading engine
│   ├── strategies/        # Trading strategies
│   ├── strategy/          # Advanced strategies (Smart DCA)
│   ├── signals/           # Signal provider system
│   ├── analytics/         # Analytics (Liquidation Heatmap)
│   ├── trading/           # Trading modules (News/Event)
│   ├── alerts/            # Advanced alert system
│   ├── integrations/      # Telegram, Slack, etc.
│   └── utils/             # Utilities
├── tests/                  # Test suite
├── config/                 # Configuration files
├── examples/               # Example configurations
├── grafana/                # Grafana dashboards
│   ├── dashboards/        # Dashboard JSON files
│   └── provisioning/      # Auto-provisioning configs
├── k8s/                    # Kubernetes manifests
├── terraform/              # Infrastructure as Code
├── .github/workflows/      # CI/CD pipelines
└── docs/                   # Documentation
```

---

## Performance

### Targets

| Metric | Minimum | Target |
|--------|---------|--------|
| Win Rate | > 55% | > 60% |
| Profit Factor | > 1.2 | > 1.5 |
| Max Drawdown | < 20% | < 10% |
| Sharpe Ratio | > 1.0 | > 1.5 |
| Order Latency | < 100ms | < 50ms |

### Optimization Tips

1. **Enable Redis caching** for frequently accessed data
2. **Use connection pooling** for database connections
3. **Deploy in same region** as exchange servers
4. **Use dedicated WebSocket** connections per symbol
5. **Enable JIT compilation** with PyPy for CPU-bound tasks

---

## Security

### Best Practices

1. **API Key Security**
   - Never commit API keys to version control
   - Use environment variables or secret management
   - Disable withdrawal permissions on API keys
   - Enable IP whitelist on exchanges

2. **Network Security**
   - Use TLS for all connections
   - Enable Kubernetes Network Policies
   - Run behind a firewall/VPN for admin access

3. **Application Security**
   - Validate all user inputs
   - Rate limit API endpoints
   - Use JWT for authentication
   - Enable audit logging

### Security Checklist

- [ ] API keys stored in environment variables
- [ ] Withdrawal disabled on exchange API keys
- [ ] IP whitelist configured on exchanges
- [ ] TLS enabled for all connections
- [ ] Secrets managed via Kubernetes Secrets
- [ ] RBAC configured for Kubernetes
- [ ] Network policies in place
- [ ] Regular security audits

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest tests/`)
5. Submit a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for public functions
- Maximum line length: 100 characters

---

## Risk Warning

⚠️ **IMPORTANT**: Trading cryptocurrencies involves substantial risk of loss. This software is provided for educational and research purposes only.

- Never trade with funds you cannot afford to lose
- Always start with paper trading mode
- Thoroughly backtest strategies before live trading
- Past performance does not guarantee future results

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Book-Based Strategies Documentation

For detailed documentation on strategies implemented from the book "Скальпинг: практическое руководство трейдера" (Scalping: A Practical Guide for Traders), see:

- **[docs/BOOK_STRATEGIES_UK.md](docs/BOOK_STRATEGIES_UK.md)** - Full documentation in Ukrainian

These strategies include professional scalping techniques adapted for the 24/7 crypto market:
- Range Trading with false breakout detection
- Session-based trading (adapted from "Morning Return")
- Trendline breakout with automatic pivot detection
- Size Bounce/Breakout wall trading
- Bid/Ask Flip manipulation detection
- Order Flow Velocity analysis
- Maker/Taker fee optimization

---

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/repo/discussions)
