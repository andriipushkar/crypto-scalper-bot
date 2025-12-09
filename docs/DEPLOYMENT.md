# Deployment Guide

Complete guide for deploying the Crypto Scalper Bot in various environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [AWS Infrastructure (Terraform)](#aws-infrastructure-terraform)
6. [CI/CD Pipelines](#cicd-pipelines)
7. [GitOps with ArgoCD](#gitops-with-argocd)
8. [Monitoring Setup](#monitoring-setup)
9. [Grafana Dashboards](#grafana-dashboards)
10. [Security Configuration](#security-configuration)
11. [Backup & Recovery](#backup--recovery)
12. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Storage | 20 GB SSD | 100+ GB SSD |
| Network | 100 Mbps | 1 Gbps |

### Software Requirements

- Python 3.10+
- Docker 20.10+
- Docker Compose 2.0+
- kubectl 1.25+
- Helm 3.10+
- Terraform 1.5+
- AWS CLI 2.0+ (for AWS deployment)

### Exchange API Keys

Before deployment, obtain API keys from:

1. **Binance Futures**
   - Create API at: https://www.binance.com/en/my/settings/api-management
   - Enable Futures trading
   - Disable withdrawal
   - Configure IP whitelist

2. **Bybit**
   - Create API at: https://www.bybit.com/app/user/api-management
   - Enable Contract trading
   - Disable withdrawal

3. **OKX**
   - Create API at: https://www.okx.com/account/my-api
   - Enable Trade permission
   - Disable withdrawal

---

## Local Development

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd crypto-scalper-bot
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### Step 3: Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt

# ML dependencies (optional)
pip install -r requirements-ml.txt
```

### Step 4: Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit with your settings
nano .env
```

**Minimal .env configuration:**

```bash
# Exchange (at least one required)
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
BINANCE_TESTNET=true

# Database (optional for local dev)
DATABASE_URL=sqlite:///./trading.db

# Redis (optional for local dev)
REDIS_URL=redis://localhost:6379/0
```

### Step 5: Start Services (Optional)

```bash
# Start Redis (if needed)
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Start PostgreSQL (if needed)
docker run -d --name postgres \
  -e POSTGRES_USER=trading \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=trading \
  -p 5432:5432 postgres:15-alpine
```

### Step 6: Run Application

```bash
# Paper trading mode (recommended for testing)
python -m src.main --paper

# With dashboard
python -m src.dashboard.app &
python -m src.main --paper

# Run tests
pytest tests/ -v
```

---

## Docker Deployment

### Step 1: Build Image

```bash
# Build production image
docker build -t crypto-scalper-bot:latest .

# Build with specific tag
docker build -t crypto-scalper-bot:v1.0.0 .
```

### Step 2: Docker Compose Setup

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  trading-bot:
    build: .
    image: crypto-scalper-bot:latest
    container_name: trading-bot
    restart: unless-stopped
    environment:
      - BINANCE_API_KEY=${BINANCE_API_KEY}
      - BINANCE_API_SECRET=${BINANCE_API_SECRET}
      - BINANCE_TESTNET=false
      - DATABASE_URL=postgresql+asyncpg://trading:password@postgres:5432/trading
      - REDIS_URL=redis://redis:6379/0
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - TELEGRAM_ALLOWED_USERS=${TELEGRAM_ALLOWED_USERS}
    depends_on:
      - postgres
      - redis
    volumes:
      - ./config:/app/config:ro
      - ./data:/app/data
      - ./logs:/app/logs
    networks:
      - trading-network

  dashboard:
    build: .
    image: crypto-scalper-bot:latest
    container_name: dashboard
    command: python -m src.dashboard.app
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://trading:password@postgres:5432/trading
      - REDIS_URL=redis://redis:6379/0
      - DASHBOARD_SECRET_KEY=${DASHBOARD_SECRET_KEY}
    depends_on:
      - postgres
      - redis
    networks:
      - trading-network

  postgres:
    image: postgres:15-alpine
    container_name: postgres
    restart: unless-stopped
    environment:
      - POSTGRES_USER=trading
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=trading
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - trading-network

  redis:
    image: redis:7-alpine
    container_name: redis
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - trading-network

volumes:
  postgres_data:
  redis_data:

networks:
  trading-network:
    driver: bridge
```

### Step 3: Start Services

```bash
# Create .env file with your secrets
cat > .env << EOF
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_ALLOWED_USERS=123456,789012
DASHBOARD_SECRET_KEY=$(openssl rand -hex 32)
EOF

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f trading-bot

# Stop services
docker-compose down
```

### Step 4: Health Checks

```bash
# Check container status
docker-compose ps

# Check trading bot logs
docker-compose logs trading-bot

# Check dashboard health
curl http://localhost:8000/health
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (EKS, GKE, AKS, or local)
- kubectl configured
- Helm 3 installed

### Step 1: Create Namespace

```bash
kubectl create namespace trading-bot
```

### Step 2: Create Secrets

```bash
# Create secret for exchange API keys
kubectl create secret generic exchange-secrets \
  --namespace trading-bot \
  --from-literal=BINANCE_API_KEY=your_key \
  --from-literal=BINANCE_API_SECRET=your_secret \
  --from-literal=BYBIT_API_KEY=your_key \
  --from-literal=BYBIT_API_SECRET=your_secret

# Create secret for Telegram
kubectl create secret generic telegram-secrets \
  --namespace trading-bot \
  --from-literal=TELEGRAM_BOT_TOKEN=your_token

# Create secret for database
kubectl create secret generic db-secrets \
  --namespace trading-bot \
  --from-literal=DATABASE_URL=postgresql+asyncpg://user:pass@host/db
```

### Step 3: Deploy with Kustomize

**Development:**

```bash
kubectl apply -k k8s/overlays/dev
```

**Production:**

```bash
kubectl apply -k k8s/overlays/prod
```

### Step 4: Deploy with Helm

```bash
# Add required repos
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Install dependencies
helm install redis bitnami/redis \
  --namespace trading-bot \
  --set auth.enabled=false

# Install trading bot
helm install trading-bot ./helm/trading-bot \
  --namespace trading-bot \
  --values helm/trading-bot/values-prod.yaml
```

### Step 5: Verify Deployment

```bash
# Check pods
kubectl get pods -n trading-bot

# Check services
kubectl get svc -n trading-bot

# View logs
kubectl logs -f deployment/trading-bot -n trading-bot

# Port forward for local access
kubectl port-forward svc/dashboard 8000:8000 -n trading-bot
```

### Step 6: Configure Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: trading-bot-ingress
  namespace: trading-bot
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - trading.yourdomain.com
      secretName: trading-tls
  rules:
    - host: trading.yourdomain.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: dashboard
                port:
                  number: 8000
```

---

## AWS Infrastructure (Terraform)

### Step 1: Configure AWS CLI

```bash
aws configure
# Enter your AWS credentials
```

### Step 2: Initialize Terraform

```bash
cd terraform/environments/prod

# Initialize
terraform init

# Review plan
terraform plan -var-file="terraform.tfvars"
```

### Step 3: Create terraform.tfvars

```hcl
# terraform.tfvars
region              = "us-east-1"
environment         = "prod"
cluster_name        = "trading-bot-prod"
vpc_cidr            = "10.0.0.0/16"

# RDS Configuration
db_instance_class   = "db.r6g.large"
db_username         = "trading"
db_name             = "trading"

# ElastiCache Configuration
redis_node_type     = "cache.r6g.large"
redis_num_nodes     = 3

# EKS Configuration
node_instance_types = ["c5.xlarge"]
node_min_size       = 2
node_max_size       = 10
node_desired_size   = 3

# Tags
tags = {
  Project     = "crypto-scalper-bot"
  Environment = "prod"
  ManagedBy   = "terraform"
}
```

### Step 4: Apply Infrastructure

```bash
# Apply infrastructure
terraform apply -var-file="terraform.tfvars"

# Save outputs
terraform output > outputs.txt
```

### Step 5: Configure kubectl

```bash
# Update kubeconfig
aws eks update-kubeconfig \
  --region us-east-1 \
  --name trading-bot-prod
```

### Step 6: Store Secrets in AWS Secrets Manager

```bash
# Create secrets
aws secretsmanager create-secret \
  --name trading-bot/exchange-keys \
  --secret-string '{
    "BINANCE_API_KEY": "your_key",
    "BINANCE_API_SECRET": "your_secret"
  }'

# Reference in Kubernetes using External Secrets Operator
```

---

## CI/CD Pipelines

### GitHub Actions Workflows

Проект використовує GitHub Actions для автоматизації CI/CD:

| Workflow | Файл | Тригер | Опис |
|----------|------|--------|------|
| CI | `.github/workflows/ci.yml` | Push, PR | Тести, лінтинг, type checking |
| CD | `.github/workflows/cd.yml` | Tag push | Збірка, деплой staging/prod |
| Security | `.github/workflows/security.yml` | Щоденно, PR | Сканування вразливостей |

### CI Pipeline

CI pipeline запускається при кожному push та PR:

```yaml
# Етапи CI pipeline
- Лінтинг (Ruff)
- Type checking (MyPy)
- Unit тести (pytest)
- Integration тести
- Coverage report
- Збірка Docker образу
```

**Налаштування secrets у GitHub:**

```
DOCKER_REGISTRY - Docker registry URL
DOCKER_USERNAME - Docker username
DOCKER_PASSWORD - Docker password
SLACK_WEBHOOK_URL - Slack notifications (опціонально)
```

### CD Pipeline

CD pipeline запускається при створенні тегу версії (v*):

```bash
# Створення релізу
git tag v1.0.0
git push origin v1.0.0
```

**Етапи деплою:**

1. **Staging (автоматично)**
   - Збірка multi-arch Docker образу (amd64, arm64)
   - Push до container registry
   - Деплой на staging кластер
   - Smoke tests

2. **Production (manual approval)**
   - Деплой на production кластер
   - Health checks
   - Slack notification
   - GitHub Release

### Security Scanning

Сканування безпеки виконується щоденно та при PR:

```yaml
# Типи сканування
- Dependency vulnerabilities (Safety, pip-audit)
- Code security (Bandit, CodeQL)
- Container scanning (Trivy)
- Secret detection (Gitleaks, TruffleHog)
```

**Налаштування:**

```bash
# Локальний запуск security scan
bandit -r src/ -f json -o bandit-report.json
safety check --full-report
trivy image crypto-scalper-bot:latest
```

### Pre-commit Hooks

Локальна перевірка перед комітом:

```bash
# Встановлення
pip install pre-commit
pre-commit install

# Ручний запуск
pre-commit run --all-files
```

**Налаштовані хуки:**

| Hook | Опис |
|------|------|
| `ruff` | Python linter |
| `ruff-format` | Форматування коду |
| `mypy` | Type checking |
| `bandit` | Security linter |
| `check-yaml` | Валідація YAML |
| `detect-secrets` | Виявлення секретів |

---

## GitOps with ArgoCD

### Step 1: Install ArgoCD

```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Get admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
```

### Step 2: Create Application

```yaml
# argocd/application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: trading-bot
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/crypto-scalper-bot.git
    targetRevision: main
    path: k8s/overlays/prod
  destination:
    server: https://kubernetes.default.svc
    namespace: trading-bot
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

### Step 3: Apply Application

```bash
kubectl apply -f argocd/application.yaml

# Access ArgoCD UI
kubectl port-forward svc/argocd-server -n argocd 8080:443
# Open https://localhost:8080
```

---

## Monitoring Setup

### Step 1: Install Prometheus Stack

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values monitoring/prometheus-values.yaml
```

### Step 2: Configure ServiceMonitor

```yaml
# servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: trading-bot
  namespace: trading-bot
spec:
  selector:
    matchLabels:
      app: trading-bot
  endpoints:
    - port: metrics
      interval: 15s
      path: /metrics
```

### Step 3: Import Grafana Dashboards

```bash
# Access Grafana
kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring

# Login: admin / prom-operator
```

---

## Grafana Dashboards

### Доступні дашборди

Готові дашборди знаходяться в `grafana/dashboards/`:

| Дашборд | Файл | Панелі | Опис |
|---------|------|--------|------|
| **Trading Overview** | `trading-overview.json` | 15+ | PnL, win rate, trades, signals |
| **Risk Management** | `risk-management.json` | 12+ | Drawdown, exposure, positions |
| **Exchange Health** | `exchange-health.json` | 14+ | Latency, errors, rate limits |
| **Strategy Performance** | `strategy-performance.json` | 16+ | Per-strategy metrics, ML, sentiment |

### Автоматичне provisioning

**Step 1: Скопіюйте конфігурацію provisioning:**

```bash
# Для Docker
cp grafana/provisioning/*.yml /etc/grafana/provisioning/
cp -r grafana/dashboards /var/lib/grafana/dashboards/

# Для Kubernetes (ConfigMap)
kubectl create configmap grafana-dashboards \
  --from-file=grafana/dashboards/ \
  -n monitoring
```

**Step 2: Налаштуйте datasources.yml:**

```yaml
# grafana/provisioning/datasources.yml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

**Step 3: Налаштуйте dashboards.yml:**

```yaml
# grafana/provisioning/dashboards.yml
apiVersion: 1
providers:
  - name: 'Crypto Scalper Dashboards'
    folder: 'Crypto Scalper'
    type: file
    options:
      path: /var/lib/grafana/dashboards
```

### Kubernetes Deployment

```yaml
# grafana-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: monitoring
spec:
  template:
    spec:
      containers:
        - name: grafana
          image: grafana/grafana:10.2.0
          volumeMounts:
            - name: dashboards
              mountPath: /var/lib/grafana/dashboards
            - name: provisioning
              mountPath: /etc/grafana/provisioning
      volumes:
        - name: dashboards
          configMap:
            name: grafana-dashboards
        - name: provisioning
          configMap:
            name: grafana-provisioning
```

### Функціонал дашбордів

**Trading Overview:**
- Total PnL (USD) з кольоровою індикацією
- Win Rate gauge
- Trades за 24h
- Current Drawdown
- Open Positions
- Cumulative PnL графік
- Trade Activity (LONG/SHORT)
- Signal Confidence

**Risk Management:**
- Drawdown gauge з порогами
- Daily Loss Limit usage
- Total Exposure gauge
- Position table з real-time P&L
- Drawdown History
- Exit Events (stop-loss, take-profit)
- Leverage tracking

**Exchange Health:**
- Connection status per exchange
- API Latency (ms)
- WebSocket Latency
- Rate Limit usage (%)
- API Errors by type
- Order Execution Time (P50, P95)
- Order Success Rate

**Strategy Performance:**
- Strategy summary table
- Cumulative PnL by strategy
- Win Rate comparison
- Signal generation rate
- ML Model Confidence
- ML Prediction Time
- Sentiment Score by source
- Fear & Greed Index gauge

### Template Variables

Кожен дашборд підтримує фільтрацію:

| Variable | Тип | Опис |
|----------|-----|------|
| `datasource` | Datasource | Prometheus instance |
| `symbol` | Query | BTCUSDT, ETHUSDT, etc. |
| `exchange` | Query | binance, bybit, okx |
| `strategy` | Query | momentum, ml_signals, etc. |

### Alerting

Дашборди підтримують threshold alerts:

```yaml
# Приклади порогів
- Drawdown > 10% → Warning
- Drawdown > 15% → Critical
- API Latency > 500ms → Warning
- Order Success Rate < 95% → Critical
```

---

### Configure Prometheus Alerts

```yaml
# alerting-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: trading-bot-alerts
  namespace: trading-bot
spec:
  groups:
    - name: trading-bot
      rules:
        - alert: TradingBotDown
          expr: up{job="trading-bot"} == 0
          for: 1m
          labels:
            severity: critical
          annotations:
            summary: "Trading bot is down"

        - alert: HighDailyLoss
          expr: trading_daily_pnl < -100
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "Daily loss exceeds $100"
```

---

## Security Configuration

### 1. Network Policies

```yaml
# networkpolicy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: trading-bot-policy
  namespace: trading-bot
spec:
  podSelector:
    matchLabels:
      app: trading-bot
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: dashboard
  egress:
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - port: 443  # Exchange APIs
        - port: 6379 # Redis
        - port: 5432 # PostgreSQL
```

### 2. Pod Security

```yaml
# pod-security.yaml
apiVersion: v1
kind: Pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
  containers:
    - name: trading-bot
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop:
            - ALL
```

### 3. Secrets Management

```bash
# Use External Secrets Operator for production
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets \
  --namespace external-secrets \
  --create-namespace
```

---

## Backup & Recovery

### Database Backup

```bash
# Automated backup script
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
pg_dump $DATABASE_URL | gzip > backup_${TIMESTAMP}.sql.gz
aws s3 cp backup_${TIMESTAMP}.sql.gz s3://your-bucket/backups/
```

### Configuration Backup

```bash
# Backup Kubernetes resources
kubectl get all -n trading-bot -o yaml > k8s-backup.yaml
kubectl get secrets -n trading-bot -o yaml > secrets-backup.yaml
```

### Recovery Procedure

1. **Restore Database:**
```bash
gunzip -c backup.sql.gz | psql $DATABASE_URL
```

2. **Restore Kubernetes:**
```bash
kubectl apply -f k8s-backup.yaml
```

3. **Verify Services:**
```bash
kubectl get pods -n trading-bot
curl http://dashboard:8000/health
```

---

## Troubleshooting

### Common Issues

#### Pod CrashLoopBackOff

```bash
# Check logs
kubectl logs -f pod/trading-bot-xxx -n trading-bot

# Common causes:
# - Missing environment variables
# - Invalid API keys
# - Database connection issues
```

#### WebSocket Disconnects

```bash
# Check network connectivity
kubectl exec -it trading-bot-xxx -- curl -v wss://fstream.binance.com

# Increase timeout settings in config
```

#### High Memory Usage

```bash
# Check memory usage
kubectl top pods -n trading-bot

# Solutions:
# - Reduce history length
# - Increase pod memory limits
# - Enable garbage collection tuning
```

### Useful Commands

```bash
# Pod debugging
kubectl describe pod trading-bot-xxx -n trading-bot
kubectl exec -it trading-bot-xxx -n trading-bot -- /bin/bash

# Log analysis
kubectl logs trading-bot-xxx -n trading-bot --since=1h | grep ERROR

# Resource usage
kubectl top nodes
kubectl top pods -n trading-bot

# Restart deployment
kubectl rollout restart deployment/trading-bot -n trading-bot

# Scale deployment
kubectl scale deployment/trading-bot --replicas=0 -n trading-bot
kubectl scale deployment/trading-bot --replicas=1 -n trading-bot
```

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Database connectivity
kubectl exec -it trading-bot-xxx -- python -c "from src.database import test_connection; test_connection()"

# Redis connectivity
kubectl exec -it trading-bot-xxx -- redis-cli -h redis ping
```
