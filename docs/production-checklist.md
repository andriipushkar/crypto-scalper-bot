# Production Deployment Checklist

## Pre-Deployment

### Code Quality
- [ ] All tests passing (`pytest --cov=src tests/`)
- [ ] Code coverage > 80%
- [ ] No critical linting errors (`flake8 src/`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Security audit clean (`python security/audit.py`)

### Configuration
- [ ] Environment variables documented
- [ ] Secrets stored in AWS Secrets Manager
- [ ] Config files reviewed for production values
- [ ] Feature flags configured
- [ ] Logging level set to INFO (not DEBUG)

### Infrastructure
- [ ] Terraform plan reviewed (`terraform plan`)
- [ ] No destructive changes in plan
- [ ] Resource limits appropriate
- [ ] Auto-scaling configured
- [ ] Multi-AZ enabled

## Exchange Setup

### API Configuration
- [ ] Production API keys created
- [ ] Keys have minimal required permissions
- [ ] IP whitelist configured
- [ ] API rate limits understood
- [ ] Withdrawal disabled on trading keys

### Trading Limits
- [ ] Maximum position size set
- [ ] Maximum daily loss limit configured
- [ ] Maximum leverage configured
- [ ] Allowed symbols whitelisted
- [ ] Emergency stop parameters set

```yaml
# Example limits configuration
trading:
  max_position_size_usd: 10000
  max_daily_loss_usd: 1000
  max_daily_loss_pct: 5
  max_leverage: 10
  max_open_positions: 5
  allowed_symbols:
    - BTCUSDT
    - ETHUSDT
```

## Database

### PostgreSQL
- [ ] Production instance sized correctly
- [ ] Multi-AZ replication enabled
- [ ] Automated backups configured
- [ ] Point-in-time recovery enabled
- [ ] Connection pooling configured
- [ ] SSL/TLS enforced
- [ ] Slow query logging enabled

### Migrations
- [ ] All migrations tested in staging
- [ ] Rollback scripts prepared
- [ ] Backup taken before migration
- [ ] Migration run in maintenance window

```bash
# Pre-migration backup
pg_dump -h $DB_HOST -U $DB_USER trading_bot > backup.sql

# Run migrations
alembic upgrade head

# Verify
alembic current
```

## Redis

### Configuration
- [ ] Production instance sized correctly
- [ ] Persistence enabled (RDB + AOF)
- [ ] Replication configured
- [ ] Memory limit set
- [ ] Eviction policy configured

### Data
- [ ] Cache TTLs appropriate
- [ ] Critical data has backups
- [ ] Connection pool sized

## Kubernetes

### Cluster
- [ ] Production cluster created
- [ ] Node groups sized correctly
- [ ] Cluster autoscaler enabled
- [ ] Network policies applied
- [ ] Pod security policies enabled

### Deployments
- [ ] Resource requests/limits set
- [ ] Liveness probes configured
- [ ] Readiness probes configured
- [ ] Pod disruption budgets set
- [ ] Anti-affinity rules applied

```yaml
# Verify deployment health
kubectl get deployments -n trading
kubectl describe deployment trading-bot -n trading
```

### Secrets
- [ ] All secrets in Kubernetes Secrets
- [ ] Secrets encrypted at rest
- [ ] RBAC configured for secrets
- [ ] Service accounts minimal permissions

## Monitoring

### Prometheus
- [ ] Metrics endpoint exposed
- [ ] ServiceMonitor configured
- [ ] Recording rules created
- [ ] Retention period set

### Alerting
- [ ] Critical alerts configured
- [ ] On-call rotation set up
- [ ] Escalation policies defined
- [ ] Alert runbooks created

```yaml
# Critical alerts checklist
alerts:
  - trading_bot_down
  - high_error_rate
  - position_limit_exceeded
  - daily_loss_limit_warning
  - exchange_connectivity_lost
  - database_connection_failed
  - high_latency
```

### Dashboards
- [ ] Trading dashboard created
- [ ] System metrics dashboard
- [ ] Business metrics dashboard
- [ ] On-call dashboard

### Logging
- [ ] Centralized logging configured
- [ ] Log retention policy set
- [ ] Sensitive data filtered
- [ ] Log levels appropriate

## Security

### Network
- [ ] VPC configured correctly
- [ ] Security groups minimal
- [ ] Network ACLs reviewed
- [ ] No public database access
- [ ] WAF configured (if applicable)

### Authentication
- [ ] API authentication enabled
- [ ] JWT tokens configured
- [ ] Token expiration set
- [ ] Rate limiting enabled

### Encryption
- [ ] TLS 1.2+ enforced
- [ ] Certificates valid
- [ ] Data encrypted at rest
- [ ] Secrets encrypted

## Disaster Recovery

### Backups
- [ ] Database backups automated
- [ ] Backup verification tested
- [ ] Backup retention configured
- [ ] Off-site backup storage

### Failover
- [ ] DR region configured
- [ ] Failover tested
- [ ] DNS failover ready
- [ ] Runbooks documented

## Go-Live

### Final Checks
- [ ] Staging environment validated
- [ ] Load testing completed
- [ ] Security scan passed
- [ ] Change approval obtained
- [ ] Rollback plan documented

### Deployment Steps

```bash
# 1. Take final staging snapshot
kubectl get all -n trading > pre-deploy-state.txt

# 2. Deploy to production
kubectl apply -k k8s/overlays/prod/

# 3. Verify deployment
kubectl rollout status deployment/trading-bot -n trading

# 4. Health check
curl https://api.trading-bot.com/health

# 5. Smoke test
./scripts/smoke-test.sh

# 6. Monitor for 30 minutes
watch kubectl get pods -n trading
```

### Post-Deployment
- [ ] All pods healthy
- [ ] No error spikes in logs
- [ ] Metrics flowing
- [ ] Alerts not firing
- [ ] Trading operations normal

## Operations

### Daily Checks
- [ ] Review overnight trades
- [ ] Check P&L against limits
- [ ] Verify position accuracy
- [ ] Review error logs
- [ ] Check system metrics

### Weekly Tasks
- [ ] Review trading performance
- [ ] Update dependencies if needed
- [ ] Review security alerts
- [ ] Check backup integrity
- [ ] Capacity planning

### Monthly Tasks
- [ ] DR drill
- [ ] Security audit
- [ ] Performance review
- [ ] Cost optimization
- [ ] Documentation update

## Emergency Procedures

### Trading Halt
```bash
# Immediate halt
kubectl scale deployment trading-bot --replicas=0 -n trading

# Cancel all orders
kubectl exec -it trading-bot-xxx -n trading -- \
    python -c "from core import cancel_all; cancel_all()"
```

### Rollback
```bash
# Rollback to previous version
kubectl rollout undo deployment/trading-bot -n trading

# Verify
kubectl rollout status deployment/trading-bot -n trading
```

### Incident Response
1. Acknowledge alert
2. Assess impact
3. Halt trading if needed
4. Investigate root cause
5. Implement fix
6. Resume operations
7. Post-mortem

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Development Lead | | | |
| DevOps Lead | | | |
| Security | | | |
| Trading Operations | | | |
| Management | | | |

---

**Deployment Date:** _______________

**Version:** _______________

**Notes:**
