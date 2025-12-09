# Disaster Recovery Plan

## Overview

This document outlines the disaster recovery (DR) procedures for the crypto trading bot system.

## Recovery Objectives

| Metric | Target | Maximum |
|--------|--------|---------|
| RTO (Recovery Time Objective) | 15 minutes | 1 hour |
| RPO (Recovery Point Objective) | 1 minute | 5 minutes |
| Data Loss Tolerance | Near zero | < 5 trades |

## Infrastructure Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PRIMARY REGION (ap-southeast-1)              │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │   EKS       │  │   RDS       │  │   Redis     │                 │
│  │  Cluster    │  │  Primary    │  │  Primary    │                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │
│         │                │                │                         │
│         │         ┌──────┴──────┐         │                         │
│         │         │  Streaming  │         │                         │
│         │         │ Replication │         │                         │
│         │         └──────┬──────┘         │                         │
└─────────┼────────────────┼────────────────┼─────────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DR REGION (ap-southeast-2)                   │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │   EKS       │  │   RDS       │  │   Redis     │                 │
│  │  Standby    │  │  Read       │  │  Standby    │                 │
│  │             │  │  Replica    │  │             │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Backup Strategy

### Database Backups

```yaml
# Automated backup configuration
backups:
  database:
    type: postgresql
    schedule: "0 */4 * * *"  # Every 4 hours
    retention: 30  # days
    storage: s3://trading-bot-backups/db/
    encryption: AES-256

  # Continuous WAL archiving
  wal_archiving:
    enabled: true
    destination: s3://trading-bot-backups/wal/
    retention: 7  # days
```

### Backup Script

```bash
#!/bin/bash
# backup.sh - Database backup script

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/tmp/backups"
S3_BUCKET="s3://trading-bot-backups"

# Create backup
pg_dump -h $DB_HOST -U $DB_USER -d trading_bot \
    --format=custom \
    --compress=9 \
    > "$BACKUP_DIR/trading_bot_$TIMESTAMP.dump"

# Upload to S3
aws s3 cp "$BACKUP_DIR/trading_bot_$TIMESTAMP.dump" \
    "$S3_BUCKET/db/trading_bot_$TIMESTAMP.dump" \
    --sse AES256

# Cleanup old local backups
find $BACKUP_DIR -name "*.dump" -mtime +1 -delete

# Verify backup
aws s3 ls "$S3_BUCKET/db/trading_bot_$TIMESTAMP.dump"

echo "Backup completed: trading_bot_$TIMESTAMP.dump"
```

### Redis Backups

```bash
#!/bin/bash
# redis-backup.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
S3_BUCKET="s3://trading-bot-backups/redis"

# Trigger RDB snapshot
redis-cli -h $REDIS_HOST BGSAVE

# Wait for completion
while [ $(redis-cli -h $REDIS_HOST LASTSAVE) == $(redis-cli -h $REDIS_HOST LASTSAVE) ]; do
    sleep 1
done

# Upload to S3
aws s3 cp /var/lib/redis/dump.rdb \
    "$S3_BUCKET/dump_$TIMESTAMP.rdb" \
    --sse AES256
```

## Disaster Scenarios

### Scenario 1: Primary Region Failure

**Detection:**
- CloudWatch alarms for region connectivity
- Route53 health checks fail
- Manual notification from AWS

**Recovery Steps:**

1. **Activate DR Region (0-5 minutes)**
   ```bash
   # Promote RDS read replica
   aws rds promote-read-replica \
       --db-instance-identifier trading-bot-dr \
       --region ap-southeast-2

   # Update DNS
   aws route53 change-resource-record-sets \
       --hosted-zone-id Z123456 \
       --change-batch file://failover-dns.json
   ```

2. **Scale DR EKS Cluster (5-10 minutes)**
   ```bash
   # Scale up DR cluster
   kubectl --context dr-cluster scale deployment trading-bot --replicas=3
   kubectl --context dr-cluster scale deployment dashboard --replicas=2
   ```

3. **Verify Services (10-15 minutes)**
   ```bash
   # Health checks
   curl https://api-dr.trading-bot.com/health
   curl https://dashboard-dr.trading-bot.com/health

   # Verify trading connectivity
   kubectl --context dr-cluster logs -l app=trading-bot --tail=100
   ```

4. **Resume Trading**
   - Verify all positions synced
   - Check pending orders
   - Enable new order placement

### Scenario 2: Database Corruption

**Detection:**
- Application errors
- Data integrity check failures
- Monitoring alerts

**Recovery Steps:**

1. **Stop Trading Immediately**
   ```bash
   kubectl scale deployment trading-bot --replicas=0
   ```

2. **Assess Damage**
   ```sql
   -- Check data integrity
   SELECT COUNT(*) FROM trades WHERE pnl IS NULL;
   SELECT COUNT(*) FROM positions WHERE quantity < 0;
   ```

3. **Restore from Backup**
   ```bash
   # Download latest backup
   aws s3 cp s3://trading-bot-backups/db/latest.dump /tmp/

   # Restore
   pg_restore -h $DB_HOST -U $DB_USER -d trading_bot_new \
       --clean --if-exists /tmp/latest.dump

   # Verify
   psql -h $DB_HOST -U $DB_USER -d trading_bot_new \
       -c "SELECT COUNT(*) FROM trades"
   ```

4. **Point-in-Time Recovery (if needed)**
   ```bash
   # Restore to specific time using WAL
   pg_restore --target-time="2024-01-15 10:30:00" ...
   ```

5. **Resume Operations**
   ```bash
   # Swap databases
   psql -c "ALTER DATABASE trading_bot RENAME TO trading_bot_old"
   psql -c "ALTER DATABASE trading_bot_new RENAME TO trading_bot"

   # Restart services
   kubectl scale deployment trading-bot --replicas=3
   ```

### Scenario 3: Exchange API Failure

**Detection:**
- Connection timeouts
- HTTP 5xx errors
- Rate limit exceeded

**Recovery Steps:**

1. **Automatic Fallback**
   ```python
   # The system automatically switches to backup exchange
   # See src/core/exchange_manager.py
   ```

2. **Manual Intervention**
   ```bash
   # Check exchange status
   curl https://api.binance.com/api/v3/ping

   # Force switch exchange
   kubectl exec -it trading-bot-xxx -- \
       python -c "from core import switch_exchange; switch_exchange('bybit')"
   ```

3. **Position Sync**
   ```bash
   # Sync positions from backup exchange
   kubectl exec -it trading-bot-xxx -- \
       python -c "from core import sync_positions; sync_positions()"
   ```

### Scenario 4: Kubernetes Cluster Failure

**Detection:**
- kubectl commands fail
- Pods not responding
- Control plane unavailable

**Recovery Steps:**

1. **Check Cluster Status**
   ```bash
   aws eks describe-cluster --name trading-bot-prod
   ```

2. **If Control Plane Down:**
   ```bash
   # AWS usually recovers automatically
   # Monitor: https://status.aws.amazon.com/
   ```

3. **If Node Group Down:**
   ```bash
   # Scale new node group
   eksctl scale nodegroup \
       --cluster trading-bot-prod \
       --name ng-fallback \
       --nodes 3
   ```

4. **Deploy to New Nodes:**
   ```bash
   # Force pod rescheduling
   kubectl delete pods -l app=trading-bot
   ```

## Runbooks

### Emergency Trading Halt

```bash
#!/bin/bash
# emergency-halt.sh

echo "EMERGENCY TRADING HALT INITIATED"

# 1. Stop all trading pods
kubectl scale deployment trading-bot --replicas=0

# 2. Cancel all open orders
kubectl run cancel-orders --rm -it --image=trading-bot:latest \
    -- python -c "from core import cancel_all_orders; cancel_all_orders()"

# 3. Close all positions (optional)
read -p "Close all positions? (yes/no): " confirm
if [ "$confirm" == "yes" ]; then
    kubectl run close-positions --rm -it --image=trading-bot:latest \
        -- python -c "from core import close_all_positions; close_all_positions()"
fi

# 4. Notify team
curl -X POST $SLACK_WEBHOOK \
    -H 'Content-type: application/json' \
    -d '{"text":"EMERGENCY: Trading halted!"}'

echo "Trading halted successfully"
```

### Resume Trading

```bash
#!/bin/bash
# resume-trading.sh

echo "Resuming trading operations..."

# 1. Health checks
kubectl run health-check --rm -it --image=trading-bot:latest \
    -- python -c "from core import health_check; health_check()"

# 2. Verify exchange connectivity
curl https://api.binance.com/api/v3/ping

# 3. Sync positions
kubectl run sync --rm -it --image=trading-bot:latest \
    -- python -c "from core import sync_all; sync_all()"

# 4. Scale up trading pods
kubectl scale deployment trading-bot --replicas=3

# 5. Monitor for 5 minutes
kubectl logs -f -l app=trading-bot --tail=100 &
sleep 300
kill %1

# 6. Verify trading active
kubectl exec -it $(kubectl get pod -l app=trading-bot -o jsonpath='{.items[0].metadata.name}') \
    -- python -c "from core import get_status; print(get_status())"

echo "Trading resumed successfully"
```

## Testing

### DR Drill Schedule

| Drill Type | Frequency | Duration | Participants |
|------------|-----------|----------|--------------|
| Backup Restore | Monthly | 2 hours | DevOps |
| Region Failover | Quarterly | 4 hours | DevOps, Trading |
| Full DR | Annually | 8 hours | All teams |

### Test Checklist

```markdown
## DR Test Checklist

### Pre-Test
- [ ] Notify stakeholders
- [ ] Ensure backups are current
- [ ] Document current state
- [ ] Prepare rollback plan

### During Test
- [ ] Execute failover
- [ ] Verify all services
- [ ] Test trading operations
- [ ] Measure RTO/RPO

### Post-Test
- [ ] Document findings
- [ ] Update procedures
- [ ] Schedule follow-up
- [ ] Update DR plan
```

## Communication

### Escalation Matrix

| Severity | Response Time | Notify |
|----------|---------------|--------|
| Critical | 5 minutes | On-call, CTO, CEO |
| High | 15 minutes | On-call, Team Lead |
| Medium | 1 hour | On-call |
| Low | 4 hours | Next business day |

### Notification Channels

```yaml
notifications:
  critical:
    - pagerduty
    - phone
    - sms
  high:
    - slack: "#trading-alerts"
    - email: ops@trading-bot.com
  medium:
    - slack: "#trading-ops"
```

## Recovery Verification

### Checklist After Recovery

```markdown
## Post-Recovery Verification

### Infrastructure
- [ ] All pods running
- [ ] Database accessible
- [ ] Redis connected
- [ ] Exchange API responsive

### Data Integrity
- [ ] Position count matches
- [ ] Trade history complete
- [ ] Account balance correct
- [ ] Open orders synced

### Trading Operations
- [ ] Can place test order
- [ ] Signals generating
- [ ] Risk limits enforced
- [ ] Notifications working

### Monitoring
- [ ] Metrics flowing
- [ ] Alerts configured
- [ ] Logs streaming
- [ ] Dashboards updated
```

## Appendix

### Key Contacts

| Role | Name | Phone | Email |
|------|------|-------|-------|
| On-call Primary | - | - | oncall@trading-bot.com |
| On-call Secondary | - | - | oncall-backup@trading-bot.com |
| AWS Support | - | - | Enterprise Support |

### External Resources

- AWS Status: https://status.aws.amazon.com/
- Binance Status: https://www.binance.com/en/system-status
- Bybit Status: https://status.bybit.com/
