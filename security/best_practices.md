# Security Best Practices

## Secrets Management

### DO
- Use environment variables for all secrets
- Use AWS Secrets Manager or HashiCorp Vault in production
- Rotate API keys regularly (every 90 days)
- Use separate API keys for development and production
- Enable IP whitelisting on exchange APIs

### DON'T
- Hardcode secrets in source code
- Commit .env files to version control
- Share API keys between team members
- Use the same credentials across environments

## API Security

### Authentication
```python
# Always use secure authentication
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if not secrets.compare_digest(api_key, settings.API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
```

### Rate Limiting
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/api/orders")
@limiter.limit("100/minute")
async def get_orders(request: Request):
    ...
```

### Input Validation
```python
from pydantic import BaseModel, Field, validator

class OrderRequest(BaseModel):
    symbol: str = Field(..., regex=r'^[A-Z]{2,10}USDT$')
    quantity: Decimal = Field(..., gt=0, le=1000000)

    @validator('symbol')
    def validate_symbol(cls, v):
        if v not in ALLOWED_SYMBOLS:
            raise ValueError('Invalid symbol')
        return v
```

## Database Security

### Parameterized Queries
```python
# GOOD - Parameterized query
cursor.execute(
    "SELECT * FROM trades WHERE symbol = %s AND user_id = %s",
    (symbol, user_id)
)

# BAD - String formatting (SQL injection risk)
cursor.execute(f"SELECT * FROM trades WHERE symbol = '{symbol}'")
```

### Connection Security
```python
# Use SSL for database connections
DATABASE_URL = "postgresql://user:pass@host:5432/db?sslmode=require"
```

## Network Security

### TLS/SSL
- Use TLS 1.2+ for all connections
- Pin certificates for exchange APIs
- Validate SSL certificates (never set verify=False)

### Firewall Rules
- Restrict inbound traffic to necessary ports only
- Use VPC for cloud deployments
- Implement network policies in Kubernetes

## Kubernetes Security

### Pod Security
```yaml
apiVersion: v1
kind: Pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
  containers:
  - name: app
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
          - ALL
```

### Network Policies
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: trading-bot-policy
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
          app: api-gateway
  egress:
  - to:
    - ipBlock:
        cidr: 0.0.0.0/0
    ports:
    - protocol: TCP
      port: 443
```

## Logging and Monitoring

### Sensitive Data
```python
# Never log sensitive data
logger.info(f"Order placed: {order.id}")  # GOOD
logger.info(f"API Key: {api_key}")  # BAD

# Mask sensitive fields
def mask_sensitive(data: dict) -> dict:
    sensitive_keys = {'api_key', 'secret', 'password', 'token'}
    return {
        k: '***' if k.lower() in sensitive_keys else v
        for k, v in data.items()
    }
```

### Audit Logging
```python
# Log security-relevant events
logger.info(
    "security_event",
    event_type="login_attempt",
    user_id=user.id,
    ip_address=request.client.host,
    success=True,
    timestamp=datetime.utcnow().isoformat(),
)
```

## Dependency Management

### Regular Updates
```bash
# Check for vulnerabilities
pip install safety
safety check

# Update dependencies
pip install pip-tools
pip-compile --upgrade requirements.in
```

### Lock Files
- Always use requirements.txt with pinned versions
- Use pip-tools or poetry for dependency management
- Review dependency updates before applying

## Incident Response

### Compromised API Keys
1. Immediately revoke the compromised key on the exchange
2. Generate new API keys
3. Update all deployments with new keys
4. Review trading history for unauthorized activity
5. Enable additional security (IP whitelist, withdrawal limits)

### Suspicious Activity
1. Pause trading immediately
2. Review recent trades and orders
3. Check for unauthorized access in logs
4. Rotate all credentials
5. Conduct full security audit

## Checklist

### Development
- [ ] No hardcoded secrets
- [ ] Input validation on all endpoints
- [ ] SQL injection prevention
- [ ] Rate limiting implemented
- [ ] Logging excludes sensitive data

### Deployment
- [ ] TLS enabled for all connections
- [ ] Network policies configured
- [ ] Pod security contexts set
- [ ] Secrets stored securely
- [ ] IP whitelisting on exchange APIs

### Operations
- [ ] Regular dependency updates
- [ ] Security audit scheduled
- [ ] Monitoring alerts configured
- [ ] Incident response plan documented
- [ ] Backup and recovery tested
