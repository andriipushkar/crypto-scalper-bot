"""
Security middleware for FastAPI.

Provides:
- Authentication middleware (JWT + API Key)
- Rate limiting middleware
- Request logging
- Security headers
"""

import time
from typing import Optional, Callable, List
from functools import wraps

from fastapi import Request, Response, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from loguru import logger

from src.utils.security import (
    JWTAuth,
    APIKeyAuth,
    RateLimiter,
    redact_secrets,
)


# =============================================================================
# Security Instances
# =============================================================================

# Global instances (initialized on startup)
jwt_auth: Optional[JWTAuth] = None
api_key_auth: Optional[APIKeyAuth] = None
rate_limiter: Optional[RateLimiter] = None

# Security schemes for OpenAPI
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def init_security(
    enable_jwt: bool = True,
    enable_api_key: bool = True,
    rate_limit_rpm: int = 60,
):
    """Initialize security components."""
    global jwt_auth, api_key_auth, rate_limiter

    if enable_jwt:
        try:
            jwt_auth = JWTAuth()
            logger.info("JWT authentication enabled")
        except Exception as e:
            logger.warning(f"JWT auth disabled: {e}")

    if enable_api_key:
        api_key_auth = APIKeyAuth()
        logger.info("API key authentication enabled")

    rate_limiter = RateLimiter(
        requests_per_minute=rate_limit_rpm,
        burst_size=min(rate_limit_rpm // 6, 20),
    )
    logger.info(f"Rate limiting enabled: {rate_limit_rpm} req/min")


# =============================================================================
# Authentication Dependencies
# =============================================================================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    api_key: str = Depends(api_key_header),
) -> Optional[dict]:
    """
    Dependency to get current authenticated user.

    Supports both JWT Bearer tokens and API keys.
    """
    # Try JWT first
    if credentials and jwt_auth:
        payload = jwt_auth.verify_token(credentials.credentials)
        if payload:
            return {"type": "jwt", "data": payload}

    # Try API key
    if api_key and api_key_auth:
        key_data = api_key_auth.verify_key(api_key)
        if key_data:
            return {"type": "api_key", "data": key_data}

    return None


async def require_auth(
    user: dict = Depends(get_current_user),
) -> dict:
    """Dependency that requires authentication."""
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def require_admin(
    user: dict = Depends(require_auth),
) -> dict:
    """Dependency that requires admin permissions."""
    if user["type"] == "api_key":
        if "admin" not in user["data"].get("permissions", []):
            raise HTTPException(status_code=403, detail="Admin permission required")
    elif user["type"] == "jwt":
        if not user["data"].get("is_admin", False):
            raise HTTPException(status_code=403, detail="Admin permission required")

    return user


# =============================================================================
# Rate Limiting Middleware
# =============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.

    Limits requests per IP address.
    """

    def __init__(self, app, requests_per_minute: int = 60, exempt_paths: List[str] = None):
        super().__init__(app)
        self.limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            burst_size=min(requests_per_minute // 6, 20),
        )
        self.exempt_paths = exempt_paths or ["/health", "/docs", "/openapi.json"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        # Get client IP
        client_ip = self._get_client_ip(request)

        # Check rate limit
        if not self.limiter.is_allowed(client_ip):
            wait_time = self.limiter.get_wait_time(client_ip)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests",
                    "retry_after": round(wait_time, 1),
                },
                headers={"Retry-After": str(int(wait_time) + 1)},
            )

        return await call_next(request)

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP from request, handling proxies."""
        # Check X-Forwarded-For header (for reverse proxies)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Get first IP in chain
            return forwarded.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct connection IP
        if request.client:
            return request.client.host

        return "unknown"


# =============================================================================
# Security Headers Middleware
# =============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Only add HSTS in production (when not localhost)
        if request.url.hostname not in ["localhost", "127.0.0.1"]:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self' ws: wss:; "
            "font-src 'self';"
        )

        return response


# =============================================================================
# Request Logging Middleware
# =============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log all requests with timing and sanitized details.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()

        # Log request
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Log with timing (redact sensitive info)
        log_path = redact_secrets(path)
        logger.info(
            f"{client_ip} - {method} {log_path} - {response.status_code} ({duration_ms:.1f}ms)"
        )

        # Add timing header
        response.headers["X-Response-Time"] = f"{duration_ms:.1f}ms"

        return response


# =============================================================================
# Authentication Middleware
# =============================================================================

class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for protected routes.

    Checks for valid JWT or API key on protected endpoints.
    """

    def __init__(
        self,
        app,
        public_paths: List[str] = None,
        auth_required: bool = True,
    ):
        super().__init__(app)
        self.public_paths = public_paths or [
            "/",
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/ws",  # WebSocket has its own auth
        ]
        self.auth_required = auth_required

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip auth for public paths
        path = request.url.path
        if any(path.startswith(p) for p in self.public_paths):
            return await call_next(request)

        # Skip if auth not required
        if not self.auth_required:
            return await call_next(request)

        # Check authentication
        user = await self._authenticate(request)

        if not user:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Store user in request state
        request.state.user = user

        return await call_next(request)

    async def _authenticate(self, request: Request) -> Optional[dict]:
        """Authenticate request."""
        # Check Bearer token
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer ") and jwt_auth:
            token = auth_header[7:]
            payload = jwt_auth.verify_token(token)
            if payload:
                return {"type": "jwt", "data": payload}

        # Check API key
        api_key = request.headers.get("X-API-Key")
        if api_key and api_key_auth:
            key_data = api_key_auth.verify_key(api_key)
            if key_data:
                return {"type": "api_key", "data": key_data}

        return None


# =============================================================================
# Decorator for Route Protection
# =============================================================================

def require_permission(permission: str):
    """
    Decorator to require specific permission for a route.

    Usage:
        @app.get("/admin/config")
        @require_permission("admin")
        async def get_config(user: dict = Depends(require_auth)):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get user from kwargs or request state
            user = kwargs.get("user")
            request = kwargs.get("request")

            if not user and request:
                user = getattr(request.state, "user", None)

            if not user:
                raise HTTPException(status_code=401, detail="Authentication required")

            # Check permission
            has_permission = False
            if user["type"] == "api_key":
                has_permission = permission in user["data"].get("permissions", [])
            elif user["type"] == "jwt":
                has_permission = permission in user["data"].get("permissions", [])

            if not has_permission:
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission '{permission}' required"
                )

            return await func(*args, **kwargs)

        return wrapper
    return decorator
