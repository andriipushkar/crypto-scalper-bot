"""
Security utilities for the trading bot.

Provides:
- API key encryption/decryption
- JWT authentication
- Password hashing
- Secrets management
- Input validation
"""

import base64
import hashlib
import hmac
import os
import re
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Optional, Dict, Any, List, Callable

from loguru import logger

# Optional cryptography imports
try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography not installed. Some security features disabled.")

# Optional JWT imports
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("PyJWT not installed. JWT authentication disabled.")


# =============================================================================
# Constants
# =============================================================================

# Patterns for detecting secrets in logs
SECRET_PATTERNS = [
    re.compile(r'api[_-]?key["\']?\s*[:=]\s*["\']?[\w-]{20,}', re.IGNORECASE),
    re.compile(r'api[_-]?secret["\']?\s*[:=]\s*["\']?[\w-]{20,}', re.IGNORECASE),
    re.compile(r'password["\']?\s*[:=]\s*["\']?[^\s"\']{8,}', re.IGNORECASE),
    re.compile(r'token["\']?\s*[:=]\s*["\']?[\w-]{20,}', re.IGNORECASE),
    re.compile(r'Bearer\s+[\w-]+\.[\w-]+\.[\w-]+', re.IGNORECASE),  # JWT
    re.compile(r'["\']?[\w]*secret[\w]*["\']?\s*[:=]\s*["\']?[^\s"\']{8,}', re.IGNORECASE),
]

# Characters to use for generating secure tokens
TOKEN_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


# =============================================================================
# Encryption
# =============================================================================

class SecretEncryptor:
    """
    Encrypt and decrypt sensitive data using Fernet (AES-128-CBC).

    Uses PBKDF2 to derive encryption key from a master password.
    """

    def __init__(self, master_password: str = None):
        """
        Initialize encryptor.

        Args:
            master_password: Master password for key derivation.
                           If not provided, uses MASTER_PASSWORD env var.
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography package required for encryption")

        self._master_password = master_password or os.getenv("MASTER_PASSWORD")
        if not self._master_password:
            raise ValueError(
                "Master password required. Set MASTER_PASSWORD env var or pass to constructor."
            )

        self._fernet = self._create_fernet()

    def _create_fernet(self) -> 'Fernet':
        """Create Fernet instance from master password."""
        # Use a fixed salt (in production, store this securely)
        salt = os.getenv("ENCRYPTION_SALT", "trading_bot_salt_v1").encode()

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )

        key = base64.urlsafe_b64encode(
            kdf.derive(self._master_password.encode())
        )

        return Fernet(key)

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a string.

        Args:
            plaintext: String to encrypt

        Returns:
            Base64-encoded encrypted string
        """
        encrypted = self._fernet.encrypt(plaintext.encode())
        return encrypted.decode()

    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt a string.

        Args:
            ciphertext: Base64-encoded encrypted string

        Returns:
            Decrypted string

        Raises:
            InvalidToken: If decryption fails
        """
        try:
            decrypted = self._fernet.decrypt(ciphertext.encode())
            return decrypted.decode()
        except InvalidToken:
            raise ValueError("Failed to decrypt: invalid token or wrong password")

    def encrypt_dict(self, data: Dict[str, str]) -> Dict[str, str]:
        """Encrypt all values in a dictionary."""
        return {k: self.encrypt(v) for k, v in data.items()}

    def decrypt_dict(self, data: Dict[str, str]) -> Dict[str, str]:
        """Decrypt all values in a dictionary."""
        return {k: self.decrypt(v) for k, v in data.items()}


def generate_encryption_key() -> str:
    """Generate a new Fernet encryption key."""
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography package required")
    return Fernet.generate_key().decode()


# =============================================================================
# Password Hashing
# =============================================================================

def hash_password(password: str, salt: bytes = None) -> tuple:
    """
    Hash a password using PBKDF2-SHA256.

    Args:
        password: Password to hash
        salt: Optional salt (generated if not provided)

    Returns:
        Tuple of (hash_hex, salt_hex)
    """
    if salt is None:
        salt = os.urandom(32)

    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode(),
        salt,
        100000  # iterations
    )

    return key.hex(), salt.hex()


def verify_password(password: str, hash_hex: str, salt_hex: str) -> bool:
    """
    Verify a password against its hash.

    Args:
        password: Password to verify
        hash_hex: Stored hash (hex)
        salt_hex: Stored salt (hex)

    Returns:
        True if password matches
    """
    salt = bytes.fromhex(salt_hex)
    computed_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(computed_hash, hash_hex)


# =============================================================================
# JWT Authentication
# =============================================================================

@dataclass
class JWTConfig:
    """JWT configuration."""
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7


class JWTAuth:
    """
    JWT authentication handler.

    Usage:
        auth = JWTAuth(JWTConfig(secret_key="your-secret"))
        token = auth.create_access_token({"user_id": "admin"})
        payload = auth.verify_token(token)
    """

    def __init__(self, config: JWTConfig = None):
        """Initialize JWT auth."""
        if not JWT_AVAILABLE:
            raise RuntimeError("PyJWT package required for JWT authentication")

        if config is None:
            secret = os.getenv("JWT_SECRET_KEY")
            if not secret:
                # Generate a random secret if not provided
                secret = secrets.token_urlsafe(32)
                logger.warning("JWT_SECRET_KEY not set, using random secret (tokens won't persist)")
            config = JWTConfig(secret_key=secret)

        self.config = config

    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: timedelta = None,
    ) -> str:
        """
        Create a new access token.

        Args:
            data: Payload data to encode
            expires_delta: Custom expiration time

        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                minutes=self.config.access_token_expire_minutes
            )

        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access",
        })

        return jwt.encode(
            to_encode,
            self.config.secret_key,
            algorithm=self.config.algorithm,
        )

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create a refresh token with longer expiration."""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(
            days=self.config.refresh_token_expire_days
        )

        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "refresh",
        })

        return jwt.encode(
            to_encode,
            self.config.secret_key,
            algorithm=self.config.algorithm,
        )

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token to verify

        Returns:
            Decoded payload or None if invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.debug("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.debug(f"Invalid token: {e}")
            return None

    def get_token_from_header(self, authorization: str) -> Optional[str]:
        """Extract token from Authorization header."""
        if not authorization:
            return None

        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None

        return parts[1]


# =============================================================================
# API Key Authentication
# =============================================================================

class APIKeyAuth:
    """
    Simple API key authentication.

    Usage:
        auth = APIKeyAuth()
        key = auth.generate_key()
        auth.add_key(key, "admin")

        if auth.verify_key(key):
            print("Valid key")
    """

    def __init__(self):
        self._keys: Dict[str, Dict[str, Any]] = {}
        self._load_keys()

    def _load_keys(self) -> None:
        """Load API keys from environment."""
        # Load dashboard API key from env
        dashboard_key = os.getenv("DASHBOARD_API_KEY")
        if dashboard_key:
            self._keys[dashboard_key] = {
                "name": "dashboard",
                "created": datetime.now(timezone.utc).isoformat(),
                "permissions": ["read", "write", "admin"],
            }

    def generate_key(self, length: int = 32) -> str:
        """Generate a new API key."""
        return secrets.token_urlsafe(length)

    def add_key(
        self,
        key: str,
        name: str,
        permissions: List[str] = None,
    ) -> None:
        """Add an API key."""
        self._keys[key] = {
            "name": name,
            "created": datetime.now(timezone.utc).isoformat(),
            "permissions": permissions or ["read"],
        }

    def remove_key(self, key: str) -> bool:
        """Remove an API key."""
        if key in self._keys:
            del self._keys[key]
            return True
        return False

    def verify_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Verify an API key and return its metadata."""
        return self._keys.get(key)

    def has_permission(self, key: str, permission: str) -> bool:
        """Check if a key has a specific permission."""
        key_data = self._keys.get(key)
        if not key_data:
            return False
        return permission in key_data.get("permissions", [])


# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for API endpoints.

    Usage:
        limiter = RateLimiter(requests_per_minute=60)

        if limiter.is_allowed("user_123"):
            # Process request
        else:
            # Return 429 Too Many Requests
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
    ):
        self.rate = requests_per_minute / 60.0  # tokens per second
        self.burst_size = burst_size
        self._buckets: Dict[str, Dict[str, float]] = {}

    def is_allowed(self, key: str) -> bool:
        """
        Check if a request is allowed for the given key.

        Args:
            key: Identifier (e.g., IP address, user ID)

        Returns:
            True if request is allowed
        """
        now = time.time()

        if key not in self._buckets:
            self._buckets[key] = {
                "tokens": self.burst_size,
                "last_update": now,
            }

        bucket = self._buckets[key]

        # Add tokens based on elapsed time
        elapsed = now - bucket["last_update"]
        bucket["tokens"] = min(
            self.burst_size,
            bucket["tokens"] + elapsed * self.rate
        )
        bucket["last_update"] = now

        # Check if we have tokens
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True

        return False

    def get_wait_time(self, key: str) -> float:
        """Get seconds to wait before next request is allowed."""
        if key not in self._buckets:
            return 0.0

        bucket = self._buckets[key]
        if bucket["tokens"] >= 1:
            return 0.0

        needed = 1 - bucket["tokens"]
        return needed / self.rate

    def reset(self, key: str = None) -> None:
        """Reset rate limit for a key or all keys."""
        if key:
            self._buckets.pop(key, None)
        else:
            self._buckets.clear()

    def cleanup(self, max_age_seconds: int = 3600) -> int:
        """Remove old entries to prevent memory leak."""
        now = time.time()
        old_keys = [
            k for k, v in self._buckets.items()
            if now - v["last_update"] > max_age_seconds
        ]
        for k in old_keys:
            del self._buckets[k]
        return len(old_keys)


# =============================================================================
# Secure Logging
# =============================================================================

class SecretFilter:
    """
    Filter to redact secrets from log messages.

    Usage:
        logger.add(sink, filter=SecretFilter())
    """

    def __init__(self, patterns: List[re.Pattern] = None):
        self.patterns = patterns or SECRET_PATTERNS

    def __call__(self, record: Dict) -> bool:
        """Filter log record, redacting secrets."""
        message = record.get("message", "")

        for pattern in self.patterns:
            message = pattern.sub("[REDACTED]", message)

        record["message"] = message
        return True


def redact_secrets(text: str) -> str:
    """Redact secrets from a string."""
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def mask_string(s: str, visible_chars: int = 4) -> str:
    """
    Mask a string, showing only first/last few characters.

    Args:
        s: String to mask
        visible_chars: Number of chars to show at start/end

    Returns:
        Masked string like "abcd****wxyz"
    """
    if len(s) <= visible_chars * 2:
        return "*" * len(s)

    return s[:visible_chars] + "*" * (len(s) - visible_chars * 2) + s[-visible_chars:]


# =============================================================================
# Input Validation
# =============================================================================

def validate_symbol(symbol: str) -> bool:
    """Validate a trading symbol."""
    if not symbol:
        return False
    # Only allow alphanumeric characters
    return bool(re.match(r'^[A-Z0-9]{2,20}$', symbol.upper()))


def validate_decimal(value: str, min_val: float = None, max_val: float = None) -> bool:
    """Validate a decimal string."""
    try:
        num = float(value)
        if min_val is not None and num < min_val:
            return False
        if max_val is not None and num > max_val:
            return False
        return True
    except (ValueError, TypeError):
        return False


def sanitize_string(s: str, max_length: int = 1000) -> str:
    """Sanitize a string for safe storage/display."""
    if not isinstance(s, str):
        return ""
    # Remove control characters except newlines/tabs
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', s)
    return s[:max_length]


# =============================================================================
# Secure Token Generation
# =============================================================================

def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure token."""
    return secrets.token_urlsafe(length)


def generate_api_key() -> str:
    """Generate a new API key in a specific format."""
    prefix = "sk"  # secret key prefix
    random_part = secrets.token_hex(24)
    return f"{prefix}_{random_part}"


def generate_nonce() -> str:
    """Generate a nonce for request signing."""
    return f"{int(time.time() * 1000)}{secrets.token_hex(8)}"


# =============================================================================
# Request Signing (for exchange APIs)
# =============================================================================

def sign_request(
    secret: str,
    params: Dict[str, Any],
    timestamp: int = None,
) -> str:
    """
    Sign request parameters (Binance-style HMAC-SHA256).

    Args:
        secret: API secret
        params: Request parameters
        timestamp: Optional timestamp (uses current if not provided)

    Returns:
        Signature hex string
    """
    if timestamp is None:
        timestamp = int(time.time() * 1000)

    params["timestamp"] = timestamp

    # Create query string
    query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))

    # Sign with HMAC-SHA256
    signature = hmac.new(
        secret.encode(),
        query.encode(),
        hashlib.sha256
    ).hexdigest()

    return signature
