"""
Authentication & Authorization module for AgentArmy.

Provides JWT token management, API key handling, and rate limiting for agents.
Supports token generation, validation, refresh, and distributed rate limiting.
"""

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

import jwt
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class TokenPayload(BaseModel):
    """JWT token payload model.

    Attributes:
        sub: Subject (agent ID)
        agent_name: Human-readable agent name
        iss: Issuer
        aud: Audience
        iat: Issued at timestamp
        exp: Expiration timestamp
        jti: JWT ID (unique token identifier)
        scope: Space-separated permission scopes
    """

    sub: str = Field(..., description="Agent ID (subject)")
    agent_name: str = Field(..., description="Agent display name")
    iss: str = Field(default="agent-army", description="Issuer")
    aud: str = Field(default="agent-army-api", description="Audience")
    iat: int = Field(..., description="Issued at (Unix timestamp)")
    exp: int = Field(..., description="Expiration (Unix timestamp)")
    jti: str = Field(default_factory=lambda: str(uuid4()), description="JWT ID")
    scope: str = Field(default="", description="Space-separated scopes")


class RateLimitBucket:
    """Token bucket for rate limiting."""

    def __init__(
        self, capacity: int, refill_rate: float, refill_period: timedelta
    ) -> None:
        """Initialize rate limit bucket.

        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens to add per refill period
            refill_period: Time period for refill
        """
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_rate = refill_rate
        self.refill_period = refill_period
        self.last_refill = datetime.now(timezone.utc)

    def allow_request(self, tokens: int = 1) -> bool:
        """Check if request is allowed and consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if request is allowed, False if rate limited
        """
        now = datetime.now(timezone.utc)
        time_passed = (now - self.last_refill).total_seconds()
        refill_period_seconds = self.refill_period.total_seconds()

        if time_passed >= refill_period_seconds:
            tokens_to_add = (time_passed / refill_period_seconds) * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class AuthManager:
    """Authentication and authorization manager for agents.

    Handles JWT token lifecycle, API key management, and rate limiting.
    """

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        token_expiry_minutes: int = 60,
        refresh_expiry_days: int = 7,
    ) -> None:
        """Initialize authentication manager.

        Args:
            secret_key: Secret key for signing tokens
            algorithm: JWT algorithm (default: HS256)
            token_expiry_minutes: Token expiration in minutes
            refresh_expiry_days: Refresh token expiration in days
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry_minutes = token_expiry_minutes
        self.refresh_expiry_days = refresh_expiry_days

        # API key storage: {key_hash: (agent_id, key_name, created_at, active)}
        self._api_keys: dict[str, tuple[str, str, datetime, bool]] = {}

        # Rate limiting buckets per agent
        self._rate_limits: dict[str, RateLimitBucket] = {}

        # Revoked tokens (in production, use Redis)
        self._revoked_tokens: set[str] = set()

    def generate_token(
        self, agent_id: str, agent_name: str, scope: str = ""
    ) -> str:
        """Generate JWT token for agent.

        Args:
            agent_id: Unique agent identifier
            agent_name: Human-readable agent name
            scope: Space-separated permission scopes

        Returns:
            JWT token string
        """
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(minutes=self.token_expiry_minutes)

        payload = TokenPayload(
            sub=agent_id,
            agent_name=agent_name,
            iat=int(now.timestamp()),
            exp=int(expiry.timestamp()),
            scope=scope,
        )

        token = jwt.encode(
            payload.model_dump(),
            self.secret_key,
            algorithm=self.algorithm,
        )

        logger.info(
            "token_generated",
            agent_id=agent_id,
            agent_name=agent_name,
            scope=scope,
            expiry=expiry.isoformat(),
        )

        return token

    def validate_token(self, token: str) -> Optional[TokenPayload]:
        """Validate JWT token.

        Args:
            token: JWT token string

        Returns:
            TokenPayload if valid, None otherwise
        """
        try:
            if token in self._revoked_tokens:
                logger.warning("token_revoked", token_prefix=token[:10])
                return None

            payload = jwt.decode(
                token, self.secret_key, algorithms=[self.algorithm]
            )

            token_data = TokenPayload(**payload)

            # Check expiration
            now = datetime.now(timezone.utc)
            if now.timestamp() > token_data.exp:
                logger.warning(
                    "token_expired",
                    agent_id=token_data.sub,
                    expiry=token_data.exp,
                )
                return None

            logger.debug("token_validated", agent_id=token_data.sub)
            return token_data

        except jwt.InvalidTokenError as e:
            logger.warning("token_invalid", error=str(e))
            return None

    def refresh_token(self, token: str) -> Optional[str]:
        """Refresh an existing token.

        Args:
            token: Current JWT token

        Returns:
            New JWT token if refresh successful, None otherwise
        """
        payload = self.validate_token(token)
        if not payload:
            logger.warning("refresh_failed_invalid_token")
            return None

        # Check if token is old enough for refresh
        now = datetime.now(timezone.utc)
        token_age_seconds = now.timestamp() - payload.iat
        min_age_seconds = 60  # Minimum 1 minute

        if token_age_seconds < min_age_seconds:
            logger.warning(
                "refresh_too_soon", token_age_seconds=token_age_seconds
            )
            return None

        # Revoke old token
        self._revoked_tokens.add(token)

        # Generate new token
        new_token = self.generate_token(
            agent_id=payload.sub,
            agent_name=payload.agent_name,
            scope=payload.scope,
        )

        logger.info(
            "token_refreshed", agent_id=payload.sub, old_jti=payload.jti
        )

        return new_token

    def revoke_token(self, token: str) -> bool:
        """Revoke a token immediately.

        Args:
            token: JWT token to revoke

        Returns:
            True if revocation successful
        """
        payload = self.validate_token(token)
        if not payload:
            return False

        self._revoked_tokens.add(token)
        logger.info("token_revoked", agent_id=payload.sub, jti=payload.jti)
        return True

    def create_api_key(self, agent_id: str, key_name: str) -> str:
        """Create API key for agent.

        Args:
            agent_id: Agent identifier
            key_name: Descriptive name for the key

        Returns:
            API key string (format: aak_{random_32_bytes})
        """
        api_key = f"aak_{uuid4().hex}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        self._api_keys[key_hash] = (
            agent_id,
            key_name,
            datetime.now(timezone.utc),
            True,
        )

        logger.info(
            "api_key_created",
            agent_id=agent_id,
            key_name=key_name,
            key_hash=key_hash[:8],
        )

        return api_key

    def validate_api_key(self, api_key: str) -> Optional[str]:
        """Validate API key and return agent ID.

        Args:
            api_key: API key string

        Returns:
            Agent ID if valid, None otherwise
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        if key_hash not in self._api_keys:
            logger.warning("api_key_invalid", key_hash=key_hash[:8])
            return None

        agent_id, key_name, created_at, active = self._api_keys[key_hash]

        if not active:
            logger.warning(
                "api_key_inactive", key_hash=key_hash[:8], agent_id=agent_id
            )
            return None

        logger.debug("api_key_validated", agent_id=agent_id, key_name=key_name)
        return agent_id

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key.

        Args:
            api_key: API key to revoke

        Returns:
            True if revocation successful
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        if key_hash not in self._api_keys:
            return False

        agent_id, key_name, created_at, active = self._api_keys[key_hash]
        self._api_keys[key_hash] = (agent_id, key_name, created_at, False)

        logger.info(
            "api_key_revoked",
            key_hash=key_hash[:8],
            agent_id=agent_id,
            key_name=key_name,
        )

        return True

    def check_rate_limit(
        self,
        agent_id: str,
        tokens: int = 1,
        capacity: int = 100,
        refill_rate: float = 10.0,
        refill_period_seconds: int = 60,
    ) -> bool:
        """Check rate limit for agent and consume token.

        Args:
            agent_id: Agent identifier
            tokens: Tokens to consume
            capacity: Bucket capacity
            refill_rate: Tokens to add per period
            refill_period_seconds: Refill period in seconds

        Returns:
            True if request allowed, False if rate limited
        """
        if agent_id not in self._rate_limits:
            self._rate_limits[agent_id] = RateLimitBucket(
                capacity=capacity,
                refill_rate=refill_rate,
                refill_period=timedelta(seconds=refill_period_seconds),
            )

        bucket = self._rate_limits[agent_id]
        allowed = bucket.allow_request(tokens)

        if not allowed:
            logger.warning(
                "rate_limit_exceeded",
                agent_id=agent_id,
                tokens_requested=tokens,
                tokens_available=int(bucket.tokens),
            )
        else:
            logger.debug(
                "rate_limit_check_passed",
                agent_id=agent_id,
                tokens_remaining=int(bucket.tokens),
            )

        return allowed

    def get_rate_limit_status(self, agent_id: str) -> Optional[dict]:
        """Get rate limit status for agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Dict with capacity, tokens, refill_rate, or None if not found
        """
        if agent_id not in self._rate_limits:
            return None

        bucket = self._rate_limits[agent_id]
        return {
            "capacity": bucket.capacity,
            "tokens": int(bucket.tokens),
            "refill_rate": bucket.refill_rate,
            "refill_period_seconds": bucket.refill_period.total_seconds(),
        }
