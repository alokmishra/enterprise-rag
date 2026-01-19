"""
Enterprise RAG System - API Key Authentication
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime
from typing import Optional
from uuid import UUID

from src.core.exceptions import AuthenticationError
from src.auth.models import APIKey


def generate_api_key(prefix: str = "rag") -> tuple[str, str, str]:
    """
    Generate a new API key.

    Returns:
        Tuple of (full_key, key_prefix, key_hash)
        - full_key: The complete API key to give to the user (only shown once)
        - key_prefix: First 8 chars for identification
        - key_hash: Hash to store in database
    """
    # Generate 32 bytes of random data
    random_bytes = secrets.token_bytes(32)
    key_body = secrets.token_urlsafe(32)

    # Create the full key with prefix
    full_key = f"{prefix}_{key_body}"

    # Get prefix for identification
    key_prefix = full_key[:8]

    # Hash the key for storage
    key_hash = hash_api_key(full_key)

    return full_key, key_prefix, key_hash


def hash_api_key(key: str) -> str:
    """Hash an API key for secure storage."""
    return hashlib.sha256(key.encode()).hexdigest()


class APIKeyAuth:
    """API Key authentication manager."""

    def __init__(self, key_repository=None):
        """
        Initialize the API key auth manager.

        Args:
            key_repository: Repository for API key storage
        """
        self.key_repository = key_repository
        self._cache: dict[str, APIKey] = {}

    async def validate_key(self, api_key: str) -> APIKey:
        """
        Validate an API key and return the key details.

        Args:
            api_key: The API key to validate

        Returns:
            APIKey model if valid

        Raises:
            AuthenticationError if invalid
        """
        key_hash = hash_api_key(api_key)

        # Check cache first
        if key_hash in self._cache:
            cached = self._cache[key_hash]
            if self._is_valid(cached):
                return cached
            else:
                del self._cache[key_hash]
                raise AuthenticationError("API key expired or inactive")

        # Look up in repository
        if self.key_repository:
            key_model = await self.key_repository.get_by_hash(key_hash)
            if key_model and self._is_valid_model(key_model):
                api_key_obj = APIKey(
                    id=key_model.id,
                    name=key_model.name,
                    key_hash=key_model.key_hash,
                    key_prefix=key_model.key_prefix,
                    tenant_id=key_model.tenant_id,
                    user_id=key_model.user_id,
                    permissions=key_model.permissions or [],
                    is_active=key_model.is_active,
                    created_at=key_model.created_at,
                    expires_at=key_model.expires_at,
                    last_used_at=key_model.last_used_at,
                )
                self._cache[key_hash] = api_key_obj

                # Update last used
                await self.key_repository.update_last_used(key_model.id)

                return api_key_obj

        raise AuthenticationError("Invalid API key")

    def _is_valid(self, key: APIKey) -> bool:
        """Check if an API key is valid."""
        if not key.is_active:
            return False
        if key.expires_at and key.expires_at < datetime.utcnow():
            return False
        return True

    def _is_valid_model(self, key_model) -> bool:
        """Check if an API key model is valid."""
        if not key_model.is_active:
            return False
        if key_model.expires_at and key_model.expires_at < datetime.utcnow():
            return False
        return True

    async def create_key(
        self,
        name: str,
        tenant_id: UUID,
        user_id: Optional[UUID] = None,
        permissions: list[str] = None,
        expires_in_days: Optional[int] = None,
    ) -> tuple[str, APIKey]:
        """
        Create a new API key.

        Args:
            name: Human-readable name for the key
            tenant_id: Tenant the key belongs to
            user_id: Optional user the key belongs to
            permissions: List of permissions for the key
            expires_in_days: Optional expiration in days

        Returns:
            Tuple of (full_key, APIKey model)
            Note: full_key is only returned once and should be shown to user
        """
        from datetime import timedelta

        full_key, key_prefix, key_hash = generate_api_key()

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        api_key = APIKey(
            name=name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            tenant_id=tenant_id,
            user_id=user_id,
            permissions=permissions or [],
            expires_at=expires_at,
        )

        if self.key_repository:
            await self.key_repository.create(api_key)

        return full_key, api_key

    async def revoke_key(self, key_id: UUID) -> bool:
        """Revoke an API key."""
        if self.key_repository:
            await self.key_repository.deactivate(key_id)

            # Clear from cache
            for hash_key, cached in list(self._cache.items()):
                if cached.id == key_id:
                    del self._cache[hash_key]
                    break

            return True
        return False

    def clear_cache(self) -> None:
        """Clear the API key cache."""
        self._cache.clear()
