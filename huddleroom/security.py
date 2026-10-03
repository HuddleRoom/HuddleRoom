import hashlib
import os
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
import bcrypt

from huddleroom.config import settings


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.jwt_expire_minutes)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    from fastapi import HTTPException, status
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return payload
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def generate_api_key() -> tuple[str, str, str]:
    """Returns (full_key, key_prefix, hashed_key)."""
    random_part = os.urandom(32).hex()
    full_key = f"{settings.api_key_prefix}{random_part}"
    key_prefix = full_key[:8]
    hashed_key = hash_api_key(full_key)
    return full_key, key_prefix, hashed_key


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()
