from datetime import datetime, timedelta, timezone
import hmac

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.settings import Settings, get_settings

bearer_scheme = HTTPBearer()


def create_access_token(username: str, settings: Settings) -> str:
    if not settings.jwt_secret_key:
        raise HTTPException(status_code=500, detail="JWT_SECRET_KEY is not configured")
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    return jwt.encode(
        {"sub": username, "exp": expires_at},
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def credentials_match(username: str, password: str, settings: Settings) -> bool:
    if not settings.auth_password:
        raise HTTPException(status_code=500, detail="AUTH_PASSWORD is not configured")
    return hmac.compare_digest(username, settings.auth_username) and hmac.compare_digest(
        password, settings.auth_password
    )


async def get_current_username(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> str:
    if not settings.jwt_secret_key:
        raise HTTPException(status_code=500, detail="JWT_SECRET_KEY is not configured")
    try:
        payload = jwt.decode(credentials.credentials, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        username = payload.get("sub")
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    if username != settings.auth_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username
