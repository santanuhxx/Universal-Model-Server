from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from core.config import get_settings

API_KEY_HEADER = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)

async def verify_api_key(
    api_key: str = Security(API_KEY_HEADER),
) -> str:
    settings = get_settings()

    if not settings.api_key_enabled:
        return "no-auth"

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header missing",
        )

    if api_key not in settings.valid_api_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return api_key