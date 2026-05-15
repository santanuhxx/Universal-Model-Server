from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime, timezone
import time

router = APIRouter()

START_TIME = time.time()

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    timestamp: str

class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, bool]

@router.get("/health", response_model=HealthResponse)
async def health():
    from core.config import get_settings
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        uptime_seconds=round(time.time() - START_TIME, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

@router.get("/ready", response_model=ReadyResponse)
async def ready():
    checks = {
        "server": True,
        "model_registry": True,
    }
    return ReadyResponse(
        ready=all(checks.values()),
        checks=checks,
    )