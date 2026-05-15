from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Server
    app_name: str = "Universal Model Server"
    app_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    debug: bool = False

    # SLA defaults (milliseconds)
    sla_urgent_ms: int = 100
    sla_normal_ms: int = 500
    sla_batch_ms: int = 5000

    # Queue
    max_queue_size: int = 1000

    # GPU
    default_device: str = "cpu"

    # Observability
    enable_tracing: bool = True
    prometheus_port: int = 9090

    # CORS
    allowed_origins: list[str] = ["http://localhost:3000"]

    # Auth
    api_key_enabled: bool = False
    api_keys: str = ""

    @property
    def valid_api_keys(self) -> set[str]:
        if not self.api_keys:
            return set()
        return set(k.strip() for k in self.api_keys.split(","))


@lru_cache()
def get_settings() -> Settings:
    return Settings()