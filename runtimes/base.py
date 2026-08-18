from abc import ABC, abstractmethod
from typing import Any
from core.schemas import ModelInfo
import time

class BaseRuntime(ABC):  
    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info
        self.is_loaded = False
        self._load_time_ms: float = 0.0

    @abstractmethod
    async def load(self) -> None:
        ...

    @abstractmethod
    async def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        ...

    @abstractmethod
    async def unload(self) -> None:
        ...

    # ── Shared helpers ──────────────────────────────

    async def __aenter__(self):
        start = time.perf_counter()
        await self.load()
        self._load_time_ms = (time.perf_counter() - start) * 1000
        self.is_loaded = True
        print(f"✅ [{self.model_info.name}] loaded in {self._load_time_ms:.1f}ms "
              f"on {self.model_info.device}")
        return self

    async def __aexit__(self, *args):
        await self.unload()
        self.is_loaded = False
        print(f"🗑️  [{self.model_info.name}] unloaded")

    def _validate_loaded(self):
        if not self.is_loaded:
            raise RuntimeError(
                f"Model '{self.model_info.name}' is not loaded. "
                f"Use 'async with runtime:' or call load() first."
            )