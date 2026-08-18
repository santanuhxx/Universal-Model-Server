import asyncio
import time
from typing import Any
from core.schemas import ModelInfo
from runtimes.base import BaseRuntime


class JAXRuntime(BaseRuntime):
    def __init__(self, model_info: ModelInfo):
        super().__init__(model_info)
        self._params = None

    async def load(self) -> None:
        try:
            import jax
            import pickle

            loop = asyncio.get_event_loop()

            def _load():
                with open(self.model_info.model_path, "rb") as f:
                    params = pickle.load(f)
                return params

            self._params = await loop.run_in_executor(None, _load)
            print(f"   JAX device: {jax.default_backend()}")

        except ImportError:
            # JAX not installed — stub mode
            print(f"   ⚠️  JAX not installed — running in stub mode")
            self._params = None

    async def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self._validate_loaded()
        start = time.perf_counter()

        if self._params is None:
            # Stub — identity passthrough when JAX is unavailable
            result = {k: v for k, v in inputs.items()}
        else:
            import jax.numpy as jnp
            loop = asyncio.get_event_loop()

            def _run():
                arr = jnp.array(list(inputs.values())[0])
                return {"output": arr.tolist()}

            result = await loop.run_in_executor(None, _run)

        elapsed = (time.perf_counter() - start) * 1000
        result["_inference_ms"] = round(elapsed, 2)
        return result

    async def unload(self) -> None:
        self._params = None