import asyncio
import time
from typing import Any
import numpy as np
from core.schemas import ModelInfo
from runtimes.base import BaseRuntime

class PyTorchRuntime(BaseRuntime):  
    def __init__(self, model_info: ModelInfo):
        super().__init__(model_info)
        self._model = None
        self._device = None

    async def load(self) -> None:
        import torch

        self._device = torch.device(self.model_info.device)

        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: torch.jit.load(
                self.model_info.model_path,
                map_location=self._device
            )
        )
        self._model.eval()

    async def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self._validate_loaded()
        import torch

        start = time.perf_counter()

        def _run():
            with torch.no_grad():
                # inputs dict → tensors
                tensors = {
                    k: torch.tensor(v, device=self._device)
                    for k, v in inputs.items()
                }
                input_tensor = list(tensors.values())[0]
                output = self._model(input_tensor)
                return {"output": output.cpu().numpy().tolist()}

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run)

        elapsed = (time.perf_counter() - start) * 1000
        result["_inference_ms"] = round(elapsed, 2)
        return result

    async def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass