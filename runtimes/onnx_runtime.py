import asyncio
import time
from typing import Any
import numpy as np
from core.schemas import ModelInfo
from runtimes.base import BaseRuntime

class ONNXRuntime(BaseRuntime):
    def __init__(self, model_info: ModelInfo):
        super().__init__(model_info)
        self._session = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []

    async def load(self) -> None:
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.model_info.device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )

        loop = asyncio.get_event_loop()
        self._session = await loop.run_in_executor(
            None,
            lambda: ort.InferenceSession(
                self.model_info.model_path,
                providers=providers
            )
        )

        self._input_names  = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]
        print(f"   Inputs : {self._input_names}")
        print(f"   Outputs: {self._output_names}")

    async def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self._validate_loaded()

        start = time.perf_counter()

        def _run():                                              # 8 spaces
            feed = {                                            # 12 spaces
                name: np.array(inputs[name], dtype=np.float32)
                for name in self._input_names
                if name in inputs
            }
            outputs = self._session.run(self._output_names, feed)
            return {
                name: out.tolist() if hasattr(out, "tolist") else out
                for name, out in zip(self._output_names, outputs)
            }

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run)

        elapsed = (time.perf_counter() - start) * 1000
        result["_inference_ms"] = round(elapsed, 2)
        return result

    async def unload(self) -> None:
        self._session = None