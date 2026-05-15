from typing import Any
import numpy as np


async def normalize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in inputs.items():
        arr = np.array(value, dtype=np.float32)
        # Min-max normalization
        min_val, max_val = arr.min(), arr.max()
        if max_val > min_val:
            arr = (arr - min_val) / (max_val - min_val)
        result[key] = arr.tolist()
    return result


async def format_output(inputs: Any) -> dict[str, Any]:
    if isinstance(inputs, dict):
        return {
            "formatted": True,
            "predictions": inputs,
        }
    return {"formatted": True, "predictions": inputs}