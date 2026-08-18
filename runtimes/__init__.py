from core.schemas import ModelInfo, RuntimeType
from runtimes.base import BaseRuntime
from runtimes.pytorch_runtime import PyTorchRuntime
from runtimes.onnx_runtime import ONNXRuntime
from runtimes.jax_runtime import JAXRuntime

def get_runtime(model_info: ModelInfo) -> BaseRuntime:
    registry = {
        RuntimeType.PYTORCH : PyTorchRuntime,
        RuntimeType.ONNX    : ONNXRuntime,
        RuntimeType.JAX     : JAXRuntime,
    }
    cls = registry.get(model_info.runtime)
    if cls is None:
        raise ValueError(
            f"Unsupported runtime: '{model_info.runtime}'. "
            f"Available: {list(registry.keys())}"
        )
    return cls(model_info)