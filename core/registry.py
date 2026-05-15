from core.schemas import ModelInfo, RuntimeType
from core.config import get_settings
import yaml
import os

class ModelRegistry:
    def __init__(self):
        self._models: dict[str, ModelInfo] = {}

    def register(self, model_info: ModelInfo) -> None:
        self._models[model_info.name] = model_info
        print(f"📦 Registered model: '{model_info.name}' "
              f"[{model_info.runtime} on {model_info.device}]")

    def get(self, name: str) -> ModelInfo:
        if name not in self._models:
            raise KeyError(
                f"Model '{name}' not found. "
                f"Available: {list(self._models.keys())}"
            )
        return self._models[name]

    def list_models(self) -> list[str]:
        return list(self._models.keys())

    def load_from_yaml(self, path: str) -> None:
        if not os.path.exists(path):
            print(f"⚠️  Model config not found: {path}")
            return
        with open(path) as f:
            data = yaml.safe_load(f)
        for m in data.get("models", []):
            self.register(ModelInfo(**m))

_registry = ModelRegistry()

def get_registry() -> ModelRegistry:
    return _registry