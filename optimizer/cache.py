import os
import json
import hashlib
from pathlib import Path


class OptimizationCache:
    def __init__(self, cache_dir: str = ".optimizer_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._meta_file = self.cache_dir / "meta.json"
        self._meta: dict = self._load_meta()

    def _load_meta(self) -> dict:
        if self._meta_file.exists():
            with open(self._meta_file) as f:
                return json.load(f)
        return {}

    def _save_meta(self) -> None:
        with open(self._meta_file, "w") as f:
            json.dump(self._meta, f, indent=2)

    def _model_hash(self, model_path: str) -> str:
        h = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def get_cached_path(
        self,
        model_path: str,
        optimization_level: str,
    ) -> str | None:
        key = f"{model_path}:{optimization_level}"
        if key not in self._meta:
            return None

        cached = self._meta[key]
        current_hash = self._model_hash(model_path)

        if cached["hash"] != current_hash:
            del self._meta[key]
            self._save_meta()
            return None

        cached_path = cached["optimized_path"]
        if os.path.exists(cached_path):
            return cached_path

        return None

    def save_to_cache(
        self,
        original_path: str,
        optimized_path: str,
        optimization_level: str,
        stats: dict,
    ) -> None:
        key = f"{original_path}:{optimization_level}"
        self._meta[key] = {
            "hash": self._model_hash(original_path),
            "optimized_path": optimized_path,
            "stats": stats,
        }
        self._save_meta()

    def get_stats(self, model_path: str, optimization_level: str) -> dict:
        key = f"{model_path}:{optimization_level}"
        return self._meta.get(key, {}).get("stats", {})