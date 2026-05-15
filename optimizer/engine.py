import asyncio
import os
import time
from pathlib import Path
from dataclasses import dataclass
from optimizer.cache import OptimizationCache


@dataclass
class OptimizationResult:
    original_path: str
    optimized_path: str
    optimization_level: str
    original_size_mb: float
    optimized_size_mb: float
    size_reduction_pct: float
    optimization_time_s: float
    from_cache: bool

    def summary(self) -> str:
        tag = "📦 cache" if self.from_cache else "⚡ fresh"
        return (
            f"{tag} | {self.optimization_level} | "
            f"{self.original_size_mb:.1f}MB → "
            f"{self.optimized_size_mb:.1f}MB "
            f"({self.size_reduction_pct:.0f}% smaller) | "
            f"{self.optimization_time_s:.2f}s"
        )


class ModelOptimizer:   
    def __init__(self, output_dir: str = "models/optimized"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache = OptimizationCache()

    async def optimize(
        self,
        model_path: str,
        optimization_level: str = "standard",
    ) -> OptimizationResult:
        # Cache check
        cached = self.cache.get_cached_path(model_path, optimization_level)
        if cached:
            stats = self.cache.get_stats(model_path, optimization_level)
            print(f"📦 Using cached optimized model: {cached}")
            return OptimizationResult(
                original_path=model_path,
                optimized_path=cached,
                optimization_level=optimization_level,
                original_size_mb=stats.get("original_size_mb", 0),
                optimized_size_mb=stats.get("optimized_size_mb", 0),
                size_reduction_pct=stats.get("size_reduction_pct", 0),
                optimization_time_s=0.0,
                from_cache=True,
            )

        # Fresh optimization
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._optimize_sync,
            model_path,
            optimization_level,
        )
        return result

    def _optimize_sync(
        self,
        model_path: str,
        optimization_level: str,
    ) -> OptimizationResult:
        import onnx
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic, QuantType

        start = time.perf_counter()
        original_size = os.path.getsize(model_path) / (1024 * 1024)

        stem = Path(model_path).stem
        out_path = str(
            self.output_dir / f"{stem}_{optimization_level}.onnx"
        )

        if optimization_level == "basic":
            self._graph_optimize(model_path, out_path, level=1)

        elif optimization_level == "standard":
            self._graph_optimize(model_path, out_path, level=2)

        elif optimization_level == "aggressive":
            # INT8 dynamic quantization
            quantize_dynamic(
                model_input=model_path,
                model_output=out_path,
                weight_type=QuantType.QInt8,
            )
        else:
            raise ValueError(
                f"Unknown optimization level: '{optimization_level}'. "
                f"Use: basic, standard, aggressive"
            )

        optimized_size = os.path.getsize(out_path) / (1024 * 1024)
        reduction = round(
            (1 - optimized_size / original_size) * 100, 1
        ) if original_size > 0 else 0.0
        elapsed = round(time.perf_counter() - start, 2)

        stats = {
            "original_size_mb":   round(original_size, 3),
            "optimized_size_mb":  round(optimized_size, 3),
            "size_reduction_pct": reduction,
        }
        self.cache.save_to_cache(
            model_path, out_path, optimization_level, stats
        )

        result = OptimizationResult(
            original_path=model_path,
            optimized_path=out_path,
            optimization_level=optimization_level,
            original_size_mb=round(original_size, 3),
            optimized_size_mb=round(optimized_size, 3),
            size_reduction_pct=reduction,
            optimization_time_s=elapsed,
            from_cache=False,
        )
        print(f"✅ Optimized: {result.summary()}")
        return result

    def _graph_optimize(
        self,
        input_path: str,
        output_path: str,
        level: int,
    ) -> None:
        """ONNX Runtime graph optimization।"""
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            if level == 1
            else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.optimized_model_filepath = output_path

        ort.InferenceSession(
            input_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )