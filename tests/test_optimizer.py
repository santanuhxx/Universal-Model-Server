import pytest
import os
from pathlib import Path


@pytest.fixture
def dummy_onnx_path():
    path = "models/echo.onnx"
    assert os.path.exists(path), "Run scripts/create_dummy_models.py first"
    return path


@pytest.mark.asyncio
async def test_optimize_basic(dummy_onnx_path):
    from optimizer.engine import ModelOptimizer
    optimizer = ModelOptimizer(output_dir="models/test_optimized")
    result = await optimizer.optimize(dummy_onnx_path, "basic")

    assert os.path.exists(result.optimized_path)
    assert result.optimization_level == "basic"
    assert result.optimized_size_mb > 0
    print(f"\n   {result.summary()}")


@pytest.mark.asyncio
async def test_optimize_standard(dummy_onnx_path):
    from optimizer.engine import ModelOptimizer
    optimizer = ModelOptimizer(output_dir="models/test_optimized")
    result = await optimizer.optimize(dummy_onnx_path, "standard")

    assert os.path.exists(result.optimized_path)
    assert result.optimization_level == "standard"
    print(f"\n   {result.summary()}")


@pytest.mark.asyncio
async def test_optimize_cache(dummy_onnx_path, tmp_path):
    from optimizer.engine import ModelOptimizer
    from optimizer.cache import OptimizationCache

    optimizer = ModelOptimizer(
        output_dir=str(tmp_path / "optimized")
    )
    optimizer.cache = OptimizationCache(
        cache_dir=str(tmp_path / "cache")
    )
    r1 = await optimizer.optimize(dummy_onnx_path, "basic")
    assert r1.from_cache is False, "First call should be fresh"

    r2 = await optimizer.optimize(dummy_onnx_path, "basic")
    assert r2.from_cache is True, "Second call should hit cache"
    print(f"\n   Cache working: {r2.from_cache}")


@pytest.mark.asyncio
async def test_optimize_invalid_level(dummy_onnx_path):
    from optimizer.engine import ModelOptimizer
    optimizer = ModelOptimizer(output_dir="models/test_optimized")

    with pytest.raises(ValueError, match="Unknown optimization level"):
        await optimizer.optimize(dummy_onnx_path, "super_ultra")