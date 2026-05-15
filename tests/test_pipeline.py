import pytest
from pipeline.dag import PipelineDAG


@pytest.mark.asyncio
async def test_pipeline_sequential():
    order = []

    async def stage_a(inputs):
        order.append("a")
        return {"from_a": True}

    async def stage_b(inputs):
        order.append("b")
        return {"from_b": True, "got_a": inputs.get("from_a")}

    pipeline = (
        PipelineDAG("test")
        .add_stage("a", stage_a)
        .add_stage("b", stage_b, depends_on=["a"])
    )

    results = await pipeline.execute({"initial": True})
    assert order == ["a", "b"]
    assert results["b"]["got_a"] is True


@pytest.mark.asyncio
async def test_pipeline_parallel():
    import asyncio
    started = []

    async def stage_x(inputs):
        started.append("x")
        await asyncio.sleep(0.01)
        return "x_done"

    async def stage_y(inputs):
        started.append("y")
        await asyncio.sleep(0.01)
        return "y_done"

    pipeline = (
        PipelineDAG("parallel_test")
        .add_stage("x", stage_x)
        .add_stage("y", stage_y) 
    )

    import time
    start = time.perf_counter()
    results = await pipeline.execute({})
    elapsed = time.perf_counter() - start

    assert elapsed < 0.035, f"Too slow: {elapsed:.3f}s — not parallel?"
    assert "x" in results and "y" in results


@pytest.mark.asyncio
async def test_shadow_manager():
    from shadow.manager import ShadowManager

    manager = ShadowManager(shadow_ratio=0.1)

    async def champion(inputs):
        return {"model": "champion", **inputs}

    async def shadow(inputs):
        return {"model": "shadow", **inputs}

    shadowed = 0
    for i in range(100):
        if manager.should_shadow():
            shadowed += 1

    # 10% ± 2% tolerance
    assert 8 <= shadowed <= 12, f"Expected ~10 shadows, got {shadowed}"