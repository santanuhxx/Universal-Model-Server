import pytest
import asyncio
from core.schemas import InferenceRequest, Priority
from scheduler.priority_queue import SLAPriorityQueue, QueueItem
from scheduler.tenant_limiter import TenantLimiter

@pytest.mark.asyncio
async def test_queue_urgent_before_normal():
    queue = SLAPriorityQueue()

    normal = InferenceRequest(
        model_name="m", inputs={}, priority=Priority.NORMAL
    )
    urgent = InferenceRequest(
        model_name="m", inputs={}, priority=Priority.URGENT
    )

    await queue.enqueue(normal)
    await queue.enqueue(urgent)

    first = await queue.dequeue()
    assert first.request.priority == Priority.URGENT

@pytest.mark.asyncio
async def test_queue_full_raises():
    queue = SLAPriorityQueue(max_size=1)
    req = InferenceRequest(model_name="m", inputs={})
    await queue.enqueue(req)

    with pytest.raises(RuntimeError, match="Queue full"):
        await queue.enqueue(req)

def test_tenant_limiter_allows_within_limit():
    limiter = TenantLimiter(max_requests_per_second=5)
    for _ in range(5):
        assert limiter.is_allowed("team_a") is True

def test_tenant_limiter_blocks_over_limit():
    limiter = TenantLimiter(max_requests_per_second=3)
    for _ in range(3):
        limiter.is_allowed("team_b")
    assert limiter.is_allowed("team_b") is False

def test_tenant_limiter_isolates_teams():
    limiter = TenantLimiter(max_requests_per_second=2)
    limiter.is_allowed("team_a")
    limiter.is_allowed("team_a")
    limiter.is_allowed("team_a")  # team_a blocked

    assert limiter.is_allowed("team_b") is True  # team_b unaffected