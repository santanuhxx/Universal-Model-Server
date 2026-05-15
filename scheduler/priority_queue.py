import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from core.schemas import InferenceRequest, Priority

@dataclass(order=True)
class QueueItem:
    absolute_deadline: float
    priority_weight: int = field(compare=True)
    enqueued_at: float = field(compare=False)
    request: InferenceRequest = field(compare=False)

    @staticmethod
    def weight(priority: Priority) -> int:
        return {
            Priority.URGENT: 0,
            Priority.NORMAL: 1,
            Priority.BATCH:  2,
        }[priority]


class SLAPriorityQueue:
    def __init__(self, max_size: int = 1000):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=max_size
        )
        self._stats: dict[str, int] = {}  

    async def enqueue(self, request: InferenceRequest) -> None:
        item = QueueItem(
            absolute_deadline=request.absolute_deadline,
            priority_weight=QueueItem.weight(request.priority),
            enqueued_at=time.time(),
            request=request,
        )
        try:
            self._queue.put_nowait(item)
            self._stats[request.tenant_id] = (
                self._stats.get(request.tenant_id, 0) + 1
            )
        except asyncio.QueueFull:
            raise RuntimeError(
                f"Queue full! Max size: {self._queue.maxsize}. "
                f"Tenant '{request.tenant_id}' request dropped."
            )

    async def dequeue(self) -> QueueItem:
        return await self._queue.get()

    def is_expired(self, item: QueueItem) -> bool:
        return time.time() > item.absolute_deadline

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> dict:
        return {
            "queue_size": self.size,
            "tenant_counts": self._stats,
        }