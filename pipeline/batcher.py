import asyncio
import time
from typing import Any
from dataclasses import dataclass, field


@dataclass
class BatchItem:
    inputs: dict[str, Any]
    future: asyncio.Future
    enqueued_at: float = field(default_factory=time.time)


class DynamicBatcher:  
    def __init__(
        self,
        inference_fn,
        max_batch_size: int = 32,
        max_wait_ms: float = 10.0,
    ):
        self.inference_fn = inference_fn
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: list[BatchItem] = []
        self._lock = asyncio.Lock()
        self._batch_task: asyncio.Task | None = None

    async def infer(self, inputs: dict[str, Any]) -> Any:
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        async with self._lock:
            self._queue.append(BatchItem(inputs=inputs, future=future))
            if self._batch_task is None or self._batch_task.done():
                self._batch_task = asyncio.create_task(
                    self._flush_after_wait()
                )
            if len(self._queue) >= self.max_batch_size:
                self._batch_task.cancel()
                asyncio.create_task(self._flush())

        return await future

    async def _flush_after_wait(self) -> None:
        await asyncio.sleep(self.max_wait_ms / 1000)
        await self._flush()

    async def _flush(self) -> None:
        async with self._lock:
            if not self._queue:
                return
            batch = self._queue[:self.max_batch_size]
            self._queue = self._queue[self.max_batch_size:]

        try:
            results = await asyncio.gather(
                *[self.inference_fn(item.inputs) for item in batch],
                return_exceptions=True,
            )
            for item, result in zip(batch, results):
                if isinstance(result, Exception):
                    item.future.set_exception(result)
                else:
                    item.future.set_result(result)
        except Exception as e:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)