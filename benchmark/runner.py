import asyncio
import time
import httpx
from benchmark.stats import LatencyTracker, RequestStat, BenchmarkReport


class BenchmarkRunner:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    async def run(
        self,
        model_name: str,
        inputs: dict,
        concurrency: int = 10,
        duration_seconds: float = 30.0,
        priority: str = "normal",
    ) -> BenchmarkReport:
        tracker = LatencyTracker(window_size=10000)
        start = time.time()

        print(f"\n🔥 Benchmark starting...")
        print(f"   Model       : {model_name}")
        print(f"   Concurrency : {concurrency}")
        print(f"   Duration    : {duration_seconds}s")
        print()

        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=10.0,
        ) as client:
            tasks = [
                self._worker(
                    client, model_name, inputs,
                    priority, tracker, duration_seconds, start
                )
                for _ in range(concurrency)
            ]
            await asyncio.gather(*tasks)

        elapsed = time.time() - start
        report = BenchmarkReport(
            model_name=model_name,
            concurrency=concurrency,
            duration_seconds=elapsed,
            stats=tracker.summary(),
        )
        report.print_report()
        return report

    async def _worker(
        self,
        client: httpx.AsyncClient,
        model_name: str,
        inputs: dict,
        priority: str,
        tracker: LatencyTracker,
        duration: float,
        start: float,
    ) -> None:
        while time.time() - start < duration:
            req_start = time.perf_counter()
            status = "error"
            try:
                r = await client.post(
                    "/infer",
                    json={
                        "model_name": model_name,
                        "inputs": inputs,
                        "priority": priority,
                    },
                )
                data = r.json()
                status = (
                    "success"
                    if data.get("status") == "done"
                    else "error"
                )
            except Exception:
                status = "error"

            latency = (time.perf_counter() - req_start) * 1000
            tracker.record(RequestStat(
                latency_ms=latency,
                status=status,
                model_name=model_name,
                priority=priority,
            ))