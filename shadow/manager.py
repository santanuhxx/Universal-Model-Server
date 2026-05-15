import asyncio
import time
from typing import Any
from dataclasses import dataclass, field


@dataclass
class ShadowResult:
    request_id: str
    champion_output: Any
    shadow_output: Any
    champion_latency_ms: float
    shadow_latency_ms: float
    timestamp: float = field(default_factory=time.time)

    @property
    def latency_diff_ms(self) -> float:
        return round(self.shadow_latency_ms - self.champion_latency_ms, 2)


class ShadowManager:
    def __init__(self, shadow_ratio: float = 0.1):
        self.shadow_ratio = shadow_ratio
        self._results: list[ShadowResult] = []
        self._request_count = 0

    def should_shadow(self) -> bool:
        self._request_count += 1
        threshold = int(1 / self.shadow_ratio)
        return self._request_count % threshold == 0

    async def run_shadow(
        self,
        request_id: str,
        champion_handler,
        shadow_handler,
        inputs: dict[str, Any],
    ) -> Any:
        champion_start = time.perf_counter()
        shadow_start = time.perf_counter()

        champion_result, shadow_result = await asyncio.gather(
            champion_handler(inputs),
            shadow_handler(inputs),
            return_exceptions=True,
        )

        champion_latency = (time.perf_counter() - champion_start) * 1000
        shadow_latency = (time.perf_counter() - shadow_start) * 1000

        if not isinstance(shadow_result, Exception):
            result = ShadowResult(
                request_id=request_id,
                champion_output=champion_result,
                shadow_output=shadow_result,
                champion_latency_ms=round(champion_latency, 2),
                shadow_latency_ms=round(shadow_latency, 2),
            )
            self._results.append(result)
            asyncio.create_task(self._log_result(result))

        return champion_result

    async def _log_result(self, result: ShadowResult) -> None:
        print(
            f"🔮 Shadow [{result.request_id[:8]}] "
            f"champion={result.champion_latency_ms}ms "
            f"shadow={result.shadow_latency_ms}ms "
            f"diff={result.latency_diff_ms:+.1f}ms"
        )

    def get_summary(self) -> dict:
        if not self._results:
            return {"total_shadow_requests": 0}

        avg_diff = sum(r.latency_diff_ms for r in self._results) / len(
            self._results
        )
        return {
            "total_shadow_requests": len(self._results),
            "avg_latency_diff_ms": round(avg_diff, 2),
            "shadow_ratio": self.shadow_ratio,
            "verdict": (
                "✅ Shadow faster" if avg_diff < 0
                else "⚠️ Shadow slower"
            ),
        }