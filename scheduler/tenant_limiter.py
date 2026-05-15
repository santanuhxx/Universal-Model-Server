import time
from collections import defaultdict, deque
from core.config import get_settings

class TenantLimiter:
    def __init__(
        self,
        max_requests_per_second: int = 100,
        window_seconds: float = 1.0,
    ):
        self.max_rps = max_requests_per_second
        self.window = window_seconds
        # tenant_id → timestamps of recent requests
        self._windows: dict[str, deque] = defaultdict(deque)

    def is_allowed(self, tenant_id: str) -> bool:
        now = time.time()
        window = self._windows[tenant_id]

        cutoff = now - self.window
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self.max_rps:
            return False

        window.append(now)
        return True

    def get_usage(self, tenant_id: str) -> dict:
        now = time.time()
        window = self._windows[tenant_id]
        cutoff = now - self.window
        active = sum(1 for t in window if t >= cutoff)
        return {
            "tenant_id": tenant_id,
            "requests_in_window": active,
            "max_allowed": self.max_rps,
            "usage_pct": round(active / self.max_rps * 100, 1),
        }