import time
import statistics
from dataclasses import dataclass, field
from collections import deque


@dataclass
class RequestStat:
    latency_ms: float
    status: str
    model_name: str
    priority: str
    timestamp: float = field(default_factory=time.time)


class LatencyTracker:
    def __init__(self, window_size: int = 1000):
        self._stats: deque[RequestStat] = deque(maxlen=window_size)

    def record(self, stat: RequestStat) -> None:
        self._stats.append(stat)

    def percentile(self, p: float) -> float:
        if not self._stats:
            return 0.0
        latencies = sorted(s.latency_ms for s in self._stats)
        idx = max(0, int(len(latencies) * p / 100) - 1) 
        return round(latencies[min(idx, len(latencies) - 1)], 2)

    def sla_breach_rate(self, sla_ms: float) -> float:
        if not self._stats:
            return 0.0
        breaches = sum(
            1 for s in self._stats
            if s.latency_ms > sla_ms
        )
        return round(breaches / len(self._stats) * 100, 2)

    def throughput(self, window_seconds: float = 60.0) -> float:
        if not self._stats:
            return 0.0
        now = time.time()
        cutoff = now - window_seconds
        recent = sum(1 for s in self._stats if s.timestamp >= cutoff)
        return round(recent / window_seconds, 2)

    def summary(self) -> dict:
        if not self._stats:
            return {"total_requests": 0}

        success = sum(1 for s in self._stats if s.status == "success")
        return {
            "total_requests"   : len(self._stats),
            "success_count"    : success,
            "error_count"      : len(self._stats) - success,
            "success_rate_pct" : round(success / len(self._stats) * 100, 1),
            "p50_ms"           : self.percentile(50),
            "p95_ms"           : self.percentile(95),
            "p99_ms"           : self.percentile(99),
            "min_ms"           : round(min(s.latency_ms for s in self._stats), 2),
            "max_ms"           : round(max(s.latency_ms for s in self._stats), 2),
            "throughput_rps"   : self.throughput(),
            "sla_breach_100ms" : self.sla_breach_rate(100),
            "sla_breach_500ms" : self.sla_breach_rate(500),
        }


@dataclass
class BenchmarkReport:
    model_name: str
    concurrency: int
    duration_seconds: float
    stats: dict

    def print_report(self) -> None:
        print(f"""
╔══════════════════════════════════════════════╗
║         BENCHMARK REPORT                     ║
╠══════════════════════════════════════════════╣
║  Model       : {self.model_name:<28} ║
║  Concurrency : {self.concurrency:<28} ║
║  Duration    : {self.duration_seconds:<28.1f} ║
╠══════════════════════════════════════════════╣
║  Total Req   : {self.stats['total_requests']:<28} ║
║  Success     : {self.stats['success_rate_pct']:<27.1f}% ║
║  Throughput  : {self.stats['throughput_rps']:<25.2f} rps ║
╠══════════════════════════════════════════════╣
║  P50 Latency : {self.stats['p50_ms']:<25.2f} ms ║
║  P95 Latency : {self.stats['p95_ms']:<25.2f} ms ║
║  P99 Latency : {self.stats['p99_ms']:<25.2f} ms ║
╠══════════════════════════════════════════════╣
║  SLA >100ms  : {self.stats['sla_breach_100ms']:<27.2f}% ║
║  SLA >500ms  : {self.stats['sla_breach_500ms']:<27.2f}% ║
╚══════════════════════════════════════════════╝
        """)