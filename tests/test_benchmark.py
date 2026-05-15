import pytest
from benchmark.stats import LatencyTracker, RequestStat


def make_stat(latency_ms: float, status: str = "success") -> RequestStat:
    return RequestStat(
        latency_ms=latency_ms,
        status=status,
        model_name="echo_model",
        priority="normal",
    )


def test_percentiles():
    tracker = LatencyTracker()
    for i in range(1, 101):
        tracker.record(make_stat(float(i)))

    assert tracker.percentile(50) == 50.0
    assert tracker.percentile(95) == 95.0
    assert tracker.percentile(99) == 99.0


def test_sla_breach_rate():
    tracker = LatencyTracker()
    # 70 fast, 30 slow
    for _ in range(70):
        tracker.record(make_stat(50.0))
    for _ in range(30):
        tracker.record(make_stat(200.0))

    # 100ms SLA — 30% breach
    assert tracker.sla_breach_rate(100) == 30.0
    # 500ms SLA — 0% breach
    assert tracker.sla_breach_rate(500) == 0.0


def test_empty_tracker():
    tracker = LatencyTracker()
    assert tracker.percentile(99) == 0.0
    assert tracker.sla_breach_rate(100) == 0.0
    assert tracker.throughput() == 0.0
    assert tracker.summary()["total_requests"] == 0


def test_summary_keys():
    tracker = LatencyTracker()
    tracker.record(make_stat(10.0))
    summary = tracker.summary()

    required_keys = [
        "total_requests", "success_count", "error_count",
        "success_rate_pct", "p50_ms", "p95_ms", "p99_ms",
        "min_ms", "max_ms", "throughput_rps",
        "sla_breach_100ms", "sla_breach_500ms",
    ]
    for key in required_keys:
        assert key in summary, f"Missing key: {key}"


def test_window_size():
    tracker = LatencyTracker(window_size=10)
    for i in range(20):
        tracker.record(make_stat(float(i)))

    assert tracker.summary()["total_requests"] == 10