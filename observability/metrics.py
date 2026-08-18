from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)
from fastapi import APIRouter
from fastapi.responses import Response

router = APIRouter()

# ── Counters ───────────────────────────────────────
REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["model_name", "tenant_id", "status"],
)

# ── Histograms ─────────────────────────────────────
REQUEST_LATENCY = Histogram(
    "inference_latency_ms",
    "Inference latency in milliseconds",
    ["model_name", "priority"],
    buckets=[5, 10, 25, 50, 100, 250, 500, 1000],
)

# ── Gauges ─────────────────────────────────────────
QUEUE_SIZE = Gauge(
    "queue_size_current",
    "Current number of requests in queue",
)

LOADED_MODELS = Gauge(
    "loaded_models_total",
    "Number of models currently loaded",
)

@router.get("/metrics")
async def metrics():
    """Prometheus metrics scrape endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )