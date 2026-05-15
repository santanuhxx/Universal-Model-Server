from fastapi import FastAPI, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Annotated
from core.config import get_settings
from core.schemas import InferenceRequest
from core.registry import get_registry
from core.auth import verify_api_key
from router import get_engine
from observability.health import router as health_router
from observability.metrics import router as metrics_router
from observability.tracing import setup_tracing
from pipeline import register_pipeline, list_pipelines, get_pipeline
from pipeline import normalize_inputs, format_output
from pipeline.dag import PipelineDAG
from shadow import get_shadow_manager
from optimizer.engine import ModelOptimizer
from benchmark.stats import LatencyTracker, RequestStat
from drift.monitor import get_drift_monitor

settings = get_settings()

# Global tracker
_tracker = LatencyTracker(window_size=5000)


def _setup_pipelines() -> None:
    preprocess_pipeline = (
        PipelineDAG("preprocess_and_infer")
        .add_stage("normalize", normalize_inputs)
        .add_stage("format",    format_output, depends_on=["normalize"])
    )
    register_pipeline(preprocess_pipeline)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 {settings.app_name} v{settings.app_version} starting...")
    if settings.enable_tracing:
        setup_tracing()
    registry = get_registry()
    registry.load_from_yaml("configs/models.yaml")
    engine = get_engine()
    await engine.startup()
    _setup_pipelines()
    yield
    print("🛑 Shutting down...")
    await get_engine().shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Framework-agnostic ML model serving platform",
        lifespan=lifespan,
    )

    # ── CORS ───────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Priority", "Authorization", "X-API-Key"],
    )

    # ── Routers ────────────────────────────────────────────
    app.include_router(health_router,  tags=["Health"])
    app.include_router(metrics_router, tags=["Observability"])

    # ── Inference ──────────────────────────────────────────

    @app.post("/infer", tags=["Inference"])
    async def infer(
        request: InferenceRequest,
        x_priority: Annotated[str | None, Header()] = None,
        api_key: str = Depends(verify_api_key),
    ):
        """
        Main inference endpoint।
        Header: X-API-Key (required if API_KEY_ENABLED=true)
        Header: X-Priority — urgent | normal | batch (optional override)
        """
        result = await get_engine().handle(
            request, priority_hint=x_priority
        )
        _tracker.record(RequestStat(
            latency_ms=result.latency_ms
                if hasattr(result, "latency_ms") else 0.0,
            status=result.status.value
                if hasattr(result, "status") else "error",
            model_name=request.model_name,
            priority=request.priority.value,
        ))
        return result

    # ── Pipeline ───────────────────────────────────────────

    @app.post("/pipeline/{pipeline_name}", tags=["Pipeline"])
    async def run_pipeline(
        pipeline_name: str,
        inputs: dict,
        api_key: str = Depends(verify_api_key),
    ):
        
        pipeline = get_pipeline(pipeline_name)
        results  = await pipeline.execute(inputs)
        return {"pipeline": pipeline_name, "results": results}

    # ── Optimizer ──────────────────────────────────────────

    @app.post("/optimize/{model_name}", tags=["Optimizer"])
    async def optimize_model(
        model_name: str,
        level: str = "standard",
        api_key: str = Depends(verify_api_key),
    ):
        registry = get_registry()
        try:
            info = registry.get(model_name)
        except KeyError:
            return {"error": f"Model '{model_name}' not found"}

        if info.runtime.value != "onnx":
            return {
                "error": "Only ONNX models support optimization currently"
            }

        optimizer = ModelOptimizer()
        result = await optimizer.optimize(info.model_path, level)
        return {
            "model_name"          : model_name,
            "optimization_level"  : result.optimization_level,
            "original_size_mb"    : result.original_size_mb,
            "optimized_size_mb"   : result.optimized_size_mb,
            "size_reduction_pct"  : result.size_reduction_pct,
            "optimization_time_s" : result.optimization_time_s,
            "from_cache"          : result.from_cache,
            "optimized_path"      : result.optimized_path,
        }

    # ── Registry ───────────────────────────────────────────

    @app.get("/models", tags=["Registry"])
    async def list_models():
        """Available models।"""
        return {"models": get_registry().list_models()}

    @app.get("/pipelines", tags=["Pipeline"])
    async def pipelines():
        """Available pipelines।"""
        return {"pipelines": list_pipelines()}

    # ── Scheduler ──────────────────────────────────────────

    @app.get("/queue/stats", tags=["Scheduler"])
    async def queue_stats():
        """Current queue depth + tenant counts।"""
        return get_engine().queue.stats

    # ── Shadow ─────────────────────────────────────────────

    @app.get("/shadow/summary", tags=["Shadow"])
    async def shadow_summary():
        """Champion vs challenger comparison।"""
        return get_shadow_manager().get_summary()

    # ── Benchmark ──────────────────────────────────────────

    @app.get("/benchmark/stats", tags=["Benchmark"])
    async def benchmark_stats():
        """Live P50 / P95 / P99 latency stats।"""
        return _tracker.summary()

    # ── Drift ──────────────────────────────────────────────

    @app.get("/drift/summary", tags=["Drift"])
    async def drift_summary():
        return get_drift_monitor().get_summary()

    @app.get("/drift/summary/{model_name}", tags=["Drift"])
    async def drift_summary_model(model_name: str):
        return get_drift_monitor().get_summary(model_name)

    @app.get("/drift/alerts", tags=["Drift"])
    async def drift_alerts():
        return {"alerts": get_drift_monitor().get_all_alerts()}

    return app