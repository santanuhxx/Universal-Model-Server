import time
from core.schemas import InferenceRequest, InferenceResponse, ErrorResponse, RequestStatus
from core.registry import get_registry
from runtimes import get_runtime
from scheduler.priority_queue import SLAPriorityQueue
from scheduler.tenant_limiter import TenantLimiter
from scheduler.sla_classifier import SLAClassifier
from observability.metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, QUEUE_SIZE, LOADED_MODELS
)
from observability.tracing import get_tracer
from optimizer.engine import ModelOptimizer
from drift.monitor import get_drift_monitor


class RouterEngine:
    def __init__(self):
        self.queue        = SLAPriorityQueue()
        self.limiter      = TenantLimiter()
        self.classifier   = SLAClassifier()
        self.registry     = get_registry()
        self._runtimes: dict = {}
        self._tracer      = get_tracer()
        self._optimizer   = ModelOptimizer()
        self._drift       = get_drift_monitor()

    async def startup(self) -> None:
        print("🔄 Loading registered models...")
        for name in self.registry.list_models():
            info = self.registry.get(name)
            try:
                optimized_path = info.model_path
                if info.runtime.value == "onnx":
                    result = await self._optimizer.optimize(
                        info.model_path,
                        optimization_level="standard",
                    )
                    optimized_path = result.optimized_path
                    print(f"   {result.summary()}")

                optimized_info = info.model_copy(
                    update={"model_path": optimized_path}
                )
                runtime = get_runtime(optimized_info)
                await runtime.load()
                runtime.is_loaded = True
                self._runtimes[name] = runtime

                self._drift.register_model(name)
                print(f"✅ Loaded: {name}")

            except Exception as e:
                print(f"⚠️  Skipped '{name}': {e}")

        LOADED_MODELS.set(len(self._runtimes))
        print(f"✅ {len(self._runtimes)} model(s) ready.")

    async def shutdown(self) -> None:
        for name, runtime in self._runtimes.items():
            await runtime.unload()
            print(f"🗑️  Unloaded: {name}")

    async def handle(
        self,
        request: InferenceRequest,
        priority_hint: str | None = None,
    ) -> InferenceResponse | ErrorResponse:
        start = time.perf_counter()

        with self._tracer.start_as_current_span("inference") as span:
            span.set_attribute("model_name", request.model_name)
            span.set_attribute("tenant_id",  request.tenant_id)
            span.set_attribute("priority",   request.priority.value)

            # 1. Rate limit
            if not self.limiter.is_allowed(request.tenant_id):
                REQUEST_COUNT.labels(
                    model_name=request.model_name,
                    tenant_id=request.tenant_id,
                    status="rate_limited",
                ).inc()
                return ErrorResponse(
                    request_id=request.request_id,
                    status=RequestStatus.FAILED,
                    error=f"Rate limit exceeded for '{request.tenant_id}'",
                    latency_ms=0.0,
                )

            # 2. SLA classify
            request = self.classifier.classify(request, hint=priority_hint)

            # 3. Model check
            if request.model_name not in self._runtimes:
                return ErrorResponse(
                    request_id=request.request_id,
                    status=RequestStatus.FAILED,
                    error=f"Model '{request.model_name}' not loaded.",
                    latency_ms=0.0,
                )

            # 4. Deadline check
            QUEUE_SIZE.set(self.queue.size)
            if time.time() > request.absolute_deadline:
                REQUEST_COUNT.labels(
                    model_name=request.model_name,
                    tenant_id=request.tenant_id,
                    status="timeout",
                ).inc()
                return ErrorResponse(
                    request_id=request.request_id,
                    status=RequestStatus.TIMEOUT,
                    error="Request expired before processing.",
                    latency_ms=round(
                        (time.perf_counter() - start) * 1000, 2
                    ),
                )

            # 5. Inference
            try:
                runtime = self._runtimes[request.model_name]
                outputs = await runtime.predict(request.inputs)
                latency = round((time.perf_counter() - start) * 1000, 2)

                # Drift monitoring
                alert = self._drift.record_output(
                    request.model_name, outputs
                )
                if alert:
                    outputs["_drift_alert"] = {
                        "severity": alert.severity.value,
                        "p_value" : alert.p_value,
                    }

                REQUEST_COUNT.labels(
                    model_name=request.model_name,
                    tenant_id=request.tenant_id,
                    status="success",
                ).inc()
                REQUEST_LATENCY.labels(
                    model_name=request.model_name,
                    priority=request.priority.value,
                ).observe(latency)
                span.set_attribute("latency_ms", latency)

                return InferenceResponse(
                    request_id=request.request_id,
                    model_name=request.model_name,
                    outputs=outputs,
                    latency_ms=latency,
                    tenant_id=request.tenant_id,
                    served_by=request.model_name,
                )

            except Exception as e:
                REQUEST_COUNT.labels(
                    model_name=request.model_name,
                    tenant_id=request.tenant_id,
                    status="error",
                ).inc()
                return ErrorResponse(
                    request_id=request.request_id,
                    status=RequestStatus.FAILED,
                    error=str(e),
                    latency_ms=round(
                        (time.perf_counter() - start) * 1000, 2
                    ),
                )