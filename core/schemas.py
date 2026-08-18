from pydantic import BaseModel, ConfigDict, Field
from typing import Any
from enum import Enum
import time
import uuid

# ── Enums ──────────────────────────────────────────

class Priority(str, Enum):
    URGENT = "urgent"
    NORMAL = "normal"
    BATCH  = "batch"

class RuntimeType(str, Enum):
    PYTORCH = "pytorch"
    ONNX    = "onnx"
    JAX     = "jax"

class RequestStatus(str, Enum):
    QUEUED     = "queued"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"
    TIMEOUT    = "timeout"

# ── Request ────────────────────────────────────────

class InferenceRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = Field(..., description="Name of the registered model in the registry")
    inputs: dict[str, Any] = Field(..., description="Input data for the model")
    priority: Priority = Priority.NORMAL
    deadline_ms: int | None = Field(None, description="Response required within this time limit")
    tenant_id: str = Field("default", description="Team or tenant making the request")
    pipeline_id: str | None = Field(None, description="Pipeline ID to execute if needed")

    def model_post_init(self, __context: Any) -> None:
        from core.config import get_settings
        settings = get_settings()
        if self.deadline_ms is None:
            mapping = {
                Priority.URGENT: settings.sla_urgent_ms,
                Priority.NORMAL: settings.sla_normal_ms,
                Priority.BATCH:  settings.sla_batch_ms,
            }
            self.deadline_ms = mapping[self.priority]

    @property
    def absolute_deadline(self) -> float:
        return time.time() + (self.deadline_ms / 1000)

# ── Response ───────────────────────────────────────

class InferenceResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    model_name: str
    outputs: dict[str, Any]
    status: RequestStatus = RequestStatus.DONE
    latency_ms: float
    tenant_id: str
    served_by: str = "unknown"

class ErrorResponse(BaseModel):
    request_id: str
    status: RequestStatus
    error: str
    latency_ms: float

# ── Model Registry ─────────────────────────────────

class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    runtime: RuntimeType
    model_path: str
    device: str = "cpu"
    max_batch_size: int = 32
    shadow_model: str | None = None