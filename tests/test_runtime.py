import asyncio
import pytest
from core.schemas import ModelInfo, RuntimeType
from runtimes import get_runtime

def test_factory_unknown_runtime():
    from pydantic import ValidationError
    from core.schemas import ModelInfo

    with pytest.raises(ValidationError):
        ModelInfo(
            name="test",
            runtime="unknown_framework", 
            model_path="fake.pt",
        )

def test_schema_deadline_auto_set():
    from core.schemas import InferenceRequest, Priority
    req = InferenceRequest(
        model_name="test_model",
        inputs={"data": [1.0, 2.0]},
        priority=Priority.URGENT,
    )
    assert req.deadline_ms == 100   # settings.sla_urgent_ms default

def test_schema_request_id_unique():
    from core.schemas import InferenceRequest
    r1 = InferenceRequest(model_name="m", inputs={})
    r2 = InferenceRequest(model_name="m", inputs={})
    assert r1.request_id != r2.request_id