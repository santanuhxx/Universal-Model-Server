from core.schemas import InferenceRequest, Priority
from core.config import get_settings

class SLAClassifier:
    def __init__(self):
        self.settings = get_settings()

    def classify(
        self,
        request: InferenceRequest,
        hint: str | None = None,
    ) -> InferenceRequest:
        
        if hint and hint.lower() in Priority._value2member_map_:
            request.priority = Priority(hint.lower())

        mapping = {
            Priority.URGENT: self.settings.sla_urgent_ms,
            Priority.NORMAL: self.settings.sla_normal_ms,
            Priority.BATCH:  self.settings.sla_batch_ms,
        }
        request.deadline_ms = mapping[request.priority]
        return request