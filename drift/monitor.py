from drift.detector import DriftDetector, DriftAlert
from typing import Any
import asyncio


class DriftMonitor:
    def __init__(self):
        self._detectors: dict[str, DriftDetector] = {}

    def register_model(
        self,
        model_name: str,
        reference_window: int = 200,
    ) -> None:
        self._detectors[model_name] = DriftDetector(
            model_name=model_name,
            reference_window=reference_window,
        )
        print(f"👁️  Drift monitoring enabled: '{model_name}'")

    def record_output(
        self,
        model_name: str,
        outputs: dict[str, Any],
    ) -> DriftAlert | None:
        if model_name not in self._detectors:
            return None

        value = self._extract_value(outputs)
        if value is None:
            return None

        return self._detectors[model_name].record(value)

    def _extract_value(self, outputs: dict[str, Any]) -> float | None:
      
        try:
            if "output_label" in outputs:
                label = outputs["output_label"]
                if isinstance(label, list):
                    return float(label[0])
                return float(label)

            if "output_probability" in outputs:
                probs = outputs["output_probability"]
                if isinstance(probs, list) and probs:
                    if isinstance(probs[0], dict):
                        return float(max(probs[0].values()))
                    return float(max(probs[0]))

            for v in outputs.values():
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, list) and v:
                    return float(v[0])
        except Exception:
            pass
        return None

    def get_summary(self, model_name: str | None = None) -> dict:
        if model_name:
            if model_name not in self._detectors:
                return {"error": f"Model '{model_name}' not monitored"}
            return self._detectors[model_name].summary()

        return {
            name: detector.summary()
            for name, detector in self._detectors.items()
        }

    def get_all_alerts(self) -> list[dict]:
        alerts = []
        for detector in self._detectors.values():
            for alert in detector.alerts:
                alerts.append({
                    "model_name"  : alert.model_name,
                    "severity"    : alert.severity.value,
                    "p_value"     : alert.p_value,
                    "ks_statistic": alert.ks_statistic,
                    "timestamp"   : alert.timestamp,
                    "recommendation": alert.recommendation,
                })
        return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)

# Global singleton
_monitor = DriftMonitor()

def get_drift_monitor() -> DriftMonitor:
    return _monitor