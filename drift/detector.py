import time
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import numpy as np


class DriftSeverity(str, Enum):
    NONE     = "none"
    WARNING  = "warning"    # p-value < 0.05
    CRITICAL = "critical"   # p-value < 0.01


@dataclass
class DriftAlert:
    model_name: str
    severity: DriftSeverity
    p_value: float
    ks_statistic: float
    reference_size: int
    current_size: int
    timestamp: float = field(default_factory=time.time)
    recommendation: str = ""

    def __post_init__(self):
        if self.severity == DriftSeverity.CRITICAL:
            self.recommendation = (
                "🚨 Immediate action required! "
                "Consider rolling back to previous model version."
            )
        elif self.severity == DriftSeverity.WARNING:
            self.recommendation = (
                "⚠️ Monitor closely. "
                "Shadow deploy new model version for comparison."
            )


class DriftDetector:
    def __init__(
        self,
        model_name: str,
        reference_window: int = 200,
        current_window: int = 100,
        warning_threshold: float = 0.05,
        critical_threshold: float = 0.01,
    ):
        self.model_name = model_name
        self.warning_threshold  = warning_threshold
        self.critical_threshold = critical_threshold

        self._reference: deque = deque(maxlen=reference_window)
        self._current:   deque = deque(maxlen=current_window)
        self._alerts: list[DriftAlert] = []
        self._is_reference_ready = False
        self._reference_window = reference_window

    def record(self, output_value: float) -> DriftAlert | None:
      
        if not self._is_reference_ready:
            self._reference.append(output_value)
            if len(self._reference) >= self._reference_window:
                self._is_reference_ready = True
                print(
                    f"📊 [{self.model_name}] "
                    f"Reference distribution ready "
                    f"({self._reference_window} samples)"
                )
            return None

        self._current.append(output_value)

        if len(self._current) >= 50:
            return self._run_ks_test()

        return None

    def _run_ks_test(self) -> DriftAlert | None:
        from scipy import stats

        ref = np.array(list(self._reference))
        cur = np.array(list(self._current))

        ks_stat, p_value = stats.ks_2samp(ref, cur)

        severity = DriftSeverity.NONE
        if p_value < self.critical_threshold:
            severity = DriftSeverity.CRITICAL
        elif p_value < self.warning_threshold:
            severity = DriftSeverity.WARNING

        if severity != DriftSeverity.NONE:
            alert = DriftAlert(
                model_name=self.model_name,
                severity=severity,
                p_value=round(float(p_value), 6),
                ks_statistic=round(float(ks_stat), 4),
                reference_size=len(ref),
                current_size=len(cur),
            )
            self._alerts.append(alert)
            self._log_alert(alert)
            return alert

        return None

    def _log_alert(self, alert: DriftAlert) -> None:
        icon = "🚨" if alert.severity == DriftSeverity.CRITICAL else "⚠️"
        print(
            f"{icon} DRIFT DETECTED [{alert.model_name}] "
            f"severity={alert.severity.value} "
            f"p={alert.p_value:.4f} "
            f"ks={alert.ks_statistic:.4f}"
        )
        print(f"   → {alert.recommendation}")

    @property
    def is_ready(self) -> bool:
        return self._is_reference_ready

    @property
    def alerts(self) -> list[DriftAlert]:
        return self._alerts

    def summary(self) -> dict:
        return {
            "model_name"       : self.model_name,
            "is_ready"         : self.is_ready,
            "reference_samples": len(self._reference),
            "current_samples"  : len(self._current),
            "total_alerts"     : len(self._alerts),
            "critical_alerts"  : sum(
                1 for a in self._alerts
                if a.severity == DriftSeverity.CRITICAL
            ),
            "warning_alerts"   : sum(
                1 for a in self._alerts
                if a.severity == DriftSeverity.WARNING
            ),
            "latest_alert"     : {
                "severity"    : self._alerts[-1].severity.value,
                "p_value"     : self._alerts[-1].p_value,
                "ks_statistic": self._alerts[-1].ks_statistic,
                "timestamp"   : self._alerts[-1].timestamp,
                "recommendation": self._alerts[-1].recommendation,
            } if self._alerts else None,
        }