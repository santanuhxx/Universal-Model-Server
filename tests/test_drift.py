import pytest
import numpy as np
from drift.detector import DriftDetector, DriftSeverity
from drift.monitor import DriftMonitor


def fill_reference(detector: DriftDetector, n: int = 200) -> None:
    rng = np.random.default_rng(42)
    for v in rng.normal(loc=0.5, scale=0.1, size=n):
        detector.record(float(v))


def test_reference_not_ready_initially():
    detector = DriftDetector("test_model", reference_window=200)
    assert detector.is_ready is False


def test_reference_ready_after_fill():
    detector = DriftDetector("test_model", reference_window=200)
    fill_reference(detector, 200)
    assert detector.is_ready is True


def test_no_drift_same_distribution():
    detector = DriftDetector("test_model", reference_window=200)
    fill_reference(detector, 200)

    rng = np.random.default_rng(99)
    alert = None
    for v in rng.normal(loc=0.5, scale=0.1, size=100):
        alert = detector.record(float(v))

    # Same distribution → no critical alert
    if alert:
        assert alert.severity != DriftSeverity.CRITICAL


def test_drift_detected_different_distribution():
    detector = DriftDetector(
        "test_model",
        reference_window=200,
        warning_threshold=0.05,
        critical_threshold=0.01,
    )
    fill_reference(detector, 200)

    # Completely different distribution — mean 5.0 vs 0.5
    rng = np.random.default_rng(7)
    alerts = []
    for v in rng.normal(loc=5.0, scale=0.1, size=100):
        alert = detector.record(float(v))
        if alert:
            alerts.append(alert)

    assert len(alerts) > 0, "Drift should be detected"
    assert alerts[0].severity in (
        DriftSeverity.WARNING, DriftSeverity.CRITICAL
    )


def test_monitor_extract_value():
    monitor = DriftMonitor()
    monitor.register_model("test_model", reference_window=10)

    # output_label format
    val = monitor._extract_value({"output_label": [1]})
    assert val == 1.0

    # output_probability format
    val = monitor._extract_value(
        {"output_probability": [{"0": 0.3, "1": 0.7}]}
    )
    assert val == 0.7


def test_monitor_summary_unknown_model():
    monitor = DriftMonitor()
    result = monitor.get_summary("nonexistent_model")
    assert "error" in result


def test_alert_has_recommendation():
    from drift.detector import DriftAlert, DriftSeverity
    alert = DriftAlert(
        model_name="test",
        severity=DriftSeverity.CRITICAL,
        p_value=0.001,
        ks_statistic=0.8,
        reference_size=200,
        current_size=100,
    )
    assert len(alert.recommendation) > 0
    assert "rolling back" in alert.recommendation.lower()