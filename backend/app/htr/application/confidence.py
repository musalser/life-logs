from __future__ import annotations

from ..domain.entities import ConfidenceLevel


class ConfidencePolicy:
    """Classifies word confidence into UI levels; thresholds are configurable."""

    def __init__(self, warning_threshold: float, critical_threshold: float):
        if critical_threshold > warning_threshold:
            raise ValueError("critical_threshold must not exceed warning_threshold")
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def classify(self, confidence: float | None) -> ConfidenceLevel:
        if confidence is None:
            return ConfidenceLevel.CRITICAL
        if confidence >= self.warning_threshold:
            return ConfidenceLevel.NORMAL
        if confidence >= self.critical_threshold:
            return ConfidenceLevel.WARNING
        return ConfidenceLevel.CRITICAL
