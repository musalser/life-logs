import pytest

from app.htr.application.confidence import ConfidencePolicy
from app.htr.domain.entities import ConfidenceLevel

POLICY = ConfidencePolicy(warning_threshold=0.90, critical_threshold=0.70)


@pytest.mark.parametrize(
    "confidence,expected",
    [
        (1.0, ConfidenceLevel.NORMAL),
        (0.95, ConfidenceLevel.NORMAL),
        (0.90, ConfidenceLevel.NORMAL),  # boundary: >= warning
        (0.8999, ConfidenceLevel.WARNING),
        (0.80, ConfidenceLevel.WARNING),
        (0.70, ConfidenceLevel.WARNING),  # boundary: >= critical
        (0.6999, ConfidenceLevel.CRITICAL),
        (0.0, ConfidenceLevel.CRITICAL),
        (None, ConfidenceLevel.CRITICAL),
    ],
)
def test_classification_boundaries(confidence, expected):
    assert POLICY.classify(confidence) == expected


def test_thresholds_are_configurable():
    policy = ConfidencePolicy(warning_threshold=0.5, critical_threshold=0.2)
    assert policy.classify(0.5) == ConfidenceLevel.NORMAL
    assert policy.classify(0.3) == ConfidenceLevel.WARNING
    assert policy.classify(0.1) == ConfidenceLevel.CRITICAL


def test_invalid_threshold_order_rejected():
    with pytest.raises(ValueError):
        ConfidencePolicy(warning_threshold=0.5, critical_threshold=0.9)
