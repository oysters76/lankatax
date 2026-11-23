from __future__ import annotations

import pytest

from lankatax_ai.backend.services.errors import ValidationError
from lankatax_ai.backend.services.utils import require_range

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("label", "value", "low", "high"),
    [
        ("Top-k", 1, 1, 50),
        ("Top-k", 50, 1, 50),
        ("Vector weight", 0.0, 0.0, 1.0),
        ("Vector weight", 0.7, 0.0, 1.0),
        ("Vector weight", 1.0, 0.0, 1.0),
    ],
)
def test_require_range_accepts_inclusive_bounds(label, value, low, high):
    require_range(label, value, low, high)


@pytest.mark.parametrize(
    ("label", "value", "low", "high", "message"),
    [
        ("Top-k", 0, 1, 50, "Top-k must be 1-50, got 0"),
        ("Top-k", 51, 1, 50, "Top-k must be 1-50, got 51"),
        ("Vector weight", -0.1, 0.0, 1.0, "Vector weight must be 0.0-1.0, got -0.1"),
    ],
)
def test_require_range_rejects_and_reports_bounds(label, value, low, high, message):
    with pytest.raises(ValidationError, match=message):
        require_range(label, value, low, high)
