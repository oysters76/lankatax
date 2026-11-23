"""Small helpers shared by service modules."""

from __future__ import annotations

from lankatax_ai.backend.services.errors import ValidationError


def require_range(label: str, value: float, low: float, high: float) -> None:
    """Raise ``ValidationError`` unless ``low <= value <= high``.

    The message is generated here so every service reports bounds identically.
    """
    if not low <= value <= high:
        raise ValidationError(f"{label} must be {low}-{high}, got {value}")
