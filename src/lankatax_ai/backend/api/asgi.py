"""ASGI entry point.

Run with::

    uv run uvicorn lankatax_ai.backend.api.asgi:app --reload
"""

from __future__ import annotations

from lankatax_ai.backend.api.app import create_app

app = create_app()
