# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.12.21 AS uv

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app \
    && chown appuser:appuser /app

WORKDIR /app

COPY --from=uv /uv /uvx /bin/

COPY --chown=appuser:appuser pyproject.toml uv.lock README.md ./
COPY --chown=appuser:appuser src ./src

USER appuser
RUN uv sync --frozen --no-dev

COPY --chown=appuser:appuser prompts ./prompts
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser alembic ./alembic

EXPOSE 8000

# Ask quotas are per process; a shared limiter is required before scaling out.
CMD ["uvicorn", "lankatax_ai.backend.api.asgi:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
