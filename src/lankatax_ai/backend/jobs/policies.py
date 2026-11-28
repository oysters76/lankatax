"""Execution policies for Hatchet workflows."""

from __future__ import annotations

from datetime import timedelta

from lankatax_ai.backend.config import settings

SCRAPE_RETRIES = 3
INGEST_RETRIES = 3
CACHE_RETRIES = 2

SCRAPE_EXECUTION_TIMEOUT = timedelta(seconds=settings.job_scrape_timeout_seconds)
INGEST_EXECUTION_TIMEOUT = timedelta(seconds=settings.job_ingest_timeout_seconds)
CACHE_EXECUTION_TIMEOUT = timedelta(seconds=settings.job_cache_timeout_seconds)
SCHEDULE_TIMEOUT = timedelta(seconds=settings.job_schedule_timeout_seconds)

REFRESH_CRON = settings.job_refresh_cron
CACHE_EXPIRE_CRON = settings.job_cache_expire_cron
