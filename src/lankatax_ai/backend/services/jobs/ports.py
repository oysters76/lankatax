"""Ports for the job use case.

The implementation lives in ``backend.jobs``. This module must stay free of any
orchestrator SDK.
"""

from __future__ import annotations

from typing import Protocol

from lankatax_ai.backend.services.jobs.types import (
    JobDetails,
    JobFilter,
    JobReceipt,
    JobRequest,
    JobStatus,
    JobSummary,
)

__all__ = ["JobGateway"]


class JobGateway(Protocol):
    """Execution backend used by the job service."""

    async def enqueue(self, request: JobRequest) -> JobReceipt: ...

    async def list_jobs(self, filters: JobFilter) -> list[JobSummary]: ...

    async def get(self, run_id: str) -> JobDetails: ...

    async def status(self, run_id: str) -> JobStatus: ...

    async def cancel(self, run_id: str) -> None: ...

    async def replay(self, run_id: str) -> None: ...
