"""Wiring for the job use case."""

from __future__ import annotations

import dataclasses

from lankatax_ai.backend.services.jobs.ports import JobGateway

__all__ = ["JobDeps"]


@dataclasses.dataclass(frozen=True, slots=True)
class JobDeps:
    """Collaborators the job use case needs, wired by the caller."""

    gateway: JobGateway
