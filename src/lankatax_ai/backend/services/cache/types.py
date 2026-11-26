"""Types for the cache use case.

The result type and the admin port live with the cache store, because the store
and its callers share them. This module only adds the wiring the use case needs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from lankatax_ai.backend.cache.types import CacheAdmin, CacheStats

__all__ = ["CacheAdmin", "CacheAdminFactory", "CacheDeps", "CacheStats"]


CacheAdminFactory = Callable[[AsyncSession], CacheAdmin]


@dataclasses.dataclass(frozen=True, slots=True)
class CacheDeps:
    """Collaborators the cache use case needs, wired by the caller."""

    admin_factory: CacheAdminFactory
