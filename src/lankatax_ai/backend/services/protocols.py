"""Protocols shared by more than one service."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession


class Embeddings(Protocol):
    async def aembed_query(self, text: str) -> list[float]: ...


SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]
