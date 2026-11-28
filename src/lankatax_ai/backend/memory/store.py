"""In-process conversation store for tests and single-process runs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from uuid import uuid4

from lankatax_ai.backend.memory.types import Turn
from lankatax_ai.backend.services.errors import NotFoundError

__all__ = ["InMemoryConversationStore"]


class InMemoryConversationStore:
    """Keep turns in a dict keyed by conversation id."""

    def __init__(self) -> None:
        self._turns: dict[str, list[Turn]] = defaultdict(list)

    async def create(self, *, fingerprint: str) -> str:
        conversation_id = str(uuid4())
        self._turns[conversation_id] = []
        return conversation_id

    async def load(self, conversation_id: str, *, limit: int) -> list[Turn]:
        if conversation_id not in self._turns:
            raise NotFoundError("Conversation not found")
        if limit <= 0:
            return []
        return list(self._turns[conversation_id][-limit:])

    async def append(self, conversation_id: str, turns: Sequence[Turn]) -> None:
        self._turns[conversation_id].extend(turns)
