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
        self._owners: dict[str, str] = {}

    async def create(self, *, fingerprint: str) -> str:
        conversation_id = str(uuid4())
        self._turns[conversation_id] = []
        self._owners[conversation_id] = fingerprint
        return conversation_id

    async def load(self, conversation_id: str, *, limit: int, owner: str) -> list[Turn]:
        self._require_owner(conversation_id, owner)
        if limit <= 0:
            return []
        return list(self._turns[conversation_id][-limit:])

    async def append(self, conversation_id: str, turns: Sequence[Turn], *, owner: str) -> None:
        self._require_owner(conversation_id, owner)
        self._turns[conversation_id].extend(turns)

    def _require_owner(self, conversation_id: str, owner: str) -> None:
        if self._owners.get(conversation_id) != owner:
            raise NotFoundError("Conversation not found")
