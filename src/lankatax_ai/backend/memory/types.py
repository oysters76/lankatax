"""Conversation types shared by the memory adapters and the ask use case."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, Protocol

__all__ = ["ConversationStore", "Turn"]

_ROLES = frozenset({"user", "assistant"})


@dataclasses.dataclass(frozen=True, slots=True)
class Turn:
    """One stored message in a conversation."""

    role: str
    content: str
    sources: tuple[dict[str, Any], ...] = ()
    tokens_used: int = 0

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"role must be one of {sorted(_ROLES)}, got {self.role!r}")


class ConversationStore(Protocol):
    """Persistence for conversation history."""

    async def create(self, *, fingerprint: str) -> str: ...

    async def load(self, conversation_id: str, *, limit: int, owner: str) -> list[Turn]: ...

    async def append(self, conversation_id: str, turns: Sequence[Turn], *, owner: str) -> None: ...
