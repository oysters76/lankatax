"""SQLAlchemy conversation store over the existing chat tables."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import select

from lankatax_ai.backend.db.models import ChatMessage, ChatSession
from lankatax_ai.backend.memory.types import Turn
from lankatax_ai.backend.services.errors import NotFoundError
from lankatax_ai.backend.services.protocols import SessionFactory

__all__ = ["SqlConversationStore"]


class SqlConversationStore:
    """Persist conversations in ``chat_sessions`` and ``chat_messages``."""

    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    async def create(self, *, fingerprint: str) -> str:
        async with self._session_factory() as session:
            conversation = ChatSession(fingerprint=fingerprint)
            session.add(conversation)
            await session.flush()
            conversation_id = conversation.id
            await session.commit()
        return str(conversation_id)

    async def load(self, conversation_id: str, *, limit: int, owner: str) -> list[Turn]:
        if limit <= 0:
            return []
        session_id = self._parse_id(conversation_id)
        async with self._session_factory() as session:
            conversation = await session.get(ChatSession, session_id)
            if conversation is None or conversation.fingerprint != owner:
                raise NotFoundError("Conversation not found")
            rows = (
                await session.scalars(
                    select(ChatMessage)
                    .where(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
                    .limit(limit)
                )
            ).all()
        return [
            Turn(
                role=row.role,
                content=row.content,
                sources=tuple(_as_sources(row.sources_json)),
                tokens_used=row.tokens_used or 0,
            )
            for row in reversed(rows)
        ]

    async def append(self, conversation_id: str, turns: Sequence[Turn], *, owner: str) -> None:
        if not turns:
            return
        session_id = self._parse_id(conversation_id)
        async with self._session_factory() as session:
            conversation = await session.get(ChatSession, session_id)
            if conversation is None or conversation.fingerprint != owner:
                raise NotFoundError("Conversation not found")
            # Server defaults share one timestamp per transaction, which would
            # make ordering arbitrary; give each turn a distinct instant.
            now = datetime.now(UTC)
            for index, turn in enumerate(turns):
                session.add(
                    ChatMessage(
                        session_id=session_id,
                        role=turn.role,
                        content=turn.content,
                        sources_json=list(turn.sources),
                        tokens_used=turn.tokens_used,
                        created_at=now + timedelta(microseconds=index),
                    )
                )
            conversation.message_count += len(turns)
            await session.commit()

    @staticmethod
    def _parse_id(conversation_id: str) -> UUID:
        try:
            return UUID(conversation_id)
        except ValueError as error:
            raise NotFoundError("Conversation not found") from error


def _as_sources(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
