from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

import pytest

from lankatax_ai.backend.memory import InMemoryConversationStore, SqlConversationStore, Turn
from lankatax_ai.backend.services.errors import NotFoundError

pytestmark = pytest.mark.unit


async def test_in_memory_round_trip_and_window():
    store = InMemoryConversationStore()
    conversation_id = await store.create(fingerprint="test")

    await store.append(
        conversation_id,
        [
            Turn(role="user", content="q1"),
            Turn(role="assistant", content="a1", sources=({"title": "doc"},), tokens_used=3),
            Turn(role="user", content="q2"),
            Turn(role="assistant", content="a2"),
        ],
    )

    assert [turn.content for turn in await store.load(conversation_id, limit=10)] == [
        "q1",
        "a1",
        "q2",
        "a2",
    ]
    assert [turn.content for turn in await store.load(conversation_id, limit=2)] == ["q2", "a2"]


async def test_conversations_are_isolated():
    store = InMemoryConversationStore()
    first = await store.create(fingerprint="a")
    second = await store.create(fingerprint="b")

    await store.append(first, [Turn(role="user", content="first")])
    await store.append(second, [Turn(role="user", content="second")])

    assert [turn.content for turn in await store.load(first, limit=5)] == ["first"]
    assert [turn.content for turn in await store.load(second, limit=5)] == ["second"]


async def test_unknown_conversation_is_not_found():
    with pytest.raises(NotFoundError):
        await InMemoryConversationStore().load("missing", limit=5)


def test_unknown_roles_are_rejected():
    with pytest.raises(ValueError, match="role"):
        Turn(role="system", content="not allowed")


@pytest.mark.db
async def test_sql_store_round_trip(clean_session):
    @asynccontextmanager
    async def factory():
        yield clean_session

    store = SqlConversationStore(factory)
    conversation_id = await store.create(fingerprint="local-user")

    await store.append(
        conversation_id,
        [
            Turn(role="user", content="How is PAYE withheld?"),
            Turn(
                role="assistant",
                content="Employers deduct it monthly [1].",
                sources=({"title": "PAYE guide", "url": "https://ird.gov.lk"},),
                tokens_used=42,
            ),
        ],
    )

    turns = await store.load(conversation_id, limit=10)

    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert turns[1].sources == ({"title": "PAYE guide", "url": "https://ird.gov.lk"},)
    assert turns[1].tokens_used == 42


@pytest.mark.db
async def test_sql_store_window_and_missing_conversation(clean_session):
    @asynccontextmanager
    async def factory():
        yield clean_session

    store = SqlConversationStore(factory)
    conversation_id = await store.create(fingerprint="local-user")
    await store.append(
        conversation_id,
        [Turn(role="user", content=f"q{index}") for index in range(4)],
    )

    assert [turn.content for turn in await store.load(conversation_id, limit=2)] == ["q2", "q3"]

    with pytest.raises(NotFoundError):
        await store.load(str(uuid4()), limit=5)
    with pytest.raises(NotFoundError):
        await store.load("not-a-uuid", limit=5)
