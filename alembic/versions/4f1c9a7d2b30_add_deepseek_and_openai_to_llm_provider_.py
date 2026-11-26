"""add deepseek and openai to llm_provider_enum

Revision ID: 4f1c9a7d2b30
Revises: 9a5c45ba434f
Create Date: 2026-11-26 10:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4f1c9a7d2b30"
down_revision: str | Sequence[str] | None = "9a5c45ba434f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the providers the application already supports."""
    op.execute("ALTER TYPE llm_provider_enum ADD VALUE IF NOT EXISTS 'deepseek'")
    op.execute("ALTER TYPE llm_provider_enum ADD VALUE IF NOT EXISTS 'openai'")


def downgrade() -> None:
    """Postgres cannot remove a value from an enum type.

    Removing these values would require recreating the type and rewriting every
    row; rows logged with them would be invalid anyway. Left as a no-op.
    """
