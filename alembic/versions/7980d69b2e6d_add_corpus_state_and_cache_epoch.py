"""add corpus state and cache epoch

Revision ID: 7980d69b2e6d
Revises: 4f1c9a7d2b30
Create Date: 2026-11-27 15:18:32.068699

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7980d69b2e6d"
down_revision: str | Sequence[str] | None = "4f1c9a7d2b30"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "corpus_state",
        sa.Column("id", sa.Integer(), server_default="1", nullable=False),
        sa.Column("epoch", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_ingest_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("last_refresh_run_id", sa.Text(), nullable=True),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("id = 1", name="ck_corpus_state_singleton"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute("INSERT INTO corpus_state (id, epoch) VALUES (1, 0)")

    op.add_column(
        "cache_entries",
        sa.Column("corpus_epoch", sa.Integer(), server_default="0", nullable=False),
    )
    op.create_index(
        "idx_cache_entries_epoch_created",
        "cache_entries",
        ["corpus_epoch", "created_at"],
        unique=False,
    )

    op.drop_table("scrape_runs")


def downgrade() -> None:
    """Downgrade schema."""
    op.create_table(
        "scrape_runs",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column(
            "started_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("pages_scraped", sa.Integer(), server_default="0", nullable=False),
        sa.Column("pages_updated", sa.Integer(), server_default="0", nullable=False),
        sa.Column("pages_failed", sa.Integer(), server_default="0", nullable=False),
        sa.Column("status", sa.Text(), server_default="running", nullable=False),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed')", name="ck_scrape_runs_status"
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.drop_index("idx_cache_entries_epoch_created", table_name="cache_entries")
    op.drop_column("cache_entries", "corpus_epoch")
    op.drop_table("corpus_state")
