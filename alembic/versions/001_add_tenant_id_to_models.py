"""Add tenant_id to documents, chunks, and query_logs tables.

Revision ID: 001_add_tenant_id
Revises:
Create Date: 2026-01-19

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "001_add_tenant_id"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add tenant_id column to documents, chunks, and query_logs tables."""

    # Add tenant_id to documents table
    op.add_column(
        "documents",
        sa.Column(
            "tenant_id",
            sa.String(36),
            nullable=False,
            server_default="default",
        ),
    )
    op.create_index("ix_documents_tenant_id", "documents", ["tenant_id"])

    # Add tenant_id to chunks table
    op.add_column(
        "chunks",
        sa.Column(
            "tenant_id",
            sa.String(36),
            nullable=False,
            server_default="default",
        ),
    )
    op.create_index("ix_chunks_tenant_id", "chunks", ["tenant_id"])

    # Add tenant_id to query_logs table
    op.add_column(
        "query_logs",
        sa.Column(
            "tenant_id",
            sa.String(36),
            nullable=False,
            server_default="default",
        ),
    )
    op.create_index("ix_query_logs_tenant_id", "query_logs", ["tenant_id"])

    # Create composite indexes for common query patterns
    op.create_index(
        "ix_documents_tenant_status",
        "documents",
        ["tenant_id", "status"],
    )
    op.create_index(
        "ix_chunks_tenant_document",
        "chunks",
        ["tenant_id", "document_id"],
    )


def downgrade() -> None:
    """Remove tenant_id column from documents, chunks, and query_logs tables."""

    # Drop composite indexes first
    op.drop_index("ix_chunks_tenant_document", table_name="chunks")
    op.drop_index("ix_documents_tenant_status", table_name="documents")

    # Drop query_logs tenant_id
    op.drop_index("ix_query_logs_tenant_id", table_name="query_logs")
    op.drop_column("query_logs", "tenant_id")

    # Drop chunks tenant_id
    op.drop_index("ix_chunks_tenant_id", table_name="chunks")
    op.drop_column("chunks", "tenant_id")

    # Drop documents tenant_id
    op.drop_index("ix_documents_tenant_id", table_name="documents")
    op.drop_column("documents", "tenant_id")
