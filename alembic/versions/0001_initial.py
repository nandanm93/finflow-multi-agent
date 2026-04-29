"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-04-26 22:55:00
"""

from alembic import op
import sqlalchemy as sa


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "client_profiles",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("age", sa.Integer(), nullable=False),
        sa.Column("risk_aversion", sa.String(length=20), nullable=False),
        sa.Column("annual_income", sa.Float(), nullable=False),
        sa.Column("liquid_assets", sa.Float(), nullable=False),
        sa.Column("raw_profile_json", sa.Text(), nullable=False),
    )
    op.create_table(
        "conversation_sessions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(length=64), nullable=False, unique=True),
        sa.Column("client_id", sa.Integer(), sa.ForeignKey("client_profiles.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_conversation_sessions_session_id",
        "conversation_sessions",
        ["session_id"],
        unique=True,
    )
    op.create_table(
        "conversation_turns",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "session_id",
            sa.Integer(),
            sa.ForeignKey("conversation_sessions.id"),
            nullable=False,
        ),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("client_message", sa.Text(), nullable=False),
        sa.Column("advisor_response", sa.Text(), nullable=False),
        sa.Column("recommendation", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "analyst_tasks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("turn_id", sa.Integer(), sa.ForeignKey("conversation_turns.id"), nullable=False),
        sa.Column("external_task_id", sa.String(length=64), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("evidence_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_analyst_tasks_external_task_id",
        "analyst_tasks",
        ["external_task_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_analyst_tasks_external_task_id", table_name="analyst_tasks")
    op.drop_table("analyst_tasks")
    op.drop_table("conversation_turns")
    op.drop_index("ix_conversation_sessions_session_id", table_name="conversation_sessions")
    op.drop_table("conversation_sessions")
    op.drop_table("client_profiles")
