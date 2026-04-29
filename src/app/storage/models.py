from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.storage.db import Base


class ClientProfileORM(Base):
    __tablename__ = "client_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120))
    age: Mapped[int] = mapped_column(Integer)
    risk_aversion: Mapped[str] = mapped_column(String(20))
    annual_income: Mapped[float] = mapped_column(Float)
    liquid_assets: Mapped[float] = mapped_column(Float)
    raw_profile_json: Mapped[str] = mapped_column(Text)

    sessions: Mapped[list["ConversationSessionORM"]] = relationship(back_populates="client")


class ConversationSessionORM(Base):
    __tablename__ = "conversation_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    client_id: Mapped[int] = mapped_column(ForeignKey("client_profiles.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    client: Mapped[ClientProfileORM] = relationship(back_populates="sessions")
    turns: Mapped[list["ConversationTurnORM"]] = relationship(back_populates="session")


class ConversationTurnORM(Base):
    __tablename__ = "conversation_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("conversation_sessions.id"))
    turn_index: Mapped[int] = mapped_column(Integer)
    client_message: Mapped[str] = mapped_column(Text)
    advisor_response: Mapped[str] = mapped_column(Text)
    recommendation: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    session: Mapped[ConversationSessionORM] = relationship(back_populates="turns")
    analyst_tasks: Mapped[list["AnalystTaskORM"]] = relationship(back_populates="turn")


class AnalystTaskORM(Base):
    __tablename__ = "analyst_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    turn_id: Mapped[int] = mapped_column(ForeignKey("conversation_turns.id"))
    external_task_id: Mapped[str] = mapped_column(String(64), index=True)
    question: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)
    evidence_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    turn: Mapped[ConversationTurnORM] = relationship(back_populates="analyst_tasks")
